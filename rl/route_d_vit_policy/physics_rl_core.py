"""
Physics-guided RL regularization core.
Route D: Physics-guided PPO with ViT-policy.

Migrated from CIFAR-100 ViT-Small (v17) physics regularization:
  - Spectral dose-response D4 (time-axis rotation) -> D5 (step-based, patch-dim FFT)
  - Matern kernel weight smoothing -> Matern value function smoothing (EMA target)
  - Feature Lipschitz loss -> attention spectral penalty
  - Mean-teacher EMA -> EMA target network for RL
  - sigma_max annealing -> field-alpha release cycle

Adapted for RL:
  - No teacher (supervised distillation) -> EMA target network (mean-teacher)
  - Epoch-based scheduling -> step-based scheduling
  - Spatial feature maps (CNN) -> patch sequence features (ViT, dim=1 FFT)
  - Fixed teacher logits -> value function + policy gradient spectral analysis

Components:
  1. spectral_decomposition: FFT on patch dim -> coarse/medium/fine
  2. spectral_weights_rl: D5 time-axis rotation (step-based, dose-neutral)
  3. feature_diffusion_loss: spectral-weighted MSE vs EMA target
  4. matern_value_loss: Matern nu=2.5 kernel on value predictions
  5. attention_spectral_penalty: Lipschitz proxy on attention matrices
  6. entropy_schedule: quench + temper (cosine)
  7. field_alpha_schedule: physics strength release cycle
  8. PhysicsRegularizer: wrapper computing total physics loss
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def spectral_decomposition(
    features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose ViT patch features into coarse/medium/fine bands.

    FFT along patch dimension (dim=1) of (B, n_patches, dim).
    Coarse = lowest 1/3 frequencies (global scene structure).
    Medium = middle 1/3 (regional patterns).
    Fine = highest 1/3 (per-patch details).

    Direct adaptation from CIFAR-100 spatial FFT -> ViT patch-sequence FFT.
    """
    n = features.shape[1]
    freq = torch.fft.fft(features, dim=1)
    third = max(n // 3, 1)

    # Masks shaped (1, n, 1) for broadcasting with freq (B, n, dim)
    mask_c = torch.zeros(n, device=features.device, dtype=torch.complex64)
    mask_c[:third] = 1.0
    mask_c = mask_c.view(1, -1, 1)
    mask_m = torch.zeros(n, device=features.device, dtype=torch.complex64)
    mask_m[third : 2 * third] = 1.0
    mask_m = mask_m.view(1, -1, 1)
    mask_f = torch.zeros(n, device=features.device, dtype=torch.complex64)
    mask_f[2 * third :] = 1.0
    mask_f = mask_f.view(1, -1, 1)

    coarse = torch.fft.ifft(freq * mask_c, dim=1).real
    medium = torch.fft.ifft(freq * mask_m, dim=1).real
    fine = torch.fft.ifft(freq * mask_f, dim=1).real
    return coarse, medium, fine


def spectral_weights_rl(
    step: int,
    total_steps: int,
    w_start: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    w_end: Tuple[float, float, float] = (0.2, 0.3, 0.5),
    rot_start_frac: float = 0.5,
    rot_end_frac: float = 0.93,
) -> Tuple[float, float, float]:
    """D5 spectral dose-response: time-axis rotation for RL.

    Same principle as CIFAR-100 D4: start coarse-heavy, end fine-heavy.
    Adapted: epoch-based -> step-based; e151/e279 -> 50%/93% of total_steps.

    Dose-neutral: sum of weights = 1 at all times (simplex interpolation).
    """
    rot_start = int(total_steps * rot_start_frac)
    rot_end = int(total_steps * rot_end_frac)
    if step <= rot_start:
        t = 0.0
    elif step >= rot_end:
        t = 1.0
    else:
        t = (step - rot_start) / max(rot_end - rot_start, 1)
    return tuple(w_start[i] * (1 - t) + w_end[i] * t for i in range(3))


def feature_diffusion_loss(
    features: torch.Tensor,
    features_ema: torch.Tensor,
    weights: Tuple[float, float, float],
) -> torch.Tensor:
    """Spectral-weighted feature diffusion loss vs EMA target.

    CIFAR-100: student vs teacher logits, band-wise weighted.
    RL: current model vs EMA target, band-wise weighted.

    This is the core spectral regularization: it encourages the model to
    learn global structure (coarse) first, then refine details (fine),
    following the dose-response schedule.
    """
    c_p, m_p, f_p = spectral_decomposition(features)
    c_e, m_e, f_e = spectral_decomposition(features_ema)
    loss = (
        F.mse_loss(c_p, c_e.detach()) * weights[0]
        + F.mse_loss(m_p, m_e.detach()) * weights[1]
        + F.mse_loss(f_p, f_e.detach()) * weights[2]
    )
    return loss


def matern_value_loss(
    value_pred: torch.Tensor,
    value_ema: torch.Tensor,
    nu: float = 2.5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Matern kernel value smoothing loss.

    CIFAR-100: Matern kernel on weight space (teacher-student smoothing).
    RL: Matern kernel on value predictions (temporal smoothing via EMA target).

    The Matern nu=2.5 kernel provides smooth (twice-differentiable) distance
    weighting, encouraging value predictions to track the EMA target smoothly.
    """
    diff = value_pred - value_ema.detach()
    r = torch.abs(diff) / sigma
    if nu == 2.5:
        s5 = math.sqrt(5.0)
        kern = (1 + s5 * r + 5 * r**2 / 3) * torch.exp(-s5 * r)
    elif nu == 1.5:
        s3 = math.sqrt(3.0)
        kern = (1 + s3 * r) * torch.exp(-s3 * r)
    else:
        kern = torch.exp(-r)
    return (1 - kern).mean()


def attention_spectral_penalty(
    attn_weights: torch.Tensor,
    bound: float = 1.0,
) -> torch.Tensor:
    """Lipschitz proxy: penalize high spectral norm of attention matrices.

    CIFAR-100: feature Lipschitz loss constrains CNN feature map Lipschitz.
    RL with ViT: penalize attention matrices with high spectral norm
    (Frobenius norm as upper bound for computational efficiency).

    attn_weights: (B, n_heads, n_patches, n_patches)
    """
    n = attn_weights.shape[-1]
    frob = torch.norm(attn_weights.flatten(-2), p="fro", dim=-1)
    # Soft-threshold: penalize excess over bound * sqrt(n_patches)
    threshold = bound * math.sqrt(n)
    return torch.relu(frob - threshold).mean()


def entropy_schedule(
    step: int,
    total_steps: int,
    start_coef: float = 0.01,
    end_coef: float = 0.005,
    quench_start_frac: float = 0.7,
    quench_end_frac: float = 0.9,
) -> float:
    """Entropy quench + temper.

    Quench: gradually reduce entropy bonus from start to end.
    Temper: cosine smooth transition (not hard cutoff).
    """
    qs = int(total_steps * quench_start_frac)
    qe = int(total_steps * quench_end_frac)
    if step <= qs:
        return start_coef
    if step >= qe:
        return end_coef
    t = (step - qs) / max(qe - qs, 1)
    t_smooth = 0.5 * (1 - math.cos(math.pi * t))
    return start_coef * (1 - t_smooth) + end_coef * t_smooth


def field_alpha_schedule(
    step: int,
    total_steps: int,
    peak_alpha: float = 1.0,
    warmup_frac: float = 0.05,
    peak_frac: float = 0.86,
) -> float:
    """Field-alpha release cycle: physics regularization strength.

    Phase 1 (warmup): 0 -> peak_alpha over warmup_frac of training.
    Phase 2 (plateau): peak_alpha until peak_frac.
    Phase 3 (release): peak_alpha -> 0 over remaining steps.

    Adapted from PPO v32: W_PHYS_PEAK=4.3M (86% of 5M), warmup=250K.
    """
    warmup_end = max(int(total_steps * warmup_frac), 1)
    peak_step = int(total_steps * peak_frac)
    if step <= warmup_end:
        return peak_alpha * step / warmup_end
    if step <= peak_step:
        return peak_alpha
    remaining = max(total_steps - peak_step, 1)
    return peak_alpha * max(1.0 - (step - peak_step) / remaining, 0.0)


class PhysicsRegularizer:
    """Wrapper computing total physics regularization loss.

    Usage in PPO update loop:
        reg = PhysicsRegularizer(total_steps=5_000_000)
        physics_loss, metrics = reg(model, ema_target, obs_batch, step)
        total_loss = ppo_loss + field_alpha * physics_loss
    """

    def __init__(
        self,
        total_steps: int,
        matern_nu: float = 2.5,
        matern_sigma: float = 1.0,
        lipschitz_bound: float = 1.0,
        w_start: Tuple[float, float, float] = (0.5, 0.3, 0.2),
        w_end: Tuple[float, float, float] = (0.2, 0.3, 0.5),
        feature_coef: float = 1.0,
        matern_coef: float = 0.5,
        lipschitz_coef: float = 0.1,
    ):
        self.total_steps = total_steps
        self.matern_nu = matern_nu
        self.matern_sigma = matern_sigma
        self.lipschitz_bound = lipschitz_bound
        self.w_start = w_start
        self.w_end = w_end
        self.feature_coef = feature_coef
        self.matern_coef = matern_coef
        self.lipschitz_coef = lipschitz_coef

    def __call__(
        self,
        model,
        ema_target,
        observations: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total physics loss.

        Args:
            model: ViTActorCritic (current policy)
            ema_target: ViTActorCritic (EMA target, no grad)
            observations: (B, 4, 84, 84)
            step: current training step

        Returns:
            total_physics_loss: scalar
            metrics: dict of component losses + weights + alpha
        """
        alpha = field_alpha_schedule(step, self.total_steps)
        weights = spectral_weights_rl(
            step, self.total_steps, self.w_start, self.w_end
        )

        # Current model features
        cur = model.get_physics_features(observations)
        # EMA target features (no grad)
        with torch.no_grad():
            ema = ema_target.get_physics_features(observations)

        # 1. Spectral-weighted feature diffusion (D5)
        # Average across transformer layers
        n_layers = min(len(cur["features"]), len(ema["features"]))
        feat_loss = 0.0
        for i in range(n_layers):
            feat_loss = feat_loss + feature_diffusion_loss(
                cur["features"][i], ema["features"][i], weights
            )
        feat_loss = feat_loss / n_layers

        # 2. Matern value smoothing
        mat_loss = matern_value_loss(
            cur["value"], ema["value"], self.matern_nu, self.matern_sigma
        )

        # 3. Attention spectral penalty (Lipschitz proxy)
        lip_loss = 0.0
        n_attn = min(len(cur["attn_weights"]), len(ema["attn_weights"]))
        for i in range(n_attn):
            lip_loss = lip_loss + attention_spectral_penalty(
                cur["attn_weights"][i], self.lipschitz_bound
            )
        lip_loss = lip_loss / max(n_attn, 1)

        total = (
            self.feature_coef * feat_loss
            + self.matern_coef * mat_loss
            + self.lipschitz_coef * lip_loss
        )

        metrics = {
            "physics_total": total.item(),
            "physics_feature": feat_loss.item(),
            "physics_matern": mat_loss.item(),
            "physics_lipschitz": lip_loss.item(),
            "field_alpha": alpha,
            "spec_w_c": weights[0],
            "spec_w_m": weights[1],
            "spec_w_f": weights[2],
        }
        return total, metrics
