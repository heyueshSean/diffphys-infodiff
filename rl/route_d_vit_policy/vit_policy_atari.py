"""
ViT-Small Actor-Critic for Atari Breakout.
Route D: Physics-guided PPO with ViT-policy.

Migrated from CIFAR-100 ViT-Small (v17):
  - BlurPool anti-aliased patch embedding (from CIFAR BlurPool)
  - StochasticDepth per-sample (p=0.1, from CIFAR 16-block SD)
  - 6-layer transformer, 384 dim, 6 heads, MLP ratio 4
  - Mean pooling (no CLS token, simpler for RL)
  - Dual heads: policy logits + value function

Adapted for RL:
  - Input: (B, 4, 84, 84) — 4 stacked grayscale frames
  - Patch size 14 -> 6x6 = 36 patches
  - Orthogonal init for policy/value heads (RL convention)
  - Physics hooks: per-layer features + attention weights

Integration:
  - forward() returns (policy_logits, value) — compatible with any PPO
  - forward_with_physics() returns (features, attn_weights) — for PhysicsRegularizer
  - init_ema_target() / update_ema() — mean-teacher for spectral regularization
"""

import copy
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlurPool(nn.Module):
    """Anti-aliased pooling (Zhang 2019), migrated from CIFAR BlurPool.

    Applies a binomial blur kernel before patch projection to reduce
    aliasing artifacts from frame differencing in Atari input.
    """

    def __init__(self, channels: int):
        super().__init__()
        k = torch.tensor([1.0, 2.0, 1.0])
        k = k / k.sum()
        k2d = k[:, None] * k[None, :]
        self.register_buffer(
            "blur", k2d[None, None].expand(channels, 1, 3, 3).contiguous()
        )
        self.channels = channels
        self.pad = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            F.pad(x, [self.pad] * 4, mode="reflect"),
            self.blur,
            groups=self.channels,
        )


class StochasticDepth(nn.Module):
    """Per-sample stochastic depth, migrated from CIFAR 16-block SD.

    Randomly drops residual branches per-sample during training.
    Scales surviving branches by 1/(1-p) to maintain expectation.
    """

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        mask = (torch.rand(shape, device=x.device) > self.p).float()
        return x * mask / (1.0 - self.p)


class PatchEmbedding(nn.Module):
    """Patch embedding with BlurPool anti-aliasing.

    Input: (B, 4, 84, 84) -> Output: (B, 36, 384)

    BlurPool applied before patch projection to reduce aliasing
    in high-frequency frame differencing artifacts.
    """

    def __init__(
        self,
        in_channels: int = 4,
        patch_size: int = 14,
        dim: int = 384,
        img_size: int = 84,
    ):
        super().__init__()
        self.blur = BlurPool(in_channels)
        self.proj = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )
        self.n_patches = (img_size // patch_size) ** 2  # 36
        self.dim = dim
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blur(x)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B, 36, 384)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block with physics hooks.

    Standard pre-norm transformer:
    - Multi-head self-attention (returns weights for spectral analysis)
    - MLP with GELU
    - StochasticDepth on both residual branches

    Physics hooks:
    - Returns attention weights (B, n_heads, n_patches, n_patches)
    - Feature output for spectral dose-response + Lipschitz
    """

    def __init__(
        self,
        dim: int = 384,
        n_heads: int = 6,
        mlp_ratio: int = 4,
        sd_prob: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, batch_first=True, dropout=0.0
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.sd = StochasticDepth(sd_prob)
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.xavier_uniform_(self.mlp[2].weight)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(
            normed, normed, normed, need_weights=True, average_attn_weights=False
        )
        x = x + self.sd(attn_out)
        x = x + self.sd(self.mlp(self.norm2(x)))
        return x, attn_weights


class ViTActorCritic(nn.Module):
    """ViT-Small Actor-Critic for Atari Breakout.

    Route D: Physics-guided PPO with ViT-policy.

    Architecture:
      PatchEmbedding(4->384, 14x14) -> 6 x TransformerBlock -> MeanPool
      -> PolicyHead(384->n_actions) + ValueHead(384->1)

    Physics integration:
      get_physics_features() returns dict for PhysicsRegularizer
      init_ema_target() / update_ema() for mean-teacher spectral regularization
    """

    def __init__(
        self,
        in_channels: int = 4,
        n_actions: int = 4,
        patch_size: int = 14,
        dim: int = 384,
        depth: int = 6,
        n_heads: int = 6,
        mlp_ratio: int = 4,
        img_size: int = 84,
        sd_prob: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.n_actions = n_actions

        self.patch_embed = PatchEmbedding(in_channels, patch_size, dim, img_size)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.patch_embed.n_patches, dim) * 0.02
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(dim, n_heads, mlp_ratio, sd_prob)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

        self.policy_head = nn.Linear(dim, n_actions)
        self.value_head = nn.Linear(dim, 1)

        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard forward: returns (policy_logits, value)."""
        features, _ = self.forward_with_physics(x)
        pooled = features[-1].mean(dim=1)
        return self.policy_head(pooled), self.value_head(pooled)

    def forward_with_physics(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward with physics feature extraction.

        Returns:
          features: list of per-layer (B, 36, 384)
          attn_weights: list of per-layer (B, n_heads, 36, 36)
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed
        features = [x]
        attn_weights = []
        for block in self.blocks:
            x, w = block(x)
            features.append(x)
            attn_weights.append(w)
        x = self.norm(x)
        features[-1] = x
        return features, attn_weights

    def get_physics_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract all physics features for regularization."""
        features, attn_weights = self.forward_with_physics(x)
        pooled = features[-1].mean(dim=1)
        policy_logits = self.policy_head(pooled)
        value = self.value_head(pooled)
        return {
            "features": features,
            "attn_weights": attn_weights,
            "patch_embed": features[0],
            "pooled": pooled,
            "policy_logits": policy_logits,
            "value": value,
        }

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        _, value = self.forward(x)
        return value

    @torch.no_grad()
    def act(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        logits, _ = self.forward(x)
        if deterministic:
            return logits.argmax(dim=-1)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

    def init_ema_target(self) -> "ViTActorCritic":
        """Create EMA target network (mean-teacher) for physics regularization.

        The EMA target provides smoothed features for spectral dose-response
        and smoothed value predictions for Matern value smoothing.
        Adapted from CIFAR-100 mean-teacher mechanism.
        """
        ema = copy.deepcopy(self)
        for p in ema.parameters():
            p.requires_grad = False
        return ema

    @torch.no_grad()
    def update_ema(self, ema_target: "ViTActorCritic", momentum: float = 0.999):
        """Update EMA target network (exponential moving average)."""
        for p, p_ema in zip(self.parameters(), ema_target.parameters()):
            p_ema.mul_(momentum).add_(p, alpha=1 - momentum)
