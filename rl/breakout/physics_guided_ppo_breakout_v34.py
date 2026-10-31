"""
Physics-Guided PPO for Atari Breakout (v34)
MULTI-SCALE SPECTRAL DECOMPOSITION of the physics loss (D3 — L3 channel)
Base: v26 (SOTA, best 215.2 @ 4.3M, final-1M 190.88, TAIL CV 4.34%).
All v23/v26 machinery (smooth, DAP, schedules, safety net, entropy) UNCHANGED.

== v34 DESIGN ==

WHERE WE STAND (information-layer audit of the physics field, PPO line):
  L0 field values        — SATURATED (v32 dose ladder: no headroom left)
  L1 gradient direction  — used (normalize + chord distance)
  L2 per-sample residual — TESTED & REJECTED (v33: b43 peak suppression
                           -21.1; concentrating dose on rough states
                           over-regularizes peak-forming states)
  L3 SCALE STRUCTURE     — HALF-USED: every field operator (value smooth /
                           DAP / Lipschitz loss) runs at the single
                           degenerate sigma=10 — the field's spectrum is
                           collapsed onto ONE low-frequency band
  => v34 opens L3: decompose the diffusion operator into a scale ladder
     and redistribute the SAME w_phys dose across bands.

MECHANICS (the ONLY change vs v26 — the physics-loss kernel):
  v26 loss:  L = L(sigma=10)                            [single band]
  v34 loss:  L = sum_b w_b * L(sigma_b),  sum_b w_b = 1 [3-band ladder]
  L(sigma_b) is EXACTLY the v26 feature-Lipschitz formula evaluated at
  sigma_b (same-action mask, row-normalized, mean over neighbor-valid
  states). The mixture sum_b w_b * K_sigma_b is a spectrally-decomposed
  diffusion operator: coarse sigma carries the low-eigenvalue (class-
  level) band, fine sigma reaches the high-eigenvalue (local) band.

  LADDER (one decade, ~geometric spacing):
    coarse  sigma=10, w=0.5  — v26's degenerate band: within same-action
                                states this is FLAT (all peers ~equal) —
                                the PROVEN variance compressor; anchor.
    mid     sigma=3,  w=0.3  — begins to resolve within-class chords.
    fine    sigma=1,  w=0.2  — LOCAL Lipschitz: nearest same-action peers
                                dominate; targets the high-frequency
                                policy roughness behind TAIL oscillation.

  WHY sigma=1 is "fine": Q-vectors are unit-norm 4-d chords; typical
  same-action neighbor distances are ~0.2-0.8. exp(-d^2/200) (sigma=10)
  is flat over that whole range (degenerate); exp(-d^2/2) (sigma=1)
  gives nearest peers ~1.3-1.5x the weight of far peers — the first
  kernel in this codebase that actually RESOLVES the neighborhood.

  DOSE DISCIPLINE (v32 saturation + v33 over-concentration lessons):
  - global w_phys schedule UNCHANGED (0 -> 0.1 ramp 4.2-4.5M, hold
    4.5-4.8M, decay -> 0.03): blocks 1-42 bit-identical to v26.
  - band weights sum to 1.0: REALLOCATION, not addition.
  - no per-sample reweighting (L2 rejected in v33): dose stays uniform
    across samples; only the SPECTRAL profile changes.
  - 50% of dose stays in the proven coarse band; only 20% reaches the
    new fine band (conservative posture after two falsifications).

  COMPUTE: dist_q / same_action computed ONCE, shared across bands;
  per band = one N x N exp + row ops on the 64-sample minibatch
  (negligible; no extra kNN pass, no extra forward, no new RNG).

DETERMINISM:
  - MS_SPECTRAL_PHYS = False              -> v26 BIT-EXACT (full 5M)
  - ladder ((10.0,), (1.0,)) [collapse]   -> v26 BIT-EXACT (the ms
    function reduces to the v26 formula: 0.0 + 1.0 * x == x in IEEE)
  - default 3-band ladder                 -> blocks 1-42 bit-identical
    to v26 (w_phys = 0 until 4.2M); divergence starts at 4.2M.

DECISION CRITERIA (500K-window granularity, seed 42, 5M window):
  WIN:   best >= 215 AND TAIL CV <= 3.0% AND TAIL low >= 180
         AND final-1M >= 195  (v26 ceiling + finer-grained stability)
  PASS:  best >= 210 AND TAIL CV <= 4.34% (v26) AND final-1M >= 190.88
  FAIL:  best < 210 (b43 peak suppression — the v33 signature: fine-band
         compression during peak formation) OR TAIL CV > 4.34%
  Monitored: b43 (4.2-4.3M) must stay >= 205 (v26 215.2, v33 194.1).

ABLATION PROTOCOL:
  A0 harness: MS_SPECTRAL_PHYS=False      -> v26 bit-exact (control)
  A1 identity: collapsed single band      -> v26 bit-exact (math control)
  A2 main:     3-band ladder (this file, defaults)
  A3 fine-only (optional): ((1.0,), (1.0,)) — isolates the fine band
  3 seeds (42/1/2) only after seed-42 PASS/WIN.

== v26 DESIGN (inherited, UNCHANGED in v34) ==

  v25 vs v23 CLEAN A/B VERDICT (same seed; blocks 1-10 bit-identical, first
  divergence at block 1.1M — exactly when w_phys activates):

    metric         v23 (no phys loss)   v25 (phys loss 1-5M)   delta
    best block     213.6 @4.3M          201.6 @5.0M            -12.0 (ceiling CAP)
    final-1M       190.21               191.71                 +1.50
    TAIL mean      190.08               196.46                 +6.38
    TAIL CV        7.51%                2.87%                  -4.6pp (variance CRUSH)
    TAIL low       172.2                185.6                  +13.4 (floor RAISED)
    mid 2.4-3.6M   52.7                 83.6                   +30.9 (steady climb vs
                                                                 stagnation + violent
                                                                 breakout)

  CONCLUSION: full-window physics loss is a VARIANCE COMPRESSOR, not a
  ceiling raiser. It converts v23's "stagnate 2.5-3.4M, then violent
  breakout to 213.6" into "steady climb converging to ~195": regularizing
  the representation smooths away BOTH the TAIL crashes AND the 213.6 spike.

  v23 SECONDARY FINDING (falsifies v22's attribution): blk45 = 174.6 is
  IDENTICAL to v21's despite the field TAIL being softened 0.0015 -> 0.002,
  and the dip STARTS at 4.4-4.5M — BEFORE any TAIL schedule change (field
  alpha and DAP are both constant until 4.5M). The blk45-46 dip is NOT
  caused by the field/DAP TAIL drops; it correlates with ent-temper
  completion (4.4M) or is endogenous TAIL variance of the un-regularized
  representation.

  => v26 HYPOTHESIS: apply the physics loss ONLY in the TAIL, where v25
    proved it compresses variance, and keep w=0 while the ceiling forms,
    where v25 proved it costs -12 peak.

  [v26 SCHEDULE] (the ONLY change vs v23):
        0-4.2M:   w_phys = 0        (v23 ceiling path, bit-exact: blocks 1-42)
        4.2-4.5M: ramp 0 -> 0.1     (warm-in during HOLD2, before TAIL starts)
        4.5-4.8M: hold 0.1          (compress TAIL variance)
        4.8-5.0M: decay 0.1 -> 0.03 (retain partial compression to the end —
                                     v26 floor is 0.03, NOT v25's 0.01: TAIL
                                     is exactly where compression is needed)

  Determinism: v26 diverges from v23 at step 4.2M (w_phys first active).
  Blocks 1-42 (0-4.2M) should be IDENTICAL to v23/v21 (same seed).

  DECISION CRITERIA:
    WIN:     peak window > 200 AND TAIL CV < 5% AND TAIL low > 180
             (v23's ceiling + v25's stability = new best config)
    PARTIAL: TAIL CV < 5% but peak window ~195 — compression is
             instantaneous, not accumulated; still the best final-1M config.
    FAIL:    dip persists despite w=0.1 in TAIL -> TAIL variance is NOT a
             representation-smoothness phenomenon; phys-loss benefit needs
             long-term accumulated exposure (v25's 1-4M ramp/hold).

== v23 BASELINE (all UNCHANGED in v26) ==

  v23 = v21 DAP schedule (single-peak + TAIL DECAY 0.04->0.02) + softened
  field TAIL (0.003->0.002). v23 peak 213.6, final-1M 190.2, TAIL CV 7.5%.
  v23 schedule (ALL UNCHANGED in v26):
    Field:  0.008 -> 0.004 (1.2M) -> 0.003 (3.0M) -> 0.002 (4.5M)
    DAP:    0.02 -> 0.04 (1.5-3.0M ramp) -> 0.04 peak -> 0.02 (4.5M decay)
    Entropy: quench 0.0 (0-4.0M) -> temper 0.005 (4.0-4.4M) -> hold
    Kernel: Gaussian sigma=10, k=12 (v24 falsified, NOT changed)
    Safety: gentle (2 blocks <= 0.5*best) + hard-collapse rollback
    PPO: clip=0.1, epochs=4, batch=64, vf_coef=0.495
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import cv2
import gym
from collections import deque
import math
import time

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==================== Physics-Guided Config (v4: all structural OFF) ====================
USE_BLURPOOL = False
USE_RESIDUAL = False
USE_SPECTRAL_REG = False
USE_MATERN_SMOOTH = False

SD_DROP_MAX = 0.0
SD_WARMUP_START = 200_000
SD_WARMUP_END = 500_000
SPECTRAL_COEF_MAX = 0.0002

# ==================== v26 Physics Loss (TAIL-ONLY, the ONLY change vs v23) ====================
W_PHYS_PEAK    = 0.1      # peak physics loss weight (TAIL variance compression)
W_PHYS_END     = 0.03     # end-of-training weight (retain compression to the end)
W_PHYS_SIGMA   = 10.0     # broad kernel (SAME as smooth/DAP — NOT selective)
W_PHYS_RAMP_S  = 4_200_000   # ramp start: w_phys=0 through ceiling formation
W_PHYS_PEAK_S  = 4_500_000   # ramp end (warm-in during HOLD2) / hold start
W_PHYS_HOLD_S  = 4_800_000   # hold end / decay start
SPEC_WARMUP_START = 500_000
SPEC_WARMUP_END = 2_000_000
SPEC_EXCLUDE_CONV1 = True

MATERN_LAM = 1.0
MATERN_NU_REG = 1.0
MATERN_S = 1.0
MATERN_NU_SMOOTH = 1.5

# ==================== v34 Multi-Scale Spectral Decomposition (D3, the ONLY change vs v26) ====================
# L_ms = sum_b w_b * L(sigma_b),  sum_b w_b = 1.0  (dose REALLOCATION, not addition)
# L(sigma_b) = EXACTLY the v26 feature-Lipschitz formula evaluated at sigma_b.
# MS_SPECTRAL_PHYS = False, or a collapsed ladder ((10.0,), (1.0,)),
# reproduces v26 bit-exactly (no new RNG anywhere in the ms path).
MS_SPECTRAL_PHYS = True        # master switch: False -> v26 bit-exact
MS_SIGMAS  = (10.0, 3.0, 1.0)  # one decade, ~geometric: coarse / mid / fine
MS_WEIGHTS = (0.5, 0.3, 0.2)   # band weights: 50% anchor in proven coarse band

# ==================== v23 Field Schedule (TAIL endpoint softened) ====================
RELEASE_CYCLE = True
REL_A_SMOOTH = 0.008
REL_A_MID    = 0.004
REL_A_LOW    = 0.003
REL_A_FINAL  = 0.002   # v23: was 0.0015 — half-way relaxation (Plan A).
                       # v22 evidence: blk45-46 dip occurs with DAP held
                       # constant, so field-alpha TAIL drop is the destabilizer.
                       # Landing at 0.002 halves the shock (delta 0.001 -> 0.0005).
REL_SMOOTH_END = 1_200_000
REL_DROP_LEN   = 100_000
REL_HOLD1_END  = 3_000_000
REL_DROP2_LEN  = 125_000
REL_HOLD2_END  = 4_500_000
REL_TAIL_LEN   = 500_000
REL_A_V4_END   = 0.004
REL_V4_DECAY_START = 1_300_000
REL_V4_DECAY_END   = 2_300_000

# ==================== v18 DAP Config (REVERTED to v16 single-peak, proven 187.9) ====================
# v17 dual-peak (early 0.04 + valley 0.025 + peak 0.04) collapsed AND pre-collapse
# W8 was 0.60x of v16 -> no benefit, possible harm. Revert to v16's schedule.
DAP_GAIN = 5.0     # safety-net bump multiplier (bump_alpha * 5 -> DAP bump)
DAP_CAP  = 0.05    # cap to prevent over-propagation (max 5% push)

# v23 DAP schedule constants (v21's proven single-peak + gentle TAIL DECAY)
# v22 held 0.04 through TAIL and was WORSE (final-1M 187.2 vs 191.9, low
# 163.8 vs 171.6) -> revert to v21's decay 0.04 -> 0.02 in 4.5-5.0M.
DAP_EARLY     = 0.02    # 0-1.5M: conservative (advantages noisy)
DAP_PEAK      = 0.04    # 3.0-4.5M: maximum push (plateau-breaking)
DAP_LATE      = 0.02    # 5.0M: gentle landing (v21's proven final level)
DAP_LOW_END   = 1_500_000    # end of low-DAP early phase
DAP_RAMP_END  = 3_000_000    # end of ramp-up to peak
DAP_PEAK_END  = 4_500_000    # end of peak hold (decay starts here)
# DAP_FINAL = 5_000_000 (implied by total_steps)

# ==================== v18 Safety Net (gentle, UNCHANGED from v15/v16/v17) ====================
CLOSED_LOOP     = True
CL_BLOCK        = 100_000
CL_HISTORY      = 10
CL_MIN_STEP     = 1_500_000
CL_CRASH_RATIO  = 0.5
CL_BEST_MIN     = 80.0
CL_COOLDOWN     = 400_000
CL_SUSTAIN      = 2
CL_DISABLE_STEP = 4_650_000
CL_REC_LEN      = 200_000
CL_ALPHA_BUMP   = 0.002
CL_ENT_BUMP     = 0.005
CL_ALPHA_CAP    = 0.008

# ==================== v23 Robustness (QUENCH+TEMPER entropy + hard-collapse rollback, KEPT from v18/v19/v20/v21) ====================
# v22 verdict: DAP TAIL HOLD falsified — final-1M 187.2 (v21: 191.9), TAIL
# low 163.8 (v21: 171.6), peak window 190.3 (v21: 195.8). Same seed, blocks
# 1-44 identical -> clean paired A/B. High DAP through TAIL amplifies
# oscillation; blk45-46 dip is caused by field-alpha TAIL relaxation.
# v23: revert DAP to v21 + soften field TAIL endpoint (0.0015 -> 0.002).
# Determinism: blocks 1-44 should reproduce v16/v20/v21/v22 exactly
# (DAP 0-4.5M identical; field alpha only differs from 4.5M onward).
ENT_COLD            = 0.0      # quench: v16 cold path (deep-tunnel skill forms here)
ENT_TEMPER          = 0.005    # temper: stress-relief floor (v20's 0.002 was too weak)
ENT_QUENCH_END      = 4_000_000  # cold through W8 (v16/v20 blk40 = 201.7 pace)
ENT_TEMPER_RAMP_END = 4_400_000  # ramp 0.0 -> 0.005 entering the danger zone
CL_HARD_THRESH     = 5.0      # hard collapse if block_mean < this (flatline)
CL_HARD_BEST_MIN   = 80.0     # ... and trailing best was real skill
CL_HARD_REC_LEN    = 300_000  # hard recovery window (longer than gentle 200K)
CL_HARD_ALPHA_BUMP = 0.004    # stronger field re-annealing during hard recovery
CL_HARD_ENT_PULSE  = 0.05     # STRONG entropy pulse to break degenerate policy

# ==================== AtariWrapper (unchanged from v4; reset() processes frames) ====================
class AtariWrapper:
    def __init__(self, env_name="BreakoutNoFrameskip-v4", frame_stack=4):
        self.env = gym.make(env_name)
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
    def _process(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (84, 84)).astype(np.float32) / 255.0
    def reset(self):
        self.env.reset()
        for _ in range(random.randint(0, 30)): self.env.step(0)
        obs = self._process(self.env.step(0)[0])
        for _ in range(self.frame_stack): self.frames.append(obs)
        return np.stack(self.frames, axis=0)
    def step(self, action):
        total_re = 0; done = False
        for _ in range(4):
            res = self.env.step(action)
            if len(res) == 5: obs, r, term, trunc, _ = res; done = term or trunc
            else: obs, r, done, _ = res
            if not np.isfinite(r):
                r = 0.0
            total_re += r
            if done: break
        self.frames.append(self._process(obs))
        return np.stack(self.frames, axis=0), total_re, done, {}

# ==================== RunningNorm (unchanged from v4) ====================
class RunningNorm:
    def __init__(self, shape=()):
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = 0
    def normalize(self, x):
        self.count += 1
        x_t = torch.tensor(x, dtype=torch.float32)
        delta = x_t - self.mean
        self.mean += delta / self.count
        delta2 = x_t - self.mean
        self.var += delta * delta2
        std = torch.sqrt(self.var / self.count) + 1e-8
        if std > 10.0:
            std = torch.tensor(10.0)
        return (x_t - self.mean) / std

# ==================== [kept, default OFF] BlurPool2d ====================
class BlurPool2d(nn.Module):
    def __init__(self, channels, stride=1):
        super().__init__()
        self.stride = stride
        k = torch.tensor([1., 2., 1.])
        filt = k.outer(k) / 16.0
        self.register_buffer(
            'filt', filt[None, None, :, :].expand(channels, 1, 3, 3).contiguous())
        self.channels = channels
    def forward(self, x):
        return F.conv2d(x, self.filt, stride=self.stride,
                        padding=1, groups=x.shape[1])

# ==================== [kept, default OFF] ResidualBlock ====================
class ResidualBlock(nn.Module):
    def __init__(self, channels, drop_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.drop_rate = drop_rate
    def set_drop_rate(self, dr):
        self.drop_rate = dr
    def zero_init(self):
        nn.init.zeros_(self.conv2.weight)
    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if self.training and self.drop_rate > 0:
            keep_prob = 1.0 - self.drop_rate
            shape = (out.shape[0],) + (1,) * (out.ndim - 1)
            mask = (torch.rand(shape, device=out.device) < keep_prob).to(out.dtype).div_(keep_prob)
            out = out * mask
        return F.relu(out + identity)

# ==================== PhysicsGuidedActorCritic (100% original, same as v4) ====================
class PhysicsGuidedActorCritic(nn.Module):
    """Actor-Critic (v18: EXACT same architecture as v4/original).
    conv1(4->32,8x8,s4) -> conv2(32->64,4x4,s2) -> conv3(64->64,3x3,s1)
    -> flatten 3136 -> fc 512 -> actor/critic. No structural changes.
    """
    def __init__(self, n_actions):
        super().__init__()
        if USE_BLURPOOL:
            self.pre_blur2 = BlurPool2d(32, stride=1)
        else:
            self.pre_blur2 = nn.Identity()
        self.conv1 = nn.Conv2d(4, 32, 8, 4); self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1); self.fc = nn.Linear(64*7*7, 512)
        self.actor = nn.Linear(512, n_actions); self.critic = nn.Linear(512, 1)
        if USE_RESIDUAL:
            self.res1 = ResidualBlock(64, drop_rate=0.0)
            self.res2 = ResidualBlock(64, drop_rate=0.0)
        else:
            self.res1 = nn.Identity()
            self.res2 = nn.Identity()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None: nn.init.zeros_(m.bias)
        if USE_RESIDUAL:
            for blk in (self.res1, self.res2):
                if isinstance(blk, ResidualBlock):
                    blk.zero_init()

    def forward(self, x, return_qv=False, return_feat=False):
        x = F.relu(self.conv1(x))
        x = self.pre_blur2(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.res1(x)
        x = self.res2(x)
        x = x.view(x.size(0), -1)
        feat = F.relu(self.fc(x))
        logits = self.actor(feat)
        v = self.critic(feat)
        if return_qv:
            adv = logits - logits.mean(dim=1, keepdim=True)
            q = v + adv
            if return_feat:
                return logits, v.squeeze(-1), q, feat
            return logits, v.squeeze(-1), q
        if return_feat:
            return logits, v, feat
        return logits, v

    def set_sd_drop(self, drop_rate):
        if USE_RESIDUAL and isinstance(self.res1, ResidualBlock):
            self.res1.set_drop_rate(drop_rate)
            self.res2.set_drop_rate(drop_rate)

# ==================== [kept, default OFF] Matern spectral reg ====================
def matern_spectral_reg(model, lam=1.0, nu=1.0, s=1.0, exclude_conv1=True):
    loss = 0.0
    n_layers = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 4:
            if exclude_conv1 and 'conv1' in name and 'conv1' in name.split('.'):
                continue
            w = param
            W = torch.fft.fft2(w, norm='ortho')
            W_mag = W.abs() ** 2
            kh, kw = w.shape[2], w.shape[3]
            fy = torch.fft.fftfreq(kh).to(w.device)
            fx = torch.fft.fftfreq(kw).to(w.device)
            freq_sq = fy[:, None]**2 + fx[None, :]**2
            matern = (1.0 + lam * freq_sq / nu) ** (-s)
            high_freq_weight = 1.0 - matern
            loss = loss + (W_mag * high_freq_weight).mean()
            n_layers += 1
    return loss / max(n_layers, 1)

# ==================== v25: Physics Loss (feature Lipschitz regularization) ====================
def physics_lipschitz_loss(feat, q_vectors, sigma=10.0):
    """v25: Feature Lipschitz regularization — structurally NEW intervention.

    For nearby states (Q-space, broad sigma=10 kernel — SAME as smooth/DAP),
    penalize large feature differences. Same-action masking (consistent with
    existing physics field design).

    KEY DIFFERENCE from smooth/DAP:
    - smooth/DAP: modify TARGETS (baseline/advantage) post-hoc, no gradient
    - physics_loss: add to PPO LOSS, gradient flows to encoder weights
      -> directly regularizes representation smoothness

    This is the ResNet v14 verified path (loss += w_phys * smooth_reg) applied
    to PPO. The ViT v3 paradigm shift (structural hooks -> loss regularization).
    """
    N = feat.shape[0]
    if N < 4:
        return feat.sum() * 0.0  # maintain grad graph

    # Q-space weights: DETACHED (no grad through Q — we regularize FEATURES,
    # not Q-distances. Q is used only to define "nearby" neighborhoods.)
    with torch.no_grad():
        q_norm = F.normalize(q_vectors, dim=1)
        dist_q = torch.cdist(q_norm, q_norm)
        W = torch.exp(-dist_q ** 2 / (2 * sigma ** 2))
        best_actions = q_vectors.argmax(dim=1)
        same_action = (best_actions.unsqueeze(0) == best_actions.unsqueeze(1)).float()
        W = W * same_action
        W.fill_diagonal_(0)  # exclude self
        row_sum = W.sum(dim=1).clamp(min=1e-8)
        has_neighbor = row_sum > 1e-6

    # Feature distance (L2-normalized — HAS grad, this is what we regularize)
    f_norm = F.normalize(feat, dim=1)
    dist_feat = torch.cdist(f_norm, f_norm)

    # Weighted average: nearby states should have similar features
    per_state_loss = (W * dist_feat).sum(dim=1) / row_sum

    if has_neighbor.sum() == 0:
        return feat.sum() * 0.0
    return per_state_loss[has_neighbor].mean()

# ==================== v34: Multi-Scale Spectral Decomposition physics loss (D3) ====================
PHYS_STATS = {"band_sum": [0.0, 0.0, 0.0], "band_n": 0}  # [BLK] logging only

def physics_lipschitz_loss_ms(feat, q_vectors, sigmas=MS_SIGMAS, weights=MS_WEIGHTS):
    """v34 D3: MULTI-SCALE SPECTRAL DECOMPOSITION of the v26 physics loss.

    L_ms = sum_b w_b * L(sigma_b),  sum_b w_b = 1.0

    L(sigma_b) is EXACTLY the v26 feature-Lipschitz formula at sigma_b:
    same-action masked, row-normalized Gaussian kernel on unit-norm Q
    chords, per-state weighted feature distance, mean over
    neighbor-valid states. The ladder turns v26's single degenerate
    band (sigma=10, flat over the typical 0.2-0.8 chord range) into a
    coarse/mid/fine decomposition — the fine band (sigma=1) is the
    first kernel here that actually RESOLVES the neighborhood.

    DOSE: band weights sum to 1.0 -> same global w_phys schedule and
    same across-sample uniformity as v26 (v33's per-sample reweighting
    is NOT used); only the SPECTRAL profile of the compression changes.

    DETERMINISM: with sigmas=(10.0,), weights=(1.0,) this reduces to
    the v26 formula bit-exactly (0.0 + 1.0 * x == x in IEEE arithmetic).
    No RNG, no extra forward, no extra kNN pass: dist_q and the
    same-action mask are computed ONCE and shared across bands.

    A band whose has_neighbor set is empty contributes 0 (weight
    effectively lost); with 64-sample minibatches and 4 actions this
    does not occur in practice (Gaussian weights never reach 0).
    """
    N = feat.shape[0]
    if N < 4:
        return feat.sum() * 0.0  # maintain grad graph

    # Q-space geometry: DETACHED, computed ONCE, shared by all bands
    with torch.no_grad():
        q_norm = F.normalize(q_vectors, dim=1)
        dist_q = torch.cdist(q_norm, q_norm)
        best_actions = q_vectors.argmax(dim=1)
        same_action = (best_actions.unsqueeze(0) == best_actions.unsqueeze(1)).float()

    # Feature distance (L2-normalized — HAS grad, this is what we regularize)
    f_norm = F.normalize(feat, dim=1)
    dist_feat = torch.cdist(f_norm, f_norm)

    total = feat.sum() * 0.0  # maintain grad graph through the band sum
    for bi, (sigma_b, w_b) in enumerate(zip(sigmas, weights)):
        with torch.no_grad():
            W = torch.exp(-dist_q ** 2 / (2 * sigma_b ** 2))
            W = W * same_action
            W.fill_diagonal_(0)  # exclude self
            row_sum = W.sum(dim=1).clamp(min=1e-8)
            has_neighbor = row_sum > 1e-6
        # per-state weighted feature distance (W detached, dist_feat has grad)
        per_state_loss = (W * dist_feat).sum(dim=1) / row_sum
        if has_neighbor.sum() > 0:
            band_val = per_state_loss[has_neighbor].mean()
            total = total + w_b * band_val
            if bi < len(PHYS_STATS["band_sum"]):
                PHYS_STATS["band_sum"][bi] += float(band_val)
        else:
            PHYS_STATS["band_sum"][bi] += 0.0
    PHYS_STATS["band_n"] += 1
    return total

def physics_loss_weight(step, total_steps=5_000_000):
    """v26: TAIL-ONLY physics loss schedule — variance compressor applied
    only where variance hurts (v25 A/B: TAIL CV 7.5%->2.9%, low +13.4,
    but best -12.0 when active during ceiling formation).

    0-4.2M:   w=0          (v23 ceiling path preserved, bit-exact blocks 1-42)
    4.2-4.5M: ramp 0->0.1  (warm-in during HOLD2, before TAIL starts)
    4.5-4.8M: hold 0.1     (compress TAIL variance)
    4.8-5.0M: decay 0.1->0.03 (keep partial compression to the end)
    """
    if step < W_PHYS_RAMP_S:
        return 0.0
    elif step < W_PHYS_PEAK_S:
        return W_PHYS_PEAK * (step - W_PHYS_RAMP_S) / (W_PHYS_PEAK_S - W_PHYS_RAMP_S)
    elif step < W_PHYS_HOLD_S:
        return W_PHYS_PEAK
    else:
        return W_PHYS_END + (W_PHYS_PEAK - W_PHYS_END) * max(0, 1 - (step - W_PHYS_HOLD_S) / (total_steps - W_PHYS_HOLD_S))

# ==================== gradient_aware_smooth (Gaussian, bidirectional, for VALUE baseline only) ====================
def gradient_aware_smooth(values, q_vectors, k, sigma, grad_threshold=2.0,
                          use_matern=False, nu=1.5):
    """Bidirectional smoothing (v18: used for VALUE baseline only, UNCHANGED)."""
    N = values.shape[0]
    if N < 3: return values
    with torch.no_grad():
        best_actions = q_vectors.argmax(dim=1)
        q_norm = F.normalize(q_vectors, dim=1)
        dist = torch.cdist(q_norm, q_norm)
        k = min(k, N-1)
        _, idx = torch.topk(dist, k=k, largest=False, dim=1)
        if use_matern:
            if nu == 1.5:
                r = math.sqrt(3.0) * dist / sigma
                W = (1 + r) * torch.exp(-r)
            elif nu == 2.5:
                r = math.sqrt(5.0) * dist / sigma
                W = (1 + r + r**2 / 3.0) * torch.exp(-r)
            else:
                W = torch.exp(-dist**2 / (2*sigma**2))
        else:
            W = torch.exp(-dist**2 / (2*sigma**2))  # ORIGINAL Gaussian
        mask = torch.zeros_like(W)
        neighbor_actions = best_actions[idx]
        same_action = (neighbor_actions == best_actions.unsqueeze(1))
        mask.scatter_(1, idx, same_action.float())
        W = W * mask
        v_std = values.std() + 1e-6
        v_row = values.unsqueeze(1).expand(N, N)
        v_col = values.unsqueeze(0).expand(N, N)
        v_diff = torch.abs(v_row - v_col)
        grad_suppress = torch.exp(-v_diff / (grad_threshold * v_std))
        W = W * grad_suppress
        row_sum = W.sum(dim=1, keepdim=True).clamp(min=1e-8)
        has_neighbor = row_sum.squeeze() > 1e-8
        P = W / row_sum
        V_smooth = P @ values
        V_smooth[~has_neighbor] = values[~has_neighbor]
        return V_smooth

# ==================== directional_advantage_propagate (UNCHANGED from v15/v16/v17: DAP) ====================
def directional_advantage_propagate(advantages, q_vectors, k, sigma,
                                     grad_threshold=2.0):
    """Directional Advantage Propagation (DAP). UNCHANGED.

    Asymmetric propagation: good states (high advantage) radiate signal
    to similar but worse states. Reverse direction (bad -> good) is blocked.

    signal[i] = Sum_j P[i,j] * max(0, Adv[j] - Adv[i])
    (always >= 0, pushes advantages UP toward better neighbors)

    Usage: Adv_new = Adv + gamma * signal  (additive, not blend)
    """
    N = advantages.shape[0]
    if N < 3:
        return torch.zeros_like(advantages)
    with torch.no_grad():
        best_actions = q_vectors.argmax(dim=1)
        q_norm = F.normalize(q_vectors, dim=1)
        dist = torch.cdist(q_norm, q_norm)
        k = min(k, N-1)
        _, idx = torch.topk(dist, k=k, largest=False, dim=1)
        W = torch.exp(-dist**2 / (2*sigma**2))  # Gaussian kernel
        mask = torch.zeros_like(W)
        neighbor_actions = best_actions[idx]
        same_action = (neighbor_actions == best_actions.unsqueeze(1))
        mask.scatter_(1, idx, same_action.float())
        W = W * mask
        adv_std = advantages.std() + 1e-6
        adv_row = advantages.unsqueeze(1).expand(N, N)
        adv_col = advantages.unsqueeze(0).expand(N, N)
        adv_diff_abs = torch.abs(adv_row - adv_col)
        grad_suppress = torch.exp(-adv_diff_abs / (grad_threshold * adv_std))
        W = W * grad_suppress
        row_sum = W.sum(dim=1, keepdim=True).clamp(min=1e-8)
        has_neighbor = row_sum.squeeze() > 1e-8
        P = W / row_sum  # normalized propagation matrix

        adv_diff_ij = advantages.unsqueeze(0).expand(N, N) - \
                      advantages.unsqueeze(1).expand(N, N)
        positive_diff = torch.clamp(adv_diff_ij, min=0.0)
        signal = (P * positive_diff).sum(dim=1)
        signal[~has_neighbor] = 0.0
        return signal

# ==================== v23 Field Schedule (TAIL endpoint softened) ====================
def release_cycle_alpha(step, total_steps):
    """v23 field schedule (only TAIL endpoint changed: 0.0015 -> 0.002).

    SMOOTH 0 - 1.2M    : 0.008
    REL1   1.2M-1.3M   : 0.008 -> 0.004
    HOLD1  1.3M-3.0M   : 0.004
    REL2   3.0-3.125M  : 0.004 -> 0.003
    HOLD2  3.125-4.5M  : 0.003
    TAIL   4.5M-5.0M   : 0.003 -> 0.002  (v23: was -> 0.0015)
    """
    if not RELEASE_CYCLE:
        if step < REL_V4_DECAY_START:
            return REL_A_SMOOTH
        if step < REL_V4_DECAY_END:
            p = (step - REL_V4_DECAY_START) / (REL_V4_DECAY_END - REL_V4_DECAY_START)
            return REL_A_SMOOTH + (REL_A_V4_END - REL_A_SMOOTH) * p
        return REL_A_V4_END

    A = REL_A_SMOOTH
    A_MID = REL_A_MID
    A_LOW = REL_A_LOW
    A_FIN = REL_A_FINAL
    S_END = REL_SMOOTH_END
    D_LEN = REL_DROP_LEN
    H1_END = REL_HOLD1_END
    D2_LEN = REL_DROP2_LEN
    H2_END = REL_HOLD2_END
    T_LEN = REL_TAIL_LEN

    if step < S_END:
        return A
    elif step < S_END + D_LEN:
        p = (step - S_END) / D_LEN
        return A + (A_MID - A) * p
    elif step < H1_END:
        return A_MID
    elif step < H1_END + D2_LEN:
        p = (step - H1_END) / D2_LEN
        return A_MID + (A_LOW - A_MID) * p
    elif step < H2_END:
        return A_LOW
    elif step < H2_END + T_LEN:
        p = (step - H2_END) / T_LEN
        return A_LOW + (A_FIN - A_LOW) * p
    else:
        return A_FIN

def release_phase_name(step):
    if not RELEASE_CYCLE:
        if step < REL_V4_DECAY_START: return "SMOOTH"
        if step < REL_V4_DECAY_END:   return "V4DECAY"
        return "V4HOLD"
    S_END = REL_SMOOTH_END
    D_LEN = REL_DROP_LEN
    H1_END = REL_HOLD1_END
    D2_LEN = REL_DROP2_LEN
    H2_END = REL_HOLD2_END
    T_LEN = REL_TAIL_LEN
    if step < S_END:           return "SMOOTH"
    if step < S_END + D_LEN:   return "REL1"
    if step < H1_END:          return "HOLD1"
    if step < H1_END + D2_LEN: return "REL2"
    if step < H2_END:          return "HOLD2"
    if step < H2_END + T_LEN:  return "TAIL"
    return "FINAL"

# ==================== v23 DAP Schedule (v21's proven single-peak + gentle TAIL DECAY) ====================
def dap_schedule_alpha(step, total_steps=5_000_000):
    """v23 DAP alpha schedule = v21's proven single-peak with gentle TAIL decay.

    v22 verdict: holding 0.04 through TAIL was WORSE (final-1M 187.2 vs
    191.9, TAIL low 163.8 vs 171.6) -> revert to v21's decay.
    v22 evidence: with DAP constant at 4.5M, the blk45-46 dip still
    occurred -> the dip is caused by field-alpha TAIL relaxation (fixed
    separately via REL_A_FINAL = 0.002), not by DAP decay.

      0 - 1.5M  : DAP_EARLY (0.02)  -- low, advantages noisy
      1.5-3.0M  : ramp 0.02 -> 0.04 -- agent learning, increase push
      3.0-4.5M  : DAP_PEAK (0.04)   -- plateau-breaking, maximum push
      4.5-5.0M  : decay 0.04 -> 0.02 -- v21's gentle landing (proven 191.9)
    """
    if step < DAP_LOW_END:                    # 0 - 1.5M: low
        return DAP_EARLY
    elif step < DAP_RAMP_END:                 # 1.5-3.0M: ramp up
        p = (step - DAP_LOW_END) / (DAP_RAMP_END - DAP_LOW_END)
        return DAP_EARLY + (DAP_PEAK - DAP_EARLY) * p
    elif step < DAP_PEAK_END:                 # 3.0-4.5M: peak hold
        return DAP_PEAK
    else:                                     # 4.5-5.0M: gentle decay (v21)
        p = min((step - DAP_PEAK_END) / (total_steps - DAP_PEAK_END), 1.0)
        return DAP_PEAK + (DAP_LATE - DAP_PEAK) * p

def dap_phase_name(step):
    if step < DAP_LOW_END:           return "DAP_LOW"
    if step < DAP_RAMP_END:          return "DAP_RAMP"
    if step < DAP_PEAK_END:          return "DAP_PEAK"
    return "DAP_DECAY"

# ==================== v23 Quench + Temper Entropy Schedule (UNCHANGED from v21/v22) ====================
def entropy_schedule_alpha(step):
    """v23: quench + temper schedule (UNCHANGED from v21/v22, temper 0.005).

    v21 proved the concept: blk1-40 reproduced v16 exactly, final-1M = 191.92
    (FIRST >=190!), peak block 213.6 (historical high). TAIL CV 8.0%
    (v20's 10.4%), TAIL low-point 171.6 (v20's 154.6, +17.0).
    Temper 0.005 is the confirmed sweet spot — UNCHANGED in v23.

      0 - 4.0M      : ENT_COLD (0.0)      -- v16 cold path (deep skill)
      4.0-4.4M      : linear 0.0 -> 0.005  -- temper ramp entering danger zone
      4.4-5.0M      : ENT_TEMPER (0.005)   -- stress relief + safety floor
    """
    if step < ENT_QUENCH_END:
        return ENT_COLD
    elif step < ENT_TEMPER_RAMP_END:
        p = (step - ENT_QUENCH_END) / (ENT_TEMPER_RAMP_END - ENT_QUENCH_END)
        return ENT_COLD + (ENT_TEMPER - ENT_COLD) * p
    else:
        return ENT_TEMPER

def ent_phase_name(step):
    if step < ENT_QUENCH_END:        return "ENT_QUENCH"
    if step < ENT_TEMPER_RAMP_END:   return "ENT_TEMPER"
    return "ENT_HOLD"

# ==================== v18 Safety Net (gentle, UNCHANGED) ====================
def detect_collapse(block_mean, hist_best, step, last_trig, crash_streak=0):
    if not CLOSED_LOOP:
        return False
    if block_mean is None or hist_best is None:
        return False
    if step >= CL_DISABLE_STEP:
        return False
    if step < CL_MIN_STEP:
        return False
    if hist_best < CL_BEST_MIN:
        return False
    if block_mean > CL_CRASH_RATIO * hist_best:
        return False
    if crash_streak < CL_SUSTAIN:
        return False
    if step - last_trig < CL_COOLDOWN:
        return False
    return True

def field_response(step, trig_step):
    """Gentle recovery response (UNCHANGED)."""
    if not CLOSED_LOOP or trig_step is None:
        return 0.0, 0.0
    t = step - trig_step
    if t < 0 or t >= CL_REC_LEN:
        return 0.0, 0.0
    p = t / CL_REC_LEN
    w = 0.5 * (1.0 + math.cos(math.pi * p))
    return CL_ALPHA_BUMP * w, CL_ENT_BUMP * w

# ==================== v18 Hard-Collapse Rollback (NEW) ====================
def detect_hard_collapse(block_mean, hist_best, step, last_hard_trig):
    """v18 NEW: HARD (flatline) collapse detector. Immediate (streak=1).

    Fires when block_mean < CL_HARD_THRESH (5) AND trailing best > 80
    (real skill existed). This is the catastrophic policy-collapse case
    that the gentle net (streak=2) is too slow for.
    """
    if not CLOSED_LOOP:
        return False
    if block_mean is None or hist_best is None:
        return False
    if step >= CL_DISABLE_STEP:
        return False
    if step < CL_MIN_STEP:
        return False
    if hist_best < CL_HARD_BEST_MIN:
        return False
    if block_mean >= CL_HARD_THRESH:
        return False
    if step - last_hard_trig < CL_HARD_REC_LEN:
        return False  # don't re-trigger within an active hard recovery
    return True

def hard_response(step, trig_step):
    """v18 NEW: STRONG recovery response for hard collapse.

    Larger alpha bump + STRONG entropy pulse (0.05) over a longer window
    (300K) to break a degenerate deterministic policy.
    """
    if not CLOSED_LOOP or trig_step is None:
        return 0.0, 0.0
    t = step - trig_step
    if t < 0 or t >= CL_HARD_REC_LEN:
        return 0.0, 0.0
    p = t / CL_HARD_REC_LEN
    w = 0.5 * (1.0 + math.cos(math.pi * p))
    return CL_HARD_ALPHA_BUMP * w, CL_HARD_ENT_PULSE * w

def effective_alpha(step, total_steps, trig_step=None, hard_trig_step=None):
    """Field alpha for VALUE baseline smoothing (UNCHANGED logic + hard bump)."""
    bump_g, _ = field_response(step, trig_step)
    bump_h, _ = hard_response(step, hard_trig_step)
    bump = max(bump_g, bump_h)
    return min(release_cycle_alpha(step, total_steps) + bump, CL_ALPHA_CAP)

def effective_dap_alpha(step, total_steps, trig_step=None, hard_trig_step=None):
    """v23: DAP alpha single-peak + gentle TAIL decay + dual recovery bumps."""
    dap_base = dap_schedule_alpha(step, total_steps)
    bump_g, _ = field_response(step, trig_step)
    bump_h, _ = hard_response(step, hard_trig_step)
    dap_bump = max(bump_g, bump_h) * DAP_GAIN  # convert field bump to DAP bump
    return min(dap_base + dap_bump, DAP_CAP)

# ==================== Warmup helpers (unchanged, inactive at defaults) ====================
def get_sd_drop(step):
    if step < SD_WARMUP_START:
        return 0.0
    if step >= SD_WARMUP_END:
        return SD_DROP_MAX
    return SD_DROP_MAX * (step - SD_WARMUP_START) / (SD_WARMUP_END - SD_WARMUP_START)

def get_spectral_coef(step):
    if step < SPEC_WARMUP_START:
        return 0.0
    if step >= SPEC_WARMUP_END:
        return SPECTRAL_COEF_MAX
    return SPECTRAL_COEF_MAX * (step - SPEC_WARMUP_START) / (SPEC_WARMUP_END - SPEC_WARMUP_START)

# ==================== Training (v25: v23 baseline + LOSS-LEVEL physics field) ====================
def train(total_steps=5_000_000, rollout_len=1024, gamma=0.99, lambda_=0.95,
          clip_eps=0.1, lr_max=2.5e-4, lr_min=1e-5,
          update_epochs=4, batch_size=64, vf_coef=0.495, diff_start=200_000):
    env = AtariWrapper("BreakoutNoFrameskip-v4")
    n_actions = env.env.action_space.n
    model = PhysicsGuidedActorCritic(n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_max, eps=1e-5)
    reward_norm = RunningNorm()

    DIFF_K = 12
    DIFF_SIGMA = 10.0
    GRAD_THRESHOLD = 2.0

    # ---- v18 field-control state ----
    cl_hist = deque(maxlen=CL_HISTORY)
    cl_block_rewards = []
    cl_trig_step = None
    cl_last_trig = -1_000_000_000
    cl_triggers_total = 0
    cl_crash_streak = 0

    # ---- v18 hard-collapse rollback state ----
    cl_hard_trig_step = None
    cl_last_hard_trig = -1_000_000_000
    cl_hard_triggers_total = 0
    best_checkpoint = None        # CPU clone of best-ever model state
    cl_best_ever = 0.0            # best-ever block mean

    # ---- v18 compute overhead instrumentation ----
    t_train_start = time.time()
    t_field = 0.0

    # Print configuration
    print("\n" + "="*60)
    print("Physics-Guided PPO v34 Configuration (MULTI-SCALE SPECTRAL DECOMPOSITION: v26 base + D3):")
    print(f"  Budget: {total_steps/1_000_000:.0f}M steps")
    print(f"  Architecture: 100% original (v4 baseline, NO structural changes)")
    print(f"  [1] Field schedule (VALUE baseline, v23: TAIL endpoint softened):")
    print(f"        0-{REL_SMOOTH_END/1_000_000:.1f}M: {REL_A_SMOOTH} SMOOTH")
    print(f"        {REL_SMOOTH_END/1_000_000:.1f}-{(REL_SMOOTH_END+REL_DROP_LEN)/1_000_000:.2f}M: REL1 -> {REL_A_MID}")
    print(f"        {(REL_SMOOTH_END+REL_DROP_LEN)/1_000_000:.2f}-{REL_HOLD1_END/1_000_000:.1f}M: HOLD1 {REL_A_MID}")
    print(f"        {REL_HOLD1_END/1_000_000:.1f}-{(REL_HOLD1_END+REL_DROP2_LEN)/1_000_000:.3f}M: REL2 -> {REL_A_LOW}")
    print(f"        {(REL_HOLD1_END+REL_DROP2_LEN)/1_000_000:.3f}-{REL_HOLD2_END/1_000_000:.1f}M: HOLD2 {REL_A_LOW}")
    print(f"        {REL_HOLD2_END/1_000_000:.1f}-{(REL_HOLD2_END+REL_TAIL_LEN)/1_000_000:.1f}M: TAIL -> {REL_A_FINAL} (v23: was 0.0015, halve the 4.5M shock)")
    print(f"  [2] DAP schedule (v23: v21's proven single-peak + gentle TAIL decay):")
    print(f"        0-{DAP_LOW_END/1_000_000:.1f}M: {DAP_EARLY} LOW (advantages noisy)")
    print(f"        {DAP_LOW_END/1_000_000:.1f}-{DAP_RAMP_END/1_000_000:.1f}M: {DAP_EARLY} -> {DAP_PEAK} RAMP (agent learning)")
    print(f"        {DAP_RAMP_END/1_000_000:.1f}-{DAP_PEAK_END/1_000_000:.1f}M: {DAP_PEAK} PEAK (plateau-breaking)")
    print(f"        {DAP_PEAK_END/1_000_000:.1f}-{total_steps/1_000_000:.1f}M: {DAP_PEAK} -> {DAP_LATE} DECAY (v23: v21 revert, v22 HOLD falsified)")
    print(f"  [3] DAP function: UNCHANGED (signal = Sum P*max(0, dAdv), additive)")
    print(f"  [4] Value baseline: bidirectional smooth KEPT (GAE stability)")
    print(f"  [5] Safety net (gentle): {CL_SUSTAIN} consecutive blocks "
          f"<= {CL_CRASH_RATIO}*best, armed>={CL_MIN_STEP/1_000_000:.1f}M, "
          f"cooldown={CL_COOLDOWN/1000:.0f}K, DISABLED>={CL_DISABLE_STEP/1_000_000:.2f}M")
    print(f"  [6] ENTROPY QUENCH + TEMPER (UNCHANGED from v21/v22, temper 0.005):")
    print(f"        0-{ENT_QUENCH_END/1_000_000:.1f}M: {ENT_COLD} QUENCH (ride the v16 cold path)")
    print(f"        {ENT_QUENCH_END/1_000_000:.1f}-{ENT_TEMPER_RAMP_END/1_000_000:.1f}M: {ENT_COLD} -> {ENT_TEMPER} TEMPER (enter danger zone)")
    print(f"        {ENT_TEMPER_RAMP_END/1_000_000:.1f}-{total_steps/1_000_000:.1f}M: {ENT_TEMPER} HOLD (damp TAIL oscillation)")
    print(f"        v19 verdict: anneal falsified — late entropy is not the gap cause;")
    print(f"        the hot EARLY path commits to a ~160 ceiling (path dependence).")
    print(f"  [7] HARD-COLLAPSE ROLLBACK (KEPT from v18): if block < {CL_HARD_THRESH} and best > {CL_HARD_BEST_MIN}")
    print(f"        -> restore best checkpoint + optimizer reset + ent pulse {CL_HARD_ENT_PULSE} for {CL_HARD_REC_LEN//1000}K steps")
    print(f"        Best-checkpoint snapshot on each new best-ever block (CPU clone ~6.7MB)")
    print(f"  [8] Compute overhead instrumented (light-equipment evidence)")
    print(f"  [9] v26 PHYSICS LOSS SCHEDULE (UNCHANGED, TAIL-ONLY):")
    print(f"        Feature Lipschitz: nearby states (Q-space, sigma={W_PHYS_SIGMA}) ->")
    print(f"        penalize feature L2 distance. Gradient -> encoder weights.")
    print(f"        Schedule: 0-{W_PHYS_RAMP_S/1_000_000:.1f}M w=0 (v26 ceiling path) / "
          f"{W_PHYS_RAMP_S/1_000_000:.1f}-{W_PHYS_PEAK_S/1_000_000:.1f}M ramp->{W_PHYS_PEAK} / "
          f"{W_PHYS_PEAK_S/1_000_000:.1f}-{W_PHYS_HOLD_S/1_000_000:.1f}M hold / "
          f"{W_PHYS_HOLD_S/1_000_000:.1f}-5M decay->{W_PHYS_END}")
    print(f"  [10] v34 MULTI-SCALE SPECTRAL DECOMPOSITION (D3 — the ONLY change vs v26):")
    print(f"        L_ms = sum_b w_b * L(sigma_b); band weights sum to 1.0 (REALLOCATION).")
    print(f"        Ladder sigma = {MS_SIGMAS} (coarse/mid/fine, one decade ~geometric),")
    print(f"        weights = {MS_WEIGHTS} — 50% anchor in v26's proven coarse band,")
    print(f"        20% in the new fine band (sigma=1: first kernel that resolves the")
    print(f"        0.2-0.8 same-action chord range; sigma=10 is flat/degenerate there).")
    print(f"        No per-sample reweighting (v33 L2 rejected); uniform dose, spectral")
    print(f"        profile only. MS_SPECTRAL_PHYS={MS_SPECTRAL_PHYS} (False -> v26 bit-exact).")
    print(f"  [EVIDENCE] v17 collapsed @4.44M. v20 first >200 (peak 203.8). v21 first >=190")
    print(f"              (peak 213.6, final-1M 191.9). v22 DAP HOLD falsified. v23 revert+soften.")
    print(f"              v24 KERNEL REBUILD FALSIFIED: peak 91.9 (43% of v21). Degenerate sigma=10")
    print(f"              was load-bearing (broad smoothing + DAP propagation). Kernel axis dead.")
    print(f"              v25: LOSS-LEVEL physics field (structural change, not kernel tweak).")
    print(f"              v18: ent 0.01 fixed -> 0 triggers but 158.5 final-1M (0.85x of v16).")
    print(f"              v19: anneal 0.01 -> 0.002 falsified (W8-W10 +-2 pts) -> path dependence.")
    print(f"              v20: quench cold (v16 path) + temper 0.002 @4.0M -> peak 203.8 (first >200!)")
    print(f"                   but HOLD 0.002 too weak (TAIL CV 10.4%, 4.70M crashed to 56.4).")
    print(f"              v21: same quench + raised temper 0.002 -> 0.005 -> final-1M 191.92 (FIRST >=190!)")
    print(f"                   peak 213.6, TAIL CV 8.0%, low-point 171.6 (+17.0). But peak window")
    print(f"                   195.76 (target 200, short 4.24) due to blk45-46 dual-shock dip.")
    print(f"              v22: DAP TAIL HOLD (0.04 no decay) FALSIFIED — final-1M 187.2 (-4.7),")
    print(f"                   TAIL low 163.8 (-7.8), peak window 190.3. blk1-44 identical to v21")
    print(f"                   (clean A/B); blk45-46 dip (174.6/173.3) STILL occurred with DAP")
    print(f"                   constant -> field-alpha TAIL drop blamed. [v23 NOTE: FALSIFIED —")
    print(f"                   dip persists with softened TAIL and starts before 4.5M].")
    print(f"              v23: revert DAP to v21 (decay ->0.02) + soften field TAIL (-> 0.002).")
    print(f"                   RESULT: peak 213.6, final-1M 190.2, TAIL CV 7.5%, low 172.2.")
    print(f"              v25 CLEAN A/B vs v23 (same seed, diverge @1.1M): best 201.6 (-12.0),")
    print(f"                   final-1M 191.7 (+1.5), TAIL CV 2.9% (vs 7.5%), TAIL low 185.6 (+13.4),")
    print(f"                   mid 2.4-3.6M 83.6 (vs 52.7). Verdict: VARIANCE COMPRESSOR,")
    print(f"                   not ceiling raiser — stabilizes representation, smooths away")
    print(f"                   both the crashes AND the 213.6 spike.")
    print(f"              v26: TAIL-ONLY phys loss -> SOTA best 215.2 @4.3M, final-1M 190.88,")
    print(f"                   TAIL CV 4.34%. v32 storm-splitting: dose saturated.")
    print(f"              v33: D2 residual weighting FALSIFIED — best 210.4 (-4.8), b43 peak")
    print(f"                   suppression -21.1 (194.1 vs 215.2); concentrating dose on")
    print(f"                   high-residual states over-regularizes peak formation.")
    print(f"              v34: D3 multi-scale spectral decomposition — same dose, same TAIL")
    print(f"                   schedule, same uniform across-sample weighting; ONLY the")
    print(f"                   spectral profile changes (coarse/mid/fine ladder).")
    print(f"  PPO: clip={clip_eps}, epochs={update_epochs}, batch={batch_size}")
    print(f"  LR: {lr_max}->{lr_min} (cosine over 5M), rollout={rollout_len}")
    print(f"  Physics smooth: diff_start={diff_start}")
    print(f"  Smooth kernel: k={DIFF_K}, sigma={DIFF_SIGMA}, grad_threshold={GRAD_THRESHOLD}")
    print(f"  v26 Physics loss schedule: w_phys peak={W_PHYS_PEAK} (TAIL-only), end={W_PHYS_END}")
    print(f"  v34 MS ladder: sigmas={MS_SIGMAS}, weights={MS_WEIGHTS} (sum=1, dose-reallocation)")
    print("="*60 + "\n")

    episode_rewards = []
    state = env.reset()
    ep_raw_reward = 0
    states, actions, rewards, dones, log_probs, values, q_vectors = [], [], [], [], [], [], []

    for step in range(1, total_steps+1):
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * step / total_steps))
        for pg in optimizer.param_groups: pg['lr'] = lr

        current_sd_drop = get_sd_drop(step)
        current_spec_coef = get_spectral_coef(step)
        model.set_sd_drop(current_sd_drop)

        mix_alpha = effective_alpha(step, total_steps, cl_trig_step, cl_hard_trig_step)
        dap_alpha = effective_dap_alpha(step, total_steps, cl_trig_step, cl_hard_trig_step)
        _, ent_pulse_g = field_response(step, cl_trig_step)
        _, ent_pulse_h = hard_response(step, cl_hard_trig_step)
        ent_coef_t = entropy_schedule_alpha(step) + max(ent_pulse_g, ent_pulse_h)

        # ---- Rollout collection (live critic value stored for GAE, v4 path) ----
        s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, v, q = model(s_t, return_qv=True)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, done, _ = env.step(action.item())
        ep_raw_reward += reward
        scaled_reward = reward_norm.normalize(reward).item()

        states.append(state)
        actions.append(action)
        rewards.append(scaled_reward)
        dones.append(done)
        log_probs.append(log_prob)
        values.append(v)
        q_vectors.append(q)

        state = next_state
        if done:
            state = env.reset()
            episode_rewards.append(ep_raw_reward)
            cl_block_rewards.append(ep_raw_reward)
            ep_raw_reward = 0

        # ---- safety net controller (once per CL_BLOCK steps) ----
        if step % CL_BLOCK == 0:
            block_mean = (float(np.mean(cl_block_rewards))
                          if cl_block_rewards else None)
            if block_mean is not None:
                cl_hist.append(block_mean)
                hist_best = max(cl_hist)

                # v18: best-ever checkpoint snapshot (on new record)
                if block_mean > cl_best_ever:
                    cl_best_ever = block_mean
                    best_checkpoint = {k: v.detach().cpu().clone()
                                       for k, v in model.state_dict().items()}

                is_crash_block = (block_mean <= CL_CRASH_RATIO * hist_best
                                  and hist_best >= CL_BEST_MIN)
                cl_crash_streak = cl_crash_streak + 1 if is_crash_block else 0

                # v18: HARD collapse check (immediate, streak=1)
                if detect_hard_collapse(block_mean, hist_best, step,
                                        cl_last_hard_trig):
                    cl_hard_trig_step = step
                    cl_last_hard_trig = step
                    cl_hard_triggers_total += 1
                    # also mark gentle cooldown so they don't conflict
                    cl_last_trig = step
                    if best_checkpoint is not None:
                        model.load_state_dict(best_checkpoint)
                        optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)
                        print(f"  [HCL] Step {step}: HARD COLLAPSE "
                              f"(block {block_mean:.1f} < {CL_HARD_THRESH}, best {hist_best:.1f}) "
                              f"-> ROLLBACK to best-ever {cl_best_ever:.1f} + ent {CL_HARD_ENT_PULSE} "
                              f"for {CL_HARD_REC_LEN//1000}K steps")
                    else:
                        print(f"  [HCL] Step {step}: HARD COLLAPSE but no checkpoint "
                              f"-> ent {CL_HARD_ENT_PULSE} for {CL_HARD_REC_LEN//1000}K steps")

                # gentle collapse check (streak=2, unchanged)
                if detect_collapse(block_mean, hist_best, step, cl_last_trig,
                                   cl_crash_streak):
                    cl_trig_step = step
                    cl_last_trig = step
                    cl_triggers_total += 1
                    print(f"  [CL] Step {step}: COLLAPSE detected "
                          f"(block {block_mean:.1f} <= {CL_CRASH_RATIO}*best {hist_best:.1f}, "
                          f"streak {cl_crash_streak})"
                          f" -> recovery +{CL_ALPHA_BUMP} alpha / ent +{CL_ENT_BUMP}")
                elapsed = time.time() - t_train_start
                ovh_pct = 100.0 * t_field / max(elapsed, 1e-9)
                dap_phase = dap_phase_name(step)
                ms_n = PHYS_STATS["band_n"]
                ms_means = ([PHYS_STATS["band_sum"][b] / ms_n if ms_n > 0 else 0.0
                             for b in range(len(MS_SIGMAS))])
                ms_means = ms_means + [0.0] * (3 - len(ms_means))  # ablation-ladder safe
                print(f"  [BLK] Step {step}: block={block_mean:.1f} best={hist_best:.1f} "
                      f"bestever={cl_best_ever:.1f} streak={cl_crash_streak} "
                      f"alpha={mix_alpha:.4f} dap_a={dap_alpha:.4f} "
                      f"ent={ent_coef_t:.4f} w_phys={physics_loss_weight(step, total_steps):.4f} "
                      f"ms=[{ms_means[0]:.3f},{ms_means[1]:.3f},{ms_means[2]:.3f}] "
                      f"hcl={cl_hard_triggers_total} ovh={ovh_pct:.2f}%")
                PHYS_STATS["band_sum"] = [0.0] * len(MS_SIGMAS)
                PHYS_STATS["band_n"] = 0
            cl_block_rewards = []

        # ---- PPO Update (when rollout is full) ----
        if len(states) >= rollout_len:
            s_next = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                _, v_last, _ = model(s_next, return_qv=True)
            v_last = v_last.squeeze(0)

            vals_tensor = torch.cat(values, dim=0).squeeze(-1)
            q_rollout = torch.cat(q_vectors, dim=0).squeeze(1)

            if step >= diff_start:
                _t0 = time.time()
                v_smooth = gradient_aware_smooth(vals_tensor, q_rollout,
                                                 k=DIFF_K, sigma=DIFF_SIGMA,
                                                 grad_threshold=GRAD_THRESHOLD,
                                                 use_matern=USE_MATERN_SMOOTH,
                                                 nu=MATERN_NU_SMOOTH)
                t_field += time.time() - _t0
                baseline_vals = (1 - mix_alpha) * vals_tensor + mix_alpha * v_smooth
            else:
                baseline_vals = vals_tensor

            returns, advantages = [], []
            gae = 0
            for t in reversed(range(rollout_len)):
                if t == rollout_len-1:
                    next_val = v_last
                else:
                    next_val = baseline_vals[t+1]
                delta = rewards[t] + gamma * next_val * (1 - dones[t]) - baseline_vals[t]
                gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + baseline_vals[t])

            S = torch.FloatTensor(np.array(states)).to(device)
            A = torch.stack(actions).squeeze(-1)
            old_lp = torch.stack(log_probs).squeeze(-1)
            R = torch.tensor(returns, dtype=torch.float32, device=device)
            Adv = torch.tensor(advantages, dtype=torch.float32, device=device)

            if torch.isnan(Adv).any() or torch.isinf(Adv).any():
                print(f"Step {step}: Advantage NaN/Inf, resetting optimizer.")
                optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)
                states, actions, rewards, dones, log_probs, values, q_vectors = [], [], [], [], [], [], []
                continue

            # v23: DAP (single-peak + TAIL decay) + quench-temper entropy in loss
            if step >= diff_start:
                _t0 = time.time()
                dap_signal = directional_advantage_propagate(Adv, q_rollout,
                                                              k=DIFF_K, sigma=DIFF_SIGMA,
                                                              grad_threshold=GRAD_THRESHOLD)
                t_field += time.time() - _t0
                Adv = Adv + dap_alpha * dap_signal

            Adv = (Adv - Adv.mean()) / (Adv.std() + 1e-8)

            for _ in range(update_epochs):
                idxs = torch.randperm(rollout_len)
                for start in range(0, rollout_len, batch_size):
                    idx = idxs[start:start+batch_size]
                    s_b, a_b, old_lp_b, R_b, Adv_b = S[idx], A[idx], old_lp[idx], R[idx], Adv[idx]

                    logits_b, v_b, q_b, feat_b = model(s_b, return_qv=True, return_feat=True)
                    probs_b = F.softmax(logits_b, dim=-1)
                    dist_b = torch.distributions.Categorical(probs_b)
                    new_lp_b = dist_b.log_prob(a_b)

                    ratio = (new_lp_b - old_lp_b).exp()
                    surr1 = ratio * Adv_b
                    surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * Adv_b
                    actor_loss = -torch.min(surr1, surr2).mean()

                    v_pred = v_b.squeeze(-1)
                    v_clip = v_pred + torch.clamp(v_pred - R_b, -clip_eps, clip_eps)
                    v_loss1 = (v_pred - R_b).pow(2)
                    v_loss2 = (v_clip - R_b).pow(2)
                    critic_loss = torch.max(v_loss1, v_loss2).mean()

                    loss = actor_loss + vf_coef * critic_loss
                    # v23: quench-temper entropy applies when >0 (+ recovery pulse on top)
                    if ent_coef_t > 0:
                        ent_b = dist_b.entropy().mean()
                        loss = loss - ent_coef_t * ent_b

                    if USE_SPECTRAL_REG and current_spec_coef > 0:
                        spec_loss = matern_spectral_reg(
                            model, lam=MATERN_LAM, nu=MATERN_NU_REG, s=MATERN_S,
                            exclude_conv1=SPEC_EXCLUDE_CONV1)
                        loss = loss + current_spec_coef * spec_loss

                    # v34: physics loss (multi-scale spectral decomposition, D3);
                    # MS_SPECTRAL_PHYS=False -> v26 single-band path bit-exact
                    w_phys_t = physics_loss_weight(step, total_steps)
                    if w_phys_t > 0:
                        if MS_SPECTRAL_PHYS:
                            phys_loss = physics_lipschitz_loss_ms(
                                feat_b, q_b, sigmas=MS_SIGMAS, weights=MS_WEIGHTS)
                        else:
                            phys_loss = physics_lipschitz_loss(feat_b, q_b, sigma=W_PHYS_SIGMA)
                        loss = loss + w_phys_t * phys_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    optimizer.step()

            states, actions, rewards, dones, log_probs, values, q_vectors = [], [], [], [], [], [], []

        # ---- Logging (v23: ent follows quench+temper schedule; +hcl count) ----
        if step % 20000 == 0:
            avg_r = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            max_r = max(episode_rewards[-10:]) if episode_rewards else 0.0
            phase = release_phase_name(step)
            rec_flag = "+REC" if ent_pulse_g > 0 else ("+HREC" if ent_pulse_h > 0 else "")
            w_phys_log = physics_loss_weight(step, total_steps)
            print(f"Step {step:8d} | Avg R: {avg_r:.2f} | Max R: {max_r:.2f} | "
                  f"alpha: {mix_alpha:.4f} | dap_a: {dap_alpha:.4f} | ent: {ent_coef_t:.4f} | "
                  f"w_phys: {w_phys_log:.4f} | "
                  f"best: {cl_best_ever:.1f} | {phase:6s}{rec_flag}")

    # ---- Final report ----
    elapsed = time.time() - t_train_start
    ovh_pct = 100.0 * t_field / max(elapsed, 1e-9)
    print(f"\nv34 Training complete ({total_steps/1_000_000:.0f}M steps). "
          f"Gentle triggers: {cl_triggers_total}, Hard triggers: {cl_hard_triggers_total}")
    print(f"Best-ever block mean: {cl_best_ever:.1f}")
    print(f"Compute overhead of physics field: {t_field:.1f}s smoothing+DAP / "
          f"{elapsed:.1f}s total wall-clock = {ovh_pct:.2f}%")
    print(f"(value smooth [bidirectional] + DAP [directional, v23 single-peak + TAIL decay] +")
    print(f" v34 physics loss [feature Lipschitz, MULTI-SCALE SPECTRAL ladder],")
    print(f" entropy quench+temper {ENT_COLD} -> {ENT_TEMPER} + hard-collapse rollback; both kNN passes")
    print(f" on rollout data already collected; physics loss on minibatch forward; no replay / extra nets)")
    return episode_rewards

if __name__ == "__main__":
    rewards = train()
    np.save("breakout_physics_guided_v34.npy", rewards)
