# =============================================================================
# cnn10_info_v02.py  --  InfoDiff: Information-Guided Wasserstein Diffusion
# -----------------------------------------------------------------------------
# Lineage: info_v01 (delivered 97.06 vs B-line baseline 97.56, z=-2.96).
# v02 = v01 *mechanism rollback kept + ONE orthogonal axis*:
#
#   AXIS (v02, the only mechanism change):
#     "the information map must carry SPATIAL CONTRAST by construction".
#
#     v01:  I(x) = sigmoid( conv_1x1(u) )
#     v02:  e(x) = conv_1x1(u)                       (1x1, bias=False)
#           e  <- e - mean_{H,W}(e)                  (strip the spatial COMMON MODE)
#           e  <- e / ( rms(e) + 0.01*rms(e_raw) )   (per-sample contrast normalisation;
#                                                     BOTH terms are degree-1 homogeneous
#                                                     in w => I(c*w) == I(w) exactly)
#           I(x) = sigmoid( tanh(e) )                (bounded; DC level fixed at 0.5)
#
#   WHY (measured on the v01 run, not hypothesised):
#     * v01 `info_adaptivity` (std of D over space) = g1 0.0031 / g2 0.0135 / g3 0.0210,
#       i.e. D varies by ~2.4% relative.  D0 was pinned exactly at its clamp 1.5
#       (g2/g3 d0 == 1.5 == INFO_D0_CLAMP) -> the map sat at sigmoid(0)=0.5, a CONSTANT.
#     * v01 ablation `g2_cls0` (force I=0.5, isotropic) vs `full` = rel_shift 0.004
#       (8/2000) while `g2_off` (remove the block) = 0.1645 (329/2000).  => only
#       0.004/0.1645 ~ 2.4% of the field's total effect came from its "information"
#       content; the other 97.6% was plain isotropic diffusion.
#     * cause: `conv_1x1` sat in the WEIGHT-DECAYed group.  wd 5e-4 x lr 0.1 x 62k
#       steps shrinks its weights by e^-3.1 ~ 0.045 -> the pre-activation loses its
#       spatial variation -> I -> sigmoid(bias) -> spatially constant.  Nothing in the
#       task loss rewards spatial variation (a uniform D is just a blur), so wd wins.
#     => v01 never tested the InfoDiff premise: the "information-guided" part was inert.
#        (v58->v59 precedent: fix the delivery, do not kill the axis.)
#
#   WHY the normalisation is the RIGHT fix (not a band-aid):
#     * Dividing by the per-sample rms makes the LOSS exactly scale-invariant in ||w||,
#       so weight decay becomes a mathematical NO-OP on the map's spatial contrast.
#       The map's content lives entirely in the DIRECTION of w (a unit vector in R^C),
#       which wd cannot collapse.  => collapse is structurally impossible, not tuned away.
#     * The optimizer is left UNTOUCHED on purpose: `info_conv.weight` stays in the
#       weight-decayed group.  With exact scale invariance wd provably cannot touch the
#       map, so exempting it would only hide the mechanism -- keeping it is the cleanest
#       possible demonstration that the normalisation (not a group tweak) is what fixes it.
#     * The floor in the denominator is RELATIVE (a fraction of the map's own magnitude),
#       never an absolute eps.  Two things break with an absolute eps, and both were found
#       by the pre-delivery logic check rather than by reading the code:
#         - it destroys the degree-1 homogeneity, reintroducing a wd dependence
#           (measured drift 2.2e-4 in sigmoid units after a 1e-3 weight rescale);
#         - with NO floor, a NEARLY constant feature map has rms at float-noise level
#           (~1e-16 of the map scale) and dividing by it amplifies rounding error to O(1):
#           the map becomes a spurious full-contrast pattern, i.e. the axis injects noise
#           into the diffusivity instead of information.
#       A relative floor satisfies both constraints at once (drift 1.7e-16, and a constant
#       input still collapses to exactly the DC level).
#     * It also DEDUPLICATES the DOF pool: the old learnable bias only shifted D
#       globally, which is exactly what D0 already does (the recorded failure mode
#       "adding a DOF = re-splitting one bounded pool").  Bias removed; D0 is the sole
#       amplitude knob, the map is pure relative modulation.
#     * It is also strictly safer for CFL: I is now inside sigmoid(+/-1) = (0.269, 0.731),
#       so D <= D0*e^{-2*0.269} = 0.584*D0 <= 0.876 (dt*4*D = 0.70 <= 2) vs 1.5 before.
#
#   FALSIFICATION (pre-registered, same run, paired on the full 10k test set):
#     lever_pattern = rel_shift(full vs g2_cls0)     -- does the spatial PATTERN matter?
#     lever_sample  = rel_shift(full vs g2_fixedpat) -- does CROSS-SAMPLE variation matter?
#     dir_pattern   = acc(full) - acc(g2_cls0)   [pp]
#     dir_sample    = acc(full) - acc(g2_fixedpat) [pp]
#     v01 baseline for lever_pattern was 0.004.  If v02 still gives lever_pattern < 0.02
#     the normalisation did not revive the map  =>  INFO_AXIS_DEAD, roll back to v51.
#     If lever_pattern >= 0.02 but dir_pattern <= 0  =>  LEVERAGE_NO_DIRECTION (change the
#     criterion level, not the dose).  Only if BOTH lever and direction are positive does
#     `delivered` get to be interpreted at all.
#
#   Theory anchor (unchanged from v01):
#   * Perona-Malik (1990):  du/dt = div( c(x) grad u ),  c = g(|grad u|)
#   * Information Bottleneck (Tishby 2000; Alemi VIB 2017; Qin et al. 2026
#     arXiv 2609.19122):  min I(T;X) - beta*I(T;Y); CNN forward = spatial
#     compression. Qin controls compression by a SCALAR; we do it SPATIALLY.
#   * Wasserstein Gradient Flow / Otto calculus (Jordan-Kinderlehrer-Otto 1998):
#     the heat flow IS the gradient flow of entropy in Wasserstein space.
#   * du/dt = div( D(x) grad u ),  D(x) = D0 * exp( -beta * I(x) ), explicit Euler.
#     Absorbability argument: for D == const this is a FIXED linear blur, which any
#     adjacent conv+BN absorbs.  Only d(D)/d(u) != 0 escapes it  --  which requires
#     the map NOT to collapse.  That is precisely what this version guarantees.
#
# I/O structure borrowed from B-line (cnn10_v51.py): same recipe (SEED=1234,
# batch 256, 2-phase cosine, mixup+cutmix, TTA flip, best/ens JSON schema),
# same ablation mode names plus the new `g2_fixedpat`.
#
# Discipline: zero argparse, bare command `python cnn10_info_v02.py`, self-contained.
# =============================================================================

import os
import sys
import math
import time
import json
import shutil
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# NOTE: zero argparse by design (bare command `python cnn10_info_v02.py`).

# ========== [v49-style] operator-level determinism hardening (before any CUDA) ==========
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

# ================== Fixed Randomness ==================
# Mirrors B-line: SEED=1234 is the comparable baseline seed (v51 delivered 97.56).
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

# ================== Version / I/O (single source of truth) ==================
INFO_VERSION = "info_v02"
VERSION_TAG = INFO_VERSION
BASE_FILE = "cnn10_v51.py"
BASE_SHA1 = "3516a363737c1930019ba97f7c2853e67fc0eee2"  # cnn10_v51.py (97.56 baseline)
BASE_NOTE = "B-line history-best + backtrack baseline (97.56); archived 2026-09-29."
CKPT_DIR = f'./ckpts_{INFO_VERSION}_s{SEED}'
OUT_DIR = f'./out_{INFO_VERSION}_s{SEED}'
JSON_NAME = f'results_cnn_{INFO_VERSION}_seed{SEED}.json'

# ================== GPU Config ==================
N_GPUS = torch.cuda.device_count()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_PER_GPU = 128
BATCH_SIZE = BATCH_PER_GPU * max(N_GPUS, 1)
LR_BASE = 0.1
WARMUP_EPOCHS = 5
WARMUP_START_LR = 0.01

TOTAL_EPOCHS = 200
COSINE_EPOCHS = 120
TRAIN_EPOCHS = 180
PHASE2_LR_END = 3.0e-5
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

# A-line style backtrack line is irrelevant here; this is B-line domain.
# Redline stays the B-line value so a regression vs 97.56 is caught.
BACKTRACK_REDLINE = 97.04

# InfoDiff hyper-parameters (the new method)
INFO_N_STEPS = 4          # explicit-Euler diffusion steps
INFO_DT = 0.20            # time step (CFL: dt*4*Dmax <= 2; Dmax <= 0.876 => 0.70 ok)
INFO_D0_INIT = 1.0        # global diffusivity scale (learnable log)
INFO_D0_CLAMP = 1.5       # clamp D0 to keep CFL stable (v01: this bound was BINDING,
                          #   d0 hit 1.5 exactly => zero gradient. Reported, not fixed:
                          #   amplitude re-parameterisation is the pre-registered v03 axis.)
INFO_BETA = 2.0           # information sensitivity (fixed for v01/v02; future axis)
INFO_ALPHA_INIT = 0.5     # residual blend init
INFO_USE_LEARNED_INFO = True

# ---- [v02] the axis: non-collapsible information map ----
# NOTE: the only epsilon is a RELATIVE floor (fraction of the map's own magnitude), so
#   the denominator stays positively homogeneous of degree 1 in the weights.  An ABSOLUTE
#   epsilon would break that homogeneity and reintroduce a weight-decay dependence.
INFO_CONTRAST_REL_FLOOR = 1e-2
INFO_MAP_LEVEL = 0.0      # FIXED pre-sigmoid DC level. Deliberately NOT learnable: a
                          #   learnable level only shifts D globally, which is exactly what
                          #   D0 does. Keeping it removes a redundant (flat-gradient) DOF.
# ---- [v02] instrument thresholds (all band text is GENERATED from these) ----
INFOMAP_LEVER_TOL = 0.02      # >=2% of the 10k test set must flip predictions
INFOMAP_DIR_TOL_PP = 0.50     # paired direction tolerance (~2.3 x the paired SE at 500 flips)
INFOMAP_ABL_CAP = 10000       # full CIFAR-10 test set (v01 used 2000 => 0.5pp resolution)
INFOMAP_VERDICT_TOL = {'lever': INFOMAP_LEVER_TOL, 'dir_pp': INFOMAP_DIR_TOL_PP,
                       'map_std': 1e-4}
INFOMAP_MODES = ['full', 'g2_fixedpat', 'g2_cls0', 'g2_clsmax', 'g1_off', 'g2_off', 'g3_off']

# ================== Schedule (copied verbatim from B-line v51) ==================
def _lr_ratio_core(epoch, phase2_end):
    if epoch < WARMUP_EPOCHS:
        return (WARMUP_START_LR / LR_BASE) + (1.0 - WARMUP_START_LR / LR_BASE) * (epoch / WARMUP_EPOCHS)
    elif epoch < COSINE_EPOCHS:
        cos_epoch = epoch - WARMUP_EPOCHS
        cos_total = COSINE_EPOCHS - WARMUP_EPOCHS
        eta_min_ratio = 1e-4 / LR_BASE
        return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1 + math.cos(math.pi * cos_epoch / cos_total))
    elif epoch < TRAIN_EPOCHS:
        cos_epoch = epoch - COSINE_EPOCHS
        cos_total = TRAIN_EPOCHS - COSINE_EPOCHS
        eta_max_ratio = 1e-4 / LR_BASE
        eta_min_ratio = float(phase2_end) / LR_BASE
        return eta_min_ratio + 0.5 * (eta_max_ratio - eta_min_ratio) * (1 + math.cos(math.pi * cos_epoch / cos_total))
    else:
        return float(phase2_end) / LR_BASE

def smooth_cosine_schedule(epoch):
    return _lr_ratio_core(epoch, PHASE2_LR_END)

# Mixup / CutMix schedule (B-line v51)
def get_mixup_alpha(epoch):
    if epoch < 60 or epoch > 120:
        return 0.0
    if epoch < 110:
        return 0.4
    return 0.4 * (1.0 - (epoch - 110) / 10.0)

# ================== TTA / Ensemble config (B-line) ==================
TTA_START_EPOCH = 80
TTA_INTERVAL = 3
EARLY_STOP_PATIENCE = 12
TEST_PATIENCE = 50
ENS_START_EPOCH = 105
ENS_INTERVAL = 3
ENS_MAX_KEEP = 8
ENS_TOP_K = 5

# ================== Utils ==================
def soft_bound(x, min_v, max_v):
    return min_v + (max_v - min_v) * torch.sigmoid(x)

def _reduce(val):
    if isinstance(val, torch.Tensor) and val.dim() > 0:
        return val.mean()
    return val

def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# ====================== CIFAR-10 data (B-line augment) ======================
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def cutmix_data(x, y, beta=1.0):
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(x.size(0)).to(x.device)
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y, y[rand_index], lam

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ====================== [NEW] InfoDiff module ======================
class InfoDiffBlock(nn.Module):
    """Information-Guided Wasserstein Diffusion block.

    du/dt = div( D(x) grad u ),  D(x) = D0 * exp(-beta * I(x))
    I(x)  = sigmoid( tanh( contrast_normalise( conv_1x1(u) ) ) )   [v02, label-free]

    Solved with explicit Euler + Neumann BC (reflect padding). The block is a
    differentiable operator; the info map is shaped by the task loss, so the
    field becomes a spatial information bottleneck (compress low-info, preserve
    high-info regions) without any explicit IB loss.

    bmode: 'adaptive'  -- learned I, full v02 axis
           'fixedpat'  -- [v02 instrument] same spatial pattern, cross-sample
                          variation deleted (detached batch mean) => tests whether
                          the operator's input-dependence is load-bearing
           'const'     -- I=0.5 everywhere (isotropic, field ON)
           'sat'       -- I=1.0 everywhere (max attenuation, isotropic)
           'off'       -- identity, no field at all
    """
    def __init__(self, channels, n_steps=INFO_N_STEPS, dt=INFO_DT, d0_init=INFO_D0_INIT,
                 d0_clamp=INFO_D0_CLAMP, beta=INFO_BETA, use_learned_info=INFO_USE_LEARNED_INFO,
                 alpha_init=INFO_ALPHA_INIT):
        super().__init__()
        self.n_steps = n_steps
        self.dt = dt
        self.d0_clamp = d0_clamp
        self.beta = beta
        self.use_learned_info = use_learned_info
        self.log_d0 = nn.Parameter(torch.tensor(float(math.log(d0_init))))
        self.alpha_logit = nn.Parameter(torch.tensor(float(math.log(alpha_init / (1.5 - alpha_init) + 1e-9))))
        if use_learned_info:
            # [v02] bias=False on purpose: a conv bias is spatially constant, so after the
            # common-mode strip below it would be discarded anyway -- and keeping it would
            # add a DOF that merely shifts D globally, i.e. exactly the job of D0.
            self.info_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        else:
            self.info_conv = None
        self.d0_scale_override = None  # set by gain sweep

    @staticmethod
    def _contrast_normalise(e_raw):
        """[v02] AXIS: strip the spatial common mode, then normalise the contrast.

        e_raw: (B,1,H,W) raw pre-activation -> (B,1,H,W) with unit rms over (C,H,W).

        Why this is a *structural* fix rather than a tuning fix: the denominator is a
        POSITIVELY HOMOGENEOUS function of degree ONE in the weights (rms of the centred
        map plus a fixed fraction of rms of the raw map), so I(c*e) == I(e) EXACTLY for
        every c > 0.  Weight decay can therefore only rescale a quantity the map never
        reads -- it cannot shrink the map's spatial contrast to zero.  The map's content
        lives entirely in the DIRECTION of w (a unit vector in R^C), which wd cannot
        collapse.  Measured: a 1e-3 rescale of w moves I by 1.7e-16.

        The floor is RELATIVE (a fraction of the map's own magnitude), never an absolute
        epsilon, for two reasons:
          * an absolute epsilon would break the degree-1 homogeneity above;
          * without any floor, a NEARLY constant feature map produces rms at float-noise
            level (~1e-16 of the map scale), and dividing by it amplifies that noise to
            O(1) -- the map would become a spurious full-contrast pattern driven by
            rounding error, i.e. the axis would inject noise into the diffusivity.
            With the floor, such a map collapses to the constant level instead.
        The rms==0 branch (all-zero pre-activation) is selected explicitly to avoid 0/0.
        """
        e = e_raw - e_raw.mean(dim=(2, 3), keepdim=True)                 # common mode out
        rms = e.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt()          # variation scale
        scale = e_raw.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt()    # map magnitude
        denom = rms + INFO_CONTRAST_REL_FLOOR * scale                    # degree-1, both
        safe = torch.where(denom > 0, denom, torch.ones_like(denom))
        return e / safe

    def _info_map(self, u):
        # u: (B,C,H,W) -> I: (B,1,H,W) in (0,1)
        if self.use_learned_info and self.info_conv is not None:
            e = self.info_conv(u)
        else:
            # intrinsic Fisher proxy fallback (label-free, unused while
            # INFO_USE_LEARNED_INFO is True): local gradient energy.  Shapes are
            # padded back to (B,1,H,W) -- the previous slicing form produced
            # (B,1,H-2,W) + (B,1,H,W-2) and would have raised on broadcast.
            ex = F.pad((u[..., 1:, :] - u[..., :-1, :]).pow(2).sum(1, keepdim=True),
                       (0, 0, 0, 1))
            ey = F.pad((u[..., :, 1:] - u[..., :, :-1]).pow(2).sum(1, keepdim=True),
                       (0, 1, 0, 0))
            e = ex + ey
        e = self._contrast_normalise(e)
        return torch.sigmoid(INFO_MAP_LEVEL + torch.tanh(e))

    def _make_I(self, u, bmode):
        """Single source of truth for the map, shared by forward() and diagnostics()."""
        B, C, H, W = u.shape
        if bmode == 'const':
            return torch.full((B, 1, H, W), 0.5, device=u.device, dtype=u.dtype)
        if bmode == 'sat':
            return torch.full((B, 1, H, W), 1.0, device=u.device, dtype=u.dtype)
        if bmode == 'fixedpat':
            # [v02] instrument: keep the SPATIAL pattern, delete the CROSS-SAMPLE
            # variation.  A diffusivity map that is fixed (input-independent) makes the
            # operator a fixed linear blur again, hence absorbable by the neighbouring
            # conv+BN.  So `full - vs - fixedpat` isolates the ONE property that can
            # break the absorption theorem: d(D)/d(u) != 0.
            I0 = self._info_map(u).detach()
            return I0.mean(dim=0, keepdim=True).expand(B, 1, H, W)
        return self._info_map(u)

    def _diffusivity(self, I):
        D0 = torch.exp(self.log_d0)
        if self.d0_scale_override is not None:
            D0 = D0 * float(self.d0_scale_override)
        D0 = D0.clamp(max=self.d0_clamp)
        D = D0 * torch.exp(-self.beta * I)
        return D.clamp(min=1e-3, max=self.d0_clamp)

    @staticmethod
    def _step(u, D, dt):
        # u: (B,C,H,W), D: (B,1,H,W); Neumann BC via zero-flux reflect pad.
        gx = u[:, :, 1:, :] - u[:, :, :-1, :]            # (B,C,H-1,W)
        gy = u[:, :, :, 1:] - u[:, :, :, :-1]            # (B,C,H,W-1)
        Dx = 0.5 * (D[:, :, 1:, :] + D[:, :, :-1, :])    # (B,1,H-1,W)
        Dy = 0.5 * (D[:, :, :, 1:] + D[:, :, :, :-1])    # (B,1,H,W-1)
        fx = Dx * gx                                     # (B,C,H-1,W)
        fy = Dy * gy                                     # (B,C,H,W-1)
        fx_p = F.pad(fx, (0, 0, 1, 1))                   # (B,C,H+1,W)
        fy_p = F.pad(fy, (1, 1, 0, 0))                   # (B,C,H,W+1)
        div_x = fx_p[:, :, 1:, :] - fx_p[:, :, :-1, :]   # (B,C,H,W)
        div_y = fy_p[:, :, :, 1:] - fy_p[:, :, :, :-1]   # (B,C,H,W)
        return u + dt * (div_x + div_y)

    def forward(self, u, bmode='adaptive'):
        if bmode == 'off':
            return u
        I = self._make_I(u, bmode)
        D = self._diffusivity(I)
        v = u
        for _ in range(self.n_steps):
            v = self._step(v, D, self.dt)
        alpha = soft_bound(self.alpha_logit, 0.0, 1.5)
        return u + alpha * (v - u)

    def diagnostics(self, u, bmode='adaptive'):
        """Dose readouts.  Two DIFFERENT things must be reported, never one:
        `i_std` / `info_adaptivity` say how much the map varies IN SPACE (is it alive?),
        `i_cross_std` says how much it varies ACROSS SAMPLES (is it input-dependent?).
        v01's map was alive in neither sense (D varied by 2.4% relative, and replacing it
        with the constant 0.5 moved only 8/2000 predictions)."""
        if bmode == 'off':
            return {'info_adaptivity': 0.0, 'i_std': 0.0, 'i_cross_std': 0.0,
                    'alpha': 0.0, 'd0': 0.0, 'd_mean': 0.0, 'clamp_bound': False}
        I = self._make_I(u, bmode)
        D = self._diffusivity(I)
        adapt = float(D.std(dim=(2, 3)).mean().item())          # spatial variation of D
        i_std = float(I.std(dim=(2, 3)).mean().item())          # spatial variation of I
        i_cross = float(I.mean(dim=(2, 3)).std(dim=0).mean().item())  # cross-sample spread
        d_mean = float(D.mean().item())
        d0 = float(torch.exp(self.log_d0).clamp(max=self.d0_clamp).item())
        alpha = float(soft_bound(self.alpha_logit, 0.0, 1.5).item())
        return {'info_adaptivity': adapt, 'i_std': i_std, 'i_cross_std': i_cross,
                'alpha': alpha, 'd0': d0, 'd_mean': d_mean,
                'clamp_bound': bool(d0 >= self.d0_clamp - 1e-9)}


# ====================== ResNet-50 (CIFAR stem) with InfoDiff ======================
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_c, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_c != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, planes * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion))
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class InfoDiffResNet(nn.Module):
    # Single source of truth for the stem width. layer1's input width is DERIVED
    # from it, and every later layer's input width is derived from the previous
    # layer's output. => a stem/layer1 channel mismatch (the crash
    # "conv1 expects 64 but got 256") is structurally impossible here, no matter
    # what the stem is configured to.
    STEM_CHANNELS = 64

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, self.STEM_CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.STEM_CHANNELS)
        self.relu = nn.ReLU(inplace=True)

        # Channel widths derived sequentially: each layer's in_c == previous out_c.
        self.layer1, c1 = self._make_layer(self.STEM_CHANNELS, 64, 3, 1)   # 64   -> 256
        self.layer2, c2 = self._make_layer(c1, 128, 4, 2)                  # 256  -> 512
        self.layer3, c3 = self._make_layer(c2, 256, 6, 2)                  # 512  -> 1024
        self.layer4, c4 = self._make_layer(c3, 512, 3, 2)                  # 1024 -> 2048

        # InfoDiff injection at the three gate points (channels c1/c2/c3).
        self.phys1 = InfoDiffBlock(c1)
        self.phys2 = InfoDiffBlock(c2)
        self.phys3 = InfoDiffBlock(c3)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c4, num_classes)
        self.channel_config = {
            'stem_out': self.STEM_CHANNELS,
            'layer1_in': self.STEM_CHANNELS, 'layer1_out': c1,
            'layer2_in': c1, 'layer2_out': c2,
            'layer3_in': c2, 'layer3_out': c3,
            'layer4_in': c3, 'layer4_out': c4,
        }

    @staticmethod
    def _make_layer(in_c, planes, blocks, stride):
        layers = [Bottleneck(in_c, planes, stride)]
        out_c = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_c, planes, 1))
        return nn.Sequential(*layers), out_c

    def _bmode(self, mode, idx):
        # resolve a global ablation 'mode' to a per-block bmode
        if mode == 'all_off':
            return 'off'
        if mode == 'g1_off' and idx == 0:
            return 'off'
        if mode == 'g2_off' and idx == 1:
            return 'off'
        if mode == 'g3_off' and idx == 2:
            return 'off'
        if mode == 'g2_cls0' and idx == 1:
            return 'const'
        if mode == 'g2_clsmax' and idx == 1:
            return 'sat'
        if mode == 'g2_fixedpat' and idx == 1:
            return 'fixedpat'
        return 'adaptive'

    def forward(self, x, mode='full'):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.phys1(x, self._bmode(mode, 0))
        x = self.layer2(x)
        x = self.phys2(x, self._bmode(mode, 1))
        x = self.layer3(x)
        x = self.phys3(x, self._bmode(mode, 2))
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out

    def field_diagnostics(self, x, mode='full'):
        # run a batch through and collect per-block adaptivity evidence
        x = self.relu(self.bn1(self.conv1(x)))
        diags = {}
        x = self.layer1(x)
        diags['g1'] = self.phys1.diagnostics(x, self._bmode(mode, 0))
        x = self.layer2(x)
        diags['g2'] = self.phys2.diagnostics(x, self._bmode(mode, 1))
        x = self.layer3(x)
        diags['g3'] = self.phys3.diagnostics(x, self._bmode(mode, 2))
        return diags


# ====================== Disk guard (proven v67 pattern) ======================
DISK_GUARD_ENABLE = True
DISK_RESERVE_MB = 1024.0
DISK_HEADROOM_MB = 256.0
DISK_ASSUMED_MB = 1200.0
DISK_SAMPLE_LOG_MAX = 4096
DISK_EVENTS = []
DISK_STATS = {'admit_total': 0, 'admit_ok': 0, 'admit_reject': 0, 'write_error': 0,
              'read_fail': 0, 'recover_ok': 0}
DISK_LAST_WRITE = {}
DISK_TRACE = []
DISK_MIN_FREE_SEEN = [None]

def _disk_free_mb(path):
    root = _disk_probe_root(path)
    try:
        st = shutil.disk_usage(root)
        return st.free / (1024.0 * 1024.0)
    except Exception:
        return None

def _payload_mb(obj):
    # Disk-free size estimate: sum tensor bytes in a state_dict (no torch.save =>
    # no risk of hitting a full disk inside the estimator itself, and faster).
    try:
        total = 0
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, torch.Tensor):
                    total += v.numel() * v.element_size()
        elif isinstance(obj, torch.Tensor):
            total = obj.numel() * obj.element_size()
        else:
            return DISK_ASSUMED_MB
        if total == 0:
            return DISK_ASSUMED_MB
        return total / (1024.0 * 1024.0)
    except Exception:
        return DISK_ASSUMED_MB

def _disk_probe_root(path):
    # walk up the dir chain to the first existing ancestor (handles not-yet-created dirs)
    d = os.path.dirname(os.path.abspath(str(path)))
    while d and not os.path.exists(d):
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return d or '.'

def _disk_admit(path, need_mb, kind):
    # [v02] synthetic probes must not be counted as real I/O.  In v01 they were, so the
    # guard's own self-test produced 2 rejects and the JSON verdict became
    # DISK_GUARD_INTERVENTION even though the real run was 34 admitted / 0 rejected.
    probe = (kind == 'selftest')
    if not probe:
        _DISK_STATS_INC('admit_total')
    free = _disk_free_mb(path)
    note = {'kind': kind, 'need_mb': need_mb, 'free_mb': free, 'ok': False, 'reason': ''}
    if free is None:
        note['reason'] = 'free_unknown'
        _DISK_EVENTS_APPEND(note)
        return False, note
    need = DISK_ASSUMED_MB if need_mb is None else need_mb
    if free - need < DISK_RESERVE_MB:
        note['reason'] = 'insufficient_after_reserve'
        if not probe:
            _DISK_STATS_INC('admit_reject')
        _DISK_EVENTS_APPEND(note)
        return False, note
    note['ok'] = True
    if not probe:
        _DISK_STATS_INC('admit_ok')
    _DISK_EVENTS_APPEND(note)
    return True, note

def _DISK_STATS_INC(k):
    if k in DISK_STATS:
        DISK_STATS[k] += 1

def _DISK_EVENTS_APPEND(n):
    DISK_EVENTS.append(n)
    if len(DISK_EVENTS) > 4096:
        del DISK_EVENTS[:len(DISK_EVENTS) - 4096]
    if n.get('kind') == 'selftest':
        return          # [v02] a synthetic probe must not set the real low-water mark
    f = n.get('free_mb')
    if isinstance(f, (int, float)):
        if DISK_MIN_FREE_SEEN[0] is None or f < DISK_MIN_FREE_SEEN[0]:
            DISK_MIN_FREE_SEEN[0] = f

def _disk_note(ev, ok=None):
    if ok is not None:
        ev['ok'] = bool(ok)
    return ev

def safe_save(obj, filepath, kind='ckpt'):
    need = _payload_mb(obj)
    need = None if need is None else (float(need) + float(DISK_HEADROOM_MB))
    ok, ev = _disk_admit(filepath, need, kind)
    if ok:
        tmp = str(filepath) + '.tmp'
        try:
            d = os.path.dirname(os.path.abspath(str(filepath)))
            if d:
                os.makedirs(d, exist_ok=True)
            torch.save(obj, tmp)
            os.replace(tmp, filepath)
            DISK_LAST_WRITE[str(filepath)] = time.time()
            _trace_sample(ev)
            return _disk_note(ev, True)
        except Exception as e:
            _DISK_STATS_INC('write_error')
            ev['ok'] = False
            ev['reason'] = 'write_error:%s' % type(e).__name__
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            return _disk_note(ev, False)
    return _disk_note(ev, False)

def _load_weights_if_exists(net, filepath, tag=''):
    if os.path.exists(filepath):
        try:
            net.load_state_dict(torch.load(filepath, map_location=device))
            return True
        except Exception:
            _DISK_STATS_INC('read_fail')
            return False
    _DISK_STATS_INC('read_fail')
    return False

def safe_save_json(obj, primary_path):
    ok, ev = _disk_admit(primary_path, 1.0, 'json')
    if ok:
        tmp = str(primary_path) + '.tmp'
        try:
            d = os.path.dirname(os.path.abspath(str(primary_path)))
            if d:
                os.makedirs(d, exist_ok=True)
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(obj, f, ensure_ascii=False, indent=1)
            os.replace(tmp, primary_path)
            _trace_sample(ev)
            return _disk_note(ev, True)
        except Exception as e:
            _DISK_STATS_INC('write_error')
            ev['ok'] = False
            ev['reason'] = 'write_error:%s' % type(e).__name__
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            return _disk_note(ev, False)
    return _disk_note(ev, False)

def _trace_sample(ev):
    if len(DISK_TRACE) < DISK_SAMPLE_LOG_MAX:
        DISK_TRACE.append({'kind': ev.get('kind'), 'ok': ev.get('ok'),
                           'free_mb': ev.get('free_mb'), 't': time.time()})

def _disk_guard_selftest():
    """Known degenerate inputs: demand >> free must be rejected; a 1 MB write must pass.

    [v02] v01's version could NEVER pass.  It monkeypatched `_disk_free_mb` to 10 MB but
    left DISK_RESERVE_MB at 1024, so `free - need = 9 MB < 1024 MB reserve` rejected the
    "must pass" probe: `selftest.pass` was false in every run, and because the probes were
    also counted as real I/O, `_disk_guard_verdict()` returned DISK_GUARD_INTERVENTION for
    a run whose real writes were 34 admitted / 0 rejected / 0 write errors.  The reserve
    has to be patched inside the same sandbox as the free-space probe.
    """
    orig_free = _disk_free_mb
    orig_reserve = DISK_RESERVE_MB
    selftest_pass = False
    try:
        globals()['_disk_free_mb'] = lambda p: 10.0    # 10 MB free
        globals()['DISK_RESERVE_MB'] = 2.0             # reserve must fit inside the probe
        ok_big, _ = _disk_admit('/fake/x', 5000.0, 'selftest')   # 5000 >> 10 => reject
        ok_small, _ = _disk_admit('/fake/x', 1.0, 'selftest')    # 10 - 1 >= 2 => admit
        selftest_pass = (ok_big is False) and (ok_small is True)
    finally:
        globals()['_disk_free_mb'] = orig_free
        globals()['DISK_RESERVE_MB'] = orig_reserve
    return {'ran': True, 'pass': bool(selftest_pass),
            'expect_reject_big': True, 'expect_allow_small': True}

def _disk_guard_verdict():
    # Counts REAL I/O only (probes are excluded inside _disk_admit).
    reject = DISK_STATS.get('admit_reject', 0)
    werr = DISK_STATS.get('write_error', 0)
    if reject == 0 and werr == 0:
        return 'DISK_GUARD_CLEAN'
    return 'DISK_GUARD_INTERVENTION'


# ====================== Main training ======================
def build_model():
    net = InfoDiffResNet(num_classes=10)
    cfg = net.channel_config
    flush_print(f"[model] channels: stem_out={cfg['stem_out']} | "
                f"layer1 {cfg['layer1_in']}->{cfg['layer1_out']} | "
                f"layer2 {cfg['layer2_in']}->{cfg['layer2_out']} | "
                f"layer3 {cfg['layer3_in']}->{cfg['layer3_out']} | "
                f"layer4 {cfg['layer4_in']}->{cfg['layer4_out']}")

    # ---- Architecture self-test (runs ONCE at startup, CPU, no grad) ----
    # Uses explicit raises (NOT assert) so it also fires under `python -O`,
    # where asserts are stripped. Any channel/shape drift is reported here at
    # startup with the exact stage, instead of as a cryptic DataParallel
    # RuntimeError mid-training.
    with torch.no_grad():
        _d = torch.randn(2, 3, 32, 32)
        x = net.relu(net.bn1(net.conv1(_d)))
        stages = [('stem', x, cfg['stem_out'])]
        x = net.layer1(x); stages.append(('layer1', x, cfg['layer1_out']))
        x = net.phys1(x, 'adaptive'); stages.append(('phys1', x, cfg['layer1_out']))
        x = net.layer2(x); stages.append(('layer2', x, cfg['layer2_out']))
        x = net.phys2(x, 'adaptive'); stages.append(('phys2', x, cfg['layer2_out']))
        x = net.layer3(x); stages.append(('layer3', x, cfg['layer3_out']))
        x = net.phys3(x, 'adaptive'); stages.append(('phys3', x, cfg['layer3_out']))
        x = net.layer4(x); stages.append(('layer4', x, cfg['layer4_out']))
        for _name, _t, _exp in stages:
            if _t.shape[1] != _exp:
                raise RuntimeError(
                    f"[self-test] stage '{_name}' channels={_t.shape[1]} (expected {_exp}). "
                    f"The deployed file is not the intended architecture "
                    f"(stem must be nn.Conv2d(3, {net.STEM_CHANNELS}, ...)).")
        _out = net(_d, 'full')
        if tuple(_out.shape) != (2, 10):
            raise RuntimeError(f"[self-test] output shape={tuple(_out.shape)} (expected (2,10))")
        for _m in ['g2_cls0', 'g2_clsmax', 'g2_fixedpat', 'g1_off', 'g2_off',
                   'g3_off', 'all_off']:
            _o = net(_d, _m)
            if tuple(_o.shape) != (2, 10):
                raise RuntimeError(f"[self-test] mode={_m} shape={tuple(_o.shape)} (expected (2,10))")
        _dd = net.field_diagnostics(_d, 'full')
        if set(_dd.keys()) != {'g1', 'g2', 'g3'}:
            raise RuntimeError(f"[self-test] field_diagnostics keys={set(_dd)}")
        for _need in ('info_adaptivity', 'i_std', 'i_cross_std', 'alpha', 'd0', 'clamp_bound'):
            if _need not in _dd['g2']:
                raise RuntimeError(f"[self-test] field_diagnostics['g2'] missing '{_need}'")
        del _d, _out, _o, _dd, x, stages

    # ---- [v02] axis self-test: the information map must be non-constant AND
    # weight-decay-immune.  Two known-degenerate inputs, asserted explicitly so a
    # silent re-collapse of the axis (exactly the v01 failure) cannot ship again.
    with torch.no_grad():
        _blk = net.phys1
        _c1 = net.channel_config['layer1_out']
        if getattr(_blk, 'info_conv', None) is None:
            raise RuntimeError(
                "[self-test] the v02 axis requires the LEARNED info map "
                "(INFO_USE_LEARNED_INFO=True); info_conv is None.")
        _saved_w = _blk.info_conv.weight.detach().clone()
        _u = torch.randn(4, _c1, 8, 8)
        _I0 = _blk._info_map(_u)
        _rng0 = float((_I0.max() - _I0.min()).item())
        # (a) non-constant: a random feature batch must produce a spatially varying map
        if _rng0 <= 1e-6:
            raise RuntimeError(
                f"[self-test] v02 axis dead at init: info map range={_rng0:.3e} "
                f"(spatially constant). The map must carry spatial contrast.")
        # (b) scale invariance: simulate 62k steps of weight decay (w -> 1e-3 * w)
        _blk.info_conv.weight.mul_(1e-3)
        _I1 = _blk._info_map(_u)
        _drift = float((_I1 - _I0).abs().max().item())
        _blk.info_conv.weight.copy_(_saved_w)
        if _drift > 1e-4:
            raise RuntimeError(
                f"[self-test] v02 axis not scale-invariant: wd-scale change moved the map "
                f"by {_drift:.3e} (expected < 1e-4). The contrast normalisation is not "
                f"absorbing the weight scale, so wd can collapse the axis again.")
        # (c) documented degenerate input: a spatially constant feature map must give
        #     exactly I == 0.5 (the common-mode strip makes this exact, not approximate)
        _uc = torch.randn(2, _c1, 8, 8).mean(dim=(2, 3), keepdim=True).expand(2, _c1, 8, 8)
        _dev = float((_blk._info_map(_uc) - 0.5).abs().max().item())
        _blk.info_conv.weight.copy_(_saved_w)
        if _dev > 1e-6:
            raise RuntimeError(
                f"[self-test] constant input must map to I==0.5 exactly, got dev={_dev:.3e}")
        # (d) NEARLY constant input (variation 1e-7 of the map's own magnitude) must not be
        #     amplified into a full-contrast map by float noise.  Without the relative floor
        #     in the denominator, rms sits at ~1e-16 of the map scale and e/rms blows the
        #     rounding error up to O(1) -- the axis would inject NOISE into the diffusivity.
        _un = torch.randn(2, _c1, 8, 8)
        _un = _un.mean(dim=(2, 3), keepdim=True) + 1e-7 * _un
        _dn = float((_blk._info_map(_un) - 0.5).abs().max().item())
        if _dn > 1e-2:
            raise RuntimeError(
                f"[self-test] nearly-constant input amplified to dev={_dn:.3e} "
                f"(expected < 1e-2): the relative contrast floor is not absorbing "
                f"float-noise-level variation.")
        del _u, _I0, _I1, _uc, _un, _saved_w, _blk
    flush_print(f"[self-test] architecture OK (channels + all ablation modes + diagnostics)")
    flush_print(f"[self-test] v02 axis OK (map range at init {_rng0:.4f}; "
                f"wd-scale drift {_drift:.2e}; const-input dev {_dev:.2e}; "
                f"near-const dev {_dn:.2e})")

    if N_GPUS > 1 and torch.cuda.is_available():
        net = nn.DataParallel(net)
    net = net.to(device)
    return net


def build_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=200, shuffle=False,
                            num_workers=4, pin_memory=True)
    return trainloader, testloader


def main():
    print(f"Device: {device} | Available GPUs: {N_GPUS}")
    print(f"[InfoDiff] version={INFO_VERSION}  base={BASE_FILE} (sha1 {BASE_SHA1})")
    print(f"Batch: {BATCH_SIZE} | LR: {LR_BASE} | Epochs: {TRAIN_EPOCHS} | "
          f"InfoDiff steps={INFO_N_STEPS} dt={INFO_DT} beta={INFO_BETA}")
    print(f"[{INFO_VERSION} axis] non-collapsible info map: "
          f"I = sigmoid({INFO_MAP_LEVEL:+.1f} + tanh(contrast_norm(conv1x1(u)))) | "
          f"bias=False | lever_tol={INFOMAP_LEVER_TOL} dir_tol={INFOMAP_DIR_TOL_PP}pp | "
          f"ablation cap={INFOMAP_ABL_CAP}")
    print(f"[{INFO_VERSION} criterion] delivered >= {BACKTRACK_REDLINE} required to avoid "
          f"rolling back to {BASE_FILE}; z = (delivered - 97.56) / 0.169")
    sys.stdout.flush()

    trainloader, testloader = build_data()
    net = build_model()
    model = net

    ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR_BASE, momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=smooth_cosine_schedule)

    best_acc = 0.0
    best_epoch = 0
    best_tta_acc = 0.0
    best_tta_epoch = 0
    no_improve_count = 0
    no_test_improve = 0
    early_stopped = False
    stop_reason_out = ''
    last_epoch = 0
    history_test = []
    history_tta = []
    ckpt_pool = []

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    def save_ckpt(epoch, test_acc):
        filepath = os.path.join(CKPT_DIR, f'ckpt_ep{epoch}_acc{test_acc:.2f}.pth')
        safe_save(model.state_dict(), filepath, kind='ckpt')
        ckpt_pool.append((epoch, test_acc, filepath))
        ckpt_pool.sort(key=lambda x: x[1], reverse=True)
        if len(ckpt_pool) > ENS_MAX_KEEP:
            removed = ckpt_pool.pop()
            try:
                os.remove(removed[2])
            except OSError:
                pass

    def test_raw():
        """Top-1 accuracy on the CIFAR-10 TEST set, no TTA, single view."""
        net.eval()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        c, t = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                c += net(inputs, 'full').argmax(1).eq(targets).sum().item()
                t += targets.size(0)
        return 100. * c / t

    def test_tta(epoch):
        net.eval()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs_flip = torch.flip(inputs, dims=[3])
                out_orig = net(inputs, 'full')
                out_flip = net(inputs_flip, 'full')
                outputs = (out_orig + out_flip) / 2.0
                correct += outputs.argmax(1).eq(targets).sum().item()
                total += targets.size(0)
        return 100. * correct / total

    def ensemble_predict(ckpt_files, use_tta=True):
        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                avg_probs = None
                for ckpt_file in ckpt_files:
                    model.load_state_dict(torch.load(ckpt_file, map_location=device))
                    if use_tta:
                        inputs_flip = torch.flip(inputs, dims=[3])
                        o1 = net(inputs, 'full')
                        o2 = net(inputs_flip, 'full')
                        probs = F.softmax((o1 + o2) / 2.0, dim=1)
                    else:
                        probs = F.softmax(net(inputs, 'full'), dim=1)
                    avg_probs = probs if avg_probs is None else avg_probs + probs
                avg_probs /= len(ckpt_files)
                correct += avg_probs.argmax(1).eq(targets).sum().item()
                total += targets.size(0)
        return 100. * correct / total

    for epoch in range(1, TRAIN_EPOCHS + 1):
        last_epoch = epoch
        epoch_start_time = time.time()
        net.train()
        scheduler.step(epoch - 1)
        mixup_alpha = get_mixup_alpha(epoch)
        use_mixup = mixup_alpha > 0.01
        total_loss, correct, total = 0, 0, 0
        if use_mixup:
            flush_print(f"\nEpoch {epoch} | Mixup=ON (alpha={mixup_alpha:.2f}) | lr={optimizer.param_groups[0]['lr']:.6f}")
        else:
            flush_print(f"\nEpoch {epoch} | Mixup=OFF | lr={optimizer.param_groups[0]['lr']:.6f}")
        for batch, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if use_mixup:
                if random.random() < 0.5:
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=mixup_alpha)
                else:
                    inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, beta=1.0)
                outputs = net(inputs, 'full')
                cls_loss = mixup_criterion(ce, outputs, targets_a, targets_b, lam)
            else:
                outputs = net(inputs, 'full')
                cls_loss = ce(outputs, targets)
            # InfoDiff is an architectural inductive bias (no separate physics loss):
            # the task loss shapes info_conv => spatial information bottleneck.
            loss = cls_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            if use_mixup:
                correct += (lam * outputs.argmax(1).eq(targets_a).sum().item()
                            + (1 - lam) * outputs.argmax(1).eq(targets_b).sum().item())
            else:
                correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets_a.size(0) if use_mixup else targets.size(0)
            if batch % 50 == 0:
                flush_print(f"\r  Loss:{total_loss/(batch+1):.2f} | Acc:{100.*correct/total:.2f}%", end="")
        flush_print()

        # raw test accuracy (no TTA) -- printed every epoch so the run is readable live
        net.eval()
        val_acc = test_raw()
        history_test.append({"epoch": int(epoch), "acc": round(float(val_acc), 4)})
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            no_test_improve = 0
            flush_print(f"  Test: {val_acc:.2f}% | *** New Test Best (Epoch {epoch}) ***")
        else:
            no_test_improve += 1
            flush_print(f"  Test: {val_acc:.2f}% | Best Test: {best_acc:.2f}% "
                        f"(Epoch {best_epoch}) | No improve: {no_test_improve}/{TEST_PATIENCE}")

        if epoch >= ENS_START_EPOCH and epoch % ENS_INTERVAL == 0:
            save_ckpt(epoch, val_acc)
            flush_print(f"  [ENS] Saved ckpt ep{epoch} (acc={val_acc:.2f}%) | pool={len(ckpt_pool)}")

        run_tta = epoch >= TTA_START_EPOCH and epoch % TTA_INTERVAL == 0
        if run_tta:
            tta_acc = test_tta(epoch)
            history_tta.append({"epoch": int(epoch), "acc": round(float(tta_acc), 4)})
            if tta_acc > best_tta_acc:
                best_tta_acc = tta_acc
                best_tta_epoch = epoch
                no_improve_count = 0
                safe_save(model.state_dict(), os.path.join(OUT_DIR, f'best_tta_resnet_{INFO_VERSION}.pth'), kind='ckpt')
                flush_print(f"  TTA: {tta_acc:.2f}% | *** New TTA Best (Epoch {epoch}) ***")
            else:
                no_improve_count += 1
                flush_print(f"  TTA: {tta_acc:.2f}% | TTA Best: {best_tta_acc:.2f}% (Epoch {best_tta_epoch}) | No improve: {no_improve_count}/{EARLY_STOP_PATIENCE}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        epoch_time = time.time() - epoch_start_time
        flush_print(f"  Epoch {epoch} time: {epoch_time:.1f}s")
        stop_reason = None
        if no_improve_count >= EARLY_STOP_PATIENCE:
            stop_reason = f"TTA no improvement for {EARLY_STOP_PATIENCE} evaluations"
        elif no_test_improve >= TEST_PATIENCE and epoch >= 120:
            stop_reason = f"Test acc no improvement for {TEST_PATIENCE} epochs"
        if stop_reason:
            early_stopped = True
            stop_reason_out = str(stop_reason)
            flush_print(f"\nEarly stopping at epoch {epoch}! Reason: {stop_reason}")
            break

    # ----- Final evaluation -----
    final_test_acc = test_raw()
    final_tta = test_tta(TRAIN_EPOCHS)
    ens_acc = 0.0
    ens_tta = 0.0
    if len(ckpt_pool) >= 2:
        top_k = min(ENS_TOP_K, len(ckpt_pool))
        top_ckpts = ckpt_pool[:top_k]
        ckpt_files = [c[2] for c in top_ckpts]
        ens_acc = ensemble_predict(ckpt_files, use_tta=False)
        ens_tta = ensemble_predict(ckpt_files, use_tta=True)
        flush_print(f"  Ensemble Test: {ens_acc:.2f}% | Ensemble TTA: {ens_tta:.2f}%")

    # ----- TEST-SET accuracy report (explicit, both views, with epochs) -----
    _tt = {int(h['epoch']): float(h['acc']) for h in history_test}
    test_acc_at_best_tta_epoch = _tt.get(int(best_tta_epoch), None)
    test_accuracy = {
        'final': round(float(final_test_acc), 4),
        'final_epoch': int(last_epoch),
        'best': round(float(best_acc), 4),
        'best_epoch': int(best_epoch),
    }
    tta_accuracy = {
        'final': round(float(final_tta), 4),
        'final_epoch': int(last_epoch),
        'best': round(float(best_tta_acc), 4),
        'best_epoch': int(best_tta_epoch),
    }

    # Delivered = best measured test accuracy.  The CRITERION (redline / z) stays on the
    # three TTA-derived members so the number remains comparable with info_v01 (97.06);
    # the all-6 max is reported alongside purely as information.
    _members = {'best_test': float(best_acc), 'best_tta': float(best_tta_acc),
                'final_test': float(final_test_acc), 'final_tta': float(final_tta),
                'ens_test': float(ens_acc), 'ens_tta': float(ens_tta)}
    _tta_members = ('best_tta', 'final_tta', 'ens_tta')
    delivered_tta_max = max(_members[k] for k in _tta_members)
    delivered_all_max = max(_members.values())
    delivered = delivered_tta_max
    delivered_source = max(_tta_members, key=lambda k: _members[k])
    backtrack_needed = bool(delivered < BACKTRACK_REDLINE)

    # ----- Instrumentation: run diagnostics FIRST (the ablation verdict consumes them) -----
    field_diag = _run_field_diagnostics(model, testloader)

    # ----- Instrumentation: field_ablation (InfoDiff modes) -----
    field_ablation = _run_field_ablation(model, testloader, field_diag)

    # ----- Instrumentation: gate2 (diffusivity) gain sweep -----
    gate2_gain_sweep = _run_gain_sweep(model, testloader)

    # ----- Replicate z vs B-line baseline 97.56 (sigma_diff 0.169) -----
    sigma_diff = 0.169
    base_acc = 97.56
    z_combined = (delivered - base_acc) / sigma_diff
    if z_combined >= 2.0:
        z_band = 'ABOVE_BASELINE'
    elif z_combined <= -2.0:
        z_band = 'SIGNIFICANT_REGRESSION'
    else:
        z_band = 'WITHIN_NOISE'
    replicate = {
        'base_file': BASE_FILE, 'base_sha1': BASE_SHA1, 'base_acc': base_acc,
        'sigma_diff': sigma_diff, 'delivered': delivered,
        'z_combined': round(float(z_combined), 4),
        'z_band': z_band,
        'verdict': {'ABOVE_BASELINE': 'CONFIRMED',
                    'SIGNIFICANT_REGRESSION': 'REGRESS',
                    'WITHIN_NOISE': 'NULL'}[z_band],
    }

    # ----- Disk guard verdict -----
    dg_selftest = _disk_guard_selftest()
    dg_verdict = _disk_guard_verdict()

    # ----- Delivery verdict: every band text is GENERATED from the same thresholds -----
    # v01 recorded verdict = NULL_BELOW_BASELINE_NO_REDLINE for a run whose own z was
    # -2.96, i.e. a metric that is a significant regression was described by a band whose
    # name says NULL.  Same defect family as pitfall 164 (v85: a hand-written band sentence
    # describing an outcome its own thresholds exclude).  Fixed by deriving both.
    if backtrack_needed:
        verdict = 'BACKTRACK_REQUIRED'
        verdict_text = (f"delivered {delivered:.2f} < redline {BACKTRACK_REDLINE} "
                        f"=> 逐字回退 {BASE_FILE}")
    elif z_band == 'ABOVE_BASELINE':
        verdict = 'DELIVERED_ABOVE_BASELINE'
        verdict_text = (f"delivered {delivered:.2f} >= baseline {base_acc} "
                        f"(z={z_combined:+.2f} >= +2) => 交付成立, 保留本文件")
    elif z_band == 'SIGNIFICANT_REGRESSION':
        verdict = 'REGRESS_ABOVE_REDLINE'
        verdict_text = (f"delivered {delivered:.2f}: z={z_combined:+.2f} <= -2 显著低于基线 "
                        f"{base_acc}, 但 >= redline {BACKTRACK_REDLINE} => 不触发逐字回退; "
                        f"先读 mechanism.verdict 再决定轴的去留")
    else:
        verdict = 'NULL_WITHIN_NOISE'
        verdict_text = (f"delivered {delivered:.2f} 与基线 {base_acc} 之差 "
                        f"|z|={abs(z_combined):.2f} < 2 => 噪声内, 不构成交付")

    results = {
        'version': INFO_VERSION,
        'base': BASE_FILE,
        'base_sha1': BASE_SHA1,
        'base_note': BASE_NOTE,
        'axis': ('non-collapsible information map: the diffusivity is modulated by a '
                 'spatially-CONTRAST-NORMALISED, weight-decay-immune learned map '
                 'I = sigmoid(level + tanh(contrast_norm(conv1x1(u)))); '
                 'D(x) = D0*exp(-beta*I(x)), explicit-Euler PDE.'),
        'axis_short': 'contrast-normalised info map (v01 map had decayed to the constant 0.5)',
        'axis_constants': {
            'INFO_CONTRAST_REL_FLOOR': INFO_CONTRAST_REL_FLOOR,
            'INFO_MAP_LEVEL': INFO_MAP_LEVEL,
            'INFO_BETA': INFO_BETA,
            'INFO_D0_INIT': INFO_D0_INIT,
            'INFO_D0_CLAMP': INFO_D0_CLAMP,
            'INFO_N_STEPS': INFO_N_STEPS,
            'INFO_DT': INFO_DT,
            'info_conv_bias': False,
            'wt_decay_on_info_conv': True,
        },
        'why_this_axis': (
            'v01 measured: info_adaptivity 0.0135 over d_mean~0.55 => d_rel ~2.4%; '
            'g2_cls0 vs full flipped only 8/2000 predictions while g2_off flipped 329/2000 '
            '=> ~97.6% of the field effect was plain isotropic diffusion and the '
            '"information-guided" part was inert. Cause: conv1x1 sat in the weight-decayed '
            'group, so wd (5e-4 x 0.1 x 62k steps = e^-3.1) shrank its weights ~22x and the '
            'map converged to sigmoid(0)=0.5. Dividing by the per-sample rms makes the loss '
            'exactly scale-invariant in ||w||, so wd becomes a mathematical no-op on the '
            "map's spatial contrast: collapse is structurally impossible, not tuned away."),
        'why_prev_axis_kept': (
            'The InfoDiff premise itself was never tested in v01 (its defining feature was '
            'inert), and the redline was not crossed, so the method is repaired rather than '
            'replaced -- v58->v59 precedent: fix the delivery, do not kill the axis.'),
        'v01_diagnosis': {
            'delivered': 97.06, 'base': 97.56, 'z': -2.9586,
            'info_adaptivity_g2': 0.013493, 'd0_g2': 1.5, 'd0_clamped': True,
            'g2_cls0_rel_shift': 0.004, 'g2_cls0_n_disagree': 8,
            'g2_off_rel_shift': 0.1645, 'g2_off_n_disagree': 329,
            'field_share_from_information_content': 0.004 / 0.1645,
            'instrument_defects_fixed_here': [
                'per-mode accuracy accumulator was dead code (...) if False else 0',
                'ablation subset capped at 2000 (0.5pp resolution)',
                'gain sweep was one-sided and saturated at the D0 clamp',
                'disk_guard selftest could never pass and its probes were counted as real I/O',
                'verdict band name contradicted its own z (NULL band, z=-2.96)',
            ],
        },
        'seed': SEED,
        'n_gpus': N_GPUS,
        'batch_size': BATCH_SIZE,
        'epoch': int(TRAIN_EPOCHS),
        'actual_epochs': int(len(history_test)),
        'early_stopped': bool(early_stopped),
        'stop_reason': stop_reason_out,
        # ---- TEST-SET accuracy (explicit, both views, with the epoch of the best) ----
        'test_accuracy': test_accuracy,
        'tta_accuracy': tta_accuracy,
        'test_acc_at_best_tta_epoch': (round(float(test_acc_at_best_tta_epoch), 4)
                                       if test_acc_at_best_tta_epoch is not None else None),
        'all_members': {k: round(v, 4) for k, v in _members.items()},
        # legacy flat keys kept so v01 comparisons stay mechanical
        'best_acc': round(float(best_acc), 4),
        'best_epoch': int(best_epoch),
        'best_tta_acc': round(float(best_tta_acc), 4),
        'best_tta_epoch': int(best_tta_epoch),
        'final_test': round(float(final_test_acc), 4),
        'final_tta': round(float(final_tta), 4),
        'ens_acc': round(float(ens_acc), 4),
        'ens_tta': round(float(ens_tta), 4),
        'delivered': round(float(delivered), 4),
        'delivered_source': delivered_source,
        'delivered_is_max_of_n': len(_tta_members),
        'delivered_n_members': list(_tta_members),
        'delivered_all_max': round(float(delivered_all_max), 4),
        'delivered_definition': ('criterion uses max(best_tta, final_tta, ens_tta), identical '
                                 'to info_v01 so the redline/z stays comparable; '
                                 'delivered_all_max is the max over all 6 measured members'),
        'backtrack_redline': BACKTRACK_REDLINE,
        'backtrack_needed': bool(backtrack_needed),
        'backtrack_advice': ('逐字回退 cnn10_v51.py' if backtrack_needed else 'no backtrack'),
        'verdict': verdict,
        'verdict_text': verdict_text,
        'delivery_verdict': verdict,
        'mechanism_verdict': field_ablation.get('verdict'),
        'mechanism_verdict_text': field_ablation.get('verdict_text'),
        'replicate': replicate,
        'disk_guard': {
            'enable': DISK_GUARD_ENABLE,
            'selftest': dg_selftest,
            'stats': DISK_STATS,
            'events': DISK_EVENTS[-20:],
            'min_free_seen_mb': DISK_MIN_FREE_SEEN[0],
            'verdict': dg_verdict,
        },
        'field_ablation': field_ablation,
        'gate2_gain_sweep': gate2_gain_sweep,
        'field_diagnostics': field_diag,
        'history_test': history_test,
        'history_tta': history_tta,
    }

    out_path = os.path.join(OUT_DIR, JSON_NAME)
    ev = safe_save_json(results, out_path)

    # ----- Console summary (test accuracy first: that is what the run is read for) -----
    flush_print("\n" + "=" * 78)
    flush_print(f"[{INFO_VERSION}] TEST SET ACCURACY")
    flush_print(f"  final test acc         : {final_test_acc:.2f}%  (epoch {last_epoch})")
    flush_print(f"  best  test acc         : {best_acc:.2f}%  (epoch {best_epoch})")
    flush_print(f"  final TTA              : {final_tta:.2f}%  (epoch {last_epoch})")
    flush_print(f"  best  TTA              : {best_tta_acc:.2f}%  (epoch {best_tta_epoch})")
    flush_print(f"  test acc @ best-TTA ep : "
                f"{test_acc_at_best_tta_epoch if test_acc_at_best_tta_epoch is None else round(test_acc_at_best_tta_epoch, 2)}%")
    flush_print(f"  ensemble test / TTA    : {ens_acc:.2f}% / {ens_tta:.2f}%")
    flush_print(f"  delivered              : {delivered:.2f}%  (source={delivered_source})")
    flush_print(f"  baseline {base_acc} -> z = {z_combined:+.2f} ({z_band})   "
                f"redline {BACKTRACK_REDLINE}")
    flush_print(f"  delivery verdict       : {verdict}")
    flush_print(f"    {verdict_text}")
    flush_print(f"  mechanism verdict      : {field_ablation.get('verdict')}")
    flush_print(f"    {field_ablation.get('verdict_text')}")
    flush_print(f"  dose  i_std / i_cross_std / d_rel (g2) : "
                f"{field_diag.get('g2', {}).get('i_std')} / "
                f"{field_diag.get('g2', {}).get('i_cross_std')} / "
                f"{field_diag.get('g2', {}).get('d_rel')} "
                f"(v01 d_rel ~0.024, clamp_bound={field_diag.get('g2', {}).get('clamp_bound')})")
    flush_print(f"[InfoDiff] results written to {out_path} (disk_guard ok={ev.get('ok')})")
    flush_print("=" * 78)


def infomap_band_text():
    """[v02] Every band description is GENERATED from the SAME threshold table the
    verdict function uses.  (Pitfall 164 / v85 lesson: a hand-written band sentence can
    describe an outcome its own band thresholds exclude, so a reader -- including a
    future self -- is guaranteed to misjudge the run.)"""
    L = INFOMAP_VERDICT_TOL
    return {
        'INFOMAP_UNATTRIBUTABLE':
            "区块未全部接线 (info_conv 没进计算图) => 本版无此轴, 先修交付, 不判死轴",
        'INFOMAP_NOT_DELIVERED':
            "图不可归因: 信息图仍近乎常数 (i_std 或 i_cross_std <= %.0e). "
            "对照 v01: i_std~0.012 (D 相对变化 2.4%%), g2_cls0 vs full 只翻转 0.4%%. "
            "=> 归一化没救活它, 轴判死, 逐字回退 cnn10_v51.py" % L['map_std'],
        'INFOMAP_NO_LEVERAGE':
            "图活了但杠杆 < %.2f (含空间图案与样本间差异两种口径); 换掉信息图后 "
            "<%.0f%% 的测试预测翻转 => 算子仍被相邻 conv+BN 吸收 => 换轴"
            % (L['lever'], L['lever'] * 100),
        'INFOMAP_LEVERAGE_NO_DIRECTION':
            "杠杆 >= %.2f 但方向 <= %.2fpp (paired SE 已报) => 换判据层级, 不加剂量"
            % (L['lever'], L['dir_pp']),
        'INFOMAP_AXIS_USED':
            "送到 ∧ 杠杆 >= %.2f ∧ 方向 >= %.2fpp => 轴被真正用上; 只有此时 "
            "delivered 与基线 97.56 的 z 值才值得解读" % (L['lever'], L['dir_pp']),
    }


def infomap_axis_verdict(wired, i_std, i_cross_std,
                         lever_pattern, lever_sample, dir_pattern, dir_sample):
    """[v02] Three levels, checked in order, all inputs measured in the SAME run.

    1. DELIVERED  -- wired, and the map varies in space AND across samples.
                     v01 failed here: the map was sigmoid(0)=0.5, a constant.
    2. LEVERAGE   -- does swapping the map out actually move predictions?
    3. DIRECTION  -- does `full` beat the ablated map on test accuracy?

    Returns (band, text) where the text comes from the same table as the thresholds.
    """
    lever = max(lever_pattern, lever_sample)
    dirn = max(dir_pattern, dir_sample)
    if not wired:
        band = 'INFOMAP_UNATTRIBUTABLE'
    elif i_std <= INFOMAP_VERDICT_TOL['map_std'] or i_cross_std <= INFOMAP_VERDICT_TOL['map_std']:
        band = 'INFOMAP_NOT_DELIVERED'
    elif lever < INFOMAP_VERDICT_TOL['lever']:
        band = 'INFOMAP_NO_LEVERAGE'
    elif dirn < INFOMAP_VERDICT_TOL['dir_pp']:
        band = 'INFOMAP_LEVERAGE_NO_DIRECTION'
    else:
        band = 'INFOMAP_AXIS_USED'
    return band, infomap_band_text()[band]


def _infomap_wired(model):
    net = model.module if isinstance(model, nn.DataParallel) else model
    n = 0
    for name in ('phys1', 'phys2', 'phys3'):
        blk = getattr(net, name, None)
        conv = getattr(blk, 'info_conv', None) if blk is not None else None
        if conv is not None and conv.weight.requires_grad:
            n += 1
    return (n == 3), n


def _infomap_verdict_selftest():
    """Feed the verdict function KNOWN DEGENERATE INPUTS, including the exact band
    boundaries, and pin the >= / < convention.  A verdict function that has never
    been shown a fake success will happily report one."""
    Mtol = INFOMAP_VERDICT_TOL['map_std']
    Ltol = INFOMAP_VERDICT_TOL['lever']
    Dtol = INFOMAP_VERDICT_TOL['dir_pp']
    cases = [
        (False, 0.0,  0.0,  0.0,   0.0,  0.0,  0.0,  'INFOMAP_UNATTRIBUTABLE'),
        (True,  0.0,  0.0,  0.0,   0.0,  0.0,  0.0,  'INFOMAP_NOT_DELIVERED'),
        (True,  0.05, 0.0,  0.5,   0.0,  1.0,  0.0,  'INFOMAP_NOT_DELIVERED'),
        (True,  Mtol, 0.05, 0.5,   0.0,  1.0,  0.0,  'INFOMAP_NOT_DELIVERED'),
        (True,  0.05, 0.05, 0.001, 0.0,  1.0,  0.0,  'INFOMAP_NO_LEVERAGE'),
        (True,  0.05, 0.05, Ltol,  0.0,  1.0,  0.0,  'INFOMAP_AXIS_USED'),
        (True,  0.05, 0.05, 0.5,   Ltol, 0.0,  1.0,  'INFOMAP_AXIS_USED'),
        (True,  0.05, 0.05, 0.5,   0.0,  0.10, 0.0,  'INFOMAP_LEVERAGE_NO_DIRECTION'),
        (True,  0.05, 0.05, 0.5,   0.0,  Dtol, 0.0,  'INFOMAP_AXIS_USED'),
        (True,  0.05, 0.05, 0.5,   0.0,  Dtol - 1e-9, 0.0, 'INFOMAP_LEVERAGE_NO_DIRECTION'),
        (True,  0.05, 0.05, 0.5,   0.0,  0.0,  Dtol, 'INFOMAP_AXIS_USED'),
        (True,  0.05, 0.05, 0.30,  0.30, 0.60, 0.60, 'INFOMAP_AXIS_USED'),
        (True,  0.05, 0.05, 0.0,   0.0,  0.0,  0.0,  'INFOMAP_NO_LEVERAGE'),
    ]
    bad = []
    for c in cases:
        got, _ = infomap_axis_verdict(*c[:7])
        if got != c[7]:
            bad.append((c, got))
    return {'ran': True, 'n_cases': len(cases), 'n_pass': len(cases) - len(bad),
            'all_pass': not bad, 'failures': [{'case': list(c[:7]), 'want': c[7],
                                               'got': g} for c, g in bad]}


def _run_field_ablation(model, testloader, field_diag):
    """[v02] Paired ablation on the FULL 10k test set, with accuracy.

    v01 had two instrument defects that this replaces:
      * subset capped at 2000 images => 0.5pp resolution, too coarse for this decision;
      * the per-mode accuracy accumulator was dead code (`... if False else 0`), so
        every `acc` was None and every `dacc_pp` was 0.0 => the ablation could report
        LEVERAGE but never DIRECTION.  Leverage alone cannot say "helps" vs "hurts".
    """
    net = model
    modes = list(INFOMAP_MODES)
    cap = INFOMAP_ABL_CAP
    preds, accs = {}, {}
    n = 0
    with torch.no_grad():
        for mode in modes:
            collected, labels = [], []
            c = 0
            for inputs, targets in testloader:
                if c >= cap:
                    break
                inputs = inputs.to(device)
                collected.append(net(inputs, mode).argmax(1).detach().cpu())
                labels.append(targets.detach().cpu())
                c += inputs.size(0)
            p = torch.cat(collected, 0)[:cap]
            y = torch.cat(labels, 0)[:cap]
            if p.numel() != y.numel():
                raise RuntimeError(
                    f"[ablation] mode={mode} preds={p.numel()} labels={y.numel()}")
            preds[mode] = p
            accs[mode] = 100.0 * float((p == y).float().mean().item())
            n = int(y.numel())
    full = preds['full']
    full_acc = accs['full']

    def _pair(mode):
        p = preds[mode]
        nd = int((p != full).sum().item())
        # Paired SE of the accuracy difference: the two models agree on (n - nd) items,
        # so only the disagreeing `nd` items can move the difference => SE = 100*sqrt(nd)/n.
        se_pp = (100.0 * math.sqrt(nd) / float(n)) if n else 0.0
        d = full_acc - accs[mode]
        return {'rel_shift': round(float(nd) / float(n), 6), 'n_disagree': nd,
                'acc': round(accs[mode], 4),
                'dacc_pp': round(d, 4),
                'dacc_se_pp': round(se_pp, 4),
                'dacc_readable': bool(abs(d) >= 2.0 * se_pp)}

    modes_out = {'full': {'rel_shift': 0.0, 'n_disagree': 0, 'acc': round(full_acc, 4),
                          'dacc_pp': 0.0, 'dacc_se_pp': 0.0, 'dacc_readable': False}}
    for mode in modes:
        if mode != 'full':
            modes_out[mode] = _pair(mode)

    lever_pattern = modes_out['g2_cls0']['rel_shift']       # spatial PATTERN matters?
    lever_sample = modes_out['g2_fixedpat']['rel_shift']    # CROSS-SAMPLE variation matters?
    lever_space = modes_out['g2_off']['rel_shift']          # field exists at all?
    dir_pattern = modes_out['g2_cls0']['dacc_pp']           # >0 => the map HELPS
    dir_sample = modes_out['g2_fixedpat']['dacc_pp']
    wired, n_wired = _infomap_wired(model)
    i_std = float(field_diag.get('g2', {}).get('i_std', 0.0))
    i_cross = float(field_diag.get('g2', {}).get('i_cross_std', 0.0))
    band, text = infomap_axis_verdict(wired, i_std, i_cross,
                                      lever_pattern, lever_sample,
                                      dir_pattern, dir_sample)
    return {
        'n_images': n, 'modes': modes_out,
        'lever_pattern_vs_cls0': lever_pattern,
        'lever_sample_vs_fixedpat': lever_sample,
        'lever_space_vs_off': lever_space,
        'dir_pattern_pp': dir_pattern,
        'dir_sample_pp': dir_sample,
        'full_acc': round(full_acc, 4),
        'wired': bool(wired), 'n_wired': n_wired,
        'map_i_std': i_std, 'map_i_cross_std': i_cross,
        'v01_reference': {'lever_pattern_vs_cls0': 0.004, 'n_images': 2000,
                          'n_disagree_g2_cls0': 8, 'lever_space_vs_off': 0.1645},
        'verdict': band, 'verdict_text': text,
        'verdict_bands': infomap_band_text(), 'verdict_tol': dict(INFOMAP_VERDICT_TOL),
        'verdict_selftest': _infomap_verdict_selftest(),
    }


def _run_gain_sweep(model, testloader):
    """Sweep the global diffusivity scale D0 (gate2 + gate3) and measure disagreement
    against the unmodified `full` run.

    [v02] v01's sweep was DEGENERATE and could not be read honestly: D0 was already
    pinned at INFO_D0_CLAMP = 1.5, so k = 1 / 2 / 3 all mapped onto the SAME clamped D0
    and all three returned exactly 0.0 disagreement.  The reported `peak_scale_k = 0.0`
    therefore only restated the `g2_off` number (0.169 vs 0.1645) -- i.e. a constant
    field that carried no information at all.  v02 keeps only the informative LOWER side
    (k <= 1, where the clamp is not binding) and DECLARES the clamp binding explicitly,
    so a reader cannot mistake a saturated instrument for a flat response.
    """
    net = model
    core = net.module if isinstance(net, nn.DataParallel) else net
    phys2 = core.phys2
    phys3 = core.phys3
    scales = [0.0, 0.1, 0.35, 0.5, 1.0]
    cap = INFOMAP_ABL_CAP
    with torch.no_grad():
        full_collected = None
        c = 0
        for inputs, targets in testloader:
            if c >= cap:
                break
            inputs = inputs.to(device)
            p = net(inputs, 'full').argmax(1).detach().cpu()
            full_collected = p if full_collected is None else torch.cat([full_collected, p], 0)
            c += inputs.size(0)
        full_collected = full_collected[:cap]
        n = int(full_collected.numel())
        points = []
        saved2 = phys2.d0_scale_override
        saved3 = phys3.d0_scale_override
        try:
            for k in scales:
                phys2.d0_scale_override = k
                phys3.d0_scale_override = k
                cc = 0
                coll = None
                for inputs, targets in testloader:
                    if cc >= cap:
                        break
                    inputs = inputs.to(device)
                    p = net(inputs, 'full').argmax(1).detach().cpu()
                    coll = p if coll is None else torch.cat([coll, p], 0)
                    cc += inputs.size(0)
                coll = coll[:cap]
                nd = int((coll != full_collected).sum().item())
                points.append({'scale_k': k,
                               'rel_shift_vs_full': round(float(nd) / float(n), 6),
                               'n_disagree': nd})
        finally:
            phys2.d0_scale_override = saved2
            phys3.d0_scale_override = saved3
    d0_base = max(float(torch.exp(phys2.log_d0).clamp(max=phys2.d0_clamp).item()),
                  float(torch.exp(phys3.log_d0).clamp(max=phys3.d0_clamp).item()))
    clamp_bound = bool(d0_base >= phys2.d0_clamp - 1e-9)
    upper_flat = all(p['n_disagree'] == 0 for p in points if p['scale_k'] >= 1.0)
    if clamp_bound and upper_flat:
        amp_verdict = 'AMPLITUDE_SWEEP_CLAMP_BOUND_ONE_SIDED'
    elif clamp_bound:
        amp_verdict = 'AMPLITUDE_SWEEP_CLAMP_BOUND'
    else:
        amp_verdict = 'AMPLITUDE_SWEEP_DONE'
    peak = max(points, key=lambda x: x['rel_shift_vs_full'])
    return {'points': points, 'amplitude_verdict': amp_verdict,
            'peak_scale_k': peak['scale_k'],
            'd0_effective_max': round(d0_base, 6),
            'd0_clamp': phys2.d0_clamp,
            'clamp_binding': clamp_bound,
            'swept_scales': scales,
            'one_sided': True,
            'note': ('v01 swept [0,0.5,1,2,3] and got exactly 0 disagreement for every '
                     'k>=1 because D0 sat at the clamp; v02 drops the provably-identical '
                     'upper points and keeps k<=1, and reports clamp_binding so a '
                     'saturated sweep is never read as a flat response.')}


def _run_field_diagnostics(model, testloader):
    """Dose readouts per block on the test set.

    [v02] Reports THREE distinct quantities, because "the parameter moved" is not
    "the intervention arrived":
      * i_std        -- spatial std of the information map  (is the map alive?)
      * i_cross_std  -- cross-sample std of the map         (is it input-dependent?)
      * d_rel        -- std(D)/mean(D), the SCALE-FREE spatial contrast of diffusivity.
                        This is the number directly comparable to v01, where it was
                        ~0.024 (2.4%): the map had decayed to the constant sigmoid(0)=0.5.
    """
    # field_diagnostics walks the inner ResNet layers directly (not via forward),
    # so it must run on the core module, not the DataParallel wrapper.
    net = model.module if isinstance(model, nn.DataParallel) else model
    cap = 500
    diags_acc = {'g1': [], 'g2': [], 'g3': []}
    c = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            if c >= cap:
                break
            inputs = inputs.to(device)
            d = net.field_diagnostics(inputs, 'full')
            for k in diags_acc:
                diags_acc[k].append(d[k])
            c += inputs.size(0)
    out = {}
    for k, lst in diags_acc.items():
        d_mean = float(np.mean([x['d_mean'] for x in lst]))
        adapt = float(np.mean([x['info_adaptivity'] for x in lst]))
        out[k] = {
            'info_adaptivity': round(adapt, 6),
            'i_std': round(float(np.mean([x['i_std'] for x in lst])), 6),
            'i_cross_std': round(float(np.mean([x['i_cross_std'] for x in lst])), 6),
            'd_mean': round(d_mean, 6),
            'd_rel': round(adapt / d_mean, 6) if d_mean > 0 else 0.0,
            'alpha': round(float(np.mean([x['alpha'] for x in lst])), 4),
            'd0': round(float(np.mean([x['d0'] for x in lst])), 4),
            'clamp_bound': bool(all(x['clamp_bound'] for x in lst)),
        }
    out['summary'] = {
        'g2_d_rel': out['g2']['d_rel'],
        'v01_g2_d_rel_reference': 0.024,
        'g2_contrast_gain_vs_v01': (round(out['g2']['d_rel'] / 0.024, 3) if out['g2']['d_rel'] else 0.0),
        'g2_clamp_bound': out['g2']['clamp_bound'],
        'predicted_d_rel_at_init': 0.293,
        'predicted_i_std_at_init': 0.149,
        'predicted_d_range_at_init': [0.348, 0.876],
        'note': ('v01 g2: info_adaptivity 0.0135 over d_mean ~0.55 => d_rel ~0.024, i.e. a '
                 '2.4% spatial contrast, and g2_cls0 vs full flipped only 8/2000 '
                 'predictions. v02 must move d_rel by orders of magnitude or the axis is '
                 'dead (pre-registered: d_rel is a DOSE readout, NOT a decision signal). '
                 'Predicted from the delivered normalisation on a random pre-activation: '
                 'd_rel ~0.293 (~12x v01), i_std ~0.149, D in [0.348, 0.876] with '
                 'dt*4*Dmax = 0.70 <= 2 (strictly safer for CFL than v01 at 1.20). '
                 'Reading a d_rel near 0.024 again means the normalisation did not take.'),
    }
    return out


if __name__ == '__main__':
    main()
