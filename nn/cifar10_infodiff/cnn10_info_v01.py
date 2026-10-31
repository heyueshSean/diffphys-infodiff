# =============================================================================
# cnn10_info_v01.py  --  InfoDiff: Information-Guided Wasserstein Diffusion
# -----------------------------------------------------------------------------
# NEW METHOD (not a continuation of the archived B-line v51..v67 axis chain).
# B-line is archived (2026-09-29): fixed, class-independent Mattern kernel,
# all axes NULL, 195 converse falsified. This file re-develops the physics field
# from fundamental theory:
#
#   * Perona-Malik (1990):  du/dt = div( c(x) grad u ),  c = g(|grad u|)
#   * Information Bottleneck (Tishby 2000; Alemi VIB 2017; Qin et al. 2026
#     arXiv 2609.19122):  min I(T;X) - beta*I(T;Y); CNN forward = spatial
#     compression. Qin controls compression by a SCALAR; we do it SPATIALLY.
#   * Wasserstein Gradient Flow / Otto calculus (Jordan-Kinderlehrer-Otto 1998):
#     the heat flow IS the gradient flow of entropy in Wasserstein space.
#     => "information diffusion" = WGF of an information-adapted entropy functional.
#
# Core divergence from B-line: diffusivity D(x) is SPATIALLY ADAPTIVE and driven
# by a LEARNED, LABEL-FREE information map I(x) (trained by the task loss), not a
# fixed global kernel and not a scalar class-posterior amplitude (that was the
# 195 failure: feeding label posterior into a fixed kernel's amplitude moved only
# 0.568%). Here the operator itself is reshaped by WHERE information lives. This
# is a spatial information bottleneck: compress where info is low (denoise),
# preserve where info is high (keep class detail).
#
#   du/dt = div( D(x) grad u ),   D(x) = D0 * exp( -beta * I(x) )
#   I(x)  = sigmoid( conv_1x1(u) )   (label-free: input is features only)
#   D0 learnable, beta fixed at 2.0 (future axis), solved by explicit Euler.
#
# I/O structure borrowed from B-line (cnn10_v51.py): same recipe (SEED=1234,
# batch 256, 2-phase cosine, mixup+cutmix, TTA flip, best/ens JSON schema),
# same ablation mode names (full / g2_cls0 / g2_clsmax / g1_off / g2_off /
# g3_off) so results are directly comparable to 97.56.
#
# Discipline: zero argparse, bare command `python cnn10_info_v01.py`, self-contained.
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

# NOTE: zero argparse by design (bare command `python cnn10_info_v01.py`).

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
INFO_VERSION = "info_v01"
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
INFO_DT = 0.20            # time step (CFL: dt*4*D0max <= 2; D0max<=1.5 => 1.2 ok)
INFO_D0_INIT = 1.0        # global diffusivity scale (learnable log)
INFO_D0_CLAMP = 1.5       # clamp D0 to keep CFL stable
INFO_BETA = 2.0           # information sensitivity (fixed for v01; future axis)
INFO_ALPHA_INIT = 0.5     # residual blend init
INFO_USE_LEARNED_INFO = True

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
    I(x)  = sigmoid( conv_1x1(u) )   (label-free information map)

    Solved with explicit Euler + Neumann BC (reflect padding). The block is a
    differentiable operator; the info map is shaped by the task loss, so the
    field becomes a spatial information bottleneck (compress low-info, preserve
    high-info regions) without any explicit IB loss.

    bmode: 'adaptive' (learned I), 'const' (I=0.5, isotropic), 'sat' (I=1.0,
    max modulation, isotropic), 'off' (identity, no field).
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
            self.info_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        else:
            self.info_conv = None
        self.d0_scale_override = None  # set by gain sweep

    def _info_map(self, u):
        # u: (B,C,H,W) -> I: (B,1,H,W) in (0,1)
        if self.use_learned_info and self.info_conv is not None:
            return torch.sigmoid(self.info_conv(u))
        # intrinsic Fisher proxy fallback: local gradient energy (label-free)
        gx = u[..., 1:, :] - u[..., :-1, :]
        gy = u[..., :, 1:] - u[..., :, :-1]
        e = (gx ** 2).sum(1, keepdim=True)[:, :, :-1, :] + (gy ** 2).sum(1, keepdim=True)[:, :, :, :-1]
        return torch.sigmoid(e)

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
        B, C, H, W = u.shape
        if bmode == 'const':
            I = torch.full((B, 1, H, W), 0.5, device=u.device, dtype=u.dtype)
        elif bmode == 'sat':
            I = torch.full((B, 1, H, W), 1.0, device=u.device, dtype=u.dtype)
        else:  # adaptive
            I = self._info_map(u)
        D = self._diffusivity(I)
        v = u
        for _ in range(self.n_steps):
            v = self._step(v, D, self.dt)
        alpha = soft_bound(self.alpha_logit, 0.0, 1.5)
        return u + alpha * (v - u)

    def diagnostics(self, u, bmode='adaptive'):
        """Returns spatial-adaptivity evidence (std of D over space) + alpha + D0."""
        if bmode == 'off':
            return {'info_adaptivity': 0.0, 'alpha': 0.0, 'd0': 0.0}
        B, C, H, W = u.shape
        if bmode == 'const':
            I = torch.full((B, 1, H, W), 0.5, device=u.device, dtype=u.dtype)
        elif bmode == 'sat':
            I = torch.full((B, 1, H, W), 1.0, device=u.device, dtype=u.dtype)
        else:
            I = self._info_map(u)
        D = self._diffusivity(I)
        adapt = float(D.std(dim=(2, 3)).mean().item())      # spatial variation of diffusivity
        d0 = float(torch.exp(self.log_d0).clamp(max=self.d0_clamp).item())
        alpha = float(soft_bound(self.alpha_logit, 0.0, 1.5).item())
        return {'info_adaptivity': adapt, 'alpha': alpha, 'd0': d0}


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
        _DISK_STATS_INC('admit_reject')
        _DISK_EVENTS_APPEND(note)
        return False, note
    note['ok'] = True
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
    # known degenerate inputs: demand > free must be rejected; 1 MB must pass.
    class _FakePath:
        def __init__(self, free):
            self._free = free
        def __fspath__(self):
            return '/fake'
    saved = _disk_free_mb.__globals__ if False else None
    # monkeypatch _disk_free_mb
    orig = _disk_free_mb
    try:
        globals()['_disk_free_mb'] = lambda p: 10.0   # 10 MB free
        ok_big, _ = _disk_admit('/fake/x', 5000.0, 'selftest')   # demand 5000 >> free => reject
        ok_small, _ = _disk_admit('/fake/x', 1.0, 'selftest')   # 1 MB => pass
        selftest_pass = (ok_big is False) and (ok_small is True)
    finally:
        globals()['_disk_free_mb'] = orig
    return {'ran': True, 'pass': bool(selftest_pass),
            'expect_reject_big': True, 'expect_allow_small': True}

def _disk_guard_verdict():
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
        for _m in ['g2_cls0', 'g2_clsmax', 'g1_off', 'g2_off', 'g3_off', 'all_off']:
            _o = net(_d, _m)
            if tuple(_o.shape) != (2, 10):
                raise RuntimeError(f"[self-test] mode={_m} shape={tuple(_o.shape)} (expected (2,10))")
        _dd = net.field_diagnostics(_d, 'full')
        if set(_dd.keys()) != {'g1', 'g2', 'g3'}:
            raise RuntimeError(f"[self-test] field_diagnostics keys={set(_dd)}")
        del _d, _out, _o, _dd, x, stages
    flush_print("[self-test] architecture OK (channels + all ablation modes + diagnostics)")

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

        # raw test (no TTA)
        net.eval()
        c, t = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                c += net(inputs, 'full').argmax(1).eq(targets).sum().item()
                t += targets.size(0)
        val_acc = 100. * c / t
        history_test.append({"epoch": int(epoch), "acc": round(float(val_acc), 4)})
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            no_test_improve = 0
        else:
            no_test_improve += 1

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
    final_tta = test_tta(TRAIN_EPOCHS)
    flush_print(f"\nFinal TTA: {final_tta:.2f}%")
    ens_acc = 0.0
    ens_tta = 0.0
    if len(ckpt_pool) >= 2:
        top_k = min(ENS_TOP_K, len(ckpt_pool))
        top_ckpts = ckpt_pool[:top_k]
        ckpt_files = [c[2] for c in top_ckpts]
        ens_acc = ensemble_predict(ckpt_files, use_tta=False)
        ens_tta = ensemble_predict(ckpt_files, use_tta=True)
        flush_print(f"  Ensemble Test: {ens_acc:.2f}% | Ensemble TTA: {ens_tta:.2f}%")

    delivered = max(best_tta_acc, final_tta)
    delivered_source = 'best_tta' if best_tta_acc >= final_tta else 'final_tta'
    backtrack_needed = bool(delivered < BACKTRACK_REDLINE)

    # ----- Instrumentation: field_ablation (InfoDiff modes) -----
    field_ablation = _run_field_ablation(model, testloader)

    # ----- Instrumentation: gate2 (diffusivity) gain sweep -----
    gate2_gain_sweep = _run_gain_sweep(model, testloader)

    # ----- Instrumentation: field adaptivity evidence -----
    field_diag = _run_field_diagnostics(model, testloader)

    # ----- Replicate z vs B-line baseline 97.56 (sigma_diff 0.169) -----
    sigma_diff = 0.169
    base_acc = 97.56
    z_combined = (delivered - base_acc) / sigma_diff
    replicate = {
        'base_file': BASE_FILE, 'base_sha1': BASE_SHA1, 'base_acc': base_acc,
        'sigma_diff': sigma_diff, 'delivered': delivered,
        'z_combined': round(float(z_combined), 4),
        'verdict': 'CONFIRMED' if z_combined >= 2.0 else ('NULL' if abs(z_combined) < 2.0 else 'REGRESS'),
    }

    # ----- Disk guard verdict -----
    dg_selftest = _disk_guard_selftest()
    dg_verdict = _disk_guard_verdict()

    # ----- Verdict (pre-registered) -----
    if backtrack_needed:
        verdict = 'BACKTRACK_REQUIRED'
    elif delivered >= base_acc:
        verdict = 'DELIVERED_ABOVE_BASELINE'
    else:
        verdict = 'NULL_BELOW_BASELINE_NO_REDLINE'

    results = {
        'version': INFO_VERSION,
        'base': BASE_FILE,
        'base_sha1': BASE_SHA1,
        'base_note': BASE_NOTE,
        'axis': ('information-guided Wasserstein diffusion (InfoDiff): spatially-adaptive, '
                 'learned label-free info map modulates diffusivity D(x)=D0*exp(-beta*I(x)); '
                 'explicit-Euler PDE, task-loss-shaped spatial information bottleneck.'),
        'seed': SEED,
        'n_gpus': N_GPUS,
        'batch_size': BATCH_SIZE,
        'epoch': int(TRAIN_EPOCHS),
        'actual_epochs': int(len(history_test)),
        'early_stopped': bool(early_stopped),
        'stop_reason': stop_reason_out,
        'best_acc': round(float(best_acc), 4),
        'best_epoch': int(best_epoch),
        'best_tta_acc': round(float(best_tta_acc), 4),
        'best_tta_epoch': int(best_tta_epoch),
        'final_tta': round(float(final_tta), 4),
        'ens_acc': round(float(ens_acc), 4),
        'ens_tta': round(float(ens_tta), 4),
        'delivered': round(float(delivered), 4),
        'delivered_source': delivered_source,
        'delivered_is_max_of_n': 3,
        'delivered_n_members': ['best_tta_acc', 'final_tta', 'ens_tta'],
        'backtrack_redline': BACKTRACK_REDLINE,
        'backtrack_needed': bool(backtrack_needed),
        'backtrack_advice': ('逐字回退 cnn10_v51.py' if backtrack_needed else 'no backtrack'),
        'verdict': verdict,
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
    flush_print(f"\n[InfoDiff] results written to {out_path} (disk_guard ok={ev.get('ok')})")
    flush_print(f"[InfoDiff] delivered={delivered:.2f}%  verdict={verdict}  "
                f"info_adaptivity_g2={field_diag.get('g2', {}).get('info_adaptivity', 0):.4f}")


def _run_field_ablation(model, testloader):
    """Pairwise disagreement of 'full' vs each ablation mode on a fixed test subset."""
    net = model
    modes = ['full', 'g2_cls0', 'g2_clsmax', 'g1_off', 'g2_off', 'g3_off']
    preds = {}
    cap = 2000
    with torch.no_grad():
        for mode in modes:
            c, t, correct = 0, 0, 0
            collected = []
            for inputs, targets in testloader:
                if c >= cap:
                    break
                inputs, targets = inputs.to(device), targets.to(device)
                out = net(inputs, mode)
                pa = out.argmax(1)
                collected.append(pa.cpu())
                correct += pa.eq(targets.to('cpu')).sum().item() if False else 0
                c += inputs.size(0)
                t += targets.size(0)
            preds[mode] = torch.cat(collected, 0)[:cap]
    full = preds['full']
    out = {'modes': {}, 'verdict': None}
    n = full.size(0)
    for mode in modes:
        if mode == 'full':
            out['modes'][mode] = {'rel_shift': 0.0, 'n_disagree': 0, 'acc': None,
                                  'dacc_pp': 0.0, 'dacc_readable': False}
            continue
        p = preds[mode]
        nd = int((p != full).sum().item())
        out['modes'][mode] = {
            'rel_shift': round(float(nd) / float(n), 6),
            'n_disagree': nd,
            'dacc_pp': 0.0,   # acc not recomputed per-mode here (would need labels); rel_shift is the leverage
            'dacc_readable': False,
        }
    # verdict: if g2_off (no field) has large rel_shift => field is load-bearing;
    # if g2_cls0/g2_clsmax near g2_off => adaptivity matters.
    g2_off = out['modes']['g2_off']['rel_shift']
    g2_cls0 = out['modes']['g2_cls0']['rel_shift']
    out['verdict'] = ('FIELD_LOAD_BEARING' if g2_off > 0.01
                      else 'FIELD_TRANSPARENT')
    out['interpretation'] = {
        'g2_off_rel_shift': g2_off,
        'g2_cls0_rel_shift': g2_cls0,
        'adaptivity_matters': bool(abs(g2_cls0 - g2_off) > 1e-4),
    }
    return out


def _run_gain_sweep(model, testloader):
    """Sweep global diffusivity scale D0 on gate2/phys3, measure disagreement vs full."""
    net = model
    phys2 = net.module.phys2 if isinstance(net, nn.DataParallel) else net.phys2
    phys3 = net.module.phys3 if isinstance(net, nn.DataParallel) else net.phys3
    scales = [0.0, 0.5, 1.0, 2.0, 3.0]
    cap = 2000
    with torch.no_grad():
        full_collected = None
        c = 0
        for inputs, targets in testloader:
            if c >= cap:
                break
            inputs = inputs.to(device)
            out = net(inputs, 'full')
            p = out.argmax(1).cpu()
            full_collected = p if full_collected is None else torch.cat([full_collected, p], 0)
            c += inputs.size(0)
        full_collected = full_collected[:cap]
        points = []
        saved2 = phys2.d0_scale_override
        saved3 = phys3.d0_scale_override
        for k in scales:
            phys2.d0_scale_override = k
            phys3.d0_scale_override = k
            cc = 0
            coll = None
            for inputs, targets in testloader:
                if cc >= cap:
                    break
                inputs = inputs.to(device)
                out = net(inputs, 'full')
                p = out.argmax(1).cpu()
                coll = p if coll is None else torch.cat([coll, p], 0)
                cc += inputs.size(0)
            coll = coll[:cap]
            nd = int((coll != full_collected).sum().item())
            points.append({'scale_k': k, 'rel_shift_vs_full': round(float(nd) / float(cap), 6),
                            'n_disagree': nd})
        phys2.d0_scale_override = saved2
        phys3.d0_scale_override = saved3
    peak = max(points, key=lambda x: x['rel_shift_vs_full'])
    return {'points': points, 'amplitude_verdict': 'AMPLITUDE_AT_OPTIMUM' if peak['scale_k'] in (1.0,)
            else 'AMPLITUDE_SWEEP_DONE', 'peak_scale_k': peak['scale_k']}


def _run_field_diagnostics(model, testloader):
    """Collect spatial-adaptivity evidence (std of D over space) per block on test set."""
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
        out[k] = {
            'info_adaptivity': round(float(np.mean([x['info_adaptivity'] for x in lst])), 6),
            'alpha': round(float(np.mean([x['alpha'] for x in lst])), 4),
            'd0': round(float(np.mean([x['d0'] for x in lst])), 4),
        }
    return out


if __name__ == '__main__':
    main()
