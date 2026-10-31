"""
Physics-Guided PPO for Atari Breakout (v10)
Dual Staircase Release (10M): v9 single staircase extended to TWO release events

== v10 DESIGN (synthesized from v9 data + release-effect hypothesis) ==

  v9 (5M) taught us:
  - The staircase (SMOOTH 0.008 -> RELEASE to 0.004 -> HOLD -> TAIL to 0.002)
    was the best of all versions: peak 288.8, late mean 186.9, CV 0.264.
  - CRITICAL: v9's window means were STILL climbing at 5M (4-4.5M: 175.4 ->
    4.5-5M: 186.9). The 5M budget CUT OFF the 0.004 plateau while it was
    still developing. -> HOLD1 must be LONGER in v10.
  - v8 showed 0.002 HOLD is a verified stability anchor (CV 0.36).
  - v7 showed full release to 0.0 is catastrophic -> any tail must keep
    alpha > 0.

  v10 extends the staircase to TWO levels over 10M (2x budget):
    0 - 2.8M   : alpha = 0.008           (SMOOTH, unchanged -- preserves the
                                         experimental-control continuity with
                                         v4/v7/v8/v9, 200K point = 9.1)
    2.8M-3.05M : alpha = 0.008 -> 0.004  (REL1, first release -- v9-verified
                                         insight trigger at 2.8M)
    3.05M-6.5M : alpha = 0.004           (HOLD1, CONSTANT peak plateau --
                                         v9's plateau was cut at 4.5M while
                                         still rising; give it 3.45M to develop)
    6.5M-6.75M : alpha = 0.004 -> 0.002  (REL2, SECOND release -- the key
                                         hypothesis: is the release effect
                                         REPEATABLE? v8 data says 0.002 is a
                                         stable, learnable anchor)
    6.75M-9.5M : alpha = 0.002           (HOLD2, CONSTANT second anchor --
                                         v8-verified stability point)
    9.5M - 10M : alpha = 0.002 -> 0.0015 (TAIL, gentle 500K landing; keep
                                         alpha > 0 -- v7 lesson: 0.0 = collapse)
  Rationale: 10M is the standard Atari budget in the literature (SB3 RL Zoo
  PPO hits 398 @ 10M). v10 chases (a) whether the 0.004 plateau keeps climbing
  with 2x time, and (b) whether a SECOND release triggers a SECOND insight
  ("release effect repeatability") -- the two questions that would make this
  a publishable empirical claim.
  GAE baseline = LIVE critic (v4 path), no target network, no re-smooth.

  v10 vs v9 delta = schedule ONLY (REL2 + HOLD2 + longer HOLD1 + fixed-window
                    v4 fallback). Architecture 100% original (v4).
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

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==================== Physics-Guided Config (v4: all structural OFF) ====================
USE_BLURPOOL = False        # [OFF] v1 lesson: blur destroys ball position info
USE_RESIDUAL = False        # [OFF] v3 lesson: zero-init residual still breaks RL
USE_SPECTRAL_REG = False    # [OFF] spec reg fights RL edge learning
USE_MATERN_SMOOTH = False   # [OFF] v2 lesson: original Gaussian kernel, NOT Matern

SD_DROP_MAX = 0.0           # [OFF] StochasticDepth off
SD_WARMUP_START = 200_000
SD_WARMUP_END = 500_000
SPECTRAL_COEF_MAX = 0.0002
SPEC_WARMUP_START = 500_000
SPEC_WARMUP_END = 2_000_000
SPEC_EXCLUDE_CONV1 = True

MATERN_LAM = 1.0
MATERN_NU_REG = 1.0
MATERN_S = 1.0
MATERN_NU_SMOOTH = 1.5

# ==================== v10 NEW: Dual Staircase Release ====================
# v9 verified: 2.8M release to 0.004 triggers insight; 0.004 HOLD develops the
#   peak; TAIL to 0.002 stabilizes. v10 (10M) repeats the staircase one level
#   lower: HOLD1 (0.004) runs 3.45M (v9 was cut off while still rising),
#   then a SECOND release to 0.002 (v8's verified stable anchor) -> HOLD2,
#   then a gentle final TAIL to 0.0015 (kept > 0: v7 showed 0.0 collapses).
RELEASE_CYCLE = True              # master switch
REL_A_SMOOTH = 0.008              # strong smoothing (v4 initial alpha)
REL_A_MID    = 0.004              # v4 peak-band alpha (HOLD1 level)
REL_A_LOW    = 0.002              # v8 stable anchor (HOLD2 level)
REL_A_FINAL  = 0.0015             # gentle final landing (kept > 0, v7 lesson)
REL_SMOOTH_END = 2_800_000        # smoothing ends -> REL1 starts (v8/v9 timing)
REL_DROP_LEN   = 250_000          # REL1 ramp length (2.8M -> 3.05M)
REL_HOLD1_END  = 6_500_000        # HOLD1(0.004) end -> REL2 starts
REL_DROP2_LEN  = 250_000          # REL2 ramp length (6.5M -> 6.75M)
REL_HOLD2_END  = 9_500_000        # HOLD2(0.002) end -> TAIL starts
REL_TAIL_LEN   = 500_000          # tail ramp length (9.5M -> 10M)
REL_A_V4_END   = 0.004            # [fallback only] true v4 endpoint
REL_V4_DECAY_START = 3_000_000    # [fallback only] v4 decay start (exact v4)
REL_V4_DECAY_END   = 5_000_000    # [fallback only] v4 decay end (FIXED window:
                                  #   v9 used total_steps -> at 10M it would
                                  #   stretch 0.008->0.004 over 3M-10M, NOT the
                                  #   real v4. Fixed 3M->5M then HOLD 0.004.)

# NOTE (v6 lesson): the Value Target Network is REMOVED (since v7, kept in
#   v8/v9/v10). GAE baseline is the LIVE critic (100% v4 path).

# ==================== AtariWrapper (unchanged from v4) ====================
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
        total_r = 0; done = False
        for _ in range(4):
            res = self.env.step(action)
            if len(res) == 5: obs, r, term, trunc, _ = res; done = term or trunc
            else: obs, r, done, _ = res
            if not np.isfinite(r):
                r = 0.0
            total_r += r
            if done: break
        self.frames.append(self._process(obs))
        return np.stack(self.frames, axis=0), total_r, done, {}

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
    """Actor-Critic (v10: EXACT same architecture as v4/original).
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

# ==================== gradient_aware_smooth (Gaussian, ORIGINAL, same as v4) ====================
def gradient_aware_smooth(values, q_vectors, k, sigma, grad_threshold=2.0,
                          use_matern=False, nu=1.5):
    """Physics-inspired value function smoothing (v10: ORIGINAL Gaussian, same as v4)."""
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

# ==================== v10 NEW: Dual Staircase Release alpha ====================
def release_cycle_alpha(step, total_steps):
    """Dual staircase release for mix_alpha (v10).

    SMOOTH 0 - 2.8M   : 0.008  (unchanged -- experimental control continuity)
    REL1   2.8M-3.05M : 0.008 -> 0.004 (first release, v9-verified insight)
    HOLD1  3.05M-6.5M : 0.004  (CONSTANT extended peak plateau -- v9 was cut
                                off at 4.5M while window means still rising)
    REL2   6.5M-6.75M : 0.004 -> 0.002 (SECOND release -- repeatability test)
    HOLD2  6.75M-9.5M : 0.002  (CONSTANT -- v8-verified stable anchor)
    TAIL   9.5M - 10M : 0.002 -> 0.0015 (gentle landing, kept > 0)

    Returns:
        float: mix_alpha at given step.
    """
    if not RELEASE_CYCLE:
        # v4 fallback: EXACT original v4 decay 0.008 -> 0.004 over a FIXED
        # 3M -> 5M window (v10 fix: independent of total_steps so a 10M
        # control run = "v4 schedule extended", NOT "v4 decay stretched"),
        # then HOLD at 0.004.
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
    S_END = REL_SMOOTH_END      # 2.8M
    D_LEN = REL_DROP_LEN        # 250K (REL1)
    H1_END = REL_HOLD1_END      # 6.5M
    D2_LEN = REL_DROP2_LEN      # 250K (REL2)
    H2_END = REL_HOLD2_END      # 9.5M
    T_LEN = REL_TAIL_LEN        # 500K

    if step < S_END:
        return A
    elif step < S_END + D_LEN:              # REL1: A -> A_MID monotonic
        p = (step - S_END) / D_LEN
        return A + (A_MID - A) * p
    elif step < H1_END:                     # HOLD1: constant peak band
        return A_MID
    elif step < H1_END + D2_LEN:            # REL2: A_MID -> A_LOW monotonic
        p = (step - H1_END) / D2_LEN
        return A_MID + (A_LOW - A_MID) * p
    elif step < H2_END:                     # HOLD2: constant second anchor
        return A_LOW
    elif step < H2_END + T_LEN:             # TAIL: A_LOW -> A_FIN gentle
        p = (step - H2_END) / T_LEN
        return A_LOW + (A_FIN - A_LOW) * p
    else:                                   # final: A_FIN
        return A_FIN

def release_phase_name(step):
    """Human-readable phase name for logging."""
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

# ==================== Training (v10) ====================
def train(total_steps=10_000_000, rollout_len=1024, gamma=0.99, lambda_=0.95,
          clip_eps=0.1, lr_max=2.5e-4, lr_min=1e-5,
          update_epochs=4, batch_size=64, vf_coef=0.495, diff_start=200_000):
    env = AtariWrapper("BreakoutNoFrameskip-v4")
    n_actions = env.env.action_space.n
    model = PhysicsGuidedActorCritic(n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_max, eps=1e-5)
    reward_norm = RunningNorm()

    # v10: NO target network (v6 bug: target never updated -> random GAE
    # baseline killed learning). GAE baseline = LIVE critic, 100% v4 path.
    DIFF_K = 12
    DIFF_SIGMA = 10.0
    GRAD_THRESHOLD = 2.0

    # Print configuration
    print("\n" + "="*60)
    print("Physics-Guided PPO v10 Configuration (Dual Staircase Release):")
    print(f"  Architecture: 100% original (v4 baseline, NO structural changes)")
    print(f"  [NEW] Dual Staircase Release alpha: {RELEASE_CYCLE}")
    print(f"        0-{REL_SMOOTH_END/1_000_000:.1f}M: alpha={REL_A_SMOOTH} (SMOOTH, v9-verified accumulation)")
    print(f"         {REL_SMOOTH_END/1_000_000:.1f}M-{(REL_SMOOTH_END+REL_DROP_LEN)/1_000_000:.2f}M: "
          f"REL1 {REL_A_SMOOTH} -> {REL_A_MID} (first release)")
    print(f"         {(REL_SMOOTH_END+REL_DROP_LEN)/1_000_000:.2f}M-{REL_HOLD1_END/1_000_000:.1f}M: "
          f"HOLD1 {REL_A_MID} (CONSTANT, extended peak plateau)")
    print(f"         {REL_HOLD1_END/1_000_000:.1f}M-{(REL_HOLD1_END+REL_DROP2_LEN)/1_000_000:.2f}M: "
          f"REL2 {REL_A_MID} -> {REL_A_LOW} (SECOND release -- repeatable?)")
    print(f"         {(REL_HOLD1_END+REL_DROP2_LEN)/1_000_000:.2f}M-{REL_HOLD2_END/1_000_000:.1f}M: "
          f"HOLD2 {REL_A_LOW} (CONSTANT, v8-verified stable anchor)")
    print(f"         {REL_HOLD2_END/1_000_000:.1f}M-{(REL_HOLD2_END+REL_TAIL_LEN)/1_000_000:.1f}M: "
          f"TAIL {REL_A_LOW} -> {REL_A_FINAL} (gentle final landing, >0)")
    print(f"  [INFORMED] v9 peak 288.8@4.5M; v9 window means still rising at 5M")
    print(f"             (4-4.5M 175.4 -> 4.5-5M 186.9) -> HOLD1 gets 3.45M")
    print(f"  [HYPOTHESIS] release effect REPEATABLE? REL2 tests it at 6.5M")
    print(f"  [FIXED vs v9] v4 fallback window FIXED 3M->5M (not total_steps)")
    print(f"  [FIXED vs v6] target network REMOVED; GAE baseline = live critic")
    print(f"  PPO: clip={clip_eps}, epochs={update_epochs}, batch={batch_size}")
    print(f"  LR: {lr_max}->{lr_min} (cosine), rollout={rollout_len}")
    print(f"  Physics smooth: diff_start={diff_start}")
    print(f"  Smooth kernel: k={DIFF_K}, sigma={DIFF_SIGMA}, grad_threshold={GRAD_THRESHOLD}")
    print("="*60 + "\n")

    episode_rewards = []
    state = env.reset()
    ep_raw_reward = 0
    states, actions, rewards, dones, log_probs, values, q_vectors = [], [], [], [], [], [], []

    for step in range(1, total_steps+1):
        # LR schedule (same as v4)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * step / total_steps))
        for pg in optimizer.param_groups: pg['lr'] = lr

        # Warmup schedules (inactive at defaults, same as v4)
        current_sd_drop = get_sd_drop(step)
        current_spec_coef = get_spectral_coef(step)
        model.set_sd_drop(current_sd_drop)

        # v10: Dual Staircase Release mix_alpha
        mix_alpha = release_cycle_alpha(step, total_steps)

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
            ep_raw_reward = 0

        # ---- PPO Update (when rollout is full) ----
        if len(states) >= rollout_len:
            s_next = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                _, v_last, _ = model(s_next, return_qv=True)
            v_last = v_last.squeeze(0)

            # GAE baseline = LIVE critic values (v4 path; v6 target-net bug removed)
            vals_tensor = torch.cat(values, dim=0).squeeze(-1)
            q_rollout = torch.cat(q_vectors, dim=0).squeeze(1)

            # Physics field smooth on LIVE values (Gaussian, v4 kernel)
            if step >= diff_start:
                v_smooth = gradient_aware_smooth(vals_tensor, q_rollout,
                                                 k=DIFF_K, sigma=DIFF_SIGMA,
                                                 grad_threshold=GRAD_THRESHOLD,
                                                 use_matern=USE_MATERN_SMOOTH,
                                                 nu=MATERN_NU_SMOOTH)
                baseline_vals = (1 - mix_alpha) * vals_tensor + mix_alpha * v_smooth
            else:
                baseline_vals = vals_tensor

            # GAE (live critic baseline, same as v4)
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

            Adv = (Adv - Adv.mean()) / (Adv.std() + 1e-8)

            for _ in range(update_epochs):
                idxs = torch.randperm(rollout_len)
                for start in range(0, rollout_len, batch_size):
                    idx = idxs[start:start+batch_size]
                    s_b, a_b, old_lp_b, R_b, Adv_b = S[idx], A[idx], old_lp[idx], R[idx], Adv[idx]

                    logits_b, v_b, _ = model(s_b, return_feat=True)
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

                    # PPO loss (no entropy, same as v4)
                    loss = actor_loss + vf_coef * critic_loss

                    # v4: spectral reg DISABLED by default
                    if USE_SPECTRAL_REG and current_spec_coef > 0:
                        spec_loss = matern_spectral_reg(
                            model, lam=MATERN_LAM, nu=MATERN_NU_REG, s=MATERN_S,
                            exclude_conv1=SPEC_EXCLUDE_CONV1)
                        loss = loss + current_spec_coef * spec_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    optimizer.step()

            # Clear rollout buffers
            states, actions, rewards, dones, log_probs, values, q_vectors = [], [], [], [], [], [], []

        # ---- Logging (v10: dual-staircase phase) ----
        if step % 20000 == 0:
            avg_r = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            max_r = max(episode_rewards[-10:]) if episode_rewards else 0.0
            phase = release_phase_name(step)
            print(f"Step {step:8d} | Avg R: {avg_r:.2f} | Max R: {max_r:.2f} | "
                  f"alpha: {mix_alpha:.4f} | {phase:6s}")

    print(f"\nv10 Training complete.")
    return episode_rewards

if __name__ == "__main__":
    rewards = train()
    np.save("breakout_physics_guided_v10.npy", rewards)
