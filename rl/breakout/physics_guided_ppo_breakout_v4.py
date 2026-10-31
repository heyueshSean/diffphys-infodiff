"""
Physics-Guided PPO for Atari Breakout (v4)
STRICT ORIGINAL BASELINE - key control experiment.

v3 training result (user report, 200K steps):
  Avg R 3.6 -> 4.6 (vs original PPO: double digits by 200K), Max R stuck
  at 11-12. v3's ONLY change vs original was 2 residual blocks with
  zero-init identity conv. => the "safe" residual blocks still broke
  early learning.

v3 ROOT CAUSE (why zero-init residual is NOT safe in RL):
  [RC-A - Adam "leap over zero"] zero-init makes forward pass identity at
     step 0, but backward gives conv2 NON-ZERO gradient immediately
     (dL/dW2 = relu(conv1(x))^T * dL/dy). Adam normalizes updates by
     gradient RMS, so the FIRST update pushes W2 away from 0 to ~lr scale
     regardless of its initial value. "Identity at init" survives only 1
     forward pass; within a few K steps the residual path is fully active.
  [RC-B - gradient path restructuring] original: conv3 -> fc (direct).
     v3: conv3 -> res1 -> res2 -> fc. The residual Jacobian (1 + df/dx)
     multiplies the gradient to conv3, adding noise terms that PPO's
     high-variance gradients amplify -> conv3 learns slower.
  [RC-C - no-BN residual stacking is unstable in early RL] CIFAR ResNet-50
     had BN to stabilize; here 2 stacked 3x3 residuals (no BN) on 64ch
     7x7 features let the fc input distribution drift during bootstrapping
     -> value estimate noise -> weaker GAE advantage -> policy stalls.
  => REMOVE RESIDUAL BLOCKS ENTIRELY (USE_RESIDUAL = False).

v4 CHANGES (ALL physics extras OFF, 100% original architecture):
  [FIX1] USE_MATERN_SMOOTH = False : original Gaussian kernel exp(-d^2/(2s^2))
  [FIX2] USE_SPECTRAL_REG  = False : spectral reg off
  [FIX3] USE_BLURPOOL      = False : no blur
  [FIX4] SD_DROP_MAX       = 0.0   : StochasticDepth off
  [FIX5] USE_RESIDUAL      = False : NO residual blocks
         res1/res2 become nn.Identity() -> forward is EXACTLY:
         conv1 -> conv2 -> conv3 -> flatten -> fc -> actor/critic
         identical to the 300+ original PPO.

[KEEP] Everything else identical to original: Gaussian gradient_aware_smooth,
reward RunningNorm, cosine lr 2.5e-4->1e-5, clip=0.1, vf_coef=0.495,
epochs=4, batch=64, rollout=1024, diff_start=200K, k=12, sigma=10,
grad_threshold=2.0, mix_alpha 0.008->0.004 from 3M, NO entropy bonus.

EXPECTATION: v4 must reproduce the original learning curve (double-digit
Avg R by 200K, 300+ by 5M). If it does, residual blocks were the culprit
and physics enhancements must NOT touch the network architecture.
If it does NOT, the original code (user's copy) must be diffed against
this file to find the remaining discrepancy.
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

# ==================== Physics-Guided Config (v4: all OFF) ====================
# v4: strict original baseline. All switches OFF (kept for future ablations).
USE_BLURPOOL = False        # [OFF] blur destroys ball position info
USE_RESIDUAL = False        # [OFF] v3 lesson: zero-init residual still breaks RL
USE_SPECTRAL_REG = False    # [OFF] spec reg fights RL edge learning
USE_MATERN_SMOOTH = False   # [OFF] original Gaussian kernel

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
MATERN_NU_SMOOTH = 1.5      # kept but unused when USE_MATERN_SMOOTH=False

# ==================== AtariWrapper (unchanged from original) ====================
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

# ==================== RunningNorm (unchanged from original) ====================
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

# ==================== [kept for ablation] BlurPool2d (default OFF) ====================
class BlurPool2d(nn.Module):
    """Anti-aliased downsampling with 3x3 binomial [1,2,1;2,4,2;1,2,1]/16.

    v4: Default OFF. v1 lesson - blurring destroys ball position info.
    """
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

# ==================== [kept for ablation] ResidualBlock (default OFF) ====================
class ResidualBlock(nn.Module):
    """Residual block. v4: NOT used (USE_RESIDUAL=False).

    v3 proved zero-init identity is NOT safe in RL: Adam's first update
    pushes conv2 away from 0 (gradient-RMS-normalized, independent of the
    zero init), so the residual path activates within a few K steps and
    the network drifts from the original CNN into a deeper no-BN stack
    that learns much slower under PPO bootstrapping.
    """
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

# ==================== PhysicsGuidedActorCritic (v4: 100% original) ====================
class PhysicsGuidedActorCritic(nn.Module):
    """Actor-Critic (v4: EXACT original architecture).

    v4 Architecture (identical to the 300+ original PPO):
      Input: 4x84x84
      -> conv1(4->32, 8x8, s4) -> 32x20x20
      -> conv2(32->64, 4x4, s2) -> 64x9x9
      -> conv3(64->64, 3x3, s1) -> 64x7x7
      -> flatten -> 3136
      -> fc -> 512
      -> actor -> n_actions
      -> critic -> 1

    NO BlurPool, NO residual, NO spectral reg. res1/res2 are Identity.
    """
    def __init__(self, n_actions):
        super().__init__()

        if USE_BLURPOOL:
            self.pre_blur2 = BlurPool2d(32, stride=1)
        else:
            self.pre_blur2 = nn.Identity()

        # Conv layers (same as original)
        self.conv1 = nn.Conv2d(4, 32, 8, 4); self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1); self.fc = nn.Linear(64*7*7, 512)
        self.actor = nn.Linear(512, n_actions); self.critic = nn.Linear(512, 1)

        # v4: residual blocks DISABLED -> Identity
        if USE_RESIDUAL:
            self.res1 = ResidualBlock(64, drop_rate=0.0)
            self.res2 = ResidualBlock(64, drop_rate=0.0)
        else:
            self.res1 = nn.Identity()
            self.res2 = nn.Identity()

        # Orthogonal init (same as original)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None: nn.init.zeros_(m.bias)

        if USE_RESIDUAL:
            for blk in (self.res1, self.res2):
                if isinstance(blk, ResidualBlock):
                    blk.zero_init()

    def forward(self, x, return_qv=False, return_feat=False):
        # v4: exact original feature flow (res1/res2 = Identity)
        x = F.relu(self.conv1(x))
        x = self.pre_blur2(x)       # Identity when USE_BLURPOOL=False
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.res1(x)            # Identity
        x = self.res2(x)            # Identity
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
        """v4: no-op (SD off). Kept for ablation use."""
        if USE_RESIDUAL and isinstance(self.res1, ResidualBlock):
            self.res1.set_drop_rate(drop_rate)
            self.res2.set_drop_rate(drop_rate)

# ==================== [kept, default OFF] Matern spectral reg ====================
def matern_spectral_reg(model, lam=1.0, nu=1.0, s=1.0, exclude_conv1=True):
    """Matern kernel spectral regularization on conv weights (v4: OFF)."""
    loss = 0.0
    n_layers = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 4:
            if exclude_conv1 and 'conv1' in name and 'conv1' in name.split('.'):
                continue
            w = param  # (out_ch, in_ch, kh, kw)
            W = torch.fft.fft2(w, norm='ortho')
            W_mag = W.abs() ** 2  # power spectrum
            kh, kw = w.shape[2], w.shape[3]
            fy = torch.fft.fftfreq(kh).to(w.device)
            fx = torch.fft.fftfreq(kw).to(w.device)
            freq_sq = fy[:, None]**2 + fx[None, :]**2  # (kh, kw)
            matern = (1.0 + lam * freq_sq / nu) ** (-s)
            high_freq_weight = 1.0 - matern
            loss = loss + (W_mag * high_freq_weight).mean()
            n_layers += 1
    return loss / max(n_layers, 1)

# ==================== gradient_aware_smooth (Gaussian, ORIGINAL) ====================
def gradient_aware_smooth(values, q_vectors, k, sigma, grad_threshold=2.0,
                          use_matern=False, nu=1.5):
    """Physics-inspired value function smoothing (v4: ORIGINAL Gaussian).

    W = exp(-d^2/(2*sigma^2)), EXACTLY as the 300+ original PPO.
    Matern path kept only for ablation studies (default OFF).
    """
    N = values.shape[0]
    if N < 3: return values
    with torch.no_grad():
        best_actions = q_vectors.argmax(dim=1)
        q_norm = F.normalize(q_vectors, dim=1)
        dist = torch.cdist(q_norm, q_norm)
        k = min(k, N-1)
        _, idx = torch.topk(dist, k=k, largest=False, dim=1)

        # Kernel choice: Gaussian (v4 DEFAULT) vs Matern (ablation only)
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

# ==================== Warmup helpers (kept, inactive at v4 defaults) ====================
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

# ==================== Training ====================
def train(total_steps=5_000_000, rollout_len=1024, gamma=0.99, lambda_=0.95,
          clip_eps=0.1, lr_max=2.5e-4, lr_min=1e-5,
          update_epochs=4, batch_size=64, vf_coef=0.495, diff_start=200_000,
          mix_alpha_initial=0.008, mix_alpha_final=0.004, mix_decay_start=3_000_000):
    env = AtariWrapper("BreakoutNoFrameskip-v4")
    n_actions = env.env.action_space.n
    model = PhysicsGuidedActorCritic(n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_max, eps=1e-5)
    reward_norm = RunningNorm()

    DIFF_K = 12
    DIFF_SIGMA = 10.0
    GRAD_THRESHOLD = 2.0

    # Print configuration
    print("\n" + "="*60)
    print("Physics-Guided PPO v4 Configuration (STRICT ORIGINAL BASELINE):")
    print(f"  [FIX1] Matern smooth: {USE_MATERN_SMOOTH} (False = ORIGINAL Gaussian kernel)")
    print(f"  [FIX2] Spectral reg:  {USE_SPECTRAL_REG} (False = disabled)")
    print(f"  [FIX3] BlurPool:      {USE_BLURPOOL} (False = no blur)")
    print(f"  [FIX4] SD drop:       {SD_DROP_MAX} (0.0 = off)")
    print(f"  [FIX5] Residual:      {USE_RESIDUAL} (False = NO residual, 100% original)")
    print(f"  PPO: clip={clip_eps}, epochs={update_epochs}, batch={batch_size}")
    print(f"  LR: {lr_max}->{lr_min} (cosine), rollout={rollout_len}")
    print(f"  Physics smooth: diff_start={diff_start}, mix_alpha={mix_alpha_initial}->{mix_alpha_final}")
    print(f"  Smooth kernel: k={DIFF_K}, sigma={DIFF_SIGMA}, grad_threshold={GRAD_THRESHOLD}")
    print("="*60 + "\n")

    episode_rewards = []
    state = env.reset()
    ep_raw_reward = 0
    states, actions, rewards, dones, log_probs, values, q_vectors = [], [], [], [], [], [], []

    for step in range(1, total_steps+1):
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * step / total_steps))
        for pg in optimizer.param_groups: pg['lr'] = lr

        # Warmup schedules (inactive at v4 defaults)
        current_sd_drop = get_sd_drop(step)
        current_spec_coef = get_spectral_coef(step)
        model.set_sd_drop(current_sd_drop)

        # Physics smoothing alpha decay
        if step >= mix_decay_start:
            decay_progress = min(1.0, (step - mix_decay_start) / (total_steps - mix_decay_start))
            mix_alpha = mix_alpha_initial + (mix_alpha_final - mix_alpha_initial) * decay_progress
        else:
            mix_alpha = mix_alpha_initial

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

        if len(states) >= rollout_len:
            s_next = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                _, v_last, _ = model(s_next, return_qv=True)
            v_last = v_last.squeeze(0)

            vals_tensor = torch.cat(values, dim=0).squeeze(-1)
            q_rollout = torch.cat(q_vectors, dim=0).squeeze(1)

            # Physics field smooth baseline (v4: ORIGINAL Gaussian kernel)
            if step >= diff_start:
                v_smooth = gradient_aware_smooth(vals_tensor, q_rollout,
                                                 k=DIFF_K, sigma=DIFF_SIGMA,
                                                 grad_threshold=GRAD_THRESHOLD,
                                                 use_matern=USE_MATERN_SMOOTH,
                                                 nu=MATERN_NU_SMOOTH)
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

                    # PPO loss (no entropy, same as original)
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

            states, actions, rewards, dones, log_probs, values, q_vectors = [], [], [], [], [], [], []

        if step % 20000 == 0:
            avg_r = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            max_r = max(episode_rewards[-10:]) if episode_rewards else 0.0
            extra = ""
            if USE_SPECTRAL_REG:
                spec_val = matern_spectral_reg(
                    model, lam=MATERN_LAM, nu=MATERN_NU_REG, s=MATERN_S,
                    exclude_conv1=SPEC_EXCLUDE_CONV1).item()
                extra = f" | spec: {spec_val:.6f} | spec_coef: {current_spec_coef:.6f}"
            sd_info = f" | sd_drop: {current_sd_drop:.4f}" if (USE_RESIDUAL and SD_DROP_MAX > 0) else ""
            print(f"Step {step:8d} | Avg R: {avg_r:.2f} | Max R: {max_r:.2f} | mix_a: {mix_alpha:.4f}{extra}{sd_info}")

    return episode_rewards

if __name__ == "__main__":
    rewards = train()
    np.save("breakout_physics_guided_v4.npy", rewards)
