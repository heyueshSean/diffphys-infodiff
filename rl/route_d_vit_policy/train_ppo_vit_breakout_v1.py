"""
Training script for Route D: Physics-guided PPO with ViT-policy on Atari Breakout.

Integrates:
  - ViTActorCritic (vit_policy_atari.py) — ViT-Small policy network
  - PhysicsRegularizer (physics_rl_core.py) — spectral dose-response + Matern + Lipschitz
  - Standard PPO (clip surrogate, GAE, multi-epoch mini-batch)

Physics integration points (marked # PHYSICS):
  1. EMA target network (mean-teacher) creation and update
  2. PhysicsRegularizer call during PPO update
  3. field_alpha * physics_loss added to total loss
  4. Entropy quench + temper scheduling
  5. Learning rate linear decay

Benchmark: SB3 PPO 398 (CNN policy, no physics). This script: ViT + physics.
Current PPO SOTA: v26 215.2@4.3M (CNN + physics, storm-splitting path A).

Run:
  python train_ppo_vit_breakout_v1.py --seed 42 --total-steps 5000000
  python train_ppo_vit_breakout_v1.py --seed 42 --no-physics   # ablation: pure ViT-PPO

v1.1 (2026-08-25) CRITICAL FIX — removed model.train() in the PPO update loop.
  StochasticDepth randomly drops residual branches under train() mode. PPO's
  importance ratio requires IDENTICAL forward semantics at collection (eval)
  and update time: old log_prob came from an eval-mode forward, new log_prob
  from a train-mode forward, so ratio drifted past clip 1+-0.1 independent of
  the parameters. The clipped branch has zero gradient -> frozen policy
  (observed: policy loss ~0.005, reward stuck at 1-2 over 480K steps, physics
  loss floored at irreducible StochasticDepth noise). The model now stays in
  eval() for the entire run; architecture regularization is supplied by the
  physics losses. Also added H (policy entropy) / KL / clip% diagnostics to
  the log line, and a regression guard in verify_ppo_vit.py.

v1.2 (2026-08-26) CRITICAL FIX — max_grad_norm 0.5 -> 5.0.
  Gradient probe on the ViT-Small policy measured PPO gradient norms of
  13.7-30.2 (vs physics 3.1-5.5). With clip_grad_norm(0.5) ~98% of the PPO
  gradient was discarded: effective lr ~ 2.5e-4 * (0.5/20) ~ 6e-6, i.e. 1/40
  of nominal. Symptoms matched exactly: KL 0.001-0.014 (below the healthy
  0.005-0.05 band), H creeping down 1.386->1.19, value loss not converging,
  reward flat at the random-policy level. SB3 PPO uses the same 0.5 clip on
  a 1.7M-param Nature CNN whose gradient norm is ~1-3, so the cap only trims
  the tail; on the 11M-param ViT it amputated the signal. Paradigm mismatch
  #3 in the CIFAR->RL transfer: gradient-scale mismatch. The log line now
  also prints gnorm (pre-clip gradient norm) so the clip ratio
  min(1, max_grad_norm/gnorm) is always visible.

Dependencies:
  pip install gymnasium[atari] torch numpy opencv-python
  autoROM --accept-license
"""

import argparse
import os
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_policy_atari import ViTActorCritic
from physics_rl_core import (
    PhysicsRegularizer,
    entropy_schedule,
    field_alpha_schedule,
)

Batch = namedtuple("Batch", ["obs", "action", "log_prob", "advantage", "return_"])


def parse_args():
    p = argparse.ArgumentParser(description="Route D: Physics-guided PPO + ViT-policy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total-steps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=5.0,
                   help="Pre-clip gradient norm cap. v1.2: 5.0 (ViT grads ~20-30; "
                        "the old 0.5 clipped ~98%% of the PPO signal)")
    p.add_argument("--ema-momentum", type=float, default=0.999)
    p.add_argument("--no-physics", action="store_true", default=False,
                   help="Disable physics (ablation: pure ViT-PPO)")
    p.add_argument("--env-id", type=str, default="ALE/Breakout-v5")
    p.add_argument("--save-dir", type=str, default="./checkpoints_ppo_vit")
    p.add_argument("--log-interval", type=int, default=10)
    return p.parse_args()


# === Environment ===

def make_envs(env_id, n_envs, seed):
    """Create n_envs Atari environments with standard preprocessing.

    Handles gymnasium >= 1.0 API (FrameStackObservation) and legacy
    gymnasium < 1.0 API (FrameStack) transparently.
    """
    try:
        import ale_py  # noqa: F401 — registers ALE namespace
        import gymnasium as gym
        gym.register_envs(ale_py)
        from gymnasium.wrappers import AtariPreprocessing
        # gymnasium >= 1.0: FrameStack -> FrameStackObservation
        try:
            from gymnasium.wrappers import FrameStackObservation as _FrameStack
            _new_api = True
        except ImportError:
            from gymnasium.wrappers import FrameStack as _FrameStack
            _new_api = False
    except ImportError:
        raise ImportError(
            "pip install 'gymnasium[atari]'  "
            "# ale_py auto-installed; no autoROM needed for gym>=1.0"
        )
    # AtariPreprocessing resizes 210x160 -> 84x84 via cv2.resize; opencv is
    # NOT pulled in by gymnasium[atari] (it lives in gymnasium[other] extras),
    # so check it up-front with an actionable message.
    try:
        import cv2  # noqa: F401
    except ImportError:
        raise ImportError(
            "pip install opencv-python  "
            "# required by AtariPreprocessing resize (or: pip install 'gymnasium[other]')"
        )
    envs = []
    for i in range(n_envs):
        # frameskip=1 disables ALE's built-in skip so AtariPreprocessing
        # owns the 4-step skip (avoids ValueError: double frame-skip)
        env = gym.make(env_id, frameskip=1)
        env = AtariPreprocessing(env, frame_skip=4, screen_size=84,
                                grayscale_obs=True, scale_obs=True)
        if _new_api:
            env = _FrameStack(env, stack_size=4)
        else:
            env = _FrameStack(env, 4)
        envs.append(env)
    return envs


def _to_tensor(obs):
    """Convert obs to (C, H, W) = (4, 84, 84) float32 tensor.

    Handles both gymnasium < 1.0 (FrameStack -> (84,84,4) HWC)
    and >= 1.0 (FrameStackObservation -> (4,84,84) CHW) formats.
    """
    o = np.array(obs)
    if o.ndim == 3 and o.shape[-1] == 4:
        # legacy HWC -> CHW
        return torch.tensor(o, dtype=torch.float32).permute(2, 0, 1)
    # already CHW (gymnasium >= 1.0)
    return torch.tensor(o, dtype=torch.float32)


def reset_envs(envs, seed):
    return torch.stack([_to_tensor(env.reset(seed=seed + i)[0])
                        for i, env in enumerate(envs)])


def env_step(envs, actions):
    obs, rew, done = [], [], []
    for i, env in enumerate(envs):
        o, r, term, trunc, _ = env.step(actions[i].item())
        d = term or trunc
        if d:
            o, _ = env.reset()
        obs.append(_to_tensor(o))
        rew.append(float(np.clip(r, -1, 1)))
        done.append(float(d))
    return torch.stack(obs), torch.tensor(rew), torch.tensor(done)


# === Rollout buffer ===

class RolloutBuffer:
    def __init__(self, n_steps, n_envs, obs_shape=(4, 84, 84)):
        self.n_steps, self.n_envs = n_steps, n_envs
        self.obs = torch.zeros(n_steps, n_envs, *obs_shape)
        self.actions = torch.zeros(n_steps, n_envs, dtype=torch.long)
        self.log_probs = torch.zeros(n_steps, n_envs)
        self.values = torch.zeros(n_steps, n_envs)
        self.rewards = torch.zeros(n_steps, n_envs)
        self.dones = torch.zeros(n_steps, n_envs)
        self.advantages = torch.zeros(n_steps, n_envs)
        self.returns = torch.zeros(n_steps, n_envs)
        self.pos = 0

    def add(self, obs, action, log_prob, value, reward, done):
        i = self.pos
        self.obs[i], self.actions[i] = obs, action
        self.log_probs[i], self.values[i] = log_prob, value
        self.rewards[i], self.dones[i] = reward, done
        self.pos += 1

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        last_gae = torch.zeros(self.n_envs)
        for t in reversed(range(self.pos)):
            next_val = last_value if t == self.pos - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            last_gae = delta + gamma * lam * (1 - self.dones[t]) * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def get_batch(self, idx, device):
        flat = lambda x: x.reshape(-1, *x.shape[2:])
        adv = flat(self.advantages)[idx]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return Batch(
            obs=flat(self.obs)[idx].to(device),
            action=flat(self.actions)[idx].to(device),
            log_prob=flat(self.log_probs)[idx].to(device),
            advantage=adv.to(device),
            return_=flat(self.returns)[idx].to(device),
        )

    def reset(self):
        self.pos = 0


def lr_schedule(step, total, start_lr):
    return start_lr * max(1.0 - step / total, 0.0)


# === Training ===

def train(args):
    print(f"[v1.2] train_ppo_vit_breakout_v1 | torch={torch.__version__} | "
          f"physics={'on' if not args.no_physics else 'OFF (ablation)'} | "
          f"seed={args.seed} | steps={args.total_steps} | "
          f"max_grad_norm={args.max_grad_norm}")
    print("[v1.2] fixes active: model.eval() whole run + grad-norm 5.0 "
          "(log line ends with 'H=... KL=... clip%=... gnorm=...')")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_physics = not args.no_physics
    os.makedirs(args.save_dir, exist_ok=True)

    envs = make_envs(args.env_id, args.n_envs, args.seed)
    n_actions = envs[0].action_space.n

    model = ViTActorCritic(n_actions=n_actions).to(device)
    # PHYSICS #1: EMA target (mean-teacher)
    ema_target = model.init_ema_target().to(device) if use_physics else None
    if ema_target:
        ema_target.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    # PHYSICS #2: regularizer
    regularizer = PhysicsRegularizer(total_steps=args.total_steps) if use_physics else None

    buffer = RolloutBuffer(args.n_steps, args.n_envs)
    obs = reset_envs(envs, args.seed).to(device)

    step = 0
    ep_rewards, ep_reward = [], [0.0] * args.n_envs
    best = 0.0
    log_every = args.log_interval * args.n_steps * args.n_envs
    n_flat = args.n_steps * args.n_envs

    while step < args.total_steps:
        # --- Collect rollouts ---
        buffer.reset()
        model.eval()
        for t in range(args.n_steps):
            with torch.no_grad():
                logits, value = model(obs)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            next_obs, reward, done = env_step(envs, action.cpu())
            buffer.add(obs.cpu(), action.cpu(), log_prob.cpu(),
                       value.squeeze(-1).cpu(), reward, done)
            for i in range(args.n_envs):
                ep_reward[i] += reward[i].item()
                if done[i]:
                    ep_rewards.append(ep_reward[i])
                    ep_reward[i] = 0.0
            obs = next_obs.to(device)

        # --- GAE ---
        with torch.no_grad():
            last_value = model.get_value(obs).squeeze(-1).cpu()
        buffer.compute_gae(last_value, args.gamma, args.gae_lambda)

        # --- PPO update ---
        # FIX (v1.1): model stays in eval() mode. PPO's importance ratio
        # requires identical forward semantics at collection (eval, above)
        # and update time. StochasticDepth drops residual branches randomly
        # under train(): new_log_prob drifted from the stored log_prob
        # independent of the parameters, ratio blew past clip 1+-0.1, the
        # clipped branch contributed zero gradient, and the policy froze.
        # eval() makes StochasticDepth an identity op (regularization is
        # supplied by the physics losses) AND makes PhysicsRegularizer's
        # current-model features deterministic, matching its eval-mode
        # EMA target — the spectral dose-response then measures real
        # model-vs-EMA structure instead of StochasticDepth noise.
        model.eval()
        n_batches = n_flat // args.batch_size
        # Diagnostics: real policy entropy H (max ln(4)=1.386), approx KL,
        # clip fraction. Healthy PPO: H>0.8, KL~0.005-0.05, clip% 5-25.
        diag_h, diag_kl, diag_clip, diag_n, diag_gn = 0.0, 0.0, 0.0, 0, 0.0
        for _ in range(args.n_epochs):
            for _ in range(n_batches):
                idx = torch.randint(0, n_flat, (args.batch_size,))
                batch = buffer.get_batch(idx, device)

                # PHYSICS #3: compute physics loss
                if use_physics:
                    physics_loss, _ = regularizer(model, ema_target, batch.obs, step)
                else:
                    physics_loss = torch.tensor(0.0, device=device)

                # PPO forward
                logits, value = model(batch.obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_prob = dist.log_prob(batch.action)
                entropy = dist.entropy()

                # Clip surrogate
                ratio = torch.exp(new_log_prob - batch.log_prob)
                s1 = ratio * batch.advantage
                s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * batch.advantage
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = F.mse_loss(value.squeeze(-1), batch.return_)

                with torch.no_grad():
                    diag_h += entropy.mean().item()
                    diag_kl += (batch.log_prob - new_log_prob).mean().item()
                    diag_clip += ((ratio - 1.0).abs() > args.clip).float().mean().item()
                    diag_n += 1

                # PHYSICS #4: entropy quench + temper
                ent_coef = entropy_schedule(step, args.total_steps) if use_physics else 0.01
                # PHYSICS #5: field-alpha weighted total
                alpha = field_alpha_schedule(step, args.total_steps) if use_physics else 0.0
                total_loss = (policy_loss + args.value_coef * value_loss
                              + ent_coef * (-entropy.mean()) + alpha * physics_loss)

                lr = lr_schedule(step, args.total_steps, args.lr)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                optimizer.zero_grad()
                total_loss.backward()
                # v1.2: capture PRE-clip gradient norm; clip_grad_norm_ returns
                # the total norm BEFORE rescaling, so gnorm/max_grad_norm < 1
                # means the cap did not bind.
                gnorm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                diag_gn += float(gnorm)
                optimizer.step()

        # PHYSICS: update EMA target
        if use_physics:
            model.update_ema(ema_target, args.ema_momentum)

        step += n_flat

        # --- Logging ---
        if step % log_every == 0 or step >= args.total_steps:
            mean_r = np.mean(ep_rewards[-100:]) if ep_rewards else 0.0
            d_n = max(diag_n, 1)
            print(f"step={step}/{args.total_steps} | reward={mean_r:.1f} | "
                  f"best={best:.1f} | policy={policy_loss.item():.4f} | "
                  f"value={value_loss.item():.4f} | physics={physics_loss.item():.4f} | "
                  f"alpha={alpha:.3f} | ent={ent_coef:.4f} | lr={lr:.6f} | "
                  f"H={diag_h / d_n:.3f} | KL={diag_kl / d_n:.4f} | "
                  f"clip%={100.0 * diag_clip / d_n:.1f} | gnorm={diag_gn / d_n:.1f}")
            if mean_r > best:
                best = mean_r
                torch.save({"model": model.state_dict(), "step": step,
                            "best": best, "args": vars(args)},
                           os.path.join(args.save_dir, "best.pt"))

    print(f"Done. Best: {best:.1f} (SB3 PPO target: 398)")
    torch.save({"model": model.state_dict(), "step": step,
                "best": best, "args": vars(args)},
               os.path.join(args.save_dir, "final.pt"))


if __name__ == "__main__":
    train(parse_args())
