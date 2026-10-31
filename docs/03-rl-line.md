# 03 · 强化学习线：PPO + Atari Breakout（v1–v34 + Route D）

> 时间跨度 2026-08-13 → 2026-08-26。**已归档 + 方向性否定**（用户 2026-08-26 明令终止）。
> 本文留档的原因：这条线是"可微物理场"**唯一大面积生效**的一条线，结论必须能被别人复用。

---

## 1. 机制：场跑在 rollout 图上，不是网络里

**关键结构性事实**：RL 线的物理场**不是网络层**。策略/价值网络 100% 是原版 NatureCNN
（1,686,693 参数，零额外开销）。场是**跑在 rollout 批次上的图扩散算子**，在 GAE 之前介入。

以 SOTA 版 `rl/breakout/physics_guided_ppo_breakout_v26.py` 为例的代码地图：

| 角色 | 函数 / 类 | 位置 |
|---|---|---|
| Actor-Critic（原版 NatureCNN） | `PhysicsGuidedActorCritic` | `v26.py:270`（forward `:299`） |
| Q 向量（图坐标，4 维 = 动作偏好） | `q = v + (logits − logits.mean())` | `v26.py:310-315` |
| **双向扩散平滑** | `gradient_aware_smooth` | `v26.py:410` |
| **方向性优势传播 DAP** | `directional_advantage_propagate` | `v26.py:451` |
| **物理损失（特征 Lipschitz 正则）** | `physics_lipschitz_loss` | `v26.py:347` |
| 场强调度 α(t) | `release_cycle_alpha` / `effective_alpha` | `v26.py:496 / 690` |
| DAP 强度调度 | `dap_schedule_alpha` / `effective_dap_alpha` | `v26.py:563 / 697` |
| 熵调度（淬火 + 回火） | `entropy_schedule_alpha` | `v26.py:595` |
| 崩溃检测 / 硬回滚 | `detect_collapse` / `detect_hard_collapse` | `v26.py:621 / 652` |

**算子**（`v26.py:415-447`，推导见 `01-foundations.md` §5）：

```
    dist   = cdist(F.normalize(q), F.normalize(q))                 # Q 空间弦距离
    W      = exp(−dist² / (2σ²))                                   # 高斯核, σ=10, k=12
    W     *= [same action]                                         # 同动作门控（硬边剪枝）
    W     *= exp(−|v_i − v_j| / (grad_threshold · std(v)))          # Perona–Malik 梯度抑制
    P      = W / rowsum(W)                                         # 行归一化随机游走矩阵
    V_smooth = P @ values                                          # 一步图扩散
```

### 三个介入点

| # | 介入点 | 代码 | 位置 |
|---|---|---|---|
| ① | **GAE baseline**：物理场平滑 V(s) 后混合 | `baseline_vals = (1−α)·V + α·V_smooth` | `v26.py:951` |
| ② | **Advantage 整流传播**（加法，非混合） | `Adv += dap_alpha * signal`，`signal[i] = Σ_j P[i,j]·max(0, Adv[j]−Adv[i])` | `v26.py:986 / 491` |
| ③ | **进 loss**：特征空间 Lipschitz 正则 | `loss += w_phys_t * physics_lipschitz_loss(feat_b, q_b, sigma=10)` | `v26.py:1028` |

**开销**：实测 **0.06%–0.07%**（`t_field` 计时，v26：11.4s / 16414s）。无额外网络、无 replay buffer。

---

## 2. 分支演进（22 个分支，完整版见 [`02-taxonomy.md`](02-taxonomy.md) 表一）

### 2.1 阶段一：证明"不能动架构"（v1–v6）

- **v1–v3（网络结构轴）**：BlurPool / ResidualBlock / StochasticDepth / 频谱正则。
  结果全线崩：v1 的 140K 平均回报 1.5–3.9；加了 ResidualBlock 的 v2 只有 20–25@4M
  （原版同期 300+）。**判死**，从此立下"场只能改机制，不能动架构"的铁律。
- **v4（基线）**：关掉全部结构增强，只留原版高斯平滑 ⇒ 峰值 **311.5**@4.52M、Max R 360。
  这个数字成为后续所有版本的参照。
- **v5（训练技巧集成）**：CheckpointPool 值集成 + CV 自适应 α + KL 早停 + 熵 0.01。
  440K 前三项机制**零次生效**；唯一生效项是熵，且有害 ⇒ v5_fix 分离后仍判负。
- **v6（Value Target Network）**：**根本性错误**——on-policy 算法要求 baseline 与当前策略一致。
  且实现上 `lcm(1024, 200000) = 3,200,000` ⇒ target **全程未更新**，GAE baseline 等于随机初始化 critic。

### 2.2 阶段二：α 释放调度——最强单轴（v7–v10）

不动结构，只动"场的强度怎么随时间释放"：

| 版本 | 峰值 | Max R | 尾段 |
|---|---|---|---|
| v7 | 242.8 | — | 5M 终点 206.3 |
| v8 | 256.4 | — | 3–5M 均值 145.2、CV 0.36 |
| **v9** | **288.8**@4.50M | **373** | 4.5–5M **186.9**、CV 0.264 |
| **v10**（10M 预算） | **295.6**@9.22M | **404** | 最后 1M **204.6** |

⇒ **调度是这条线最大的杠杆**——但它调的是场，不是网络。

### 2.3 阶段三：从"平滑 value"到"传播 advantage"（v13–v16）

- **v13**：把场的作用对象从 value 升级为 **advantage**（`adv_alpha = mix_alpha*5`，cap 0.05）
  ⇒ 末期1M 168.3、final Avg R **247.9**、CV 0.063，且水位 100/130 各比 v10 **快 1.5M**。
  定性：**加速器**。
- **v16（DAP）**：把"双向平滑"改成**不对称整流传播**
  `signal[i] = Σ_j P[i,j]·max(0, Adv[j] − Adv[i])`——只让**更好的邻居**把优势传给你。
  ⇒ 末期1M **186.4**、峰值块 **207.8**、Max R **413**。这是本线的**核心算子**。

### 2.4 阶段四：熵的淬火与回火（v18–v21）

- v18 158.5 → v19 158.97 → v20 182.9（峰值块 **203.8，首破 200**）
- **v21**：末期1M **191.92**，首次 PASS 目标线；峰值块 213.6、Max 414、TAIL CV 8.0%。

### 2.5 阶段五：物理损失进 loss（v25–v27）——本线 SOTA

| 版本 | best block | final-1M | TAIL CV | TAIL low | 触发 |
|---|---|---|---|---|---|
| v25（全窗口） | 201.6 | 191.71 | **2.70%** | 185.6 | — |
| **v26（TAIL-only）** | **215.2**@4.3M | **190.88** | 4.34% | **180.5** | **0** |
| v27（hold 0.15） | — | 188.42 | 8.20% | 172.5 | — |

**v26 vs v23 的干净 A/B 是本线最重要的因果证据**：
TAIL CV 7.51% → 2.87%、low +13.4，但 **best −12.0**。
⇒ **物理损失是方差压缩器，不是天花板提升器**。天花板 ~215 由容量 / PPO 动力学 / 预算决定，
不由表示平滑度决定。

### 2.6 阶段六：全部后续分支 FAIL（v28–v34）

| 版本 | 分支 | best | final-1M | 结论 |
|---|---|---|---|---|
| v28 | ENT-TEMPER 时间轴 | 212.1 | 190.14 | NULL（但产出科学结论：谷随 ent 移动） |
| v29 | 能量扩散损失（热半群一致性） | 211.8 | **177.75**（−13.13） | FAIL |
| v30 | 容量升级 1.69M → 4.10M | **63.2** | ~48.1 | **FAIL（−70.7%，决定性）** |
| v31 | 自适应闭环（成熟度触发） | 194.7 | 162.1 | FAIL |
| v32 | 调度风暴拆分 | 213.9 | 187.92 | FAIL（但证伪"四项叠加风暴"假说） |
| v33 | D2 残差逐样本加权 | 210.4 | 185.37 | FAIL（b43 压制 −21.1） |
| v34 | D3 多尺度谱分解 | 201.7 | 182.73 | FAIL（CV **3.26%** 系列最优，但 b43 −16.5） |

**统一签名**：v33（按残差选样本）与 v34（按核局部性选尺度）是**完全不同的集中轴**，
却给出**同样的峰值压制签名**（b43 各自 −21.1 / −16.5）。
⇒ 任何"朝高能态集中剂量"的正则都会压峰；v26 的**均匀** σ=10 因保留了近邻自由度而胜出。

### 2.7 Route D：ViT 策略网络（结构性失败）

`rl/route_d_vit_policy/`：`physics_rl_core.py` + `train_ppo_vit_breakout_v1.py` + `vit_policy_atari.py`。
把 CIFAR-100 ViT 线上的物理正则整套搬到 RL：谱分解（patch 维 FFT → coarse/mid/fine）、
D5 时间轴旋转（剂量中性）、Matérn value loss（ν=2.5）、attention 谱惩罚、熵调度、场 α 释放周期。

**三层 bug 全修后仍无起色**（train/eval 破坏 importance ratio → v1.1；版本未同步 → banner；
`max_grad_norm=0.5` 对 11M ViT 裁剪 97–99.7% 梯度 → v1.2 cap 5.0）⇒ 排除实现层后判定为
**结构性失败**：from-scratch 无归纳偏置大参数架构做 RL 策略网络，样本效率比 CNN 慢 **5–20×**。
reward 卡在 **1.0–1.3**@133K（2.7% 预算）。

---

## 3. 全线 SOTA：v26

| 指标 | v26 | 为什么算最好 |
|---|---|---|
| best block (100K) | **215.2**@4.3M | 全线块级最高 |
| final-1M 均值 | **190.88** | 5M 预算内最高之一（v21 191.92 但 TAIL CV 8.0%） |
| TAIL CV | **4.34%** | 稳定性量级领先（v23 7.5%、v21 8.0%、v31 9.1%） |
| TAIL low | **180.5** | 地板最高（v23 172.2、v28 164.0、v31 141.6） |
| 崩溃触发 | **0** | 鲁棒性目标达成 |
| 开销 | **0.06%** | 无额外网络、无 replay |

**竞争者（口径不同，不可直接相减）**

- **v10（10M 预算）**：峰值 **295.6**@9.22M、Max R **404**、最后 1M 204.6 ——
  **绝对回报全线最高**，但预算是 v26 的 2 倍。
- **v9（5M，20K 数据点口径）**：峰值 **288.8**，但那是 20K 单采样点；v26 的 215.2 是 100K 块均值。
- **学术对标**（同期记录）：SB3 PPO 398@10M、rl-algo 383@10M、XingTian 341@10M、RLlib 104@10M。
  v9 的 288.8@5M ≈ 人类归一化 9.5×，定位为"同预算正常偏上，非 SOTA 非失败"。

---

## 4. 入库版本清单

| 文件 | 角色 | 实测 |
|---|---|---|
| `rl/breakout/physics_guided_ppo_breakout_v4.py` | 原版基线 | 峰值 311.5@4.52M、Max 360 |
| `rl/breakout/physics_guided_ppo_breakout_v9.py` | α 调度（5M） | 峰值 288.8、Max 373 |
| `rl/breakout/physics_guided_ppo_breakout_v10.py` | α 调度（10M） | 峰值 295.6、Max **404** |
| `rl/breakout/physics_guided_ppo_breakout_v13.py` | Advantage 平滑 | final Avg R 247.9、CV 0.063 |
| `rl/breakout/physics_guided_ppo_breakout_v16.py` | DAP 传播 | 末期1M 186.4、峰值块 207.8 |
| `rl/breakout/physics_guided_ppo_breakout_v21.py` | 熵淬火 + 回火 | 末期1M **191.92**（首次 PASS） |
| **`rl/breakout/physics_guided_ppo_breakout_v26.py`** | **★ 全线 SOTA** | **best 215.2 / final-1M 190.88 / CV 4.34%** |
| `rl/breakout/physics_guided_ppo_breakout_v34.py` | 多尺度谱（CV 单项最优） | CV **3.26%**、best 201.7 |
| `rl/breakout/physics_guided_ppo_breakout_v24.py` | **反例**：核函数"修复" | 峰值 **91.9**（−57%） |
| `rl/breakout/physics_guided_ppo_breakout_v30.py` | **反例**：容量升级 | best **63.2**（−70.7%） |
| `rl/route_d_vit_policy/physics_rl_core.py` | Route D 核心 | 结构性失败留档 |
| `rl/route_d_vit_policy/train_ppo_vit_breakout_v1.py` | Route D 训练脚本 | reward 1.0–1.3@133K |
| `rl/route_d_vit_policy/vit_policy_atari.py` | Route D 网络 | 10,964,741 参数 |

---

## 5. 这条线为什么停了

用户 2026-08-26 原话：**「专注做 vit，ppo 的思路经过验证是错误的，刚才的思路可以否定了，
以后再探索的时候不要再重蹈覆辙」**。资源全部转 ViT / CIFAR-100 主线。
2026-08-27 起的日志中再无 RL 线条目 ⇒ **线已停，未再重启**。

⚠ **方法论硬伤（离发表的距离）**，留档时必须一并披露：
1. **单 seed**——v10 与 v9 的 SMOOTH 段 200K 4.7 vs 9.1 就是不同 seed 的方差。
2. **从未训过 α=0 的 vanilla 对照** ⇒ **无法声称"物理场优于 vanilla"**；
   v9 的收益可能主要来自"调度"而非"物理特征"。
3. 单游戏、无统计检验 / 置信区间。

⇒ 因此本线的正确读法是：**"把场接在信用分配上有可观增益"** 是一个**现象**，
不是一个已完成验证的**结论**。
