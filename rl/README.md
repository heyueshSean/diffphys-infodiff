# RL 线 · PPO + Atari Breakout

**状态：已归档 + 方向性否定（2026-08-26 终止）。** 留档原因：这是可微物理场**唯一大面积生效**的一条线。

## 机制一句话

场**不是网络层**——它是跑在 rollout 批次上的**图扩散算子**，在 GAE 之前介入
value baseline 与 advantage。网络 100% 是原版 NatureCNN（1,686,693 参数，零额外开销）。

```
    dist   = cdist(normalize(q_i), normalize(q_j))          # q = v + (logits − mean logits)
    W      = exp(−dist²/(2σ²)) · [同动作] · exp(−|v_i−v_j|/(grad_threshold·std(v)))
    P      = W / rowsum(W)                                   # 行归一化随机游走矩阵
    V_smooth = P @ V                                         # 一步图扩散
```

三个介入点：① GAE baseline 混合；② Advantage 整流传播（DAP）；③ 物理 Lipschitz 损失进 loss。
实测开销 **0.06%**。

## 文件

| 文件 | 分支 | 实测 |
|---|---|---|
| `breakout/physics_guided_ppo_breakout_v4.py` | 原版基线 | 峰值 **311.5**@4.52M、Max 360 |
| `breakout/physics_guided_ppo_breakout_v9.py` | α 释放调度（5M） | 峰值 **288.8**、Max **373** |
| `breakout/physics_guided_ppo_breakout_v10.py` | α 释放调度（10M） | 峰值 **295.6**、Max **404**、最后1M 204.6 |
| `breakout/physics_guided_ppo_breakout_v13.py` | Advantage 平滑 | final Avg R **247.9**、CV 0.063 |
| `breakout/physics_guided_ppo_breakout_v16.py` | DAP 方向性优势传播 | 末期1M **186.4**、峰值块 **207.8** |
| `breakout/physics_guided_ppo_breakout_v21.py` | 熵淬火 + 回火 | 末期1M **191.92**（首次 PASS）、Max 414 |
| **`breakout/physics_guided_ppo_breakout_v26.py`** | **★ 全线 SOTA**（物理损失 TAIL-only） | **best 215.2@4.3M / final-1M 190.88 / TAIL CV 4.34% / 0 触发** |
| `breakout/physics_guided_ppo_breakout_v34.py` | 多尺度谱（CV 单项最优） | CV **3.26%**、best 201.7 |
| `breakout/physics_guided_ppo_breakout_v24.py` | **反例**：核函数"修复" | 峰值 **91.9**（−57%） |
| `breakout/physics_guided_ppo_breakout_v30.py` | **反例**：容量升级 | best **63.2**（−70.7%） |
| `route_d_vit_policy/physics_rl_core.py` | Route D 核心（谱分解 / Matérn value / 注意力谱惩罚） | 结构性失败留档 |
| `route_d_vit_policy/train_ppo_vit_breakout_v1.py` | Route D 训练脚本 | reward 卡 1.0–1.3@133K |
| `route_d_vit_policy/vit_policy_atari.py` | Route D 网络 | 10,964,741 参数 |

## 跑

```bash
python rl/breakout/physics_guided_ppo_breakout_v26.py    # 零参数
```

## ⚠ 读这批数字的前提

1. **单 seed**（v10 与 v9 的 SMOOTH 段 200K 4.7 vs 9.1 就是 seed 方差）；
2. **从未训过 α=0 的 vanilla 对照** ⇒ 不能声称"物理场优于 vanilla"，收益可能主要来自**调度**；
3. 单游戏、无统计检验 / CI；
4. v4–v13 用 20K 数据点、v14+ 用 100K 块均值 —— **两期数字不可直接相减**。

详见 [`docs/03-rl-line.md`](../docs/03-rl-line.md)。
