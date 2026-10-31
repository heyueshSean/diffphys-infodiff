# 04 · 神经网络线：CIFAR-10 / CIFAR-100 / InfoDiff

> 三条子线：**B 线**（ResNet-50 + 场，CIFAR-10，v29→v119）、
> **A 线**（ViT-Small + 场，CIFAR-100，v51→v111）、**C 线**（InfoDiff，`du/dt = ∇·(D(x)∇u)`，未并入主线）。
> 前两条**原理互鉴、代码独立**：不许搬实现、不许拿对方读数当证据。

---

## 1. 机制：场是网络里的一个可学频域滤波层（B 线）

以 `nn/cifar10/cnn10_v119.py` 为例的代码地图：

| 符号 | 位置 | 作用 |
|---|---|---|
| `class MaternDiffusion` | `cnn10_v119.py:3955` | 可微 Matérn 扩散场（phys2、phys3） |
| `class FixedDiffusion` | `cnn10_v119.py:4062` | 固定核场（phys1，`s=0.8 / λ=0.5` 写死） |
| `class Gate` | `cnn10_v119.py:4088` | 加性注入门（gate1/2/3） |
| `class GradientStablePhysResNetV3` | `cnn10_v119.py:4110` | ResNet-50 + 场 + gate 的装配体 |
| `def soft_bound` | `cnn10_v119.py:1134` | 参数软界 `min_v + (max_v−min_v)·sigmoid(x)` |
| `def physics_reg` | `cnn10_v119.py:4341` | 越界约束，系数 0.05 |
| `def get_w_phys` | `cnn10_v119.py:4360` | w_phys 调度（0.05 → 0.15 → 0.02） |
| `def structure_alignment_loss` | `cnn10_v119.py:4514` | 场 ↔ 骨干特征对齐损失 |
| `def refine_physics` | `cnn10_v119.py:4821` | **独立优化器写场通路**（e31–100，冻结主干） |
| `def _v119_complement` | `cnn10_v119.py:4525` | [v119 唯轴] 对齐目标 = 低通补集 |

### 1.1 核

```
    m(k) = ( 1 + λ_g · |k|²_g / ν_g )^{−s_g}                      # cnn10_v119.py:4033
    |k|²_g = kx² + a_g·ky² ,  a_g = 1 + ANISO_RANGE·tanh(aniso)   # :4032 / :4025, ANISO_RANGE=1.0
```

四个可学参数（经 `soft_bound`，另有 EMA buffer `running_s/lam/nu`，按 `0.9x + 0.1·running` 混合）：

| 参数 | 含义 | 区间 | 额外 clamp |
|---|---|---|---|
| `s` | 平滑度指数 | (0.3, 2.0) | `clamp(0.3, 1.3)` |
| `λ` | 长度尺度 | (0.1, 5.0) | — |
| `ν` | Matérn 形状 | (0.5, 3.0) | — |
| `a` | 各向异性 | (0, 2]（tanh 映射） | — |

前向：分组通道均值 `x_g` → 逐组 `fft2 · kernel · ifft2` → `repeat_interleave` 回原通道。

### 1.2 注入点与 gate

| 场 | 类型 | 挂载位置 | 通道 | gate |
|---|---|---|---|---|
| phys1 | `FixedDiffusion(256, groups=8)` | layer1 后 | 256 | gate1 **固定** `GATE1_INJECT=0.9` |
| phys2 | `MaternDiffusion(512, groups=64)` | layer2 后 | 512 | gate2 可学 |
| phys3 | `MaternDiffusion(1024, groups=8)` | layer3 后 | 1024 | gate3 可学 |

```
    p   = bn( proj(phys) )            # proj = 1×1 分组卷积, groups = min(ch, 8)
    out = feat + scale · a · p        # a = soft_bound(alpha, 0.05, 0.9), scale 初值 2.0
```

### 1.3 损失项

```
    loss = cls_loss + w_phys · smooth + PINN_WEIGHT · pinn_reg + reg_loss
```

| 项 | 值 / 形式 | 位置 |
|---|---|---|
| `w_phys` | 0.05 (e<60) → 0.15 (60–120) → 0.02（末） | `:4360-4366` |
| `smooth` | **[v118 轴]** `roughness(x_g)`，即场**输入**的差分平方均值（旧 = 场输出 `phys`） | `:4055-4057` |
| `PINN_WEIGHT` | **0.04** | `:950` |
| `pinn_reg` | **[v113 轴]** 粗糙度算在 `h2 = layer2 输出（gate2 之前）`（旧 = layer4 输出 `deep`） | `:4144`, `:4158-4164` |
| `physics_reg` | `0.05 · Σ (越界 s 的平方)` | `:4341-4356` |

### 1.4 写场通路 `refine_physics`

e31–100、独立 `phys_optimizer`、**冻结主干**、每 epoch 3 步：

```
    loss = structure_alignment_loss(ph2, target2) + structure_alignment_loss(ph3, target3)
```

`structure_alignment_loss` 本体 = `1 − cos(phys_norm, feat_norm) + 0.1·mse`（灰度化 + per-map 标准化）。
**这条通路是本线最后一条"显式把场塑形成某个目标"的通路**，也是 `v119` 的唯轴所在（见 §2.5）。

---

## 2. B 线：分支演进（完整表见 [`02-taxonomy.md`](02-taxonomy.md) 表二）

基线 `cnn10_v51.py` = **97.56**（seed1234）；`σ_diff = 0.169pp`；红线 97.04。

### 2.1 加性注入族（作用面 A）—— 整体归档

- **v89（SE 通道注意力）**：97.40，NULL 且 `SE_GATE_COLLAPSED` ⇒ 回退。
- **v90 / v101 / v102（各向异性）**：97.38 / **97.48** / 97.42。机制判词 `ANISO_DELIVERED`，
  但精度全部 NULL。v90 的实战裁定是 `ANISO_SELFTEST_FAILED`（自测**假红**）⇒ 轴不带入。
- **v98（gate2 通道粒度 8→64）**：97.42，NULL。
- **v111（频带夹带）**：97.46，`BAND_DELIVERED` + 精度 NULL。
  ⇒ **决定性结论**：夹带只改"场有多粗"，没改"场是否携带新信息" ⇒ **夹带族整族判死**。
- **总族裁定**：场输出是骨干特征的**滤波摹本** ⇒ 信息上必然冗余 ⇒ **加性注入通道整体归档**。
  证据：`band_frac_mean = 0.0968`（场 90% 能量在 DC）。

### 2.2 参数侧度量族（作用面 C）—— 判死

- **v88（SAM）**：97.19，触红线 ⇒ BACKTRACK。
- **v110（场诱导的 SAM-Sobolev 信任域）**：**97.50**，判词 `SAM_SOBOLEV_DELIVERED`。
  ★★ 但决定性诊断 **`dir_ratio_mean = 0.9129`**：Sobolev 度量只把更新方向拧了 **8.7%**
  ⇒ **事实上几乎就是裸 SAM**。
  数学上必然：卷积核梯度 DC 主导且 `K(0)=1` ⇒ 低通无法旋转已 DC 主导的方向。
  ⇒ **参数侧度量族判死**；连带的"裸 SAM（K≡1）对照跑"作废（省 ~7.5h 机时）。

### 2.3 读数 / 收获族（作用面 F）—— 判死

- **v112（视图群）**：delivered 97.52，但**同权重配对 Δ = −0.18pp**（McNemar p=0.05045）。
  立项假设"各向同性核先验 ⇒ 应在平移不变群上平均"被逐视图数据证伪：
  中心位 97.40 / 翻转 97.38 vs 四角位均值 **96.9325**（低 **0.4575pp**）。
  ⇒ **读数侧视图族判死**。
- **集成（ens）**：v43–v44 起**五次负贡献** ⇒ 该轴彻底判死。

### 2.4 约束通道（作用面 B）—— 双双 `CUT_NO_CONTENT`

这是本线做过的最干净的一类实验：**拓扑切断梯度通路，看场还剩不剩内容**。

| 版本 | 作用面 | 切断证据 | 内容证据 | 判词 |
|---|---|---|---|---|
| **v117** | pinn：`deep(gate3 后)` → `h2(layer2 前)` | `grad_frac_new = 0.0`（精确零）、`grad_frac_old = 3.6357e-5 > 0` | `band_frac_mean = 0.05526` ≪ 0.20 | `PINN_FACE_CUT_NO_CONTENT` |
| **v118** | smooth：`phys(场输出)` → `x_g(场输入)` | `grad_frac_smooth_new = 0.0`、`old = 1.261e-6 > 0` | `band_frac_mean = 0.038904` ≪ 0.20 | `SMOOTH_FACE_CUT_NO_CONTENT` |

**读数来之不易**：v113 因模块级 use-before-def（`NameError`）崩；v114 device 不匹配；
v115 `math domain error`（对负的未约束 `nu` 开方）。**三次都没测到那根轴** ⇒ v117 是第四次重测才拿到。

**v118 的独立负向证据**：`smooth_level_new = 6.546e-4` vs `old = 1.492e-5` ⇒ **43.9× 跃升**，
精度仍 NULL ⇒ 该正则项对精度**完全惰性**，"剂量大 ⇒ 有用"被实测否掉。

### 2.5 对齐目标族（作用面 E）—— v119，最后一张牌

**为什么不直接归档**：v118 已判"约束通道死"，但 `refine_physics` 里
`structure_alignment_loss(ph, feat)` 这条**写场通路从未读过数、从未被切过**，
而它的形式正是**显式把场推成骨干特征的摹本**＝v111「冗余 / DC」的**预测表型**。
⇒ 依「未测 ≠ 判负」必须先测。

**唯轴**：对齐目标 `feat`（摹本）→ `feat − avgpool_3(feat)`（**低通补集 / 互补**）。
目标在 `no_grad` + `.detach()` 下产出 ⇒ **不接受梯度、零新增参数**；
不改 loss 本体、不改 refine 窗口（e31–100）、不加开关。

**预注册判词**（沿用 V113_*，一个数不改）：

| 闸门 / 判据 | 阈值 | 判词 |
|---|---|---|
| `hp_frac < 0.02` | — | `REFINE_HP_TARGET_EMPTY` |
| `grad_max_hp ≤ 0` | — | `REFINE_HP_GRAD_DEAD` |
| **`band_frac ≥ 0.20`** | **唯一成败判据** | `REFINE_HP_DELIVERED` |
| `regime == COPY_ALIGNED` 且 `max\|corr\| ≥ 0.50` | — | `REFINE_HP_COPY_ATTRACTOR` |
| 其余 | — | `REFINE_HP_NO_CONTENT` |
| 探针崩 | — | `REFINE_HP_READOUT_ERROR` |

**分支动作（已写进 `criterion.note`，不许事后换口径）**：
- `NO_CONTENT` / `COPY_ATTRACTOR` / `GRAD_DEAD` ⇒ **物理场整条线归档**，
  v120 起只许动**目标函数主项**（分类损失形式 / 标签结构 / 损失权重结构），**不许**再回来动场。
- `DELIVERED` ⇒ **复活注入通道**。
- ⚠ v119 **改了训练数学**（场被推向高频补集并经 gate 注入主干）⇒ 精度可能分叉；触红线 97.04 ⇒ 逐字回退 v110。

---

## 3. A 线：ViT-Small / CIFAR-100

`σ_seed = 0.71pp`。

| 阶段 | 分支 | 代表版本 | delivered | 结论 |
|---|---|---|---|---|
| 各向异性热核 | `k²` 加方向自由度 | **v77** | **70.66** | 当时历史最高 |
| θ 朝向 | 核朝向参数化 | v78 | 70.41 | NULL，**判死** |
| per-head ar / 环带·反应-扩散 | 每头各向异性 / 环带调制 | v79 / v80 / v81 | — | **判死** |
| 注入代数 | 改注入的代数形式 | v82 | — | 判词最高档但**是假绿**，用户明令回退 ⇒ **判死** |
| 剂量 / 交付路由 | swa_avg / best_ckpt / teacher_snapshot | v84 / **v95** / v93 | 71.08 / **71.11** / 70.84 | NULL；跨版口径混用 |
| MIX 杠杆 + 软界→有理界 | 蒸馏混合权重参数化 | v101 / v102 | 70.03 / 70.18 | MIX 轴 z=2.64 可读 |
| 跨盆地 SWA 隔离 | SWA 窗口门控 `eff≥320` → `eff≥320 且 epoch≥720` | v103 | **75.88** | **不可归因到 v103**，且与 800ep 跨盆地不可比 |
| **尾窗平均窗口** | w150 → w250 / w400 | **v110 / v111** | v110 **75.34**；v111 未跑 | 剂量边际递增 ⇒ 顶点未到 |

**v110 的关键读数**（`SWA_PAIR_DELIVERED`）：
隔离臂（e720-800，n=81）= **74.67**；跨盆地臂（e320-400 + e720-800，n=162）= **65.33**；
差 **−9.34pp** ⇒ 跨盆地确是污染。**单 epoch 峰值指纹 = 74.03@e791**。
剂量-响应 w40 74.18 → w80 74.60 → w150 75.34（边际 +0.42 → +0.74pp，**递增**）⇒ 顶点未到。

**v110 的假归档**：`v109_seg_swa.dose_note` 写死「三者相差 <0.1pp ⇒ SWA 窗口轴归档」，
而它自己的 rows 极差 1.16pp ⇒ 自相矛盾 ⇒ **未判即归档**（纪律 389）。v111 就是为此立项。

---

## 4. C 线：InfoDiff

`nn/cifar10_infodiff/`，理论三锚点：Perona–Malik / Information Bottleneck / Wasserstein Gradient Flow。

```
    ∂u/∂t = ∇·( D(x) ∇u ) ,   D(x) = D0 · exp( −β · I(x) ) ,   I(x) = sigmoid( conv_1x1(u) )
```

显式 Euler（`n_steps=4 / dt=0.20`）+ Neumann 边界；`β=2.0`；注入点沿用 B 线三处 gate。
**刻意移除** B 线的 `physics_reg` / `PINN_WEIGHT` / `get_w_phys` / `w_phys*smooth`
⇒ `loss = cls_loss`，是**架构归纳偏置而非正则项**。

| 版本 | delivered | 读数 |
|---|---|---|
| `cnn10_info_v01.py` | **97.06** | 信息图没活（`i_std ≈ 0.012`） |
| `cnn10_info_v02.py` | **97.12**（best_tta@ep132；final 97.01；ens 97.06；168/180ep 早停） | **轴判死** `INFOMAP_NO_LEVERAGE` ⇒ 逐字回退 v51 |

**三层分解（本线最有价值的读数）**：算子存在性 **12.19pp** ≫ 幅度 **1.72pp** ≫ **信息项 0.03pp（不可读）**。
⇒ "扩散率应由可学、输入依赖的信息图调制"这个前提，在 CIFAR-10 上被**最干净地证伪**。

> 注：C 线是"内容依赖调制"四连死的最后一环（v56/v57 role_mix → v59 判别性先验 →
> v60~v63 样本条件化 → 信息图）。

---

## 5. 入库版本清单

| 文件 | 分支 | delivered |
|---|---|---|
| `nn/cifar10/cnn10_v51.py` | 全族基线 | **97.56**（seed1234） |
| `nn/cifar10/cnn10_v101.py` | 各向异性 Matérn 核 | 97.48 |
| `nn/cifar10/cnn10_v110.py` | 参数侧度量族（判死） | **97.50** |
| `nn/cifar10/cnn10_v111.py` | 频带夹带（判死） | 97.46 |
| `nn/cifar10/cnn10_v112.py` | 读数侧视图族（判死） | 97.52 |
| `nn/cifar10/cnn10_v117.py` | 约束通道·pinn 作用面 | **97.59** ⚠ 不同尺 |
| `nn/cifar10/cnn10_v118.py` | 约束通道·smooth 作用面 | 97.53 |
| `nn/cifar10/cnn10_v119.py` | 对齐目标族（**待跑**） | 未跑 |
| `nn/cifar100/vit100_v77.py` | 各向异性热核 | 70.66 |
| `nn/cifar100/vit100_v95.py` | 交付路由（teacher_snapshot） | 71.11 |
| `nn/cifar100/vit100_v103.py` | 跨盆地 SWA 隔离 | 75.88（不可归因） |
| `nn/cifar100/vit100_v110.py` | 尾窗平均（可归因最高） | **75.34** |
| `nn/cifar100/vit100_v111.py` | 尾窗剂量外推（**待跑**） | 未跑 |
| `nn/cifar10_infodiff/cnn10_info_v01.py` | InfoDiff 首版 | 97.06 |
| `nn/cifar10_infodiff/cnn10_info_v02.py` | InfoDiff 修复版 | **97.12**（轴判死） |

> ⚠ **`cnn10_v117` 的 97.59 与 `v110` 的 97.50 不同尺**：v117 存在执行层回归——
> `_v117_guarded_determinism_probe()` 的 `finally` 无条件 `uda(False)` ⇒ 180ep 全程跑在
> **非确定性算子**下，而 v110–v115 全为 True。
> ⇒ **同尺口径下的最佳是 `v110 = 97.50`**（八版里六口径全面最高、唯一跑满 180ep）。
