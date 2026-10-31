# 可微物理场的信息/能量扩散：双线实验档案

**Differentiable Physics Fields for Information / Energy Diffusion — a two-line experimental archive**

> 一个可微的扩散算子，插进学习系统的**不同作用面**，会产生完全不同、且可判定的后果。
> 本仓库是这条命题在**强化学习**与**神经网络**两条线上、横跨 ~150 个版本的完整实验记录，
> 包括全部正面结果、全部负面结果，以及每个分支的最佳代码版本。

---

## 一句话结论

**作用面（where you attach the field）决定一切，剂量（how strong）几乎无效。**

两条线独立地撞上同一堵墙：

| | 强化学习线（PPO / Breakout） | 神经网络线（CIFAR-10 / CIFAR-100） |
|---|---|---|
| 场**有效**的作用面 | **信用分配**：value baseline 平滑 + advantage 方向性传播 | **信用分配之外几乎都无效** |
| 最强一版 | `v26`：best block **215.2**、final-1M **190.88**、TAIL CV **4.34%**、开销 **0.06%** | `v117`：delivered **97.59**（但同尺口径最佳为 `v110` = **97.50**） |
| 场**无效**的作用面 | 架构 / 核函数 / 容量 / 自适应闭环 | 加性注入、正则剂量、参数侧度量、读数聚合 |
| 剂量证据 | 三条正则轴（加剂量 / 智能分配 / 换形式）全灭 | smooth 剂量抬 **43.9×**，精度仍 NULL |
| 归档判定 | Route D（ViT 策略网络）结构性失败，线已停 | 约束通道 `CUT_NO_CONTENT`；`v119` 为最后一张牌 |

---

## 目录地图

```
diffphys-infodiff/
├── README.md                  ← 你在这里
├── MANIFEST.md                ← 全部入库文件的 sha1 / 字节 / 行数 / 换行约定
├── docs/
│   ├── 00-overview.md         研究总纲：问题、方法族、两线地图、怎么读本仓库
│   ├── 01-foundations.md      ★ 基础原理：扩散/热半群/Matérn/WGF/IB + 六类作用面分类学
│   ├── 02-taxonomy.md         ★ 分支总表：两线 × 22+18 个分支 × 代表版本 × 实测数字 × 结论
│   ├── 03-rl-line.md          强化学习线全文（PPO/Breakout，v1–v34 + Route D）
│   ├── 04-nn-line.md          神经网络线全文（CIFAR-10 / CIFAR-100 / InfoDiff）
│   ├── 05-negative-results.md ★ 负面结果总汇（本仓库最有价值的部分）
│   ├── 06-methodology.md      测量协议与判据写法（跨线通用）
│   └── 07-reproducibility.md  环境、指纹核对、发车命令
├── rl/breakout/               PPO+Breakout 各分支最佳版本（v4 v9 v10 v13 v16 v21 v24 v26 v30 v34）
├── rl/route_d_vit_policy/     Route D：ViT 策略网络（结构性失败，留档）
├── nn/cifar10/                ResNet-50 + 场  （v51 v101 v110 v111 v112 v117 v118 v119）
├── nn/cifar10_infodiff/       C 线 InfoDiff：du/dt = ∇·(D(x)∇u)（v01 v02）
├── nn/cifar100/               ViT-Small + 场  （v77 v95 v103 v110 v111）
├── results/                   真跑产出的 JSON / verify 输出
└── tools/                     幂等生成器 + 部署级/执行级自证 + 发车路径真跑（方法论样板）
```

---

## 六类作用面（本仓库的分类学骨架）

可微物理场是一个**谱乘子** `m(k)`（可能输入依赖）。它插到哪里，决定它是什么：

| 编号 | 作用面 | 场做了什么 | RL 线结论 | NN 线结论 |
|---|---|---|---|---|
| **A** | 加性注入（前向相加） | 把扩散后的信号加回表征 | — | **判死**（场是骨干特征的摹本 ⇒ 冗余） |
| **B** | 约束 / 正则（进 loss） | 惩罚"不平滑"，塑形场参数 | 有效但只是**方差压缩器** | **判死**（切断后场仍近 DC） |
| **C** | 度量 / 优化器 | 用核给梯度做低通，改更新方向 | — | **判死**（`dir_ratio=0.9129` ⇒ 事实上裸 SAM） |
| **D** | **信用分配**（value / advantage） | 在邻近状态间重分配回报信号 | **★ 唯一有效的面** | 不适用 |
| **E** | 对齐目标（写场通路的监督） | 决定"场应该长成什么样" | — | 待测（`v119`） |
| **F** | 读数 / 收获（推理期聚合） | 改变读出的方式 | — | **判死**（视图族） |

完整推导见 [`docs/01-foundations.md`](docs/01-foundations.md)；逐分支证据见 [`docs/02-taxonomy.md`](docs/02-taxonomy.md)。

---

## 每分支最佳版本（速查）

### 强化学习线（PPO / Atari Breakout，5M 预算，seed 42）

| 分支 | 最佳版本 | 实测 |
|---|---|---|
| 原版基线（0 结构增强） | `rl/breakout/physics_guided_ppo_breakout_v4.py` | 峰值 **311.5**@4.52M |
| **α 释放调度（最强单轴）** | `..._v9.py` / `..._v10.py` | v9 峰值 **288.8**/Max **373**；v10 峰值 **295.6**/Max **404**（10M） |
| Advantage 平滑 | `..._v13.py` | final Avg R **247.9**，CV 0.063 |
| DAP 方向性优势传播 | `..._v16.py` | 末期1M **186.4**，峰值块 **207.8** |
| 熵淬火 + 回火 | `..._v21.py` | 末期1M **191.92**（首次 PASS） |
| **★ 物理损失（TAIL-only）= 全线 SOTA** | **`..._v26.py`** | **best 215.2@4.3M**、final-1M **190.88**、TAIL CV **4.34%**、0 触发、ovh **0.06%** |
| 多尺度谱（仅稳定性单项最优） | `..._v34.py` | TAIL CV **3.26%**（系列最优），但 best 201.7 |
| 反例：核函数"修复" | `..._v24.py` | 峰值 **91.9**（−57%）⇒ 核的"退化"是 load-bearing |
| 反例：容量升级 | `..._v30.py` | best **63.2**（−70.7%）⇒ 容量轴在冻结调度下不可行 |

### 神经网络线

| 子线 | 分支 | 最佳版本 | 实测 delivered |
|---|---|---|---|
| CIFAR-10 | 全族基线 | `nn/cifar10/cnn10_v51.py` | **97.56**（seed1234） |
| CIFAR-10 | 各向异性 Matérn 核 | `nn/cifar10/cnn10_v101.py` | 97.48 |
| CIFAR-10 | 参数侧度量族 | `nn/cifar10/cnn10_v110.py` | **97.50**（判死） |
| CIFAR-10 | 频带夹带 | `nn/cifar10/cnn10_v111.py` | 97.46（判死） |
| CIFAR-10 | 读数侧视图族 | `nn/cifar10/cnn10_v112.py` | 97.52（判死） |
| CIFAR-10 | 约束通道·pinn 作用面 | `nn/cifar10/cnn10_v117.py` | **97.59** |
| CIFAR-10 | 约束通道·smooth 作用面 | `nn/cifar10/cnn10_v118.py` | 97.53 |
| CIFAR-10 | 对齐目标族（**待跑**） | `nn/cifar10/cnn10_v119.py` | 未跑 |
| CIFAR-100 | 各向异性热核 | `nn/cifar100/vit100_v77.py` | 70.66 |
| CIFAR-100 | 交付路由（teacher_snapshot） | `nn/cifar100/vit100_v95.py` | 71.11 |
| CIFAR-100 | 尾窗平均（可归因最高） | `nn/cifar100/vit100_v110.py` | **75.34**（峰值指纹 74.03@e791） |
| CIFAR-100 | 尾窗剂量外推（**待跑**） | `nn/cifar100/vit100_v111.py` | 未跑 |
| InfoDiff | 信息图调制的扩散率 | `nn/cifar10_infodiff/cnn10_info_v02.py` | 97.12（轴判死） |

> ⚠ **口径警告**：`cnn10_v117` 的 97.59 与 `cnn10_v110` 的 97.50 **不同尺**（v117 存在执行层回归，
> 180ep 全程跑在非确定性算子下）。同尺口径下的最佳是 **v110 = 97.50**。详见 [`docs/04-nn-line.md`](docs/04-nn-line.md)。

---

## 怎么跑

每个脚本都是**零命令行参数**的直跑脚本（配方写死在顶部常量里，改配方 = 改代码）：

```bash
python rl/breakout/physics_guided_ppo_breakout_v26.py      # RL 线 SOTA
python nn/cifar10/cnn10_v119.py                            # NN 线最新（180ep, seed 42）
python nn/cifar100/vit100_v111.py                          # A 线最新（800ep, seed 42）
```

**发车前必须核对 sha1**（否则会"跑了旧版却以为改过了"——这条纪律是用一整次 180ep 训练换来的）：

```bash
grep "cnn10_v119.py" MANIFEST.md      # 拿 sha1 与行数
sha1sum nn/cifar10/cnn10_v119.py
wc -l   nn/cifar10/cnn10_v119.py
```

环境要求见 [`docs/07-reproducibility.md`](docs/07-reproducibility.md)。

---

## 引用与许可

- 许可：**MIT**（见 [`LICENSE`](LICENSE)）。
- 如果你要用这里的负面结论，请直接引用 [`docs/05-negative-results.md`](docs/05-negative-results.md) 中的编号，
  它们都绑定了具体的版本号与实测数字。
- 本仓库的读数纪律（配对判据、σ 底、max-of-N 口径、sha1 provenance）写在
  [`docs/06-methodology.md`](docs/06-methodology.md)，复用时请一并带上。

---

## 状态

| 线 | 状态 | 下一动作 |
|---|---|---|
| RL（Breakout） | **已归档 + 方向性否定**（2026-08-26 终止） | 不再重启 |
| NN·CIFAR-10 | **进行中**：`v119` 待跑（最后一张牌） | 按 `v119_refineface.verdict` 六分支走；若为 `NO_CONTENT`/`COPY_ATTRACTOR`/`GRAD_DEAD` ⇒ 物理场整条线归档 |
| NN·CIFAR-100 | **进行中**：`v111` 待跑 | 按 `v111_tail_dose2.attribution_note` 四分支走 |
| NN·InfoDiff | **已判死**（`INFOMAP_NO_LEVERAGE`） | 逐字回退 `cnn10_v51.py` |
