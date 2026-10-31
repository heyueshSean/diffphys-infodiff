# NN 线 · CIFAR-10 / CIFAR-100 / InfoDiff

三条子线**原理互鉴、代码独立**：不许搬实现、不许拿对方读数当证据。

## 机制一句话

场是网络里的一个**可学频域滤波层**：

```
    m(k) = ( 1 + λ_g·|k|²_g / ν_g )^{−s_g} ,    |k|²_g = kx² + a_g·ky²
```

四个可学参数 `s / λ / ν / a`（经 `soft_bound`），三个场 phys1/2/3 挂在 layer1/2/3 之后，
经 gate1/2/3 加性注入。参与损失的项：`w_phys·smooth` + `PINN_WEIGHT·pinn_reg` + `physics_reg`。
另有一条**写场通路** `refine_physics`（e31–100、独立优化器、冻结主干）。

## 文件

### CIFAR-10（B 线，ResNet-50，σ_diff = 0.169pp，红线 97.04）

| 文件 | 分支 | delivered |
|---|---|---|
| `cifar10/cnn10_v51.py` | 全族基线（seed1234） | **97.56** |
| `cifar10/cnn10_v101.py` | 各向异性 Matérn 核 | 97.48 |
| `cifar10/cnn10_v110.py` | 参数侧度量族（**判死**） | **97.50** ← 同尺最佳 |
| `cifar10/cnn10_v111.py` | 频带夹带（**判死**） | 97.46 |
| `cifar10/cnn10_v112.py` | 读数侧视图族（**判死**） | 97.52 |
| `cifar10/cnn10_v117.py` | 约束通道·pinn 作用面 | 97.59 ⚠ 不同尺 |
| `cifar10/cnn10_v118.py` | 约束通道·smooth 作用面 | 97.53 |
| `cifar10/cnn10_v119.py` | 对齐目标族（**待跑**，最后一张牌） | 未跑 |

> ⚠ `v117` 的 97.59 与 `v110` 的 97.50 **不同尺**：v117 的守卫壳 `finally` 无条件关掉了
> 确定性算子 ⇒ 180ep 全程跑在非确定性下，而 v110–v115 全为 True。
> **同尺口径下的最佳是 v110 = 97.50。**

### CIFAR-100（A 线，ViT-Small + 知识蒸馏，σ_seed = 0.71pp）

| 文件 | 分支 | delivered |
|---|---|---|
| `cifar100/vit100_v77.py` | 各向异性热核 | 70.66 |
| `cifar100/vit100_v95.py` | 交付路由（teacher_snapshot） | 71.11 |
| `cifar100/vit100_v103.py` | 跨盆地 SWA 隔离 | 75.88（**不可归因**） |
| `cifar100/vit100_v110.py` | 尾窗平均（可归因最高） | **75.34**（峰值指纹 74.03@e791） |
| `cifar100/vit100_v111.py` | 尾窗剂量外推（**待跑**） | 未跑 |

### InfoDiff（C 线，`du/dt = ∇·(D(x)∇u)`）

| 文件 | 分支 | delivered |
|---|---|---|
| `cifar10_infodiff/cnn10_info_v01.py` | 信息图调制的空间自适应扩散率 | 97.06（图没活，`i_std≈0.012`） |
| `cifar10_infodiff/cnn10_info_v02.py` | 同上 + 归一化修复（救活图） | **97.12**（**轴判死** `INFOMAP_NO_LEVERAGE`） |

C 线三层分解：**算子存在性 12.19pp ≫ 幅度 1.72pp ≫ 信息项 0.03pp（不可读）**。

## 跑

```bash
python nn/cifar10/cnn10_v119.py       # 180 epoch, seed 42
python nn/cifar100/vit100_v111.py     # 800 epoch, seed 42
```

发车前核对 sha1（见根目录 `MANIFEST.md`）。

## 结论速览

| 作用面 | 结果 |
|---|---|
| A 加性注入 | **判死**（场 = 骨干摹本 ⇒ 冗余） |
| B 约束正则 | **判死**（v117 / v118 双双 `CUT_NO_CONTENT`） |
| C 度量 / 优化器 | **判死**（v110 `dir_ratio = 0.9129` ⇒ 事实上裸 SAM） |
| E 对齐目标 | 待测（v119） |
| F 读数 / 收获 | **判死**（v112 视图族；ens 五次负贡献） |

详见 [`docs/04-nn-line.md`](../docs/04-nn-line.md)。
