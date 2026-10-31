# 07 · 复现指南：环境、指纹核对、发车命令

---

## 1. 环境

| 角色 | 配置 | 说明 |
|---|---|---|
| **训练机**（跑实验） | `/usr/bin/python`，PyTorch **1.10.2+cu102**，CUDA 10.2 | **禁升级**；只补 `torchvision == 0.11.3+cu102` |
| 开发机（写代码 / 静态检查） | venv，PyTorch 2.13.0+cpu | **本地验证不能替代训练机跑分** |

⚠ 版本敏感处：老 torch 下 `torch.abs(float)` 会 `TypeError`；`torch.fft` 命名在新版有变化；
`use_deterministic_algorithms` 的行为跨版本不同。**在训练机上核对过的版本才算数。**

Python 版本：3.6 兼容语法（不用 walrus / f-string 之外的 3.8+ 特性作为硬依赖）。

---

## 2. 发车前必做：核对 sha1 与行数

```bash
# 1) 从 MANIFEST 取期望值
grep "cnn10_v119.py" MANIFEST.md

# 2) 在训练机上核对
sha1sum nn/cifar10/cnn10_v119.py
wc -l   nn/cifar10/cnn10_v119.py
```

**为什么强制**：历史上有一次在训练机上跑的是**首版**而非最终片，
整轮 180 epoch 跑完才发现 ⇒ 那根轴白测一次。sha1 是唯一能在一秒内区分"改了"和"没改"的手段。

重新生成清单：

```bash
python tools/make_manifest.py
```

---

## 3. 运行命令（全部零参数）

| 线 | 命令 | 预算 |
|---|---|---|
| RL（SOTA） | `python rl/breakout/physics_guided_ppo_breakout_v26.py` | 5M 步 |
| RL（绝对回报最高） | `python rl/breakout/physics_guided_ppo_breakout_v10.py` | 10M 步 |
| NN·CIFAR-10（最新） | `python nn/cifar10/cnn10_v119.py` | 180 epoch，seed 42 |
| NN·CIFAR-10（同尺最佳） | `python nn/cifar10/cnn10_v110.py` | 180 epoch，seed 42 |
| NN·CIFAR-100（最新） | `python nn/cifar100/vit100_v111.py` | 800 epoch，seed 42 |
| NN·InfoDiff | `python nn/cifar10_infodiff/cnn10_info_v02.py` | 180 epoch |

**配方写死在文件顶部常量里**：没有 `--epochs` / `--lr` / `--physics` 之类开关。
**改配方 = 改代码。** 这是刻意的——避免出现"裸命令跑成了另一个实验"。

---

## 4. preflight 闸门 G1–G8（`tools/lint_train_script.py`）

这些是**静态**检查，抓的是 `py_compile` 抓不到、但会在跑到第 N 个 epoch 时才炸的 bug。

```bash
python tools/lint_train_script.py nn/cifar10/cnn10_v119.py
```

| 闸门 | 抓什么 | 真实代价 |
|---|---|---|
| **G1** | 调用点 kwargs 名 / 元数与函数签名不符 | 一整轮 800ep 跑完才 TypeError |
| **G2** | MAYBE-UNBOUND 局部读取（只在某个 if 分支里绑定） | 跑到分支才炸 |
| **G3** | `history[...]` 的键用了但从未初始化 | 收尾 `KeyError`，**终局 JSON 全丢** |
| **G4** | `torch.save` 没有写前查盘守卫 | 磁盘满 ⇒ 崩在 C++ 侧，**不可恢复** |
| **G5** | 读了没绑定的名字（运行时 `NameError`） | 模块级 use-before-def，整跑当场死 |
| **G6** | **模块级** use-before-def（顶层顺序） | 同上 |
| **G7** | 自检字段的字面量不可能出现在它检查的常量里（假绿） | 自检恒 PASS 但什么都没验 |
| **G8** | 模块级私有助手定义在**模块级 for 循环之后** | 定义被跳过 ⇒ `NameError` |

> G4 是**继承债**：父本若已有未守卫的 `torch.save`，子版本不新增即可（`子 ≤ 父`）。

---

## 5. 三层自证（发车前跑）

```bash
python tools/selfcheck/_deploy_check_cnn10_v119.py     # 部署级（静态），期望 56/56
python tools/selfcheck/_cnn10_v119_exec_selftest.py    # 执行级（动态），期望 55/55
python tools/selfcheck/_v119_launch_path_check.py       # 发车路径真跑，期望 ALL GREEN
```

（这三个脚本以 v119 为样板；其它版本同构。）

---

## 6. 读数顺序（跑完之后先看什么）

**NN·CIFAR-10（v119）**：

1. 横幅 `[v119] ★ 唯一一根轴 = refine_physics 的**对齐目标**…`
2. `[v119] 可运行性自检：ok=True | missing=[]`
3. `[v119] 对齐目标探针：ok=…` + `regime=… | loss_copy=… | loss_hp=…` + `hp_frac / |corr|` + `grad_max_hp`
4. `[v119] ★ 对齐目标族终局判词 = REFINE_HP_*`
5. JSON 块 `v119_refineface`：`verdict` / `regime` / `band_frac_mean` / `corr_l2|l3` / `grad_max_hp` / `delivered_acc`

**NN·CIFAR-100（v111）**：`TAIL_DOSE2_*` 判词 + 峰值指纹 **74.03@e791** 必须复现。

**RL**：看 100K 块均值，不要看单点终点（相邻 20K 采样可差 ±50%）。

---

## 7. 已知会出现的"非缺陷"现象

| 现象 | 说明 |
|---|---|
| `best_physics_resnet_v51.pth` 的 `FileNotFoundError` | 环境缺预训练权重；发车路径真跑脚本会报这个，**不算代码缺陷** |
| 指纹 MISMATCH 但代码零改动 | 梯度非位级确定 ⇒ 零改动跑必分叉（B 线 `σ_diff = 0.169` 的来源） |
| `horizon_advice = DO_NOT_EXTEND` | 已核算过：延长 horizon 的期望增益 ≈ 0.1σ |
| A 线 `v103` 的 75.88 | **不可归因**到该版本，且与 800ep 跨盆地口径不可比 |

---

## 8. 目录约定

- 权重 / checkpoint 落 `./ckpts_<VERSION>_s<SEED>/`，结果落 `./out_<VERSION>_s<SEED>/`。
- 结果 JSON 里带 `self_sha1` —— **provenance 自证**：跑出来的 JSON 自己记录它是哪个字节文件产的。
- 判完一版就归档并清理 `ckpts_vXX_s42/`（磁盘是真实约束：历史上因盘满丢过整轮结果）。
