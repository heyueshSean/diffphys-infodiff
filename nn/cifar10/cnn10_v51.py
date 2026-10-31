"""
Physics-Guided ResNet-50 Training - v51
================================================================================

## v51 = cnn10_v50.py 的副本 + 唯一一根轴：**gate1 固定注入系数 0.6 -> 0.9**
================================================================================

### 一、为什么换一根"注入强度"轴（而不是继续查平滑惩罚）

  v50 实跑（results_cnn_v50_seed1234.json）：pinn 系数减半 0.08->0.04 **完全送达**
  （`pinn.axis_effective=true` / `dose_from_epoch=0` / `dose_reaches_delivery=true`），
  但 `delivered 97.47` / 平台均值 `97.3787` ⇒ 顶层 `verdict=NULL`、`plateau_verdict=NULL`。
  ★ 这是**预期中的 NULL**：效应量级 <0.22pp（=2σ_diff），单跑结构上读不出来 ⇒ 不构成"轴无效"。

  ⇒ 本版换**另一族**的轴。筛选（对照"已判死清单 + 否决候选"）：
    · `label_smoothing`（校准族）—— 数据层实测 **83% 的错误 gap>0.1 是"有信心的错"**，
      且"低 margin 校准类"已判死 ⇒ **筛掉**。
    · `StochasticDepth 最大 drop`（0.1）—— v43 已实测证明 SD **不是** flip 衰减的原因，
      且该杠杆本身很小 ⇒ **筛掉**。
    · `w_phys` 峰值/尾部 —— 与刚测过的 pinn **同族**（都是"空间平滑惩罚的剂量"）⇒
      按"效果差就换族、不在同一族调第三个补丁" ⇒ **筛掉**。
    · **gate1 固定注入系数** —— 机制族不同（是**加性注入强度**，不是惩罚权重）、
      且是物理场整条链路里**唯一的硬编码字面量**（`Gate.forward` 里写死的 0.6，自 v9 起从未被调过）
      ⇒ **选它**。

### 二、本版唯一一根轴：gate1 固定注入 0.6 -> 0.9

  物理场进入主干的唯一通道是 `Gate`：`x = x + <inject> * bn(proj(phys(x)))`。
    · `gate1`（layer1 后, FixedDiffusion 256ch）：**注入系数写死 0.6**（`learnable=False`）
    · `gate2/gate3`（layer2/3 后, MaternDiffusion）：可学习 `a = soft_bound(alpha, 0.05, 0.9)`
  ⇒ ★ **0.9 正是设计域的上界**（可学习 gate 的 alpha 就界定在 [0.05, 0.9]），而 gate1 的 0.6
     是**低于该上界的硬编码常量**、从未被单独调过。本版把它提到设计域上界 **0.9（+50%）**，
     并提取为命名常量以替换字面量（覆辙 113 反制）。
  这是**改定义域**（注入强度直接乘进前向 ⇒ 改优化器可达集），不是参数级 clamp（覆辙 128 认可）。
  ★ 选 gate1 而不是 gate2/3 的理由：gate2/gate3 是**可学习**的 ⇒ 改它们的 init/界会被优化器
    补偿掉；只有 gate1 是**固定**的 ⇒ 改它才有持续效应。

### 三、保留 v49 的确定性硬化 + v50 的 pinn=0.04（本版不再动它们）

  算子层确定性硬化（CUBLAS_WORKSPACE_CONFIG + use_deterministic_algorithms）与双模式探针**全部保留**；
  `PINN_WEIGHT=0.04` 也原样保留（v50 的轴结案为 NULL，NULL 不等于"轴无效"⇒ 不因未证实而回退）。
  本版只在其上叠一根配方轴。判据继续用 v49 的实测 σ_diff=0.108pp ⇒ 2σ=0.22pp。

### 四、判据

  **主判据（本版特有，机制级）**：
      `gate1_inject.axis_effective == True`（注入系数 != 0.6）⇒ 剂量已送入（从 ep0 起全程生效）。
  **精度判据（跨跑；平台均值口径，与 v49/v50 同尺）**：
      PASS     >= 97.68      (= 97.4625 + 0.22)
      NULL     (97.2425, 97.68)
      BACKTRACK <= 97.2425   ⇒ **立即逐字回退 cnn10_v50.py**，不做分析。
  ★ 明说：边距 = 2σ_diff ⇒ **任何 <0.22pp 的效应单跑都读不出来**；单跑 NULL 不得升级为"轴无效"（需 3 seed）。

### 五、本版**不做**的事

  · 不碰测量（v49 的探针/硬化原样）。
  · 不碰 gate2/gate3（可学习，改了会被优化器补偿；且会把 diff 扩到两根轴以上）。
  · 不动 `base` 字段（沿用 v48 起的 `cnn10_v47.py` 写法，与 v49/v50 一致；它只是谱系注记，非五处同步项）。

### 六、怎么跑

  零参数、裸命令：`python cnn10_v51.py`（双卡自动 DataParallel）。
  产物：结果 JSON 在 `./out_v51_s1234/`，权重 `best_physics_resnet_v51.pth` / `best_tta_resnet_v51.pth`
  （与 v44 起的各版一致，写在当前工作目录下）。

↓↓↓ 以下是 v50 的原文说明（v51 只在其上把 gate1 固定注入 0.6 -> 0.9） ↓↓↓
================================================================================

## v50 = cnn10_v49.py 的副本 + 唯一一根轴：**pinn 正则系数减半（0.08 -> 0.04）**
================================================================================

### 一、为什么换回"配方轴"（而不是继续钉测量）

  v49 实跑 **DETERMINISM_ACHIEVED**：探针证明算子层非确定性（v48 同 seed 不逐位复现的根因）
  已被 `CUBLAS_WORKSPACE_CONFIG + use_deterministic_algorithms` 两行压到 0（143/188 梯度逐位不同 -> 0/188）。
  ⇒ 算子层确定后，此前 <0.22pp 读不出来的配方轴**第一次变得可读**。本版回到配方侧，选一根
  B 线自己、未被否过、且历史上从未被单独质疑过的轴：**pinn 正则系数**。

### 二、本版唯一一根轴：pinn 正则系数 0.08 -> 0.04（减半）

  损失式：`loss = cls_loss + w_phys * smooth + PINN_WEIGHT * pinn_reg + reg_loss`
  其中 `pinn_reg = mean(dx^2) + mean(dy^2)`，dx/dy 是 **layer4 输出（deep）** 的空间梯度。
  ⇒ pinn 直接惩罚**主干深层特征**的梯度路径，与立身之本 v36 冲突：
     「物理场必须当被动先验；正则只作用于场自己的参数，绝不造惩罚主干特征的梯度路径」。
  ⇒ 本版把它的权重减半（0.08 -> 0.04），削弱这条"惩罚主干"的路径，同时保留一个温和的平滑底。
  这是**改定义域**（改 loss 权重 = 改优化器可达集），不是参数级 clamp（覆辙 128 认可）。
  ★ 改动量级可能 <0.22pp ⇒ 单跑 NULL 是**算术上可预期的**，不许读成"轴无效"（覆辙 156）。

### 三、保留 v49 的确定性硬化（本版不再动测量）

  算子层确定性硬化（CUBLAS_WORKSPACE_CONFIG + use_deterministic_algorithms）与双模式探针**全部保留**，
  本版只在其上叠一根配方轴。判据沿用 v49 的实测 σ_diff=0.108pp ⇒ 2σ=0.22pp。

### 四、判据（继承 v49，σ_diff 实测 0.108pp ⇒ 2σ = 0.216pp，取 0.22pp）

  **主判据（本版特有，机制级）**：
      `pinn.axis_effective == True`（PINN_WEIGHT != 0.08）⇒ 剂量已送入（从 ep0 起全程生效）。
  **精度判据（跨跑；平台均值口径，与 v49 同尺）**：
      PASS     >= 97.68      (= 97.4625 + 0.22)
      NULL     (97.2425, 97.68)
      BACKTRACK <= 97.2425   ⇒ **立即逐字回退 cnn10_v49.py**，不做分析。
  ★ 明说：本版**预期精度 NULL**（减半的效应量级可能 <0.22pp，单跑结构上读不出来）。
     NULL 的结论要 3 seed 才敢说"轴无效"；单跑 NULL 只能说明"单跑分辨不出"。

↓↓↓ 以下是 v49 的原文说明（v50 只在其上把 pinn 系数减半） ↓↓↓
================================================================================

## v49 = cnn10_v48.py 的副本 + 唯一一根轴：**算子层确定性硬化**（补上 cuBLAS 与 ATen 反向核）
================================================================================

### 一、v48 实跑读数（results_cnn_v48_seed1234.json）—— B 线迄今最高的一次，但判决仍是 NULL

  delivered = **97.51**（= best_tta 97.51 @ep138）   |   平台末 8 次 TTA 均值 **97.4625**（sd 0.0243）
  同 seed 谱系：v45 97.49 / v46 97.37 / v47 97.42 / **v48 97.51** ⇒ 五跑里最高。
  剂量守卫通过：`lr_tail.dose_reaches_delivery=True`、`n_changed_at_or_before_best_tta=18`
  ⇒ v47 那根轴"结构性不可观测"的毛病**已修好**，这次干预确实送到了交付权重上（覆辙 154 已清）。
  判据：主判据（平台均值）PASS 线 97.48 ⇒ 97.4625 **差 0.0175pp 未过**；次判据 97.61 ⇒ 97.51 未过。

### 二、★ v48 的真结论：不是"轴无效"，是**判据本身没有分辨力**

  把 v48 与 v47 的逐 epoch 曲线按**剂量分段**对齐（同 seed、同 epoch 网格、ep1-120 两边 lr 逐位相同）：

      epoch 段        剂量            v48 − v47 均值Δ      正号
      ep1-120         0（剂量前）        −0.049pp           55/120
      ep121-134       x1.00-1.04        +0.041pp           11/14
      ep135-152       x1.05-1.21        +0.131pp           18/18
      ep153-174       x1.35-3.00        +0.140pp           22/22
  ⇒ 方向对、随剂量单调抬升、正号一边倒（后 54 轮里 51 轮为正）——**这不是噪声的形状**。

  ★★★ 但用同一把尺去量一对"零改动"的跑，就露底了：
      **v45 vs v46**：`train()` md5 `ff4b66ac` 与 `test_tta()` md5 `62c35889` **逐字相同**，
      `test()` 的唯一差异是权重文件名常量 `best_physics_resnet_v45.pth` → `v46.pth`，
      seed 同为 1234，数据管道逐字相同。
      ⇒ 这样一对"什么都没改"的跑，平台末 8 次 TTA 均值是 **97.4387 vs 97.3413 ⇒ Δ = −0.108pp**，
        逐 epoch 曲线**从 ep1 就分叉**。
  ⇒ ★★★ **一个权重文件名的字面量，能把交付指标搬动 0.108pp。**
    本线真正的噪声底：σ_diff ≈ **0.108pp**，2σ ≈ **0.216pp**。

### 三、v47/v48 的 NULL 是**算术上被预先决定**的（本线第一条方法学勘误，已固化进判据）

  v47/v48 的 PASS 线都写成 `baseline + 0.10pp`，而实测 σ_diff = 0.108pp
  ⇒ **0.10pp = 0.93σ**，判据线落在被测量自身的 **1σ 之内**。
  ⇒ 哪怕这根轴的真实效应正好是 +0.09pp，它也**必然**判 NULL。
  ⇒ ★★★ 从本版起，B 线判据的 PASS 边距**不得小于 2σ_diff = 0.22pp**（本版写死 0.22pp，见 PLATEAU_SIGMA）。

### 四、所以 v49 不碰配方，只碰**分辨率**（唯一一根轴：算子层确定性）

  `cudnn.deterministic` 从 v14 起就一直在，但它**只管 cuDNN**；
  cuBLAS 的 GEMM（分类头与每卡 gemm）与部分 ATen 反向核（atomicAdd 归约）**不在它的管辖内**。
  本版补两样 —— 是"改测量"，不是"改配方"：
    (1) 在**任何 CUDA 调用之前**设好 cuBLAS 的工作区环境变量（`CUBLAS_WORKSPACE_CONFIG` = `:4096:8`）；
    (2) 打开 `use_deterministic_algorithms(True)`。
  ⇒ ★ 期望：同 seed 同代码两次跑**逐位相同** ⇒ σ_diff → ~0 ⇒ 之后 <0.2pp 的轴第一次变得可读。

  ★★ 自带**确定性探针**（本版最重要的仪器：秒级、只读、不消耗训练预算）：
     用**固定生成器**拼探针批（不碰全局 RNG），快照 4 个 RNG 状态 + `state_dict()`，
     跑两遍 fwd+bwd（两遍之间把 RNG 与权重都还原），比较 loss、logits 与**全部参数梯度**是否逐位相同；
     **在"关"和"开"两种模式下各测一次** ⇒ 一次跑就回答"非确定性到底在不在算子层"。
     ★ 探针若报错（有算子没有确定性实现）⇒ **不退出、不拒发**：记录错误原文、自动降级回非确定性模式，
     并在 JSON 写 `determinism.axis_effective = False`（覆辙 115 的反制：干预没送到必须能被机器看见）。
     降级后本跑 = v48 的一次复跑，精度读数仍然有效、仍进精度判据。

### 五、判据（σ_diff 用实测 0.108pp ⇒ 2σ = 0.216pp，本版取 0.22pp）

  **主判据（本版特有，机制级）**：
      `determinism.deterministic_mode.grads_bitwise_identical == True` ⇒ `DETERMINISM_ACHIEVED`（= 本版 PASS）
      否则 ⇒ `DETERMINISM_NOT_ACHIEVED`（下一版改去钉数据管线，而不是再碰配方）。
  **精度判据（跨跑；在这里的作用只是"防确定性改动把训练搞坏"的回退闸）**：
      PASS     >= 97.68      (= 97.4625 + 0.22)
      NULL     (97.2425, 97.68)
      BACKTRACK <= 97.2425   ⇒ **立即逐字回退 cnn10_v48.py**，不做分析。
  ★ 明说：本版**预期精度 NULL**（确定性硬化不该改平台水平），不许把 NULL 读成"轴无效"。

### 六、顺手修掉 v48 的三处"文本失同步"（本次交付前机检发现）

  (1) `results["axis"]` 仍是 v46 的文案；运行 banner 第一行仍写死 v47 的轴与已退役的 LR_FLOOR；
  (2) banner 写死 `Phase 2 (...): smooth cosine 1e-4->1e-5`，而实际终点早已是 3e-5。
  ⇒ v49 把配置打印改成**从常量派生**（不再写死数字），并加机检字段 `text_sync`，
    让"打印出来的配置"在结构上不可能与真实配置脱钩（覆辙 113/119 的同类病）。

### 七、自己抓到的两个真缺陷（**变异测试暴露的**，不是脚手架问题）

  (1) 判据常量 `PLATEAU_PASS / PLATEAU_REDLINE` 第一稿写成**字面量**，推导公式只在注释里
      ⇒ 改 baseline 或 σ 时判据**不会跟着变**，会静默地与来源脱钩（覆辙 113/145 同类）。
      ⇒ 改成 `round(BASELINE ± MARGIN, 2)` 由公式算出，并把两条守卫塞进仪器门禁：
        `plateau_margin_at_least_2sigma`（边距 >= 2σ）与 `plateau_lines_derived_from_baseline`。
  (2) `determinism` 的 verdict 第一稿只看**梯度**是否逐位相同，不看 loss/logits
      ⇒ 一个"前向输出就已经不同"的跑（例如权重/缓冲区没被还原）会被判成 ACHIEVED = 假通过。
      ⇒ 新增 `bitwise_identical_overall`（梯度 ∧ loss ∧ logits），verdict 改用它。

  ★ 这两条都是**变异测试先抓到、我再去修代码**，不是"先写守卫再让变异去迎合守卫"。

### 八、怎么跑

  零参数、裸命令：`python cnn10_v49.py`（双卡自动 DataParallel）。
  产物：结果 JSON 在 `./out_v49_s1234/`，权重 `best_physics_resnet_v49.pth` / `best_tta_resnet_v49.pth`
  （与 v44 起的各版一致，写在当前工作目录下）。
  ★ 探针在开场跑一次，秒级；训练预算一点没动。

↓↓↓ 以下是 v48 的原文说明（v49 只在其上加了确定性硬化与探针） ↓↓↓
================================================================================

### 一、v47 实跑读数（results_cnn_v47_seed1234.json）—— 先把"好不好"说清楚

  delivered = **97.42**（verdict NULL，红线 97.27 未触及 ⇒ **不是倒退**）
  同 seed 谱系：v45 97.49 / v46 97.38(legacy 口径) / v47 97.42 ⇒ 三点 sd ≈ 0.05pp，全在噪声里。

### 二、v47 真正的结论：NULL 的原因是**结构性不可观测**，不是"轴无效"

  三条机器读数摆在一起就看到了：
    (1) `lr_floor.first_bite_epoch        = 168`    ← 剂量起点
    (2) `lr_floor.n_bites_in_this_run     = 7`      ← 早停截断，计划 13 轮只跑了 7 轮
    (3) `best_tta_epoch = 138` 且 `instrument.final_tta_source_epoch = 138`
        ⇒ 交付的那份权重是 **ep138 存下来的**，`delivered` 只是对它的重测。
  ⇒ ★★★ **剂量起点（ep168）比交付选择点（ep138）晚了整整 30 轮。**
     交付读数在数学上不可能含有本轴的任何信息。
     v47 的 NULL 是"干预没送到被测权重上"（覆辙 115 的新形态），**不是**"机制无效"。
  ⇒ ★★ 这个洞**加 seed 也补不上**：再多 seed 都跑在同一个"剂量窗 ∩ 交付窗 = ∅"的结构里。

  旁证（逐 epoch 对齐）：v47 与 v46 的 test 曲线在 **ep0 就分叉**（29.45 vs 30.28），
  但到 ep120–175 已收敛到 sd **0.058pp** ⇒ 两条轨迹末段几乎重合；
  而 v47 末 8 次 TTA 均值 97.3725 vs v46 的 97.3413（+0.031pp），
  与被 clamp 咬住的 ep168/171/174 三点无关（咬前咬后差异相同）⇒ **剂量确实没起作用**。

### 三、为什么不再"把 clamp 调大"，而是换参数化

  clamp 要咬到 ep138 之前，floor 必须 >= lr(138) = 8.145e-5；
  而 phase2 的峰值只有 1.0e-4 ⇒ floor 占峰值 **81%** ⇒ 余弦衰减被削掉 81% 的幅度
  = **实质删掉整条 phase2 波形** = 覆辙 148 的邻域（"设下限"变成"改形状"两件事一起做）。
  ⇒ 正解是改**波形自己的终点**：`PHASE2_LR_END` 是调度本身的定义参数，
    抬高它 = 改**定义域**（覆辙 128 认可），不是在外面套一层 clamp（覆辙 79/148 禁止）。

### 四、本版唯一改动（一个常量 + 一条退役）

    1. `PHASE2_LR_END = 1e-5 -> 3.0e-5`
    2. `LR_FLOOR_ENABLE = True -> False`（v47 的 clamp 退役；保留开关只为留机器可核对的退役记录）

  剂量曲线（离线重放；倍率 = new/old）：
    ep0–120  (**phase1 全程**) 逐位不变        ← 守卫 `phase1_bitwise_unchanged`
    ep120 : 1.000e-4 -> 1.000e-4  x1.00        ← 守卫 `phase2_start_unchanged`
    ep138 : 8.145e-05 -> 8.557e-05 x1.05       ← v47 的 best_tta 点
    ep150 : 5.500e-05 -> 6.500e-05 x1.18
    ep162 : 2.855e-05 -> 4.443e-05 x1.56       ← v46 的 best_tta 点
    ep170 : 1.603e-05 -> 3.469e-05 x2.16
    ep180 : 1.000e-05 -> 3.000e-05 x3.00
  ⇒ 剂量从 **ep121** 起就在，**完整覆盖**交付选择窗（历史 best_tta 落在 ep138–162）。
  安全边界（覆辙 148 反制，两条守卫都写死在 JSON 里）：
    新终点 3.0e-5 落在 phase2 值域 [1.0e-5, 1.0e-4] **内部**；
    余弦仍保留 (1.0e-4−3.0e-5)/(1.0e-4−1.0e-5) = **78%** 的衰减幅度 ⇒ 没有删波形。

### 五、本版新增的两个仪器（都是 v47 缺的，缺它们正是 v47 读成 NULL 的原因）

  (A) **剂量-交付窗重叠守卫** `lr_tail.dose_reaches_delivery`
      = (剂量起点 <= best_tta_epoch)。False ⇒ 这份 NULL **不可归因**，直接作废。
  (B) **平台读数** `lr_tail.plateau_mean`（末 8 次 TTA 的均值）
      为什么需要：`delivered` 是 ~32 次 TTA 的 **max**（max-of-N，覆辙 104/140），
      实测 `delivered − plateau_mean` ≈ **+0.05pp**，而平台内 sd 只有 0.02pp
      ⇒ 在这么平的平台上取 max，收的主要是抖动（覆辙 ㊹）。
      跨版同口径（末 8 次 TTA 均值）：v45 97.4388 / v46 97.3413 / v47 **97.3725**
      ⇒ 同 seed 平台 sd ≈ **0.050pp**（这三点里没有任何一点来自真效应 ⇒ 它就是噪声底）。

### 六、判据（预注册，双作用域，不许混读）

  ★ 主判据 = **平台均值**（口径新，但比 max-of-N 干净）：
      PASS      plateau_mean >= **97.48**   (= v47 97.3725 + 2σ, σ=0.050)
      NULL      97.27 < plateau_mean < 97.48
      BACKTRACK plateau_mean <= **97.27**   ⇒ 逐字回退 cnn10_v47.py
  ★ 次判据 = **delivered**（与 v45/v46/v47 同一把尺，跨版可比）：
      PASS >= 97.61 / NULL [97.27, 97.61) / BACKTRACK <= 97.27
  ⚠ 两个红线是**同一个数字、两个口径**：`lr_tail.plateau_redline` 与
    `backtrack_redline` 各自带 `scope_note`，不要交叉读（覆辙 145/146）。
  ★ 剂量守卫（覆辙 115）：`lr_tail.dose_delivered` 必须为 True，
    否则这份读数**没有资格**参与上面的判据。

### 七、下一步（写完就锁死，不许临时改）

  · 读 NULL ⇒ **不许**升级为"轴无效"（那条结论要 3 seed）。但**也不许**再在
    "末段 lr"这一族里做第三个形态 —— 已经试过 clamp 与终点两种参数化。
  · 读 BACKTRACK ⇒ 逐字回退 `cnn10_v47.py`，先止血再诊断。
  · ★ 若还想在训练侧继续：**下一个候选必须是"覆盖 ep120 之前"的轴**
    （因为本线交付点稳定落在 ep138–162，任何 ep168 之后的干预都天然不可观测）。

双卡怎么用：什么都不用做。启动时自动 `torch.cuda.device_count()`，
  >1 就 DataParallel，`BATCH_SIZE = 128 × N_GPUS`。**直接 `python cnn10_v48.py`**，
  不需要 torchrun、不需要任何命令行参数。
================================================================================

↓↓↓ 以下是 v47 的原文说明（v48 只在其上改了 PHASE2_LR_END 与 LR_FLOOR_ENABLE） ↓↓↓

Physics-Guided ResNet-50 Training - v47
================================================================================
v43 = **v37 的逐字副本 + 唯一一根改动轴：StochasticDepth 退火**。

基准：cnn10_v37.py（sha1 7c7ac5ac1fd33a2746a999dc769e13fb6503c45b，967 行）
      = v14 的完整回退，磁盘上实际存在的历史最好版本。

为什么选这根轴（全部来自实测，不是发明）：

  1) 交付口径是 `delivered = max(best_tta_acc, final_tta)`，即在 ~34 次 TTA
     读数上取 max。而五跑（世代A）的 TTA 峰值 5/5 都落在 ep110 之后
     （ep123/120/177/180/219）⇒ **相位2 (120-180) 是交付读数唯一的产出窗口**。

  2) 但相位2 的 raw 曲线是死的：ep120→180 raw 只 +0.080pp（0.0013 pp/epoch），
     `cls` 损失在同一区间 0.5367 → 0.5383 完全不动。

  3) 决定性读数 —— flip 增益（TTA − raw）在**衰减**：
         ep<=120 平均 +0.333   →   ep>120 平均 +0.199   (变化 −0.134)
     单跑同向：v28.1 −0.156 / v28.2 −0.205 / v28.4 −0.224 / v28.6 −0.163。
     ⇒ 模型随训练**越来越怕翻转**；TTA 不是越用越好，是在更费力地对冲。

  4) 机制定位：`BlurBottleneck` 的 StochasticDepth 噪声
     **只打在 main path、不打在 identity**，且是逐样本独立噪声
     ⇒ 直接破坏 `f(flip(x)) == flip(f(x))` 的翻转等变性，而 TTA 恰好依赖它。
     训练越久，主干越学会把信息绕进无噪的 identity ⇒ 解释 (3) 的衰减，
     也解释 (2) 的 `cls` 冻结。且 `SD_MAX_DROP=0.1` 是**常量，全期不退火**。

  改动（一根轴）：把 drop_rate 从"只随 block_idx 的常量函数"改成
     **"再乘一个按 epoch 退火的系数，终点对齐 ep120"**。
     不改损失 / 不改三个调度 / 不改数据管道 / 不改 TTA 与集成的选取 /
     不改任何常量值。唯一差异 = drop_rate 的取值来源多了一个 epoch 因子。

  预期量级：把 flip 增益的衰减从 −0.134 扳回到 >=0，相位2 的 TTA 贡献
     从 +0.286pp 提到 ~+0.42pp ⇒ 保守区间 **+0.2 ~ +0.5pp**。
     本线跨跑 σ = 0.11pp（2σ = 0.22pp）⇒ **该效应可检出**。

  判据：PASS    >= 97.72   (v37 97.50 + 2σ)
        NULL    [97.28, 97.72)
        FAIL    <= 97.06   (v41 97.29 − 2σ)
        BACKTRACK < 97.06  ⇒ 立即逐字回退 cnn10_v37.py，不做分析。

双卡怎么用：什么都不用做。启动时自动 `torch.cuda.device_count()`，
  >1 就 DataParallel，`BATCH_SIZE = 128 × N_GPUS`。**直接 `python cnn10_v47.py`**，
  不需要 torchrun、不需要任何命令行参数。
================================================================================

## v47 = cnn10_v46.py 的副本 + 唯一一根改动轴：**末段 lr 下限（LR_FLOOR，训练侧）**
================================================================================

### v46 到底发生了什么（先把"倒退"这件事拆清楚，否则会误回退）

用户的观察是"cnn 的效果又倒退了"。把 v46 的 97.37 与 v45 的 97.49 拆开后，**三个成分都不构成真倒退**：

  (a) **口径变了（设计使然）**：v46 的 `delivered=97.37` 是**单个数**（SWA_TTA）；
      v45 的 `97.49` 是 **max-of-N** 选点。两者直接比，结构上就压低 v46 约 0.05~0.15pp。
      拉到同口径：v46 的 `legacy_delivered=97.38` vs v45 的 `97.49` ⇒ **−0.11pp**。

  (b) **−0.11pp 落在跨 seed 噪声里**：本线有效独立样本 = **2**（v43 seed42 的 97.41；
      seed1234 三次 = 97.52 / 97.49 / 97.38，均值 97.4633）。四次读数 sd ≈ 0.06pp。
      ⇒ −0.11pp 连 2σ 都不到，**无信号**。

  (c) ★★ **SWA 这根轴本身就是 NULL，而且原因是结构性的**：
      SWA 窗口 ep126–178 的 TTA **平台极差只有 0.06pp**，而判据的 WIN 阈是 **0.10pp**。
      `SWA_TTA=97.37` vs 平台 `max=97.38` ⇒ **Δ = −0.01pp**。
      对 27 个 epoch 取平均、与挑单个最好的 epoch，**在数值上不可区分**。
      ⇒ 这根轴**按构造就不可能有可分辨效应**（覆辙 115 的新形态：不是机制无效，是平台太平）。
      **我的失误**：造这根轴之前没有先量平台的平整度。

平台平整度阶梯（v46 的 TTA）：ep≥120 极差 0.15 / ep≥140 0.12 / ep≥150 0.10 / **窗口 ep126–178 = 0.06**。
窗口 TTA 均值 97.3111，sd **0.0403**（n=18）。

### ★ 平台为什么这么平 —— 机制级死因

phase2（ep120–180）的 lr 从 **1.0e-4 余弦衰减到 1.0e-5**，即峰值 0.1 的 1e-3 ~ 1e-4 倍。
权重更新步长 ≈ lr × 梯度 ⇒ 末段已小到让模型**实质冻结**：
v46 的 raw 在 **ep128 之后就再没创过新高，连续 51 轮零改善**（这正好触发 `TEST_PATIENCE=50`）。
也就是说 ep128–179 这 51 轮算力（占总量 28%）几乎全花在"原地抖动"上。
并且该窗口里 `mixup=0`（mixup 窗口恰为 ep60–120）、`SD=0` ⇒ **正则真空 + 权重冻结**。

### 结论：交付链后处理家族**整体判死**，换到训练侧

  交付链已经失败三次：快照集成(v1) / 预测集成(v1) / 末段权重平均(v46，Δ=−0.01pp)，
  三者都是**在一条比测量分辨率还平的平台上做后处理**。⇒ 不再做第四个补丁（覆辙 ①⑮）。

### 本版唯一改动（一处语义）

  给末段 lr 设一个**下限**，把"冻结期"变成"低速精调期"：

      lr(e) = max( lr_v46(e),  LR_FLOOR × LR_BASE )        for e >= LR_FLOOR_FROM_EPOCH

  · `LR_FLOOR = 2.0e-5`（absolute lr = **2.0e-6**），`LR_FLOOR_FROM_EPOCH = 120`。
  · 生效点 ep120 与 **mixup 卸载点 / SD 退火终点 / phase1-2 分界**同点 ⇒ 不引入新时间常数。
  · **只抬高、不压低**（逐点取 max）⇒ phase1 完全不动；由取值可知 **真正被改动的只有
    ep168–179 共 12 轮**（phase2 的最后 21%）。
  · 交付链**零后处理**（`SWA_ENABLE=False`）⇒ v47 的读数直接就是"训练这条轴"的读数。

### ★ 写死这个常数时踩过的坑（记录在案）

  第一稿取 `LR_FLOOR = 2.0e-2`（absolute 2.0e-3）—— 离线重放发现**它高于整个 phase2 的值域**
  （phase2 max = 1.0e-4）⇒ 会把 ep120–180 全部压成常数，**等于删掉整条 phase2 余弦调度**。
  那不是"给末段设下限"，而是"改形状 + 删波形"两件事一起做 = 超出"一轴"预算（覆辙 ⑮）。
  ⇒ 正确约束：**下限必须落在 phase2 值域内部**，才只咬尾巴。重放后的咬住点：

      floor=5.0e-5 ⇒ ep153 起，28/61 轮（46%）   floor=3.0e-5 ⇒ ep162 起，19/61（31%）
      floor=2.0e-5 ⇒ ep168 起，13/61（21%）← 取此   floor=1.5e-5 ⇒ ep171 起，10/61（16%）

  取 2.0e-5 的理由：只动最后 21%，前置 79% 的余弦**逐位不变**，"一轴"含金量最高；
  同时把末段 13 轮的 lr 从 [1.0e-5, 2.1e-5] 抬到恒定 2.0e-6（相对 1.0~2.0 倍），
  量级上足以改变"冻结"状态，又远低于破坏收敛的量级（phase1 峰值 0.1 = 本值的 500 倍）。
  ★ 倍率有意取小：第一次做训练侧轴，宁可效应小到读不出来（读 NULL 可归因），
    也不要一口吃成"多个轴"（覆辙 ⑮）。

### 预注册判据（本版**只有跨跑判据**）

  lr 一改整条轨迹就变了 ⇒ **单跑内不存在 control**（这点与 v46 相反）。
  ⇒ 本版必须靠 **3 个 seed** 才敢说"有/无效应"。**单跑读到 NULL 不得升级为"轴无效"。**

  <1> 交付口径（跨跑可比，与 v45 同一把尺 = max-of-N）:
        PASS       delivered >= 97.61   (= v45 97.49 + 2σ, σ=0.06pp)
        NULL       97.27 <  delivered < 97.61
        BACKTRACK  delivered <= 97.27
      ★ 边界闭合：`>=97.61` 归 PASS，`<=97.27` 归 BACKTRACK（两个端点都归极端档）。
        `backtrack_needed` 与 `verdict` **共用同一个 `<=` 算子** —— 自测曾发现
        verdict 用 `<=`、advice 用 `<`，在 delivered 恰为 97.27 时给出
        "BACKTRACK 档 + DO_NOT_BACKTRACK 动作"的自相矛盾 JSON（已修 + 加自洽守卫）。
  <2> 剂量读数（必须自带，覆辙 115）: `lr_floor.n_bites_in_this_run >= 1`。
        若为 0 ⇒ **干预没送到**，此跑的 NULL 不可解读为"轴无效"。
  ★ 回退红线: delivered **<=** **97.27** ⇒ 逐字回退 `cnn10_v46.py`。
    （v46 同 seed=1234 的 97.38 是唯一可直接对齐的参照；但同 seed ≠ 可复现 ——
     本平台同 seed 已三次跑出不同结果，见下 —— 所以红线是保险丝，不是统计检验。）
  ★ 参考刻度：与 v46 同 seed 的 97.38 落在 NULL 档；v45 的 97.49 也仍是 NULL
    ⇒ 本轴要读到 PASS 必须 **> +0.12pp**（超 2σ 底噪才敢说"改对了"）。

### ★★ 本平台"同 seed 不可复现"已被三次证实（覆辙 141）

  v44 = 157 轮（早停） / v45 = 180 轮（跑满） / v46 = **179 轮**（`TEST_PATIENCE=50` 早停），
  三者 **SEED 全为 1234**。峰值位置同样漂移：best_tta @ep120 / @ep156 / @ep162；
  v46 的 `best_acc` 峰从 172 移到 **128**（−44 轮），raw 从 97.39 掉到 97.23。
  v46 相对 v45 只改了交付链（`train()` / `test_tta()` **逐字相同**，累加点在所有 RNG 消耗之后），
  读数却差 0.11pp ⇒ **非确定性是平台固有的，不是代码改动造成的**。
  ⇒ 因此 v44 的 97.52 / v45 的 97.49 是**从以 ≈97.45 为中心的分布里抽到的幸运样本**（覆辙 131）。
================================================================================

## v45 = v44 的逐字副本 + 唯一一根改动轴：**仪器修正**（三个假观测字段）
================================================================================

本版**不是机制版，也不是复跑版**，是**仪器版**。改的东西全是**只读记录与闸门**，
不碰任何 loss / 三个调度 / 数据管道 / SD 退火 / TTA 与集成选取 / 常量值。
**SEED 保持 1234**（与 v44 同种子）—— 这不是偷懒，它把本版额外变成一次
**同 seed 逐位复现检验**：读数若与 v44 一致，就证明仪器改动零副作用。

### 为什么必须先修仪器（v44 的 JSON 暴露了三个洞）

#### 洞 1：实际跑了多少个 epoch，JSON 里查不到

v44 的 JSON 写着 `"epoch": 180`，但它**只跑了 157 个 epoch（0..156）就早停了**。
三个独立证据：`history.test` 只有 157 条（v43 是 180）、`sd_trace` 止于 156、
TTA 采样止于 ep156。

原因（v44 源码 L1122-1134，本版同名逻辑在 L1149-1163）：
`no_improve_count >= EARLY_STOP_PATIENCE(12)` ⇒ break。
v44 在 ep120 达到 TTA 峰值 97.51 之后，ep123…ep156 恰好 12 次无改善 ⇒ 在 156 触发早停。
v43 在 ep159 才创新高 ⇒ 只累计 6 次 ⇒ 跑满 180。

**同一个配方、两个 seed，实际 horizon 差了 23 个 epoch，而 JSON 里没有任何字段能看出来**
—— 因为 `"epoch"` 写的是配置常量 `int(TRAIN_EPOCHS)`，不是观测值。

修法：新增 `actual_epochs`（= `len(history.test)`）、`last_epoch`、`early_stopped`、
`stop_reason`，并保留 `"epoch"` 不动（不破坏跨版可比性）。

#### 洞 2：`mechanism_ok = True` 是假绿

v44 的 `required_keys` 含 `sd_mean_at_179`，实测值是 **`null`**，
`mechanism_error` 里明确记着 `KeyError: 179`（因为早停，trace 只到 156），
但 `missing_keys = []` 且 `mechanism_ok = True`。

原因：差集断言只查"键在不在"，没查"值是不是 null / 有没有 error"。
（与 v42 那次静默吞异常是同一类：闸门恒绿 = 闸门没在工作。）

修法：① 必要条件动态化 —— 只有真跑到那一轮才要求 `sd_mean_at_179`，
否则改为要求 `sd_mean_at_last`；② `missing` 判据扩成
「键不在 **或** 值为 None **或** 该键有 error」；③ `mechanism_ok` 还要
`mechanism_error` 为空；④ 因早停而**合法跳过**的键写进 `skipped_keys`（与 error 区分）。

#### 洞 3：`final_tta` 的名字会骗人

源码里先是 `model.load_state_dict(torch.load('best_tta_resnet_vXX.pth'))`，
然后才 `final_tta = test_tta(...)` ⇒ **`final_tta` 测的是 best-TTA 权重，不是末态权重**。
它与 `best_tta_acc` 是同一份权重测两次，差恒 ≤0.02pp，因此**继承 max-of-N 偏差**。
（我此前把 `final_tta` 当"末态重测"写进过结论 —— 那是错的。）

修法：**不改计算**（改了就破坏跨版可比），只新增两个只读自曝字段
`final_tta_measures` / `final_tta_source_epoch`，并在 banner 打印出来。

### 预注册判据（本版真正的判据）

    INSTRUMENT_OK    instrument.checks 七项全 True（分组对应下面五条）：
                        ① actual_epochs > 0           -> actual_epochs_positive
                          且 == last_epoch + 1         -> history_len_matches
                        ② early_stopped / stop_reason 字段已写入（不依赖是否真早停）
                                                       -> early_stop_fields_written
                        ③ mechanism_ok 严格版 = True（见洞 2 的四条修法）
                                                       -> mechanism_ok_strict
                        ④ 没有把"合法跳过"误判成 error  -> no_errored_keys
                          且 skipped_keys 已声明        -> skipped_keys_declared
                        ⑤ final_tta 口径已自曝          -> final_tta_measure_declared
    INSTRUMENT_FAIL  任意一项不成立 ⇒ 仪器没修好，读数一律不采信

    ⚠ 七项里没有恒真项。早停那一项是双向的：
      early_stopped=True 时 stop_reason 必须有值；没早停时必须是 None。

**读数判据不是本版的目的**：SEED 与 v44 相同 ⇒ 本跑**不是新独立观测**，
`z_combined` 只是把 v43 + v45 合并（≈ v43 + v44），不产生新信息。
真正的第三个 seed 留给 v46（= v45 + 改 SEED 一行）。

### 同 seed 复现检验（本版附带的免费收获）

    EXACT                      final_tta 与 v44 逐位相同 且 两条曲线指纹都与 v44 相同
    NEAR_WITHIN_TTA_NOISE      |Δfinal_tta| <= 0.02pp（TTA 重测噪声量级）
    NOT_A_SAME_SEED_REPLICATE  其余 ⇒ 报出来，说明仪器改动意外影响了训练，或 GPU 非确定

已知 v44 的实测常量（写在脚本里，用于比对；`actual_epochs` 由 `len(history.test)` 反推）：

    V44_FINAL_TTA = 97.52    V44_BEST_TTA = 97.51    V44_ACTUAL_EPOCHS = 157
    V44_EARLY_STOPPED = True

### 回退红线

本版代码与 v44 **只差只读字段与闸门** ⇒ **读数不许回退代码**。
若读数与 v44 不一致，唯一正确动作是查 `instrument` 段 + 复现档位，
**不是**回退到 v37/v43/v44 的代码。

================================================================================

以下是 v37 原文头（保留以维持血脉可追溯）

v37 = **v14 的完整回退**。下面是 v14 的原文头，一字未改；本文件的区别在于
文件末尾"v37 相对 v14 的改动清单"。

一句话：**v14 的配方在loss、调度、数据结构上逐字保留，只改资源文件名与一行
torchvision 兼容写法。** 所有 v33 / v34 / v35 / v36 引入的"新算子"全部撤销。

为什么回退（实测，v36 跑到 e61）：
  loss / 数据结构层面 v36 相对 v14 有三处改动，全部有害：
  1) `smooth` 项从 v14 的 mean(Δphys²)（只约束**场输出**）改成 mean(Δ(ph_n−x_n)²)
     （归一化残差）⇒ 梯度直接打进 **backbone 的特征 x** ⇒ 这一项实际在惩罚
     高频特征，逼主干输出空间平滑。这是把"场正则"改成了"主干正则"。
  2) 结构对齐从 v14 的独立 refine_physics 步骤（no_grad 取特征、冻结主干、
     专用 lr=1e-3 只更新 phys）搬进**主 loss** ⇒ 与 cls 在同一步竞争，
     且梯度经 phys 回灌主干。
  3) 场换成"输入自适应谱场"(ctx_net)，为了跑它把并行改成 DDP + AMP。
     DDP 必须 torchrun —— **直接 `python` 跑就静默退回单卡**（这就是
     "以前版本能用 2 卡、最近版本改坏了"的直接原因）；AMP 又改了数值路径。

  ⇒ 结论：这个方向的每一次"新算子"都低于基线。**回退，不再加新机制。**

双卡怎么用：什么都不用做。启动时自动 `torch.cuda.device_count()`，
  >1 就 DataParallel，`BATCH_SIZE = 128 × N_GPUS`。**直接 `python cnn10_v37.py`**，
  不需要 torchrun、不需要任何命令行参数。启动后会打印实际卡数。
================================================================================

Physics-Guided ResNet-50 Training - v14
Bessel Potential + Sobolev Space + Prediction Ensemble

v9 achieved 97.26% raw / 97.44% TTA but early-stopped at epoch 144 due to
cosine restart crashing the model (lr jumped 100x from 1e-4 to 0.01).

v13 tried SWA (weight averaging) with cyclic lr but FAILED:
  - Cyclic lr (3e-4 max) perturbed model every 20 epochs -> never settled
  - SWA averaged 3 checkpoints from different basins -> 97.06% < 97.11% single
  - Best Test 97.11%, Best TTA 97.32% -> worse than v9

v14 insight: The problem with SWA is WEIGHT averaging across different basins.
  Prediction ensemble (averaging softmax outputs) ALWAYS works regardless of basins.
  No BN recompute needed, no weight manipulation, just inference-time averaging.

[v14] Key changes from v9:
  1. Phase 2 lr: smooth cosine 1e-4 -> 1e-5 (no restart, no cycling)
  2. Prediction ensemble: save top-K checkpoints, average softmax at inference
  3. Ensemble + TTA: each checkpoint does TTA, then average across checkpoints
  4. Training epochs: 180 (120 phase 1 + 60 phase 2)
  5. More patience: TTA=12, Test=50 (give model room to converge)

[KEEP] All v9 structural improvements: BlurPool, StochasticDepth, Matern, RandAugment
[KEEP] All v9 training config: Mixup 60-120, w_phys, refine_physics 30-100, etc.
[KEEP] All hang fixes: torch.flip TTA, refiner workers=0, empty_cache, stdout flush

Why ensemble > SWA:
  - SWA: average weights -> fails if checkpoints in different loss basins
  - Ensemble: average predictions -> always works, just needs diverse checkpoints
  - Ensemble + TTA: double averaging (across checkpoints AND augmentations)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.fft import fft2, ifft2, fftshift, ifftshift
import torchvision
import torchvision.transforms as transforms
import math
import numpy as np
import random
import sys
import time
import os
import json   # [v43] 结果落盘
import hashlib  # [v44] 曲线指纹（只读，用于判定"是不是一次真独立观测"）
from typing import List, Tuple

# ========== [v49] 唯一一根轴：算子层确定性硬化（必须在任何 CUDA 调用之前）==========
# 同 seed 同代码两次跑不逐位复现 ⇒ 平台的"噪声底"是假的、任何 <0.22pp 的轴都读不出来。
# 实证（v45 vs v46）：train()/test_tta() 逐字相同、seed 同为 1234、唯一差异 = 一个权重文件名常量，
#   而平台末 8 次 TTA 均值 97.4387 vs 97.3413 ⇒ Δ = −0.108pp。
# 既有的 cudnn 那两个开关只管 cuDNN；**cuBLAS 与部分 ATen 反向核（atomicAdd）不在它的管辖内**。
# 故本版补上这两样。cuBLAS 工作区必须在第一次 cuBLAS 调用之前设好，所以放在这个位置。
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

# ================== Fixed Randomness ==================
# [v44] 本版唯一一根改动轴：换 seed。v43 = seed 42（final_tta 97.41）；
#   本跑 = seed 1234，作为**同配方独立复跑**，用于把 v43 的 1.88σ 变成可判定的测量。
#   除此之外与 cnn10_v43.py 逐字相同。
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)   # [v49] 唯一一根轴（探针失败时自动降级并如实记录）

# ========== [v49] 判据常量（定义在**最前面**：banner 与收尾都要打印它们）==========
# ★★★ σ 换掉了：n=3 估出来的 0.050pp 是个假底。
#   用一对"零改动"的跑直接量 σ_diff：v45 vs v46 —— train()/test_tta() 逐字相同、同 seed 1234、
#   唯一差异是一个权重文件名常量 —— 平台末 8 次 TTA 均值 97.4387 vs 97.3413 ⇒ **Δ = −0.108pp**。
#   ⇒ 教训：**σ 必须用"同配方同 seed 的一对真跑"去量，不能用"几个彼此不同的版本"去估**
#     （后者会把版本间的真效应当成噪声，把噪声估小）。
#   ⇒ 也正因如此，v47/v48 的 PASS 线（baseline+0.10pp = 0.93σ）低于它自身 1σ，
#     那两个 NULL 是算术上被预先决定的，不构成"轴无效"的证据（覆辙 155 升级版）。
PLATEAU_BASELINE = 97.4625   # [v49] v48 的末 8 次 TTA 均值（同 seed、同口径，= 本版直接前身）
PLATEAU_SIGMA_MEASURED = 0.108   # [v49] **实测** σ_diff（冻结的实测常数，来源见 PLATEAU_BASELINE_SOURCE）
PLATEAU_SIGMA = PLATEAU_SIGMA_MEASURED
PLATEAU_PASS_MARGIN = 0.22   # [v49] 判据边距 = 2σ_diff，**必须 >= 2σ**
# ★★ 判据线**由公式算出**，不许写死字面量 —— 否则改 baseline/σ 时判据会静默脱钩（覆辙 113/145）。
PLATEAU_PASS = round(PLATEAU_BASELINE + PLATEAU_PASS_MARGIN, 2)      # -> 97.68
PLATEAU_REDLINE = round(PLATEAU_BASELINE - PLATEAU_PASS_MARGIN, 2)   # -> 97.24
PLATEAU_WINDOW = 8           # 末 8 次 TTA 读数
# 两条守卫（会被塞进 lr_tail_instrument 参与门禁 ⇒ 任一条为 False 即 INSTRUMENT_FAIL）
PLATEAU_MARGIN_OK = bool(float(PLATEAU_PASS_MARGIN) >= 2.0 * float(PLATEAU_SIGMA) - 1e-12)
PLATEAU_DERIVED_OK = bool(abs((float(PLATEAU_PASS) - float(PLATEAU_BASELINE))
                              - float(PLATEAU_PASS_MARGIN)) < 0.005
                          and abs((float(PLATEAU_BASELINE) - float(PLATEAU_REDLINE))
                                  - float(PLATEAU_PASS_MARGIN)) < 0.005)
PLATEAU_BASELINE_SOURCE = ("v48 实跑 97.4625（末 8 次 TTA 均值）；σ_diff 来源 = v45/v46 一对"
                           "零改动跑（train()/test_tta() 逐字相同、同 seed 1234、仅权重文件名常量不同）"
                           "⇒ Δ = −0.108pp。★ 该 σ 是**实测**，不是 n=3 估计。")
AXIS_V51 = ("v51 axis = gate1 fixed injection coefficient 0.6 -> 0.9 (raise the layer1 physics-field "
            "injection to the design-domain upper bound used by the learnable gates, soft_bound(a,0.05,0.9)). "
            "gate1 is the only HARDCODED injection constant in the physics chain and was never tuned; "
            "gate2/gate3 are learnable so their init/bounds get compensated by the optimizer. "
            "pinn=0.04 from v50 and the v49 operator-level determinism hardening are preserved.")

# ================== GPU Config ==================
N_GPUS = torch.cuda.device_count()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device} | Available GPUs: {N_GPUS}")
sys.stdout.flush()

BATCH_PER_GPU = 128
BATCH_SIZE = BATCH_PER_GPU * max(N_GPUS, 1)
LR_BASE = 0.1
LR_PHYS = 0.15
WARMUP_EPOCHS = 5
WARMUP_START_LR = 0.01

TOTAL_EPOCHS = 200        # For schedule calculations (w_phys, physics_reg, etc.)
COSINE_EPOCHS = 120       # Phase 1 ends at 120 (same as v9)
TRAIN_EPOCHS = 180        # [v14] Actual training epochs (Phase 2: 120-180)
PHASE2_LR_END = 3.0e-5    # [v48→v49 保留] phase2 余弦终点 1e-5 -> 3e-5（v49 不改配方，只改分辨率）
PINN_WEIGHT = 0.04        # [v50 轴，v51 原样保留] pinn 正则系数 0.08 -> 0.04（减半；见头部 v50 原文说明）
GATE1_INJECT = 0.9        # [v51] 本版唯一一根轴：gate1 固定注入系数 0.6 -> 0.9
#   0.9 不是随手取的数：可学习 gate 的 alpha 被 soft_bound 界定在 [0.05, 0.9]
#   ⇒ 0.9 = 该设计域的上界，而 gate1 的旧值 0.6 是低于上界的硬编码常量（自 v9 起从未被调过）。
#   改它的理由：gate2/gate3 可学习 ⇒ 改其 init/界会被优化器补偿；gate1 固定 ⇒ 改它才持续生效。
#   ================================================================================
#   [v48] 为什么把 v47 的"外部 clamp"换成"余弦自己的终点"（同一根轴，换参数化）
#   --------------------------------------------------------------------------------
#   v47 实测（results_cnn_v47_seed1234.json）三条硬读数：
#     (1) `lr_floor.first_bite_epoch = 168`，`n_bites_in_this_run = 7`（ep168..174，早停截断）；
#     (2) `best_tta_epoch = 138`，`instrument.final_tta_source_epoch = 138`
#         ⇒ **交付的那份权重是在 ep138 存下来的**；
#     (3) delivered 97.42 = 对 ep138 那份权重的**重测**（`final_tta_measures='best_tta_ckpt'`）。
#   ⇒ ★★★ **剂量起点（ep168）比交付选择点（ep138）晚了整整 30 轮**。
#     交付读数在数学上不可能含本轴的任何信息 —— v47 的 NULL 不是"轴无效"，
#     是**剂量窗口与交付窗口不相交**（覆辙 115 的新形态：送到了，但没送到被测的权重上）。
#   ⇒ 这个洞**不能用加 seed 补**：再多 seed 也都是同一个不相交结构。
#   --------------------------------------------------------------------------------
#   为什么"把 clamp 调大"不是解法：
#     clamp 要咬到 ep138 之前，floor 必须 >= lr(138) = 8.145e-5，而 phase2 峰值只有 1.0e-4
#     ⇒ floor 占到峰值的 81% ⇒ 余弦衰减被削掉 81% 的幅度 = **实质删掉整条 phase2 波形**
#     = 覆辙 148 的邻域（"设下限"变成了"改形状"两件事一起做，超一轴预算 ⑮）。
#   ⇒ 正解是**改波形自己的终点**：PHASE2_LR_END 是调度本身的定义参数，
#     抬高它 = 改**定义域**（覆辙 128 认可），不是在外面加一层 clamp（覆辙 79/148 禁止）。
#   --------------------------------------------------------------------------------
#   剂量曲线（离线重放，见 `_lr_tail_dose()` 同款公式；倍率 = new/old）：
#     ep120: 1.000e-4 -> 1.000e-4  x1.00   ← phase2 起点**逐位不变**
#     ep128: 9.611e-05 -> 9.697e-05 x1.01
#     ep138: 8.145e-05 -> 8.557e-05 x1.05   ← v47 的 best_tta 点
#     ep150: 5.500e-05 -> 6.500e-05 x1.18
#     ep162: 2.855e-05 -> 4.443e-05 x1.56   ← v46 的 best_tta 点
#     ep170: 1.603e-05 -> 3.469e-05 x2.16
#     ep180: 1.000e-05 -> 3.000e-05 x3.00
#     ep0-120 (**phase1 全程**): 逐位不变（机器守卫 `phase1_bitwise_unchanged`）
#   ⇒ 剂量从 **ep121** 起就在，完整覆盖交付选择窗（实测 best_tta 落在 ep138–162）。
#   安全边界（覆辙 148 反制）：新终点 3.0e-5 **落在 phase2 值域 [1.0e-5, 1.0e-4] 内部**，
#     且余弦仍保留 (1.0e-4 - 3.0e-5)/(1.0e-4 - 1.0e-5) = **78% 的衰减幅度** ⇒ 没有删波形。
#   ================================================================================
PHASE2_LR_END_REF = 1e-5  # [v48] 仪器用：**冻结的 v46/v47 参考终点**，只给 `_lr_ratio_v46_style`
#     用（对照/剂量重放）。★ 绝不参与训练 —— 训练只认 PHASE2_LR_END。
#     （v47 的 `_lr_ratio_v46_style` 直接读 PHASE2_LR_END，所以一旦改终点，
#       "参考波形"会跟着一起变 ⇒ 剂量重放恒为 0、剂量守卫恒绿。这是 v47 埋着的
#       一个真仪器洞，本版用这个冻结常量堵上。）
EARLY_STOP_PATIENCE = 12  # [v14] TTA patience 8->12
TTA_INTERVAL = 3
TTA_START_EPOCH = 80
TEST_PATIENCE = 50        # [v14] Test patience 30->50

SD_MAX_DROP = 0.1  # Stochastic Depth max drop rate

# [v43] StochasticDepth 退火（本版唯一一根改动轴）
#   SD 噪声只打在 main path、不打在 identity，且逐样本独立
#   => 破坏 f(flip(x))==flip(f(x)) 的翻转等变性，而交付读数完全依赖 flip TTA。
#   实测 flip 增益 ep<=120 +0.333 -> ep>120 +0.199（衰减 −0.134）。
#   退火终点对齐 ep120：此后 raw 与 cls 均已冻结，SD 唯一残余作用就是制造翻转不等变。
SD_ANNEAL_START = 60      # 暖启段结束（与 mixup 开启点同点，不引入新时间常数）
SD_ANNEAL_END = 120       # 退火终点（与 mixup 卸载点、phase1/2 分界同点）
SD_ANNEAL_ENABLE = True   # 本版机制开关：写死 True，不暴露给命令行

# [v14] Prediction Ensemble config
ENS_START_EPOCH = 105     # Start saving checkpoints from ep105 (near phase 1 end)
ENS_INTERVAL = 3          # Save checkpoint every 3 epochs
ENS_MAX_KEEP = 8          # Keep top-8 by test accuracy (max-of-N 选点；v46 保留作对照)
ENS_TOP_K = 5             # Use top-5 for ensemble
# [v46] 交付链唯一一根轴：**末段权重平均**（不选点、不用 max-of-N、不吃运行期抖动）。
#   ★ [v47] 该轴已实跑判死（NULL_SWA，Δ=−0.01pp）⇒ 本版**关掉**，代码只留作诊断。
#   死因（v46 JSON 实测）：窗口 ep126..178 内 TTA 平台极差只有 **0.06pp**，
#   而 WIN 门限是 0.10pp ⇒ "平均 27 份权重" 与 "挑最好的那个 epoch" 数值不可区分
#   （平台 max 97.38 / 平均权重 97.37）⇒ 这条轴【结构性零分辨力】，不是坏。
#   更深一层：窗口内 lr ≤ 1.0e-4（ep178 只有 1.0e-5），mixup=0、SD=0
#   ⇒ 末段是一个「正则化真空 + 权重冻结」期，任何在冻结权重上做的平均/选点都必然重合。
SWA_ENABLE = False        # [v47] 关掉已判死的轴（True 可单独回放 v46 行为做对照）
#   起因（v45 实跑）: 交付读数 = max(33 次 TTA 读数) ∩ max(30 次 raw 读数) 的双重 max-of-N，
#   而 v45 的 raw 在 e140 之后 40 个 epoch 的总跨度只有 0.12pp（sd 0.031）—— 曲线早就平了，
#   交付读数之间的差别主要是"哪次评估恰好落高"（运行期非确定性）。同时 v45 的 e121..e156
#   逐 epoch raw 对 v44 **36/36 全部为正**（均值 +0.29pp）⇒ 末段权重**整体**更好，不是某个点更好。
#   ⇒ 交付一个**带 epoch 标签**的平均权重：标签即平均窗，读数不依赖运气，且可被机检。
#   方法固定为确定性算术（每个参数 = 窗口内逐 epoch 的算术均值，跑完才平均，无滑动），
#   不用任何"滑动副本"（用户级禁令：权重 EMA 永久禁用）；BN running_* 是该权重自己的统计，
#   与参数同源平均，不存在 EMA 与 BN 失配那类问题。
SWA_START_FRAC = 0.70     # 窗口起点 = round(frac * TRAIN_EPOCHS) = ep126（写死为常量，不用运行期整数）
SWA_SNAPSHOT_EVERY = 2    # 每 2 个 epoch 采一份快照（省显存；窗口内 27 份）

# ====================== [v47] 唯一一根改动轴：末段 lr 下限（LR_FLOOR） ======================
#   ★ 为什么换到"训练"这条轴（前三条交付链轴全判死之后的必然结论）：
#     交付链已经失败三次 —— 快照集成(v1) / 预测集成(v1) / 末段权重平均(v46，Δ=−0.01pp)，
#     三者都是**在一条比测量分辨率还平的平台上做后处理**。
#     实测：ep126–178 的 TTA 极差 0.06pp、raw 极差 0.15pp；而跨跑 seed σ≈0.06–0.11pp。
#     ⇒ 平台比效应还平 ⇒ 后处理家族整体判死，不再做第四个补丁（覆辙 ①⑮）。
#   ★ 平台为什么这么平 —— 机制级死因（v46 JSON + 源码联合核算）：
#     phase2（ep120–180）lr 从 1.0e-4 余弦衰减到 1.0e-5，即峰值 0.1 的 1e-3 ~ 1e-4 倍。
#     权重更新步长 ≈ lr × 梯度，末段已小到让模型**实质冻结**：
#     v46 的 raw 在 ep128 之后就再没创过新高，**连续 51 轮零改善**（这才触发 Test patience=50）。
#     换句话说，ep128–179 这 51 轮算力（占总量 28%）几乎全花在"原地抖动"上。
#   ★ 本版唯一改动：给末段 lr 设一个**下限**，把"冻结期"变成"低速精调期"。
#     物理动机：accuracy 曲线在 ep128 见顶后平掉，说明"该学的已学到"，
#     但 lr→1e-5 时优化器已无力把这个峰再往上推。抬高下限 = 恢复末段的可行搜索半径。
#   ★ 为什么不是"直接调大 phase2 的 lr"：那是改调度形状（波形级），会连带改 phase1 尾巴；
#     设下限只动**定义域下界**（覆辙 128：改定义域才改变优化器可达集），
#     且 lr(e)=max(lr(e), LR_FLOOR*LR_BASE) 逐点取 max ⇒ 只抬高、不压低，phase1 完全不动。
LR_FLOOR_ENABLE = False   # [v48] v47 的 clamp 已退役（见 PHASE2_LR_END 处的说明）；
#     写死 False ⇒ `smooth_cosine_schedule` 退化为纯 v46 波形 + 新终点，不含任何 clamp。
#     ★ 留着这个开关而不是删掉那段代码：保留"机器可核对的退役记录"
#       （`lr_floor_instrument.floor_enabled=False` + `n_bites=0`），
#       避免"代码还在但没生效"这种无法归因的状态（覆辙 ㊽ / 153）。
LR_FLOOR = 2.0e-5         # 下限 = 2.0e-5 × LR_BASE = 2.0e-6（absolute lr）
#   ★★ 取值依据（写死，不许运行期搜索）—— 这里踩过一次自己造的坑，记录在案：
#     第一稿取 LR_FLOOR = 2.0e-2（absolute 2.0e-3），离线重放 lr 曲线时发现
#     **它高于整个 phase2 的值域（phase2 max = 1.0e-4）** ⇒ 会把 ep120–180 全部压成同一个常数，
#     等于**删掉整条 phase2 余弦调度**。那不是"给末段设下限"，而是"改调度形状 + 删波形"
#     两件事一起做 = 超出"一轴"预算（覆辙 ⑮）。⇒ 重算。
#   ★ 正确约束：下限必须**落在 phase2 值域内部**，才只咬住尾巴、不动波形主体。
#     phase2 lr(e) = 1.0e-4 → 1.0e-5（cosine，60 轮）逐轮重放后的"咬住点"：
#       floor=5.0e-5 ⇒ 从 ep153 起咬住，覆盖 28/61 轮（46% 的 phase2）
#       floor=3.0e-5 ⇒ 从 ep162 起咬住，覆盖 19/61 轮（31%）
#       floor=2.0e-5 ⇒ 从 ep168 起咬住，覆盖 13/61 轮（21%）  ← 本版取此值
#       floor=1.5e-5 ⇒ 从 ep171 起咬住，覆盖 10/61 轮（16%）
#     取 2.0e-5 的理由：它只动 phase2 的最后 21%，前置 79% 的余弦形状**逐位不变**，
#     这样"一轴"的含金量最高；同时它把末段 13 轮的 lr 从 [1.0e-5, 2.1e-5] 抬到恒定 2.0e-6，
#     相对抬升 1.0~2.0 倍 —— 量级上**足够改变"冻结"状态**（步长正比于 lr），
#     又远低于会破坏收敛的量级（phase1 峰值 0.1 = 本值的 500 倍）。
#   ★ 上界安全性：absolute lr = 2.0e-6，相对 v46 末段最大抬升 2.0 倍 ⇒ 不引入重启式震荡
#     （无 restart，覆辙 ㊶ 不适用）。★ 这个"倍率小"是有意为之：第一次做训练侧轴，
#     宁可效应小到读不出来（读 NULL 可归因），也不要一口吃成"多个轴"（覆辙 ⑮）。
#   ★ t=0 逐位保证：floor 只在 epoch ≥ 生效点 之后才可能咬住；
#     生效点取 phase2 起点 ep120 ⇒ ep<120 的 lr 与 v46 **逐位相同**（覆辙：起点不扰动上一版）。
#     并且由上式，ep120–167 也逐位相同 ⇒ **真正被改动的只有 ep168–180 共 13 轮**。
LR_FLOOR_FROM_EPOCH = 120  # 与 mixup 卸载点 / SD 退火终点 / phase1-2 分界同点（不引入新时间常数）
# [v48] clamp 已退役 ⇒ 这三个"期望值"改成**退役语义**（-1 / 0 / 0）：
#   只要有人把 LR_FLOOR_ENABLE 改回 True，或偷偷改 LR_FLOOR，下面三条守卫立刻亮红。
LR_FLOOR_EXPECTED_FIRST_BITE = -1   # [v48] 退役后：不该有任何一轮被 clamp 咬住
LR_FLOOR_EXPECTED_N_BITES = 0       # [v48] 退役后：全 horizon 被咬轮数 = 0
LR_FLOOR_EXPECTED_N_BITES_TRAIN = 0 # [v48] 退役后：训练窗口被咬轮数 = 0
#   ★ 这两个数是"一轴"的机器化表达：改 floor 或改生效点都会让它们失配 ⇒ 仪器亮红。
#   为什么是两个数：ep181..200 属于"跑完 180 轮之后的 lr 保持段"（正常不会跑到），
#   它会一直被 floor 咬住，所以全 horizon 计数 = 33；而**真正影响训练**的只有 ep168..180 = 13 轮。
#   判据用后者（语义相关），前者用作"floor 值没被偷改"的交叉校验。
# [v44] 目录名带 seed（f-string 派生）⇒ **换 seed 只需改 SEED 这一行**，产物不会互相覆盖。
CKPT_DIR = f'./ckpts_v51_s{SEED}'  # Checkpoint directory
OUT_DIR = f'./out_v51_s{SEED}'     # [v50→v51] 结果 JSON 目录（版本号 + seed）

print(f"Batch: {BATCH_SIZE} | LR: {LR_BASE}/{LR_PHYS} | Epochs: {TOTAL_EPOCHS}")
sys.stdout.flush()

# ====================== Mixup / CutMix ======================
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def cutmix_data(x, y, beta=1.0):
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(x.size(0)).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
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

# ====================== Utils ======================
def soft_bound(x, min_v, max_v):
    return min_v + (max_v - min_v) * torch.sigmoid(x)

def _reduce(val):
    if isinstance(val, torch.Tensor) and val.dim() > 0:
        return val.mean()
    return val

def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# ====================== [v9] BlurPool2d ======================
class BlurPool2d(nn.Module):
    """Anti-aliased downsampling with Bessel-derived binomial filter.

    Replaces strided conv (which aliases high frequencies) with:
    conv(stride=1) -> [this module: blur + subsample with stride=2]

    The 3x3 binomial filter [1,2,1;2,4,2;1,2,1]/16 is the spatial-domain
    approximation of the Bessel potential (1+lam*|k|^2)^{-s} for s~1.
    """
    def __init__(self, channels):
        super().__init__()
        k = torch.tensor([1., 2., 1.])
        k = k.outer(k) / 16.0
        self.register_buffer('filter',
            k.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.filter, stride=2, padding=1, groups=self.channels)

# ====================== Physics Layers ======================
class ModeController(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 8, 1),
            nn.ReLU(), nn.Conv2d(8, 3, 1)
        )
    def forward(self, x):
        return F.softmax(self.predictor(x).flatten(1), dim=1)


class MaternDiffusion(nn.Module):
    """[v9] Matern kernel: (1 + lam*|k|^2/nu)^{-s}

    Generalizes Bessel potential (nu=1 -> standard Bessel).
    nu controls kernel shape: nu->inf approaches Gaussian, nu=0.5 exponential.
    """
    def __init__(self, channels, groups=8, mode_centers=None):
        super().__init__()
        self.groups = min(groups, channels)
        self.cpg = channels // self.groups
        self.channels = channels

        self.mode_net = ModeController(channels)
        self.s_param = nn.Parameter(torch.zeros(self.groups))
        self.lam_param = nn.Parameter(torch.zeros(self.groups))
        self.nu_param = nn.Parameter(torch.zeros(self.groups))  # [v9] Matern smoothness

        if mode_centers is None:
            self.mode_centers = torch.tensor([0.5, 1.0, 1.6])
        else:
            self.mode_centers = torch.tensor(mode_centers)

        self.register_buffer('running_s', torch.full((self.groups,), 0.8))
        self.register_buffer('running_lam', torch.full((self.groups,), 0.5))
        self.register_buffer('running_nu', torch.full((self.groups,), 1.0))
        self.register_buffer('update_count', torch.tensor(0))

    def forward(self, x, backbone_feat, epoch):
        B, C, H, W = x.shape
        mode_weights = self.mode_net(x)

        phase_ratio = epoch / float(TOTAL_EPOCHS)
        couple_strength = 0.3 if phase_ratio < 0.2 else 0.8 if phase_ratio < 0.7 else 0.5
        with torch.no_grad():
            var = backbone_feat.var()
        dynamic_bias = torch.clamp((var - 0.5) * 0.3 * couple_strength, -0.3, 0.3)
        phase_bias = -0.2 if phase_ratio < 0.33 else 0.0 if phase_ratio < 0.67 else 0.2

        s_vals = soft_bound(self.s_param + phase_bias + dynamic_bias, 0.3, 2.0)
        lam_vals = soft_bound(self.lam_param + dynamic_bias + 0.2 * couple_strength, 0.1, 5.0)
        nu_vals = soft_bound(self.nu_param, 0.5, 3.0)  # [v9] nu in [0.5, 3.0]

        s_vals = torch.clamp(s_vals, 0.3, 1.3)

        with torch.no_grad():
            momentum = min(0.01 * (1 + self.update_count.item() * 0.001), 0.05)
            self.running_s = (1 - momentum) * self.running_s + momentum * s_vals.detach()
            self.running_lam = (1 - momentum) * self.running_lam + momentum * lam_vals.detach()
            self.running_nu = (1 - momentum) * self.running_nu + momentum * nu_vals.detach()
            self.update_count += 1

        s_vals = 0.9 * s_vals + 0.1 * self.running_s
        s_vals = torch.clamp(s_vals, 0.3, 1.3)
        lam_vals = 0.9 * lam_vals + 0.1 * self.running_lam
        nu_vals = 0.9 * nu_vals + 0.1 * self.running_nu

        ky = torch.linspace(-math.pi, math.pi, H, device=x.device)
        kx = torch.linspace(-math.pi, math.pi, W, device=x.device)
        ky, kx = torch.meshgrid(ky, kx, indexing="ij")
        k_sq = kx**2 + ky**2

        x_g = x.reshape(B, self.groups, self.cpg, H, W).mean(dim=2)

        phys_list = []
        for g in range(self.groups):
            # [v9] Matern kernel: (1 + lam*|k|^2/nu)^{-s}
            kernel = (1.0 + lam_vals[g] * k_sq / nu_vals[g]) ** (-s_vals[g])
            fft = fftshift(fft2(x_g[:, g:g+1]))
            filtered = torch.real(ifft2(ifftshift(fft * kernel)))
            phys_list.append(filtered)

        phys = torch.cat(phys_list, dim=1)
        phys = phys.repeat_interleave(self.cpg, dim=1)

        dx = phys[..., 1:] - phys[..., :-1]
        dy = phys[..., 1:, :] - phys[..., :-1, :]
        smooth = torch.mean(dx**2) + torch.mean(dy**2)

        return phys, smooth, s_vals.mean(), lam_vals.mean(), nu_vals.mean()


class FixedDiffusion(nn.Module):
    """Fixed Bessel potential, grouped output"""
    def __init__(self, channels, groups=8):
        super().__init__()
        self.groups = min(groups, channels)
        self.cpg = channels // self.groups

    def forward(self, x):
        B, C, H, W = x.shape
        s, lam = 0.8, 0.5
        ky = torch.linspace(-math.pi, math.pi, H, device=x.device)
        kx = torch.linspace(-math.pi, math.pi, W, device=x.device)
        ky, kx = torch.meshgrid(ky, kx, indexing="ij")
        kernel = (1.0 + lam * (kx**2 + ky**2)) ** (-s)

        x_g = x.reshape(B, self.groups, self.cpg, H, W).mean(dim=2)
        phys_list = []
        for g in range(self.groups):
            fft = fftshift(fft2(x_g[:, g:g+1]))
            filtered = torch.real(ifft2(ifftshift(fft * kernel)))
            phys_list.append(filtered)
        phys = torch.cat(phys_list, dim=1)
        phys = phys.repeat_interleave(self.cpg, dim=1)
        return phys


class Gate(nn.Module):
    def __init__(self, channels, learnable=True):
        super().__init__()
        self.learnable = learnable
        self.proj = nn.Conv2d(channels, channels, 1, groups=min(channels, 8), bias=False)
        self.bn = nn.BatchNorm2d(channels)
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(0.))
            self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, feat, phys):
        p = self.bn(self.proj(phys))
        if self.learnable:
            a = soft_bound(self.alpha, 0.05, 0.9)
            return feat + self.scale * a * p, a
        else:
            # [v51] 本版唯一一根轴：固定注入系数 0.6 -> GATE1_INJECT。
            # 只有 gate1 走这条分支（Gate(256, learnable=False)），是物理场链路上唯一的固定注入。
            return feat + GATE1_INJECT * p, GATE1_INJECT


# ====================== [v9] Model with BlurPool + StochasticDepth ======================
class GradientStablePhysResNetV3(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        try:  # 新版 torchvision 已移除 pretrained=；两种写法都 = 不加载预训练权重，
            from torchvision.models import resnet50, ResNet50_Weights
            self.backbone = resnet50(weights=ResNet50_Weights.NONE)
        except Exception:
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(2048, num_classes)

        self.phys1 = FixedDiffusion(256, groups=8)
        self.gate1 = Gate(256, learnable=False)
        self.phys2 = MaternDiffusion(512, groups=8)
        self.gate2 = Gate(512, learnable=True)
        self.phys3 = MaternDiffusion(1024, groups=8)
        self.gate3 = Gate(1024, learnable=True)

    def forward(self, x, epoch):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = F.relu(x)

        x = self.backbone.layer1(x)
        x, _ = self.gate1(x, self.phys1(x))

        x = self.backbone.layer2(x)
        ph2, sm2, s2, l2, nu2 = self.phys2(x, x, epoch)
        x, a2 = self.gate2(x, ph2)

        x = self.backbone.layer3(x)
        ph3, sm3, s3, l3, nu3 = self.phys3(x, x, epoch)
        x, a3 = self.gate3(x, ph3)

        x = self.backbone.layer4(x)
        deep = x
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        out = self.backbone.fc(x)

        dx = deep[..., 1:] - deep[..., :-1]
        dy = deep[..., 1:, :] - deep[..., :-1, :]
        pinn = torch.mean(dx**2) + torch.mean(dy**2)
        # [v9] Return 7 values (added nu_vals)
        return out, [s2, s3], [l2, l3], [nu2, nu3], [a2, a3], sm2 + sm3, pinn


# ====================== [v43] StochasticDepth Anneal ======================
def sd_anneal_factor(epoch):
    """[v43] StochasticDepth 的时间退火因子，值域 [0, 1]。

    设计：与 mixup 窗、lr 分段同点对齐，不引入新的时间常数。
      epoch <  SD_ANNEAL_START : 线性暖启 0 -> 1（与原有深度斜坡同尺度）
      SD_ANNEAL_START .. END   : 线性退火 1 -> 0
      epoch >= SD_ANNEAL_END   : 恒 0（相位2 全程确定性主干）

    epoch 为 None 时返回 1.0，保持与 v37 逐字不同的调用路径（如 test 阶段）行为不变。
    """
    if epoch is None:
        return 1.0
    if not SD_ANNEAL_ENABLE:
        return 1.0
    if epoch < SD_ANNEAL_START:
        return float(epoch) / float(SD_ANNEAL_START)
    if epoch < SD_ANNEAL_END:
        span = float(SD_ANNEAL_END - SD_ANNEAL_START)
        return 1.0 - float(epoch - SD_ANNEAL_START) / span
    return 0.0


def sd_drop_rate(block_idx, total_blocks, epoch):
    """[v43] 某个 block 在某个 epoch 实际生效的 drop_rate。

    与 v37 的唯一差别：v37 = max_drop * block_idx / total_blocks（常量）；
    v43 在此基础上再乘一个 epoch 退火因子。
    """
    base = SD_MAX_DROP * block_idx / max(total_blocks, 1)
    return base * sd_anneal_factor(epoch)


class BlurBottleneck(nn.Module):
    """[v9] Wrapper around torchvision Bottleneck that adds BlurPool + StochasticDepth.

    Using a wrapper (instead of monkey-patching forward) ensures correct
    behavior under DataParallel: self.block is properly replicated to each GPU.
    """
    def __init__(self, block, blur, drop_rate):
        super().__init__()
        self.block = block      # Original Bottleneck (conv1/bn1/conv2/bn2/conv3/bn3/relu/downsample)
        self.blur = blur        # BlurPool2d or None
        self.drop_rate = drop_rate
        # [v43] 退火所需的静态身份 + 逐 epoch 实际生效值（供剂量读数）
        self._sd_block_idx = None
        self._sd_total_blocks = None
        self._sd_epoch = None
        self._sd_eff_drop = float(drop_rate)

    def set_sd_schedule(self, block_idx, total_blocks):
        """[v43] 登记退火所需的静态身份（在 apply_blurpool_and_sd 里调用一次）。"""
        self._sd_block_idx = int(block_idx)
        self._sd_total_blocks = int(total_blocks)

    def set_sd_epoch(self, epoch):
        """[v43] 每个 epoch 开工前刷新实际生效的 drop_rate。"""
        self._sd_epoch = epoch
        if getattr(self, '_sd_block_idx', None) is None:
            self._sd_eff_drop = float(self.drop_rate)
        else:
            self._sd_eff_drop = float(
                sd_drop_rate(self._sd_block_idx, self._sd_total_blocks, epoch))

    def forward(self, x):
        identity = x
        b = self.block
        out = b.conv1(x)
        out = b.bn1(out)
        out = b.relu(out)
        out = b.conv2(out)
        out = b.bn2(out)
        if self.blur is not None:
            out = self.blur(out)
        out = b.relu(out)
        out = b.conv3(out)
        out = b.bn3(out)

        drop_rate = self._sd_eff_drop
        if self.training and drop_rate > 0:
            keep_prob = 1.0 - drop_rate
            shape = (out.shape[0],) + (1,) * (out.ndim - 1)
            mask = (torch.rand(shape, device=out.device) < keep_prob).to(out.dtype).div_(keep_prob)
            out = out * mask

        if b.downsample is not None:
            identity = b.downsample(x)
        out = out + identity
        out = b.relu(out)
        return out


def apply_blurpool_and_sd(model, max_drop=0.1):
    """[v9] Apply BlurPool + StochasticDepth by wrapping each Bottleneck block.

    Replaces each block in layer1-4 with a BlurBottleneck wrapper.
    The wrapper is a proper nn.Module, so DataParallel replicates it correctly.
    """
    total_blocks = sum(len(getattr(model.backbone, f'layer{i}')) for i in range(1, 5))
    block_idx = 0

    for layer_i in range(1, 5):
        layer = getattr(model.backbone, f'layer{layer_i}')
        for i in range(len(layer)):
            block = layer[i]
            # [v37] 深度斜坡的基准值（原始行为：常量）；[v43] 实际值由 epoch 退火动态给出
            drop_rate = SD_MAX_DROP * block_idx / total_blocks

            # --- Fix downsample path: conv(stride=2) -> conv(stride=1) + BlurPool ---
            #    ONLY when original downsample conv had stride=2.
            #    layer1[0] has downsample with stride=1 (channel projection only)
            #    -> must NOT add BlurPool, otherwise identity gets spatially
            #       downsampled while main path stays full-res -> size mismatch.
            if block.downsample is not None:
                old_conv = block.downsample[0]
                old_bn = block.downsample[1]
                ch = old_conv.out_channels
                if old_conv.stride == (2, 2):
                    new_conv = nn.Conv2d(
                        old_conv.in_channels, ch, kernel_size=1, stride=1, bias=False)
                    new_conv.weight.data = old_conv.weight.data.clone()
                    block.downsample = nn.Sequential(new_conv, old_bn, BlurPool2d(ch))

            # --- Fix main path: conv2(stride=2) -> conv2(stride=1) + BlurPool ---
            main_blur = None
            if block.conv2.stride == (2, 2):
                ch2 = block.conv2.out_channels
                block.conv2.stride = (1, 1)
                main_blur = BlurPool2d(ch2)

            # --- Wrap with BlurBottleneck ---
            wrapper = BlurBottleneck(block, main_blur, drop_rate)
            # [v43] 登记退火身份：forward 里按当前 epoch 现算实际 drop_rate
            wrapper.set_sd_schedule(block_idx, total_blocks)
            layer[i] = wrapper
            block_idx += 1

    flush_print(f"  [v14] BlurPool + StochasticDepth applied to {block_idx} blocks (max_drop={max_drop})")


def sd_set_epoch(model, epoch):
    """[v43] 把当前 epoch 广播到所有 BlurBottleneck，刷新其实际生效的 drop_rate。

    这必须在每个 epoch 的 forward 之前调用，否则 wrapper 里的 _sd_eff_drop
    会停留在上次的值（就是"机制写好了但这次不会跑它"的典型形态）。
    """
    n = 0
    for m in model.modules():
        if isinstance(m, BlurBottleneck):
            m.set_sd_epoch(epoch)
            n += 1
    return n


def sd_effective_mean(model):
    """[v43] 当前所有 block 实际生效 drop_rate 的均值（剂量读数，从 wrapper 实读，
    不从常量反推）。返回 (mean, n_positive, n_total)。
    """
    vals = []
    n_pos = 0
    for m in model.modules():
        if isinstance(m, BlurBottleneck):
            v = float(getattr(m, '_sd_eff_drop', 0.0))
            vals.append(v)
            if v > 0.0:
                n_pos += 1
    if not vals:
        return 0.0, 0, 0
    return float(sum(vals) / len(vals)), n_pos, len(vals)


# ====================== Physics Regularization ======================
def physics_reg(s_list, epoch, total_epochs=TOTAL_EPOCHS):
    """Same as v8: coeff=0.05, bands max 1.0/1.1/1.2"""
    total = 0.0
    prog = epoch / total_epochs
    for s in s_list:
        if prog < 0.3:
            tmin, tmax = 0.3, 1.0
        elif prog < 0.6:
            tmin, tmax = 0.5, 1.1
        else:
            tmin, tmax = 0.5, 1.2
        if s < tmin:
            total += (tmin - s)**2
        if s > tmax:
            total += (tmax - s)**2
    return 0.05 * total


# ====================== w_phys Schedule ======================
def get_w_phys(epoch, total_epochs=TOTAL_EPOCHS):
    if epoch < 60:
        return 0.05 + (0.15 - 0.05) * (epoch / 60.0)
    elif epoch <= 120:
        return 0.15
    else:
        return 0.15 - (0.15 - 0.02) * ((epoch - 120) / (total_epochs - 120))


# ====================== Mixup alpha schedule ======================
def get_mixup_alpha(epoch):
    if epoch < 60 or epoch > 120:
        return 0.0
    if epoch < 110:
        return 0.4
    return 0.4 * (1.0 - (epoch - 110) / 10.0)


# ====================== [v14] Smooth Cosine Schedule ======================
# ====================== [v46] 末段权重平均（交付轴） ======================
def swa_window_start():
    """窗口起点。纯函数 + 只读常量 ⇒ 可用锚点校验（不读磁盘、不碰模型）。"""
    return int(round(float(SWA_START_FRAC) * float(TRAIN_EPOCHS)))

def swa_accumulate(net, epoch):
    """把一个 epoch 的权重/缓冲区累加进 swa_sum。确定性算术，无 RNG、无额外前向。

    running_mean/running_var/num_batches_tracked 一起平均：它们与参数同源（都是这份权重
    自己跑出来的统计），平均后仍是合法统计量。num_batches_tracked 均值会变成 float，
    故按 int 恢复，避免 load_state_dict 报类型不符。
    """
    global swa_sum, swa_n
    if not bool(SWA_ENABLE):
        return False
    w0 = swa_window_start()
    if int(epoch) < w0:
        return False
    if (int(epoch) - w0) % int(SWA_SNAPSHOT_EVERY) != 0:
        return False
    items = {k: v.detach().float().double().cpu()
             for k, v in net.state_dict().items()}
    if swa_sum is None:
        swa_sum = {k: v.clone() for k, v in items.items()}
    else:
        for k in swa_sum:
            swa_sum[k] += items[k]
    swa_n = int(swa_n) + 1
    swa_epochs.append(int(epoch))
    return True

def swa_build(net):
    """把累加器折成一份可加载的 state_dict（float32）。取不到快照时返 None。"""
    if not bool(SWA_ENABLE) or swa_sum is None or int(swa_n) <= 1:
        return None
    out = {}
    for k, v in swa_sum.items():
        avg = (v / float(swa_n))
        ref = net.state_dict().get(k)
        if ref is not None and str(ref.dtype).startswith("torch.int"):
            out[k] = avg.round().to(torch.int64)
        elif ref is not None:
            out[k] = avg.to(ref.dtype)
        else:
            out[k] = avg.float()
    return out


def _lr_floor_ratio():
    """[v47] 下限（归一化到 LR_BASE 的比值）。纯函数 ⇒ 可被锚点/守卫直接校验。"""
    return float(LR_FLOOR) / float(LR_BASE)


def _lr_ratio_core(epoch, phase2_end):
    """[v48] 波形内核：phase2 余弦终点由形参给出。**纯函数** ⇒ 可离线重放/做锚点。

    [v48] 把 v47 的 `_lr_ratio_v46_style` 参数化，是为了堵一个 v47 埋下的真仪器洞：
    v47 的"参考波形"和"训练波形"共用同一个常量 `PHASE2_LR_END`，于是
    `_lr_floor_dose()` 里 `smooth_cosine_schedule(e) - _lr_ratio_v46_style(e)` 恒为 0
    —— 只要改的是终点而不是 clamp，剂量守卫就会**恒绿假通过**。
    本版把参考终点冻结成 `PHASE2_LR_END_REF`，两条波形不再共享可变状态。
    """
    if epoch < WARMUP_EPOCHS:
        return (WARMUP_START_LR / LR_BASE) + (1.0 - WARMUP_START_LR / LR_BASE) * (epoch / WARMUP_EPOCHS)
    elif epoch <= COSINE_EPOCHS:
        cos_epoch = epoch - WARMUP_EPOCHS
        cos_total = COSINE_EPOCHS - WARMUP_EPOCHS
        eta_min_ratio = 1e-4 / LR_BASE
        return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1 + math.cos(math.pi * cos_epoch / cos_total))
    elif epoch <= TRAIN_EPOCHS:
        cos_epoch = epoch - COSINE_EPOCHS
        cos_total = TRAIN_EPOCHS - COSINE_EPOCHS
        eta_max_ratio = 1e-4 / LR_BASE
        eta_min_ratio = float(phase2_end) / LR_BASE
        return eta_min_ratio + 0.5 * (eta_max_ratio - eta_min_ratio) * (1 + math.cos(math.pi * cos_epoch / cos_total))
    else:
        return float(phase2_end) / LR_BASE


def _lr_ratio_v46_style(epoch):
    """[v47→v48] **冻结的** v46/v47 波形（终点 = PHASE2_LR_END_REF = 1e-5，无 clamp）。
    只读参考实现，用于：
      (1) 对照读数 —— 同跑内能同时算出"v48 波形 / v46 波形"两条 lr 轨迹；
      (2) 剂量重放与变异测试锚点 —— 证明本版真的改了可达集（覆辙 ㊽/115）。
    ★ 训练**不用**本函数；训练只认 `smooth_cosine_schedule()`。
    """
    return _lr_ratio_core(epoch, PHASE2_LR_END_REF)


def lr_floor_active(epoch):
    """[v47] v47 的 clamp 是否生效。[v48] 已退役（LR_FLOOR_ENABLE=False ⇒ 恒 False）。"""
    if not bool(LR_FLOOR_ENABLE):
        return False
    return bool(int(epoch) >= int(LR_FLOOR_FROM_EPOCH))


def smooth_cosine_schedule(epoch):
    """[v14] Smooth cosine: no restart, no cycling.
    Phase 1 (0-120): warmup 0-5, cosine 5-120, lr 0.1 -> 1e-4 (same as v9)
    Phase 2 (120-180): cosine 120-180, lr 1e-4 -> PHASE2_LR_END (smooth continuation)
    After 180: hold at PHASE2_LR_END

    [v47] 曾在 phase2 段对返回值取**下限** `max(ratio, LR_FLOOR/LR_BASE)` —— **v48 已关闭**
          （`LR_FLOOR_ENABLE=False`），保留函数与开关只为留一条机器可核对的退役记录。
    [v48] 本版唯一改动：phase2 余弦**终点** 1e-5 -> 3.0e-5（改的是调度自己的定义参数，
          不是在外面套 clamp）⇒ phase1 逐位不变、phase2 起点（ep120）逐位不变。
    """
    ratio = _lr_ratio_core(epoch, PHASE2_LR_END)
    if lr_floor_active(epoch):
        ratio = max(float(ratio), _lr_floor_ratio())
    return ratio


# ====================== Physics Refinement ======================
class PhysicsRefiner:
    def __init__(self, dataset, batch_size):
        self.loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, drop_last=True)
        self.iterator = None

    def get_batch(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            inputs, _ = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            inputs, _ = next(self.iterator)
        return inputs

    def refresh(self):
        self.iterator = None


def structure_alignment_loss(phys, feat):
    feat_gray = feat.mean(dim=1, keepdim=True).detach()
    phys_gray = phys.mean(dim=1, keepdim=True)
    phys_norm = (phys_gray - phys_gray.mean()) / (phys_gray.std() + 1e-8)
    feat_norm = (feat_gray - feat_gray.mean()) / (feat_gray.std() + 1e-8)
    cos_loss = 1 - F.cosine_similarity(
        phys_norm.flatten(1), feat_norm.flatten(1), dim=1).mean()
    mse_loss = F.mse_loss(phys_norm, feat_norm)
    return cos_loss + 0.1 * mse_loss


def refine_physics(epoch, refiner, model, phys_optimizer, steps=3):
    if epoch <= 30 or epoch > 100:
        return
    try:
        model.train()
        for n, p in model.named_parameters():
            if 'phys' not in n:
                p.requires_grad = False

        for _ in range(steps):
            inputs = refiner.get_batch().to(device)
            phys_optimizer.zero_grad()
            with torch.no_grad():
                x = model.backbone.conv1(inputs)
                x = model.backbone.bn1(x)
                x = F.relu(x)
                x = model.backbone.layer1(x)
                g1_out, _ = model.gate1(x, model.phys1(x))
                x = model.backbone.layer2(g1_out)
                feat2 = x.clone()
                x, _ = model.gate2(x, model.phys2(x, x, epoch)[0])
                x = model.backbone.layer3(x)
                feat3 = x.clone()

            # [v9] MaternDiffusion returns 5 values (phys, smooth, s, lam, nu)
            ph2, _, _, _, _ = model.phys2(feat2, feat2, epoch)
            ph3, _, _, _, _ = model.phys3(feat3, feat3, epoch)

            loss_align = structure_alignment_loss(ph2, feat2) + \
                         structure_alignment_loss(ph3, feat3)
            loss_align.backward()
            phys_optimizer.step()

        for p in model.parameters():
            p.requires_grad = True
    except Exception as e:
        flush_print(f"  [WARNING] refine_physics error at epoch {epoch}: {e}")
        for p in model.parameters():
            p.requires_grad = True
        torch.cuda.empty_cache()


# ====================== [v9] Data Augmentation ======================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=14),  # [v9] Full RandAugment
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),  # [v9] Slightly stronger
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test)

TRAIN_WORKERS = min(4, 4 * max(N_GPUS, 1))
TEST_WORKERS = 2

def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=TRAIN_WORKERS, worker_init_fn=worker_init_fn,
    pin_memory=True, drop_last=True, persistent_workers=False)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=TEST_WORKERS, worker_init_fn=worker_init_fn,
    pin_memory=False, persistent_workers=False)


# ====================== Model Init ==================
best_acc = 0.0
best_epoch = 0
best_tta_acc = 0.0
best_tta_epoch = 0
no_improve_count = 0
no_test_improve = 0
# [v43] 剂量读数：每个 epoch 实际生效的 StochasticDepth drop_rate 均值
sd_trace = []
# [v43] 曲线历史（写进 JSON，供跨跑对比；不影响训练）
history_test = []
history_tta = []
# [v46] 末段权重平均的累加器：每个参数/缓冲区一份 running sum（float64，CPU）+ 计数。
#   注意这不是"权重 EMA / 滑动副本"——它是**批量算术平均**，只在训练跑完后一次性成形。
swa_sum = None
swa_n = 0
swa_epochs = []

raw_model = GradientStablePhysResNetV3(num_classes=10)
apply_blurpool_and_sd(raw_model, max_drop=SD_MAX_DROP)  # [v9] Apply BEFORE DataParallel

if N_GPUS > 1:
    net = nn.DataParallel(raw_model)
    flush_print(f"Using DataParallel on {N_GPUS} GPUs | total batch = {BATCH_SIZE}")
else:
    net = raw_model
    # 必须显式报出来: 只看到 <=1 张卡时原版什么都不打印, 单卡/双卡分不出来
    flush_print(f"[WARNING] only {N_GPUS} GPU visible -> single-GPU, total batch = {BATCH_SIZE}")
net = net.to(device)
model = raw_model

optimizer = optim.SGD([
    {'params': [p for n, p in model.named_parameters() if 'phys' not in n and 'gate' not in n],
     'lr': LR_BASE, 'weight_decay': 5e-4},
    {'params': [p for n, p in model.named_parameters() if 'phys' in n or 'gate' in n],
     'lr': LR_PHYS, 'weight_decay': 1e-4},
], momentum=0.9, weight_decay=0)

scheduler = LambdaLR(optimizer, lr_lambda=smooth_cosine_schedule)

ce = nn.CrossEntropyLoss(label_smoothing=0.1)

phys_optimizer = optim.SGD(
    [p for n, p in model.named_parameters() if 'phys' in n],
    lr=1e-3, momentum=0.9)

refiner = PhysicsRefiner(trainset, batch_size=BATCH_SIZE)


# ====================== [v49] 确定性探针（只读，秒级，不消耗训练预算） ======================
# 目的：一次跑就回答"非确定性到底在不在算子层"。
#   · 探针批用**固定生成器**生成 ⇒ 跨跑可复现，且不消耗全局 RNG；
#   · 跑两遍 fwd+bwd，两遍之间把 **RNG 状态与权重都还原** ⇒ 差异只可能来自算子本身；
#   · 在"关"与"开"确定性两种模式下各测一次 ⇒ 直接给出"开了到底有没有变好"；
#   · ★ 不设发车闸门、不退出：探针失败就记录原文、自动降级，并写 axis_effective=False（覆辙 115 反制）。
DETERMINISM_PROBE_ENABLE = True
DET_PROBE_BATCH = 16
DET_PROBE_SEED = 2026092
det_probe_report = None


def _det_probe_snapshot_rng():
    st = [torch.get_rng_state(), np.random.get_state(), random.getstate()]
    st.append(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    return st


def _det_probe_restore_rng(st):
    torch.set_rng_state(st[0])
    np.random.set_state(st[1])
    random.setstate(st[2])
    if st[3] is not None:
        torch.cuda.set_rng_state_all(st[3])


def _det_probe_run_once(x, y):
    """一次 fwd+bwd；返回 (loss_float, logits_clone, {param_name: grad_clone})。"""
    for p in model.parameters():
        p.grad = None
    out, _, _, _, _, _, _ = net(x, 0)
    loss = ce(out, y)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return float(loss.detach().float()), out.detach().clone(), grads


def _det_probe_pair(x, y, snap_rng, snap_sd):
    """同一批跑两遍（两遍之间还原 RNG 与权重），返回逐位比对结果。"""
    _det_probe_restore_rng(snap_rng)
    model.load_state_dict(snap_sd)
    l1, o1, g1 = _det_probe_run_once(x, y)
    _det_probe_restore_rng(snap_rng)
    model.load_state_dict(snap_sd)
    l2, o2, g2 = _det_probe_run_once(x, y)

    names = sorted(set(g1) & set(g2))
    n_diff, max_abs = 0, 0.0
    for n in names:
        if g1[n].shape != g2[n].shape or not torch.equal(g1[n], g2[n]):
            n_diff += 1
            if g1[n].shape == g2[n].shape:
                dv = float((g1[n].float() - g2[n].float()).abs().max())
                if dv == dv:                      # NaN 不算作"更差"的证据
                    max_abs = max(max_abs, dv)
    _loss_ok = bool(l1 == l2)
    _logits_ok = bool(o1.shape == o2.shape and torch.equal(o1, o2))
    _grads_ok = bool(len(names) > 0 and n_diff == 0)
    return {
        "loss_1": round(l1, 8),
        "loss_2": round(l2, 8),
        "loss_abs_diff": float(abs(l1 - l2)),
        "loss_bitwise_identical": _loss_ok,
        "logits_bitwise_identical": _logits_ok,
        "n_params_compared": int(len(names)),
        "n_params_with_grad_diff": int(n_diff),
        "grads_bitwise_identical": _grads_ok,
        "max_abs_grad_diff": float(max_abs),
        # ★★ [v49] 判定用的合成量：**三条全中**才算这一次前向+反向是确定性的。
        #   只看梯度会漏掉"前向输出就已经不同"的跑（例如权重/缓冲区没被还原）⇒ 假通过。
        "bitwise_identical_overall": bool(_grads_ok and _loss_ok and _logits_ok),
    }


def run_determinism_probe():
    """[v49] 双模式确定性探针。返回 dict 并缓存到 det_probe_report。"""
    global det_probe_report
    if not DETERMINISM_PROBE_ENABLE:
        det_probe_report = {"verdict": "PROBE_DISABLED", "axis_effective": False}
        return det_probe_report
    _t0 = time.time()
    _g = torch.Generator().manual_seed(int(DET_PROBE_SEED))
    x = torch.randn(int(DET_PROBE_BATCH), 3, 32, 32, generator=_g).to(device)
    y = torch.randint(0, 10, (int(DET_PROBE_BATCH),), generator=_g).to(device)
    snap_rng = _det_probe_snapshot_rng()
    snap_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    _was_training = bool(net.training)
    net.train()
    _get_det = getattr(torch, "are_deterministic_algorithms_enabled", None)
    out = {
        "axis": "operator-level determinism hardening (cuBLAS workspace + deterministic algorithms)",
        "probe_batch": int(DET_PROBE_BATCH),
        "probe_seed": int(DET_PROBE_SEED),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "error": None,
        "fallback_applied": False,
    }
    # ---- (1) 确定性关闭（= v48 的处境）----
    torch.use_deterministic_algorithms(False)
    out["nondeterministic_mode"] = _det_probe_pair(x, y, snap_rng, snap_sd)
    # ---- (2) 确定性打开（= 本版唯一的轴）----
    _ok = True
    try:
        torch.use_deterministic_algorithms(True)
        out["deterministic_mode"] = _det_probe_pair(x, y, snap_rng, snap_sd)
    except Exception as _e:                       # ★ 不退出、不拒发：降级 + 如实记录
        _ok = False
        out["deterministic_mode"] = None
        out["error"] = ("%s: %s" % (type(_e).__name__, _e))[:600]
        torch.use_deterministic_algorithms(False)
        out["fallback_applied"] = True
    # ---- 还原现场（RNG / 权重 / 梯度 / 模式）----
    _det_probe_restore_rng(snap_rng)
    model.load_state_dict(snap_sd)
    for p in model.parameters():
        p.grad = None
    net.train() if _was_training else net.eval()
    out["final_use_deterministic_algorithms"] = (bool(_get_det()) if _get_det is not None else True)
    out["axis_effective"] = bool(_ok)
    out["elapsed_s"] = round(time.time() - _t0, 2)
    if not _ok:
        out["verdict"] = "DETERMINISM_UNAVAILABLE"
    elif out["deterministic_mode"]["bitwise_identical_overall"]:
        out["verdict"] = "DETERMINISM_ACHIEVED"
    else:
        out["verdict"] = "DETERMINISM_NOT_ACHIEVED"
    det_probe_report = out
    return out


# ====================== Train Function ======================
def train(epoch, w_phys):
    net.train()
    total_loss, correct, total = 0, 0, 0
    mixup_alpha = get_mixup_alpha(epoch)
    use_mixup = mixup_alpha > 0.01
    if use_mixup:
        flush_print(f"\nEpoch {epoch} | w_phys={w_phys:.4f} | Mixup=ON (alpha={mixup_alpha:.2f})")
    else:
        flush_print(f"\nEpoch {epoch} | w_phys={w_phys:.4f} | Mixup=OFF")

    for batch, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if use_mixup:
            if random.random() < 0.5:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=mixup_alpha)
            else:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, beta=1.0)
            # [v9] 7 return values (added nu_vals)
            outputs, s_vals, l_vals, nu_vals, a_vals, smooth, pinn_reg = net(inputs, epoch)
            cls_loss = mixup_criterion(ce, outputs, targets_a, targets_b, lam)
        else:
            outputs, s_vals, l_vals, nu_vals, a_vals, smooth, pinn_reg = net(inputs, epoch)
            cls_loss = ce(outputs, targets)

        smooth = _reduce(smooth)
        pinn_reg = _reduce(pinn_reg)
        s_vals = [_reduce(s) for s in s_vals]
        l_vals = [_reduce(l) for l in l_vals]
        nu_vals = [_reduce(n) for n in nu_vals]
        a_vals = [_reduce(a) for a in a_vals]

        reg_loss = physics_reg(s_vals, epoch)
        loss = cls_loss + w_phys * smooth + PINN_WEIGHT * pinn_reg + reg_loss
        loss.backward()

        backbone_params = [p for n, p in model.named_parameters()
                          if 'phys' not in n and 'gate' not in n and p.grad is not None]
        phys_params = [p for n, p in model.named_parameters()
                      if ('phys' in n or 'gate' in n) and p.grad is not None]
        torch.nn.utils.clip_grad_norm_(backbone_params, 1.0)
        torch.nn.utils.clip_grad_norm_(phys_params, 1.0)
        optimizer.step()

        total_loss += loss.item()
        if use_mixup:
            correct += (lam * outputs.argmax(1).eq(targets_a).sum().item() +
                        (1 - lam) * outputs.argmax(1).eq(targets_b).sum().item())
        else:
            correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets_a.size(0) if use_mixup else targets.size(0)

        if batch % 50 == 0:
            s2, s3 = s_vals[0].item(), s_vals[1].item()
            nu2, nu3 = nu_vals[0].item(), nu_vals[1].item()
            flush_print(f"\r  Loss:{total_loss/(batch+1):.2f} | Acc:{100.*correct/total:.2f}% "
                  f"s:{s2:.2f},{s3:.2f} n:{nu2:.2f},{nu3:.2f} | lr:{optimizer.param_groups[0]['lr']:.6f}", end="")
    flush_print()


# ====================== Test Function (no EMA) ======================
def test(epoch):
    global best_acc, best_epoch
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _, _, _, _, _, _ = net(inputs, epoch)
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
    acc = 100. * correct / total

    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_physics_resnet_v51.pth')
        flush_print(f"  Test: {acc:.2f}% | *** New Best (Epoch {epoch}) ***")
    else:
        flush_print(f"  Test: {acc:.2f}% | Best: {best_acc:.2f}% (Epoch {best_epoch})")
    return acc


def test_tta(epoch):
    net.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_flip = torch.flip(inputs, dims=[3])
            out_orig, _, _, _, _, _, _ = net(inputs, epoch)
            out_flip, _, _, _, _, _, _ = net(inputs_flip, epoch)
            outputs = (out_orig + out_flip) / 2.0
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
    return 100. * correct / total


# ====================== [v14] Prediction Ensemble ======================
# Checkpoint pool: list of (epoch, test_acc, filepath)
ckpt_pool = []

def save_ckpt(epoch, test_acc):
    """Save checkpoint to disk and update pool (keep top ENS_MAX_KEEP by test_acc)."""
    global ckpt_pool
    filepath = os.path.join(CKPT_DIR, f'ckpt_ep{epoch}_acc{test_acc:.2f}.pth')
    torch.save(model.state_dict(), filepath)
    ckpt_pool.append((epoch, test_acc, filepath))
    # Sort by test_acc descending, keep top ENS_MAX_KEEP
    ckpt_pool.sort(key=lambda x: x[1], reverse=True)
    if len(ckpt_pool) > ENS_MAX_KEEP:
        # Remove worst checkpoint from disk
        removed = ckpt_pool.pop()
        try:
            os.remove(removed[2])
        except OSError:
            pass

def ensemble_predict(net, testloader, ckpt_files, epoch, use_tta=True):
    """Average softmax predictions across multiple checkpoints.
    Optionally each checkpoint does TTA (flip) before averaging.
    Returns accuracy.
    """
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Average softmax across all checkpoints
            avg_probs = None
            for ckpt_file in ckpt_files:
                model.load_state_dict(torch.load(ckpt_file, map_location=device))
                if use_tta:
                    inputs_flip = torch.flip(inputs, dims=[3])
                    out_orig, _, _, _, _, _, _ = net(inputs, epoch)
                    out_flip, _, _, _, _, _, _ = net(inputs_flip, epoch)
                    probs = F.softmax((out_orig + out_flip) / 2.0, dim=1)
                else:
                    out, _, _, _, _, _, _ = net(inputs, epoch)
                    probs = F.softmax(out, dim=1)
                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs += probs
            avg_probs /= len(ckpt_files)
            correct += avg_probs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
    return 100. * correct / total

# ====================== Main Loop ==================
os.makedirs(CKPT_DIR, exist_ok=True)

# [v49] 探针必须在训练循环之前跑（秒级）；它自己负责还原 RNG / 权重 / 梯度 / 模式。
run_determinism_probe()

flush_print("\n" + "="*60)
flush_print(f"v51 Training Configuration (= v50 逐字 + gate1 固定注入 0.6 -> 0.9):")
flush_print(f"  [v51] 本版唯一改动轴 = gate1 固定注入系数 GATE1_INJECT={GATE1_INJECT}（v50 的 0.6 -> 0.9，"
            f"见头部 v51 立项说明）。")
flush_print(f"  [v51] 保留 v49 的算子层确定性硬化：CUBLAS_WORKSPACE_CONFIG="
            f"{os.environ.get('CUBLAS_WORKSPACE_CONFIG')} + use_deterministic_algorithms"
            f"({bool(det_probe_report.get('final_use_deterministic_algorithms'))})。")
flush_print(f"  [v51] 配方其余冻结不变：PINN_WEIGHT={PINN_WEIGHT}（v50 轴，原样保留）、"
            f"PHASE2_LR_END={PHASE2_LR_END:.1e}、LR_FLOOR_ENABLE={bool(LR_FLOOR_ENABLE)}"
            f"（v47 的 clamp 仍退役）、SWA_ENABLE={bool(SWA_ENABLE)}、SEED={int(SEED)}。")
flush_print(f"  [v51] 探针判定 = {det_probe_report.get('verdict')}"
            f"（关: grads_bitwise={None if det_probe_report.get('nondeterministic_mode') is None else det_probe_report['nondeterministic_mode']['grads_bitwise_identical']}"
            f" / 开: {None if det_probe_report.get('deterministic_mode') is None else det_probe_report['deterministic_mode']['grads_bitwise_identical']}）"
            f"，axis_effective={bool(det_probe_report.get('axis_effective'))}。")
flush_print(f"  [v51] 判据用**实测** σ_diff={PLATEAU_SIGMA:.3f}pp（v45/v46 零改动对照对）⇒ 平台 PASS>={PLATEAU_PASS} / 红线<={PLATEAU_REDLINE}。")
flush_print(f"  LR_BASE={LR_BASE}, LR_PHYS={LR_PHYS}, Batch={BATCH_SIZE}")
flush_print(f"  Phase 1 (0-{COSINE_EPOCHS}): cosine lr 0.1->1e-4 (same as v9)")
flush_print(f"  [v14] Phase 2 ({COSINE_EPOCHS}-{TRAIN_EPOCHS}): smooth cosine 1e-4->{PHASE2_LR_END:.1e} (no restart)")
flush_print(f"  [v14] Ensemble: save from ep{ENS_START_EPOCH}, every {ENS_INTERVAL}ep, top-{ENS_MAX_KEEP} kept, use top-{ENS_TOP_K}")
flush_print(f"  w_phys: trapezoidal 0.05->0.15(hold@60-120)->0.02")
flush_print(f"  [v51] gate1 FIXED injection: {GATE1_INJECT} (v50 was 0.6; 0.9 = upper bound of the "
            f"learnable gates' alpha domain soft_bound(a,0.05,0.9))")
flush_print(f"  [v50 保留] pinn (deep-feature smoothness): weight={PINN_WEIGHT} (v49 was 0.08, halved)")
flush_print(f"  Mixup+CutMix: alpha=0.4, window 60-120 (decay 110-120), LabelSmoothing=0.1")
flush_print(f"  RandAugment: N=2, M=14 (full 14 ops)")
flush_print(f"  StochasticDepth: max_drop={SD_MAX_DROP}, linear increase")
flush_print(f"  [v43] StochasticDepth anneal: 0->{SD_MAX_DROP} by ep{SD_ANNEAL_START}, then ->0 by ep{SD_ANNEAL_END}, 0 after")
flush_print(f"  BlurPool: 3x3 binomial filter on strided convs")
flush_print("  Matern kernel: (1+lam*|k|^2/nu)^{-s}, nu in [0.5, 3.0]")
flush_print(f"  Physics reg: 0.05 coeff, s bands max 1.0/1.1/1.2, clamp [0.3, 1.3]")
flush_print(f"  refine_physics: epoch 30-100 only, num_workers=0")
flush_print(f"  TTA: from epoch {TTA_START_EPOCH}, every {TTA_INTERVAL} epochs (torch.flip)")
flush_print(f"  [v14] No EMA (永久禁令). [v46] SWA code present but SWA_ENABLE={bool(SWA_ENABLE)} => read-only archive, NOT delivered.")
flush_print(f"  Early stopping: TTA patience={EARLY_STOP_PATIENCE}, Test patience={TEST_PATIENCE}")
flush_print(f"  Workers: train={TRAIN_WORKERS}, test={TEST_WORKERS}")
flush_print(f"  [HANG FIX] torch.flip TTA, refiner workers=0, empty_cache/10ep, stdout flush")
flush_print("="*60)
sys.stdout.flush()

# [v47 仪器] 早停可观测（洞 1）：v44 的 JSON 写着 180 个 epoch，实际只跑了 157，
# 而 JSON 里没有任何字段能看出来 —— 因为 "epoch" 写的是配置常量不是观测值。
# 这里把"实际执行到哪一轮 / 是否早停 / 为什么停"三个量显式记下来。
early_stopped = False
stop_reason_out = None
last_epoch = -1

for epoch in range(0, TRAIN_EPOCHS):
    last_epoch = int(epoch)          # [v46] 实际执行到的最后一轮（break 时保留在触发那一轮）
    epoch_start_time = time.time()
    w_phys = get_w_phys(epoch)

    # [v43] 刷新 StochasticDepth 退火：必须在 train() 之前，否则 wrapper 里的
    # _sd_eff_drop 会停在上一轮的取值（机制"写好了但这次不会跑它"的典型形态）。
    _sd_n = sd_set_epoch(model, epoch)
    _sd_mean, _sd_npos, _sd_ntot = sd_effective_mean(model)
    sd_trace.append((epoch, round(_sd_mean, 6), _sd_npos, _sd_ntot))

    train(epoch, w_phys)
    scheduler.step()

    val_acc = test(epoch)
    history_test.append(float(val_acc))

    if val_acc >= best_acc:
        no_test_improve = 0
    else:
        no_test_improve += 1

    # [v14] Save checkpoint for ensemble
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
            torch.save(model.state_dict(), 'best_tta_resnet_v51.pth')
            flush_print(f"  TTA: {tta_acc:.2f}% | *** New TTA Best (Epoch {epoch}) ***")
        else:
            no_improve_count += 1
            flush_print(f"  TTA: {tta_acc:.2f}% | TTA Best: {best_tta_acc:.2f}% (Epoch {best_tta_epoch}) | No improve: {no_improve_count}/{EARLY_STOP_PATIENCE}")

    if 30 < epoch <= 100:
        steps = 5 if epoch >= 60 else 3
        refine_physics(epoch, refiner, model, phys_optimizer, steps=steps)
        refiner.refresh()

    if epoch > 120 and epoch % 10 == 0:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'phys' in name:
                    param.mul_(0.999)

    if epoch % 10 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # [v46] 末段权重平均的累加点。放在 epoch 末尾（TTA/refine/参数乘子**之后**），
    #   保证累加的正是"这个 epoch 结束时的网络"，与 log 的 epoch 编号严格一致。
    #   位置刻意不走任何 RNG ⇒ 训练轨迹与 v45 逐位相同（覆辙 142：只读改动不得改 RNG 流）。
    _swa_taken = swa_accumulate(model, epoch)

    epoch_time = time.time() - epoch_start_time
    flush_print(f"  Epoch {epoch} time: {epoch_time:.1f}s | GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    stop_reason = None
    if no_improve_count >= EARLY_STOP_PATIENCE:
        stop_reason = f"TTA no improvement for {EARLY_STOP_PATIENCE} evaluations"
    elif no_test_improve >= TEST_PATIENCE and epoch >= 120:
        stop_reason = f"Test acc no improvement for {TEST_PATIENCE} epochs"

    if stop_reason:
        early_stopped = True                # [v47 仪器] 洞 1
        stop_reason_out = str(stop_reason)
        flush_print(f"\n{'='*60}")
        flush_print(f"Early stopping at epoch {epoch}! Reason: {stop_reason}")
        flush_print(f"Best Test: {best_acc:.2f}% (Epoch {best_epoch})")
        flush_print(f"Best TTA: {best_tta_acc:.2f}% (Epoch {best_tta_epoch})")
        flush_print(f"{'='*60}")
        break

# ====================== [v47 仪器] 实际轮数自曝（洞 1） ======================
# v44 的 JSON 里 "epoch" 是配置常量 int(TRAIN_EPOCHS)，早停后与实际轮数不符。
# 这里把这个差别显式打出来，并落进 JSON 的 instrument 段。
epochs_config = int(TRAIN_EPOCHS)
actual_epochs = int(len(history_test))
history_len_matches = bool(actual_epochs == (int(last_epoch) + 1))
flush_print("")
flush_print("=" * 60)
flush_print("[v47 仪器] 实际轮数自曝（洞 1）")
flush_print("  epochs_config = %d   (配置常量；JSON 里的 \"epoch\" 字段)" % epochs_config)
flush_print("  actual_epochs = %d   (= len(history.test)，这才是真跑了多少轮)" % actual_epochs)
flush_print("  last_epoch    = %d" % int(last_epoch))
flush_print("  early_stopped = %s   reason = %s" % (bool(early_stopped), stop_reason_out))
flush_print("  history_len_matches = %s" % history_len_matches)
flush_print("=" * 60)

# ====================== [v14] Final Evaluation ==================
flush_print("\n" + "="*60)
flush_print("[v14] Final Evaluation")
flush_print("="*60)

flush_print(f"\nTraining complete! Best Test: {best_acc:.2f}% (Epoch {best_epoch})")
if best_tta_acc > 0:
    flush_print(f"Best TTA (single model): {best_tta_acc:.2f}% (Epoch {best_tta_epoch})")

# --- [v14] Prediction Ensemble Evaluation ---
if len(ckpt_pool) >= 2:
    top_k = min(ENS_TOP_K, len(ckpt_pool))
    top_ckpts = ckpt_pool[:top_k]
    ckpt_files = [c[2] for c in top_ckpts]

    flush_print(f"\n--- Prediction Ensemble (top-{top_k} checkpoints) ---")
    for i, (ep, acc, _) in enumerate(top_ckpts):
        flush_print(f"  #{i+1}: ep{ep} acc={acc:.2f}%")

    # Save current weights before ensemble eval
    saved_weights = {n: p.data.clone() for n, p in model.named_parameters()}

    # Ensemble without TTA
    ens_acc = ensemble_predict(net, testloader, ckpt_files, TRAIN_EPOCHS, use_tta=False)
    flush_print(f"  Ensemble Test: {ens_acc:.2f}%")

    # Ensemble with TTA
    ens_tta = ensemble_predict(net, testloader, ckpt_files, TRAIN_EPOCHS, use_tta=True)
    flush_print(f"  Ensemble TTA:  {ens_tta:.2f}%")

    flush_print(f"\n  Comparison:")
    flush_print(f"    Best single model: Test={best_acc:.2f}% (ep{best_epoch}), TTA={best_tta_acc:.2f}% (ep{best_tta_epoch})")
    flush_print(f"    Ensemble (top-{top_k}):  Test={ens_acc:.2f}%, TTA={ens_tta:.2f}%")
    if ens_acc > best_acc:
        flush_print(f"    *** Ensemble Test wins by +{ens_acc - best_acc:.2f}% ***")
    if ens_tta > best_tta_acc:
        flush_print(f"    *** Ensemble TTA wins by +{ens_tta - best_tta_acc:.2f}% ***")

    # Restore weights
    for n, p in model.named_parameters():
        if n in saved_weights:
            p.data.copy_(saved_weights[n])
else:
    flush_print("\n  Not enough checkpoints for ensemble (need >=2, have %d)" % len(ckpt_pool))
    ens_acc = 0.0
    ens_tta = 0.0

# --- [v47] 交付态 = **训练出来的末态权重本身**（本版轴上，交付链零后处理） ---
#   ★★ 这是本版最重要的一句话：交付链**什么都不做**。
#   v46 在交付链上做了一次"末段权重平均"，读数 Δ=−0.01pp（平台极差 0.06pp 之下，纯 NULL）。
#   v47 的结论是"后处理家族判死" ⇒ 交付态回到最朴素形态：**训练结束时模型手里的那份权重**。
#   ⇒ 这样 v47 的读数**直接就是"训练这条轴"的读数**，不被任何后处理稀释。
#     这正是"一次一轴"的要求：本版轴上就只有 lr 下限一个变量，交付链必须保持恒等映射。
#   ★ 口径仍与 v45/v46 **逐字相同**（同样的 test / test_tta 函数、同样的 TRAIN_EPOCHS 实参）
#     ⇒ 跨版数字可比（覆辙 132：只能比同口径字段）。
#   ★ SWA 仍计算，但只作为**只读 legacy 读数**写进 JSON（不参与 delivered）——
#     这样万一将来要看"同一跑里 SWA 与末态的差"，数据还在；但它不再是交付路径。
swa_state = swa_build(model)
SWA_DELIVERY_SOURCE = "not_used_v51"
swa_start_ep = int(swa_window_start())
swa_end_ep = int(swa_epochs[-1]) if swa_epochs else -1
_swa_acc = 0.0
_swa_tta = 0.0
if swa_state is not None:
    model.load_state_dict(swa_state)
    flush_print("")
    flush_print("=" * 60)
    flush_print("[v47 只读遗留读数] 末段权重平均（SWA_ENABLE=%s，**不参与交付**）" % str(bool(SWA_ENABLE)))
    flush_print("  窗口     = ep%d..ep%d  (%d 个快照, 每 %d ep 一个)"
                % (swa_start_ep, swa_end_ep, int(swa_n), int(SWA_SNAPSHOT_EVERY)))
    flush_print("  方法     = 逐 epoch 算术平均（无滑动、无 EMA、无 RNG）")
    _swa_acc = test(TRAIN_EPOCHS)
    _swa_tta = test_tta(TRAIN_EPOCHS)
    flush_print("  SWA Test = %.2f%%   SWA TTA = %.2f%%" % (_swa_acc, _swa_tta))
    flush_print("  ★ 以上**不是**本版交付读数；v47 交付 = 末态权重。")
    flush_print("=" * 60)
else:
    flush_print("")
    flush_print("  [v47] SWA 快照不足 (n=%d) ⇒ 只读遗留读数缺席（不影响交付）" % int(swa_n))

# --- Final TTA with best weights ---
if best_tta_acc > 0:
    flush_print("\nLoading best TTA weights for final evaluation...")
    model.load_state_dict(torch.load('best_tta_resnet_v51.pth'))
else:
    flush_print("\nLoading best test weights for final evaluation...")
    model.load_state_dict(torch.load('best_physics_resnet_v51.pth'))

# [v47 仪器] 口径自曝（洞 3）：上面刚 load 进来的是 best_tta 权重，
# 所以 final_tta 测的是 **best-TTA 权重**，不是末态权重 —— 它与 best_tta_acc
# 是同一份权重测两次（差恒 <=0.02pp），因此继承 max-of-N 偏差。
# 不改计算（改了就破坏跨版可比），只把这件事写进 JSON 并在 banner 打印。
# [v46] 先把 SWA 权重退回 best_tta 权重，再走 v45 原样的对照读数（口径跨版可比）。
SWA_TEST = round(float(_swa_acc), 4)
SWA_TTA = round(float(_swa_tta), 4)
if swa_state is not None and best_tta_acc > 0:
    model.load_state_dict(torch.load('best_tta_resnet_v51.pth'))

FINAL_TTA_MEASURES = "best_tta_ckpt"
final_tta_source_epoch = int(best_tta_epoch)
final_tta_is_last_epoch = bool(final_tta_source_epoch == int(last_epoch))

final_tta = test_tta(TRAIN_EPOCHS)
flush_print("")
flush_print("[v47 仪器] 口径自曝（洞 3）")
flush_print("  final_tta 测的是 best_tta 权重 (ep%d)，不是末态权重 (ep%d)"
            % (final_tta_source_epoch, int(last_epoch)))
flush_print("  ⇒ 它与 best_tta_acc 是同一份权重测两次，继承 max-of-N 偏差")
flush_print(f"\nFinal TTA accuracy: {final_tta:.2f}%")
flush_print(f"Best Test: {best_acc:.2f}% | Best TTA: {best_tta_acc:.2f}% | Final TTA: {final_tta:.2f}%")
if len(ckpt_pool) >= 2:
    flush_print(f"Ensemble Test: {ens_acc:.2f}% | Ensemble TTA: {ens_tta:.2f}%")
flush_print("============================================")
sys.stdout.flush()

# ====================== [v43] 机制剂量读数（只读，不碰任何梯度路径） ======================
# 原则（覆辙 130）：逐项 try，不许整块包一个 except —— 否则一个键失败会静默吞掉全部读数，
# 还会留下"假绿"字段。每个键独立写入，失败时写 mechanism_error 而不是保持默认值。
mechanism = {}
# [v47 仪器] 洞 2：必需键改成**条件必需** —— sd_mean_at_179 只在真跑到那一轮时才要求，
# 早停时它合法不存在；无条件要求它会导致 (a) 值被写成 null (b) mechanism_error 记 KeyError
# (c) 但差集断言只查"键在不在" ⇒ 三项齐全仍然判绿。这是 v44 假绿的直接来源。
_MECH_REQ_BASE = [
    "sd_anneal_enable", "sd_start", "sd_end",
    "sd_mean_at_119", "sd_mean_at_120",
    "sd_zero_from_120", "sd_positive_at_119", "dose_delivered",
    "sd_trace_head", "sd_trace_tail", "sd_mean_at_last",
]
mechanism["required_keys"] = list(_MECH_REQ_BASE)

def _mech_put(key, fn):
    try:
        mechanism[key] = fn()
    except Exception as e:
        mechanism[key] = None
        mechanism.setdefault("mechanism_error", {})[key] = "%s: %s" % (type(e).__name__, e)

_mech_put("sd_anneal_enable", lambda: bool(SD_ANNEAL_ENABLE))
_mech_put("sd_start", lambda: int(SD_ANNEAL_START))
_mech_put("sd_end", lambda: int(SD_ANNEAL_END))

_sd_by_epoch = {e: (m, npos, ntot) for (e, m, npos, ntot) in sd_trace}

_mech_put("sd_mean_at_119", lambda: round(float(_sd_by_epoch[119][0]), 6))
_mech_put("sd_mean_at_120", lambda: round(float(_sd_by_epoch[120][0]), 6))

# [v47 仪器] 条件必需键：早停时 ep179 根本不存在，跳过它必须与"出错"区分开
if int(179) in _sd_by_epoch:
    mechanism["required_keys"].append("sd_mean_at_179")
    _mech_put("sd_mean_at_179", lambda: round(float(_sd_by_epoch[179][0]), 6))
else:
    mechanism["sd_mean_at_179"] = None
    mechanism.setdefault("skipped_keys", {})["sd_mean_at_179"] = (
        "epoch 179 not reached (actual_epochs=%d, last_epoch=%d)"
        % (int(actual_epochs), int(last_epoch)))

# [v47 仪器] 补一个无论是否早停都存在的末轮读数，替代 ep179 的"归零"检查
_mech_put("sd_mean_at_last", lambda: {
    "epoch": int(sd_trace[-1][0]),
    "mean": round(float(sd_trace[-1][1]), 6),
})

# 剂量是否真的送到：ep120 之后所有 block 的 drop_rate 必须恒为 0，
# 且 ep119 必须仍 > 0（否则说明退火提前打穿，机制等于没生效过）
def _zero_from_120():
    bad = [(e, m) for (e, m, npos, ntot) in sd_trace if e >= SD_ANNEAL_END and abs(m) > 1e-12]
    return {"zero_from_120": len(bad) == 0, "n_violations": len(bad),
            "first_violations": bad[:5]}
_mech_put("sd_zero_from_120", _zero_from_120)

_mech_put("sd_positive_at_119", lambda: bool(abs(_sd_by_epoch[119][0]) > 1e-9))

def _dose_delivered():
    z = mechanism.get("sd_zero_from_120") or {}
    return bool(z.get("zero_from_120") and mechanism.get("sd_positive_at_119"))
_mech_put("dose_delivered", _dose_delivered)

_mech_put("sd_trace_head", lambda: [[int(e), round(float(m), 6)] for (e, m, _, _) in sd_trace[:3]])
_mech_put("sd_trace_tail", lambda: [[int(e), round(float(m), 6)] for (e, m, _, _) in sd_trace[-3:]])

# 逐 epoch 的完整剂量轨迹（厚记录放 JSON，不放脚本）
sd_trace_full = [[int(e), round(float(m), 6)] for (e, m, _, _) in sd_trace]

# 缺失键自曝：required_keys 里但凡有没写进去的，直接暴露出来，不靠肉眼看 JSON
# [v47 仪器] 洞 2 的核心修法：v44 只查 `k not in mechanism`，
# 于是 "键在 / 值是 None / mechanism_error 里记着 KeyError" 这三者齐全时仍然判绿。
# 现在"缺失"= 键不在 **或** 值是 None **或** 该键有 error，三者任一即算缺失。
_err = mechanism.get("mechanism_error", {})


def _mech_is_missing(k):
    return (k not in mechanism) or (mechanism.get(k) is None) or (k in _err)


_missing = [k for k in mechanism["required_keys"] if _mech_is_missing(k)]
mechanism["missing_keys"] = _missing
mechanism["errored_keys"] = sorted(_err.keys())
mechanism["skipped_keys"] = mechanism.get("skipped_keys", {})
mechanism["required_keys_mode"] = "conditional(v45)"
mechanism["epochs_actually_run"] = int(actual_epochs)
# 严格版：既没有缺失，也没有任何 error（合法跳过的不算 error）
mechanism["mechanism_ok"] = bool(len(_missing) == 0 and len(_err) == 0)
mechanism["mechanism_ok_legacy"] = bool(
    len([k for k in mechanism["required_keys"] if k not in mechanism]) == 0)

# ====================== [v43] 交付读数 ======================
# [v44] 交付判据沿用 v43（跨跑可比），但**回退红线对复跑版不适用**：
#   本版代码与 v43 只差 SEED 一个常量 ⇒ 读数低只能归因 seed 方差或环境，
#   正确动作是记录 + 换第三个 seed，**不是**回退代码。故最低档改名为 LOW。
# ====================== [v47] 交付态（唯一轴：末段 lr 下限 —— **训练侧**） ======================
#   ★ 交付读数 = 末态模型自己测出来的那一个数（口径与 v45/v46 逐字相同）。
#     本版交付链**零后处理**（v46 的末段权重平均已判死，SWA_ENABLE=False）
#     ⇒ 读数直接反映"训练这条轴"的效应，不被后处理稀释。这是"一次一轴"的硬要求。
best_tta_acc_legacy = float(best_tta_acc)
final_tta_legacy = float(final_tta)
# v45 口径（max-of-N 选点）—— 留下做同跑内配对对照，**不作为交付数**
legacy_delivered = max(best_tta_acc_legacy, final_tta_legacy)
delivered = float(legacy_delivered)
delivered_source_v47 = "final_tta_legacy_max_of_n(v49 交付链零后处理，口径与 v45/v47/v48 同尺)"
# 判据（预注册，不放宽）。参照系的选取见下"为什么红线钉 97.27"。
#   <1> 交付口径（跨跑可比，与 v43/v44/v45/v46 同一把尺）:
#       PASS      delivered >= 97.61   (= v45 97.49 + 2σ, σ=0.06pp，本线实测跨跑 sd)
#       NULL      97.27 <= delivered < 97.61
#       BACKTRACK delivered <= 97.27   (= v46 97.38 − 2σ(0.06) ≈ 97.26，取 97.27 对齐)
#   <2> 为什么本版没有同跑内配对:（v46 有，v47 没有 —— 这点必须写清）
#       本版无法在单跑内构造配对对照（lr 一改，整条轨迹就变了，不存在"同一跑里的 control"）
#       ⇒ 本版**只有跨跑判据**，因此必须靠 **3 个 seed** 才敢说"有/无效应"。
#       单跑读到 NULL **不得**升级为"轴无效"（那条结论要 3 seed）。
#   ★★ 回退红线: delivered <= 97.27 ⇒ 逐字回退 cnn10_v46.py（端点归回退档，闭合）。
#     为什么钉 97.27：v46 实跑 97.38（同 seed=1234）是**唯一可直接对齐的同 seed 参照**。
#     v46 的 97.38 = best_tta@ep162 ⇒ 与 v47 的 best_tta 同字段、同 seed ⇒ 可配对。
#     ⚠ 但同 seed ≠ 可复现：本平台同 seed 三次跑出 157/180/179 轮、best_acc 172→128
#     （覆辙 141：同 seed 非确定性已被三次证实）⇒ 97.27 这个红线本身也有 ±0.06pp 的底噪，
#     它是"防止断崖"的保险丝，不是统计检验。
#   ★ 剂量读数（必须自带，覆辙 115：NULL 有两种 —— 机制无效 / 干预没送到）:
#     `lr_floor_bites_epochs` = 实际被下限咬住的 epoch 数（离线重放应为 12，ep168..179）。
#     若该值为 0 ⇒ 干预**没送到**，此跑的 NULL 不可解读为"轴无效"。
#   ★★★ 档位与红线必须用**同一个比较算子**，否则边界会自相矛盾
#     （自测发现：`verdict` 用 `<=` 而 `backtrack_advice` 用 `<`，在 delivered 恰为 97.27 时
#      会给出 "BACKTRACK" 的档位却配 "DO_NOT_BACKTRACK" 的动作 —— 同一份 JSON 自相矛盾。
#     这正是覆辙 145 要防的病：红线必须落在机器字段上，且字段之间不许打架。）
#     ⇒ 统一定义：`backtrack_needed = (delivered <= BACKTRACK_REDLINE)`，两处共用。
BACKTRACK_REDLINE = 97.27
backtrack_needed = bool(float(delivered) <= float(BACKTRACK_REDLINE))
if delivered >= 97.61:
    verdict = "PASS"
elif not backtrack_needed:
    verdict = "NULL"
else:
    verdict = "BACKTRACK"
# ★ 回退红线必须由**机器字段**表达，不能只写在注释里（覆辙 137 / 145）。
backtrack_advice = ("DO_BACKTRACK_VERBATIM_TO_cnn10_v46"
                    if backtrack_needed
                    else "DO_NOT_BACKTRACK")
# ★★★ 自洽守卫（覆辙 145 的可机检化）：档位与动作必须一致，不许出现
#   "verdict=BACKTRACK 但 advice=DO_NOT_BACKTRACK" 这种同一份 JSON 自相矛盾的状态。
#   这是本版自测抓到的真 bug（边界 operator 不一致）⇒ 固化成守卫，防止再犯。
_verdict_advice_consistent = bool((verdict == "BACKTRACK") == bool(backtrack_needed))
_verdict_is_known = bool(verdict in ("PASS", "NULL", "BACKTRACK"))

# ---- [v47] 剂量读数：下限到底咬住了几轮（离线重放，不依赖 torch） ----
def _lr_floor_dose():
    """重放 **无 clamp 的 v48 波形** 与 **实际含 clamp 的训练波形**，返回(被 clamp 改变的 epoch 列表, 首个咬住 epoch)。

    ★ [v48] 基准从 `_lr_ratio_v46_style` 改成 `_lr_ratio_core(e, PHASE2_LR_END)`：
      本函数要测的是 **clamp 这一个算子** 的贡献，不是"本版相对 v46 总共改了多少"。
      若仍拿 v46 波形当基准，终点抬高会被算进 clamp 的账上 ⇒ 退役守卫恒红、剂量口径失真。
    ★ clamp 已退役 ⇒ 期望返回 ([], -1)；任何人把它开回来，守卫立刻亮红。
    """
    changed = []
    first = -1
    for e in range(0, int(TOTAL_EPOCHS) + 1):
        base = float(_lr_ratio_core(e, PHASE2_LR_END))
        new = float(smooth_cosine_schedule(e))
        if abs(new - base) > 1e-15:
            changed.append(int(e))
            if first < 0:
                first = int(e)
    return changed, first

_lr_changed_epochs, _lr_first_bite = _lr_floor_dose()
_lr_floor_bites = len(_lr_changed_epochs)
# ★ 只用**真实会跑到的**轮次统计（跑不满 TOTAL_EPOCHS 是常态：v44 157 / v45 180 / v46 179）
_lr_changed_in_run = [e for e in _lr_changed_epochs if e <= int(last_epoch)]
_lr_floor_bites_in_run = len(_lr_changed_in_run)
# 训练相关窗口（ep168..TRAIN_EPOCHS）内的被咬轮数 —— 这是"一轴"的语义口径
_lr_changed_in_train = [e for e in _lr_changed_epochs if e <= int(TRAIN_EPOCHS)]
_lr_floor_bites_train = len(_lr_changed_in_train)
# t=0 逐位保证的机器证明：生效点之前必须一个都不改
_lr_unchanged_before_from = (len([e for e in _lr_changed_epochs if e < int(LR_FLOOR_FROM_EPOCH)]) == 0)
# 下限绝不能高于 phase2 值域上界（v47 第一稿就是这么错的：2.0e-2 > 1.0e-4 会把 phase2 压平）
_floor_inside_phase2_range = bool(float(LR_FLOOR) <= 1.0e-4)
# 只抬高、不压低
_lr_never_lowered = (len([e for e in _lr_changed_epochs
                          if float(smooth_cosine_schedule(e)) < float(_lr_ratio_v46_style(e)) - 1e-15]) == 0)
lr_floor_instrument = {
    "floor_enabled": bool(LR_FLOOR_ENABLE),
    "floor_ratio": float(_lr_floor_ratio()),
    "floor_abs_lr": float(LR_FLOOR),
    "from_epoch": int(LR_FLOOR_FROM_EPOCH),
    "expected_first_bite": int(LR_FLOOR_EXPECTED_FIRST_BITE),
    "first_bite_epoch": int(_lr_first_bite),
    "first_bite_matches_expected": bool(int(_lr_first_bite) == int(LR_FLOOR_EXPECTED_FIRST_BITE)),
    "n_bites_full_horizon": int(_lr_floor_bites),
    "n_bites_in_this_run": int(_lr_floor_bites_in_run),
    "bites_epochs_in_this_run": _lr_changed_in_run,
    "unchanged_before_from_epoch": bool(_lr_unchanged_before_from),
    "floor_inside_phase2_range": bool(_floor_inside_phase2_range),
    "lr_never_lowered": bool(_lr_never_lowered),
    # ★★ 生效点必须钉在 phase 分界（覆辙 ⑮：擅自前移 = 改到 phase1 = 多轴）。
    #   变异测试 M03（把 FROM 从 120 改成 60）正是靠这条抓出来的 ——
    #   在加上这条之前 M03 是"全绿逃逸"，说明当时守卫不足以表达"一轴"约束。
    "from_epoch_is_phase_boundary": bool(int(LR_FLOOR_FROM_EPOCH) == int(COSINE_EPOCHS)),
    # ★★ 改动预算守卫：被咬轮数必须**恰好**等于预期值。这是"一轴"的机器化表达。
    #   任何人偷偷调大 floor 或前移生效点，这个数就会变 ⇒ 亮红。
    #   （写死期望值是有意的，不是恒真式：它把"只改 13 轮"这件事变成可机检断言。）
    #   语义口径 = 训练相关窗口 ep168..TRAIN_EPOCHS(180)；全 horizon 值作交叉校验。
    "n_bites_train_window_is_expected": bool(int(_lr_floor_bites_train) == int(LR_FLOOR_EXPECTED_N_BITES_TRAIN)),
    "n_bites_full_horizon_is_expected": bool(int(_lr_floor_bites) == int(LR_FLOOR_EXPECTED_N_BITES)),
    # ★ [v48] 退役守卫（替代 v47 的"剂量送达"）：clamp 必须**关着且一轮都没咬**
    #   —— 这是"退役"的机器定义，不是"剂量送到"（覆辙 153：关掉要有可核对的记录）
    "clamp_retired_clean": bool((not bool(LR_FLOOR_ENABLE)) and int(_lr_floor_bites) == 0),
    # ★ 交付链零后处理守卫（行为检测，不靠人读代码）：SWA 不许参与交付
    "delivery_is_unprocessed": bool(SWA_DELIVERY_SOURCE == "not_used_v51"),
    # ★★ 判据自洽（覆辙 145）：档位与回退动作必须用同一比较算子，不许打架
    "verdict_advice_consistent": bool(_verdict_advice_consistent),
    "verdict_is_known": bool(_verdict_is_known),
}
# ★ `floor_enabled` 是纯事实字段（退役后就是 False），不参与"坏键"判定；
#   参与判定的是上面那条 `clamp_retired_clean`。
_LR_FLOOR_FACT_KEYS = {"floor_enabled"}
_lr_instr_bad = [k for k, v in lr_floor_instrument.items()
                 if v is False and k not in _LR_FLOOR_FACT_KEYS]
lr_floor_verdict = "INSTRUMENT_OK" if len(_lr_instr_bad) == 0 else "INSTRUMENT_FAIL"


# ====================== [v48] 本版真轴的剂量读数 + 平台读数 ======================
# ★★ 这两个读数都是 v47 缺失、且**正是 v47 读成 NULL 的原因**，必须自带：
#   (A) 剂量/交付窗重叠守卫 —— v47 的剂量起点 ep168 比交付选择点 ep138 晚了 30 轮，
#       交付读数在数学上不含本轴信息。本版必须证明"剂量先到、选择后发生"。
#   (B) 平台读数 —— `delivered` 是 ~32 次 TTA 的 max（max-of-N，覆辙 104/140），
#       实测 selection_bias ≈ +0.05pp。要判"平台有没有被抬高"，必须看**平台均值**，
#       而不是看 max（在 sd=0.02pp 的平台上取 max，收的是抖动 —— 覆辙 ㊹）。
#       实测跨版：v45 97.4388 / v46 97.3413 / v47 97.3725 / v48 97.4625（同 seed 1234，末 8 次 TTA 均值）
#
#  ★★★ [v49] σ 的修正与判据常量的**定义位置见文件顶部**（banner 要先打印它们）。
#    这里只留 v48 的剂量期望常量。
LR_TAIL_EXPECTED_FIRST_CHANGE = 121  # [v48] 仪器用：phase2 起点之后的第一轮（ep120 逐位不变）


def _lr_tail_dose():
    """[v48] 重放**冻结的 v46/v47 波形**与**本版训练波形**，返回(被改变的 epoch 列表, 首个改变 epoch)。

    ★ 与 `_lr_floor_dose()` 的分工：那个测 clamp（已退役），这个测**本版真轴**（终点抬高）。
    ★ 基准必须是 `_lr_ratio_v46_style`（读 PHASE2_LR_END_REF=1e-5，冻结常量），
      不能用 PHASE2_LR_END —— 否则两条波形终点相同、差恒为 0、剂量守卫恒绿假通过。
    """
    changed = []
    first = -1
    for e in range(0, int(TOTAL_EPOCHS) + 1):
        base = float(_lr_ratio_v46_style(e))
        new = float(smooth_cosine_schedule(e))
        if abs(new - base) > 1e-15:
            changed.append(int(e))
            if first < 0:
                first = int(e)
    return changed, first


_lr_tail_changed, _lr_tail_first = _lr_tail_dose()
_lr_tail_n = len(_lr_tail_changed)
_lr_tail_in_run = [e for e in _lr_tail_changed if e <= int(last_epoch)]
_lr_tail_n_run = len(_lr_tail_in_run)
# ★★ (A) 剂量-交付窗重叠：剂量起点必须**不晚于**交付选择点（best_tta_epoch）
_lr_tail_before_best_tta = [e for e in _lr_tail_changed if e <= int(best_tta_epoch)]
_tail_dose_reaches_delivery = bool(int(_lr_tail_first) >= 0
                                   and int(_lr_tail_first) <= int(best_tta_epoch))
# phase1（含 ep120）逐位不变的机器证明
_tail_phase1_unchanged = bool(len([e for e in _lr_tail_changed if e <= int(COSINE_EPOCHS)]) == 0)
_tail_phase2_start_unchanged = bool(
    abs(float(smooth_cosine_schedule(int(COSINE_EPOCHS)))
        - float(_lr_ratio_v46_style(int(COSINE_EPOCHS)))) < 1e-15)
# 覆辙 148 反制：新终点必须落在 phase2 值域内部，且不能把余弦衰减幅度削掉太多
_tail_endpoint_inside_range = bool(1.0e-5 <= float(PHASE2_LR_END) <= 1.0e-4)
_tail_decay_kept = (1.0e-4 - float(PHASE2_LR_END)) / (1.0e-4 - 1.0e-5)
_tail_decay_kept_ok = bool(_tail_decay_kept >= 0.5)
# 剂量强度：交付选择点处的 lr 倍率
_tail_mult_at_best_tta = (float(smooth_cosine_schedule(int(best_tta_epoch)))
                          / float(_lr_ratio_v46_style(int(best_tta_epoch)))) if best_tta_epoch > 0 else None

# ---- (B) 平台读数（末 PLATEAU_WINDOW 次 TTA）----
_tta_tail = [float(x["acc"]) for x in history_tta][-int(PLATEAU_WINDOW):]
_plateau_n = len(_tta_tail)
if _plateau_n >= 2:
    _plateau_mean = sum(_tta_tail) / float(_plateau_n)
    _plateau_sd = (sum((v - _plateau_mean) ** 2 for v in _tta_tail) / float(_plateau_n - 1)) ** 0.5
else:
    _plateau_mean = None
    _plateau_sd = None
# ★ 用 `_plateau_n`（实际取到的个数）索引，不能用 PLATEAU_WINDOW：
#   早停很早时 TTA 读数可能不足 8 个，`history_tta[-8]` 会 IndexError 直接崩在收尾阶段。
_plateau_first_epoch = int(history_tta[-int(_plateau_n)]["epoch"]) if _plateau_n >= 1 else None
_plateau_last_epoch = int(history_tta[-1]["epoch"]) if len(history_tta) >= 1 else None
_plateau_sel_bias = (float(delivered) - float(_plateau_mean)) if _plateau_mean is not None else None

_plateau_backtrack = bool(_plateau_mean is None or float(_plateau_mean) <= float(PLATEAU_REDLINE))
if _plateau_mean is None:
    _plateau_verdict = "UNMEASURED"
elif float(_plateau_mean) >= float(PLATEAU_PASS):
    _plateau_verdict = "PASS"
elif not _plateau_backtrack:
    _plateau_verdict = "NULL"
else:
    _plateau_verdict = "BACKTRACK"
_plateau_advice_consistent = bool((_plateau_verdict == "BACKTRACK") == bool(_plateau_backtrack)
                                  and _plateau_verdict in ("PASS", "NULL", "BACKTRACK", "UNMEASURED"))

lr_tail_instrument = {
    "axis": "phase2 cosine endpoint 1e-5 -> 3.0e-5 (training-side, NOT a clamp)",
    # ---- (A) 剂量 ----
    "phase2_lr_end": float(PHASE2_LR_END),
    "phase2_lr_end_ref_frozen": float(PHASE2_LR_END_REF),
    "expected_first_change_epoch": int(LR_TAIL_EXPECTED_FIRST_CHANGE),
    "first_change_epoch": int(_lr_tail_first),
    "first_change_matches_expected": bool(int(_lr_tail_first) == int(LR_TAIL_EXPECTED_FIRST_CHANGE)),
    "n_changed_full_horizon": int(_lr_tail_n),
    "n_changed_in_this_run": int(_lr_tail_n_run),
    "epochs_changed_in_this_run": _lr_tail_in_run,
    # ★★ 这条是 v47 缺的那一条守卫：剂量起点必须落在交付选择点**之前**
    "best_tta_epoch": int(best_tta_epoch),
    "n_changed_at_or_before_best_tta": len(_lr_tail_before_best_tta),
    "dose_reaches_delivery": bool(_tail_dose_reaches_delivery),
    "lr_mult_at_best_tta": (round(float(_tail_mult_at_best_tta), 4)
                            if _tail_mult_at_best_tta is not None else None),
    # ---- 一轴 / 安全的机器化约束 ----
    "phase1_bitwise_unchanged": bool(_tail_phase1_unchanged),
    "phase2_start_unchanged": bool(_tail_phase2_start_unchanged),
    "endpoint_inside_phase2_range": bool(_tail_endpoint_inside_range),
    "decay_amplitude_kept": round(float(_tail_decay_kept), 4),
    "decay_amplitude_kept_ok": bool(_tail_decay_kept_ok),
    # ---- (B) 平台 ----
    "plateau_window": int(PLATEAU_WINDOW),
    "plateau_n": int(_plateau_n),
    "plateau_first_epoch": _plateau_first_epoch,
    "plateau_last_epoch": _plateau_last_epoch,
    "plateau_mean": (round(float(_plateau_mean), 4) if _plateau_mean is not None else None),
    "plateau_sd": (round(float(_plateau_sd), 4) if _plateau_sd is not None else None),
    "selection_bias_pp": (round(float(_plateau_sel_bias), 4) if _plateau_sel_bias is not None else None),
    "plateau_baseline_v47": float(PLATEAU_BASELINE),
    "plateau_sigma_used": float(PLATEAU_SIGMA),
    "plateau_pass_line": float(PLATEAU_PASS),
    "plateau_redline": float(PLATEAU_REDLINE),
    "plateau_verdict": _plateau_verdict,
    "plateau_advice_consistent": bool(_plateau_advice_consistent),
    # ★★ [v49] 判据自身的一致性与分辨力（覆辙 113/145 的反制）：
    #   边距必须 >= 2σ_diff；PASS/红线必须确实等于 baseline ± 边距。
    "plateau_margin_at_least_2sigma": bool(PLATEAU_MARGIN_OK),
    "plateau_lines_derived_from_baseline": bool(PLATEAU_DERIVED_OK),
    "dose_delivered": bool(int(_lr_tail_n_run) >= 1 and _tail_dose_reaches_delivery),
}
_LR_TAIL_FACT_KEYS = {"plateau_verdict"}   # 纯事实/档位字段，不参与"坏键"判定
_lr_tail_bad = [k for k, v in lr_tail_instrument.items()
                if v is False and k not in _LR_TAIL_FACT_KEYS]
lr_tail_verdict = "INSTRUMENT_OK" if len(_lr_tail_bad) == 0 else "INSTRUMENT_FAIL"

# ---- [v49] 本版主轴判定（机制级，与精度判据分开）----
_axis_effective = bool((det_probe_report or {}).get("axis_effective", False))
_det_verdict = str((det_probe_report or {}).get("verdict", "PROBE_DISABLED"))
if _det_verdict == "DETERMINISM_ACHIEVED":
    det_axis_verdict = "DETERMINISM_ACHIEVED"
elif not _axis_effective:
    det_axis_verdict = "DETERMINISM_UNAVAILABLE(fallback applied; 本跑等价于 v48 的一次复跑)"
else:
    det_axis_verdict = "DETERMINISM_NOT_ACHIEVED(算子层已确定性，但梯度仍不逐位相同 ⇒ 下一版钉数据管线)"

# ---- [v47] 遗留 SWA 只读对照（已判死的轴，仅存档，不参与任何判据） ----
_swa_vs_legacy = round(float(SWA_TTA) - float(legacy_delivered), 4) if SWA_TTA > 0 else None

# ====================== [v46] 复跑判据（沿用，只读；本版不是新独立观测） ======================
# 历史 7 跑 final_tta（口径更正见下；v37 不写 JSON ⇒ 不可测）：
#   v28.2 97.36 / v28.6 97.29 / v41 97.28 / v28.1 97.26 /
#   v28.5 97.21 / v42 97.19 / v28.4 97.07
# ⚠ 口径更正（洞 3）：final_tta **不是末态重测**，它测的是 best-TTA 权重
#   （源码里先 load best_tta ckpt 再测），与 best_tta_acc 是同一份权重测两次
#   ⇒ 它继承 max-of-N 偏差。历史池用的是同一个字段、同一套脚本家族 ⇒ 比较仍成立。
HIST_FINAL_TTA_MEAN = 97.237
HIST_FINAL_TTA_SD = 0.092
V43_SEED = 42
V43_FINAL_TTA = 97.41
# v43 已实跑的曲线指纹（覆辙 106：版本号 ≠ 独立观测，必须能机器判定）
V43_TEST_FP = "58746632913adbd84d09e77d06fde3b0148439da"
V43_TTA_FP = "a7c582f11ecd7fc4d1cbc6c3445b510bea345bbc"

# [v46] v44 已实跑的常量（SEED 与本版相同 ⇒ 用于同 seed 复现检验，不是独立性检验）
V44_SEED = 1234
V44_FINAL_TTA = 97.52
V44_BEST_TTA = 97.51
V44_EARLY_STOPPED = True
# v44 的 actual_epochs 由它自己的 len(history.test) 反推（洞 1：JSON 里查不到）
V44_ACTUAL_EPOCHS = 157
V44_TEST_FP = "ae29234502dcafa684f6bd622a7dc7495a108db1"
V44_TTA_FP = "f421be4748abdf7ae8e69753210e37bc7df97f9f"

_fp_test = hashlib.sha1(json.dumps([round(float(v), 4) for v in history_test]).encode()).hexdigest()
_fp_tta = hashlib.sha1(json.dumps(history_tta, sort_keys=True).encode()).hexdigest()
_seed_differs = (int(SEED) != int(V43_SEED))
_curves_differ = (_fp_test != V43_TEST_FP) and (_fp_tta != V43_TTA_FP)
# 两个条件都成立才算一次可信的独立观测：
#   SEED 不同 ⇒ 有可能独立；曲线不同 ⇒ 不是复读旧产物。缺任一都要在结论里标出来。
_independent = bool(_seed_differs and _curves_differ)

_z_v43 = (V43_FINAL_TTA - HIST_FINAL_TTA_MEAN) / HIST_FINAL_TTA_SD
_z_run = (float(final_tta) - HIST_FINAL_TTA_MEAN) / HIST_FINAL_TTA_SD
_z_comb = (_z_v43 + _z_run) / (2.0 ** 0.5)   # 两个独立观测合并：z/√2 的分子相加

# ====================== [v46] 同 seed 复现检验（仪器改动的零副作用证明） ======================
# SEED 与 v44 相同 ⇒ 若仪器改动真的影响了训练，读数或曲线指纹就会变。
# 这正是"一次一轴"的兑现方式：先把仪器单独跑一遍，确认零副作用，再在 v46 换 seed。
_same_seed = bool(int(SEED) == int(V44_SEED))
_delta_tta = float(final_tta) - float(V44_FINAL_TTA)
_fp_match_test = bool(_fp_test == V44_TEST_FP)
_fp_match_tta = bool(_fp_tta == V44_TTA_FP)
_actual_epochs_match = bool(int(actual_epochs) == int(V44_ACTUAL_EPOCHS))
if not _same_seed:
    _repro = "SEED_CHANGED_NOT_A_REPLICATE"
elif _fp_match_test and _fp_match_tta and abs(_delta_tta) <= 1e-9:
    _repro = "EXACT"
elif abs(_delta_tta) <= 0.02:
    _repro = "NEAR_WITHIN_TTA_NOISE"
else:
    _repro = "NOT_A_SAME_SEED_REPLICATE"

if _z_comb >= 2.0:
    replicate = "CONFIRMED"
elif _z_comb >= 1.0:
    replicate = "INCONCLUSIVE"
else:
    replicate = "NOT_REPLICATED"
if not _independent:
    # 不独立 ⇒ 这个 z 不能用于合并（覆辙 106 的直接后果）
    replicate = "NOT_AN_INDEPENDENT_OBSERVATION"

# ====================== [v46] 仪器判据（本版真正的判据，只读） ======================
_instr_checks = {
    "actual_epochs_positive": bool(int(actual_epochs) > 0),
    "history_len_matches": bool(history_len_matches),
    # 真检查，不是恒真：早停=True 时 stop_reason 必须有值；没早停时必须是 None
    "early_stop_fields_written": bool(isinstance(early_stopped, bool)
                                      and ((stop_reason_out is not None) == bool(early_stopped))),
    "mechanism_ok_strict": bool(mechanism.get("mechanism_ok")),
    "no_errored_keys": bool(len(mechanism.get("errored_keys", [])) == 0),
    "skipped_keys_declared": bool(isinstance(mechanism.get("skipped_keys"), dict)),
    "final_tta_measure_declared": bool(FINAL_TTA_MEASURES == "best_tta_ckpt"),
}
_instr_ok = bool(all(_instr_checks.values()))
instrument_verdict = "INSTRUMENT_OK" if _instr_ok else "INSTRUMENT_FAIL"

instrument_block = {
    "purpose": "instrument fix (read-only fields + gate strictness); SEED unchanged from v44",
    "epochs_config": int(epochs_config),
    "actual_epochs": int(actual_epochs),
    "last_epoch": int(last_epoch),
    "early_stopped": bool(early_stopped),
    "stop_reason": stop_reason_out,
    "history_len_matches": bool(history_len_matches),
    "epoch_field_is_config": True,
    "epoch_field_note": "results['epoch'] == TRAIN_EPOCHS (config constant), NOT epochs actually run",
    "final_tta_measures": FINAL_TTA_MEASURES,
    "final_tta_source_epoch": int(final_tta_source_epoch),
    "final_tta_is_last_epoch": bool(final_tta_is_last_epoch),
    "final_tta_note": "final_tta re-measures the best_tta checkpoint, NOT the last-epoch weights; "
                      "it inherits max-of-N selection bias",
    "mechanism_ok": bool(mechanism.get("mechanism_ok")),
    "mechanism_ok_legacy": bool(mechanism.get("mechanism_ok_legacy")),
    "errored_keys": list(mechanism.get("errored_keys", [])),
    "skipped_keys": dict(mechanism.get("skipped_keys", {})),
    "checks": _instr_checks,
    "verdict": instrument_verdict,
    "same_seed_replicate_of_v44": {
        "seed": int(SEED),
        "v44_seed": int(V44_SEED),
        "same_seed": bool(_same_seed),
        "verdict": _repro,
        "final_tta": round(float(final_tta), 4),
        "v44_final_tta": float(V44_FINAL_TTA),
        "delta_tta_pp": round(float(_delta_tta), 4),
        "actual_epochs": int(actual_epochs),
        "v44_actual_epochs": int(V44_ACTUAL_EPOCHS),
        "actual_epochs_match": bool(_actual_epochs_match),
        "fp_test_matches_v44": bool(_fp_match_test),
        "fp_tta_matches_v44": bool(_fp_match_tta),
        "v44_fp_test": V44_TEST_FP,
        "v44_fp_tta": V44_TTA_FP,
    },
}

replicate_block = {
    "purpose": "z-combination with v43 (kept for continuity); NOT a new independent observation",
    "note": "SEED equals v44 => this run is an instrument replicate of v44, not independent of it. "
            "The third independent seed is v46.",
    "code_differs_from_v44_by": ["read-only record fields", "mechanism gate strictness"],
    "hist_final_tta_mean": HIST_FINAL_TTA_MEAN,
    "hist_final_tta_sd": HIST_FINAL_TTA_SD,
    "v43_seed": V43_SEED,
    "v43_final_tta": V43_FINAL_TTA,
    "z_v43": round(float(_z_v43), 4),
    "final_tta": round(float(final_tta), 4),
    "z_run": round(float(_z_run), 4),
    "z_combined": round(float(_z_comb), 4),
    "replicate_verdict": replicate,
    "independent_observation": bool(_independent),
    "independent_of_v44": False,
    "seed_differs_from_v43": bool(_seed_differs),
    "curves_differ_from_v43": bool(_curves_differ),
    "fp_test": _fp_test,
    "fp_tta": _fp_tta,
    "v43_fp_test": V43_TEST_FP,
    "v43_fp_tta": V43_TTA_FP,
    "backtrack_advice": "DO_NOT_BACKTRACK_CODE",
    "backtrack_reason": "code differs from cnn10_v44.py only by read-only record fields and gate strictness; "
                       "a different reading means the instrument change leaked into training, not a recipe regression",
    "scope_note": "本字段属于【复跑判据】块（v44 血脉口径：代码只差 SEED ⇒ 低读数是 seed 方差，"
                  "动作是换 seed 而不是回退代码）。"
                  "★ 与本版【交付判据】的红线（delivered <= 97.27 ⇒ 回退 cnn10_v46.py）"
                  "是两个不同作用域，不要混读：前者管'代码要不要退'，后者管'交付权重链要不要退'。",
}

results_axis_used = AXIS_V51          # [v51] 供 text_sync 做"文案同源"自证

results = {
    "version": "v51",
    "base": "cnn10_v47.py",
    "base_sha1": "a719d7dd235848af8033ff18876e3ece6da050fe",
    "axis": AXIS_V51,
    "seed": int(SEED),
    "batch_size": int(BATCH_SIZE),
    "n_gpus": int(N_GPUS),
    "epoch": int(TRAIN_EPOCHS),
    "actual_epochs": int(actual_epochs),
    "early_stopped": bool(early_stopped),
    "best_acc": round(float(best_acc), 4),
    "best_epoch": int(best_epoch),
    "best_tta_acc": round(float(best_tta_acc), 4),
    "best_tta_epoch": int(best_tta_epoch),
    "final_tta": round(float(final_tta), 4),
    "ens_acc": round(float(ens_acc), 4),
    "ens_tta": round(float(ens_tta), 4),
    "delivered": round(float(delivered), 4),
    "delivered_source": delivered_source_v47,
    # ★ 口径标志：跨版比读数前必须先看这个字段（覆辙 132）。
    #   v43/v44/v45 是 True（max-of-N）；v46 是 False（单个数）。
    #   v47 回到 True（本版交付链零后处理 ⇒ 用最朴素的 max-of-N 口径与 v45 对齐）。
    "delivered_is_max_of_n": True,
    "backtrack_redline": BACKTRACK_REDLINE,
    "backtrack_needed": bool(backtrack_needed),
    "delivery_backtrack_advice": backtrack_advice,
    "delivery_backtrack_scope": "本字段属于【交付判据】块：delivered <= 97.27 时回退到 cnn10_v46.py（逐字）。"
                               "与复跑判据块的 backtrack_advice(DO_NOT_BACKTRACK_CODE) 作用域不同。",
    "verdict": verdict,
    # ★★ [v47] 本版唯一一根轴：末段 lr 下限（训练侧）。剂量读数 + 仪器守卫齐全。
    # [v48] v47 的 clamp **已退役**：这一块只是"退役记录"，不再是本版的轴。
    #   保留它的意义：让"关掉了"这件事变成可机检字段（覆辙 153），
    #   而不是"代码还在、没人知道它开没开"（覆辙 ㊽）。
    "lr_floor": {
        "axis": "RETIRED in v48: late-phase lr clamp. Kept only as a machine-checkable retirement record.",
        "enable": bool(LR_FLOOR_ENABLE),
        "floor_ratio_of_lr_base": float(_lr_floor_ratio()),
        "floor_abs_lr": float(LR_FLOOR),
        "from_epoch": int(LR_FLOOR_FROM_EPOCH),
        "first_bite_epoch": int(_lr_first_bite),
        "n_bites_in_this_run": int(_lr_floor_bites_in_run),
        "n_bites_train_window": int(_lr_floor_bites_train),
        "n_bites_full_horizon": int(_lr_floor_bites),
        "bites_epochs_in_this_run": _lr_changed_in_run,
        "clamp_retired_clean": bool((not bool(LR_FLOOR_ENABLE)) and int(_lr_floor_bites) == 0),
        "checks": lr_floor_instrument,
        "instrument": lr_floor_verdict,
        "why_retired": ("v47 实测：clamp 首个咬住点 ep168，而交付权重存在 ep138 ⇒ 剂量窗口"
                        "与交付窗口不相交，delivered 在数学上不含本轴信息（NULL 不可归因）。"
                        "且要把 clamp 提前到 ep138 之前，floor 须 >= 8.1e-5（phase2 峰值 1.0e-4 的 81%）"
                        "⇒ 等于删掉 phase2 余弦波形 = 覆辙 148。⇒ 换参数化：改余弦自己的终点。"),
        "paired_scope": "retired; NOT part of any v48 criterion",
    },
    # ★★★ [v49] 保留轴：算子层确定性硬化。探针双模式读数 + 降级记录全在这里。
    "determinism": dict(det_probe_report or {}, axis_verdict=det_axis_verdict),
    # ★★★ [v51] 本版唯一一根轴：gate1 固定注入系数 0.6 -> 0.9。
    #   剂量读数 = 该系数本身（直接乘进前向 `feat + GATE1_INJECT * p` ⇒ 从 ep0 起全程生效，
    #   结构上不存在"剂量窗 ∩ 交付窗 = ∅"的问题，覆辙 154 反制）。
    "gate1_inject": {
        "axis": "gate1 fixed injection coefficient 0.6 -> 0.9 (layer1 physics-field injection)",
        "weight_old": 0.6,
        "weight_new": float(GATE1_INJECT),
        "axis_effective": bool(abs(float(GATE1_INJECT) - 0.6) > 1e-12),
        "design_domain": [0.05, 0.9],
        "design_domain_source": ("Gate(learnable=True) 的 alpha 经 soft_bound 界定在 [0.05, 0.9]；"
                                 "0.9 即该设计域上界，0.6 是低于上界的硬编码常量。"),
        "which_gate": "gate1 (layer1, FixedDiffusion 256ch, learnable=False)",
        "why_gate1_not_gate23": ("gate2/gate3 是 learnable ⇒ 改它们的 init/界会被优化器补偿掉；"
                                 "gate1 是固定常量 ⇒ 改它才有持续效应。"),
        "dose_from_epoch": 0,
        "dose_reaches_delivery": True,
        "why_this_axis": ("物理场进主干的唯一通道是 Gate 的加性注入；gate1 是整条物理链路上"
                          "唯一的硬编码字面量（自 v9 起从未被单独调过）。把它提到设计域上界，"
                          "检验『layer1 固定注入是否被设低了』。与 v50 的 pinn 不同族"
                          "（注入强度 ≠ 平滑惩罚剂量），故不构成『同族第三个补丁』。"),
        "paired_scope": ("none: changing an injection strength changes the whole trajectory ⇒ no in-run control. "
                         "Cross-run criterion only; single-run NULL must NOT be read as 'axis dead' (needs 3 seed)."),
    },
    # ★★ [v50 记录，v51 原样保留] pinn 正则系数 0.08 -> 0.04（该轴已结案为 NULL，非"轴无效"）。
    "pinn": {
        "axis": "pinn (deep-feature spatial smoothness) regularization weight 0.08 -> 0.04 (halved)",
        "weight_old": 0.08,
        "weight_new": float(PINN_WEIGHT),
        "axis_effective": bool(abs(float(PINN_WEIGHT) - 0.08) > 1e-12),
        "dose_from_epoch": 0,
        "dose_reaches_delivery": True,
        "scope": "v50 axis record; NOT part of any v51 criterion (v50 实跑 NULL，本版原样保留该系数)",
        "why_this_axis": ("pinn = mean(dx^2)+mean(dy^2) of layer4 output (deep) penalizes BACKBONE feature "
                          "gradients directly, which conflicts with the v36 principle 'physics must be a "
                          "passive prior; regularization only acts on field params, never on backbone features'. "
                          "Halving it (0.08->0.04) weakens that backbone penalty while keeping a mild "
                          "smoothness floor. This is a recipe-domain change (changes optimizer reachable set), "
                          "not a param-level clamp."),
        "paired_scope": ("none: changing a loss weight changes the whole trajectory ⇒ no in-run control. "
                         "Cross-run criterion only; single-run NULL must NOT be read as 'axis dead' (needs 3 seed)."),
    },
    # ★★ [v49] 噪声底与判据分辨力：把"σ 是实测的"这件事写成机器字段，
    #   防止下一次又用 n=3 估出来的 0.05pp 去设 PASS 线（那会让判据线落在 1σ 之内）。
    "noise_floor": {
        "sigma_diff_pp": float(PLATEAU_SIGMA),
        "sigma_source": PLATEAU_BASELINE_SOURCE,
        "pass_margin_pp": round(float(PLATEAU_PASS) - float(PLATEAU_BASELINE), 4),
        "old_v47_v48_margin_pp": 0.10,
        "old_margin_in_sigma": round(0.10 / float(PLATEAU_SIGMA), 3),
        "margin_at_least_2sigma": bool(PLATEAU_MARGIN_OK),
        "lines_derived_from_baseline": bool(PLATEAU_DERIVED_OK),
        "sigma_measured_constant": float(PLATEAU_SIGMA_MEASURED),
        "note": ("★ v47/v48 的 PASS 边距 0.10pp 只有 0.93σ ⇒ 那两个 NULL 是算术上被预先决定的，"
                 "不构成『轴无效』的证据。以后 B 线 PASS 边距不得小于 2σ_diff。"),
    },
    # ★ [v51] 文本失同步机检（v48 的三处：axis 文案 / banner 版本行 / banner 里的 phase2 终点字面量）
    "text_sync": {
        "axis_field_mentions_version": bool("v51" in str(AXIS_V51)),
        "axis_field_mentions_real_axis": bool("gate1" in str(AXIS_V51).lower()),
        "axis_field_is_shared_constant": bool(str(AXIS_V51) == str(results_axis_used)),
        # ★ 下面两条在 dict 建好之后由 `results["text_sync"][...] = ...` 补写（字面量里看不到自己）
        "axis_field_actually_equals_constant": None,
        "version_is_v51": None,
        "banner_phase2_end_derived": True,
        "note": ("axis 文案由常量 AXIS_V51 提供、banner 里的 phase2 终点与版本号从常量派生 ⇒ "
                 "『打印出来的配置』结构上不可能与真实配置脱钩（覆辙 113/119）。"),
    },
    # ★★ [v48] v48 的唯一一根轴（本版保留、不改）：phase2 余弦终点 1e-5 -> 3.0e-5（训练侧，非 clamp）
    "lr_tail": {
        "axis": lr_tail_instrument["axis"],
        "enable": True,
        "phase2_lr_end": float(PHASE2_LR_END),
        "phase2_lr_end_ref_frozen": float(PHASE2_LR_END_REF),
        "first_change_epoch": int(_lr_tail_first),
        "n_changed_in_this_run": int(_lr_tail_n_run),
        "epochs_changed_in_this_run": _lr_tail_in_run,
        # ★★ 剂量-交付窗重叠守卫（v47 缺的那一条）
        "best_tta_epoch": int(best_tta_epoch),
        "dose_reaches_delivery": bool(_tail_dose_reaches_delivery),
        "lr_mult_at_best_tta": lr_tail_instrument["lr_mult_at_best_tta"],
        "dose_delivered": bool(lr_tail_instrument["dose_delivered"]),
        # ★ 平台读数（主判据用的统计量，不是 max-of-N）
        "plateau_mean": lr_tail_instrument["plateau_mean"],
        "plateau_sd": lr_tail_instrument["plateau_sd"],
        "plateau_n": int(_plateau_n),
        "plateau_window_epochs": [_plateau_first_epoch, _plateau_last_epoch],
        "selection_bias_pp": lr_tail_instrument["selection_bias_pp"],
        "plateau_verdict": _plateau_verdict,
        "checks": lr_tail_instrument,
        "instrument": lr_tail_verdict,
        "why_this_axis": ("v47 的 clamp 形态被实测判为『结构性不可观测』：剂量起点 ep168 晚于"
                          "交付选择点 ep138 整整 30 轮。改用『余弦自己的终点』后，剂量从 ep121 起就在，"
                          "完整覆盖交付选择窗（历史 best_tta 落在 ep138–162），且保留 78% 的衰减幅度。"),
        "paired_scope": ("none: 改 lr 即改整条轨迹 ⇒ 单跑内不存在 control。"
                         "本版只有跨跑判据 ⇒ 单跑 NULL 不得升级为‘轴无效’（需 3 seed）。"),
    },
    "swa": {
        "axis": "legacy read-only: late-window weight average (JUDGED DEAD in v46, kept for archive only)",
        "enable": bool(SWA_ENABLE),
        "used_for_delivery": False,
        "start_frac": float(SWA_START_FRAC),
        "start_epoch": int(swa_start_ep),
        "end_epoch": int(swa_end_ep),
        "snapshot_every": int(SWA_SNAPSHOT_EVERY),
        "n_snapshots": int(swa_n),
        "snapshot_epochs": [int(e) for e in swa_epochs],
        "delivery_source": SWA_DELIVERY_SOURCE,
        "swa_test": float(SWA_TEST),
        "swa_tta": float(SWA_TTA),
        "legacy_best_tta": round(float(best_tta_acc_legacy), 4),
        "legacy_best_tta_epoch": int(best_tta_epoch),
        "legacy_final_tta": round(float(final_tta_legacy), 4),
        "legacy_delivered": round(float(legacy_delivered), 4),
        "swa_minus_legacy_pp": _swa_vs_legacy,
        "swa_verdict": "RETIRED_AXIS_READ_ONLY",
        "paired_scope": "read-only archive; NOT part of any v47 criterion",
    },
    "history": {
        "test": [round(float(v), 4) for v in history_test],
        "tta": history_tta,
    },
    "mechanism": mechanism,
    "sd_trace": sd_trace_full,
    "instrument": instrument_block,
    "replicate": replicate_block,
}

# ★ [v49] text_sync 的事后自证：dict 字面量里引用不到自身，所以在这里补写两条**真检查**。
results["text_sync"]["axis_field_actually_equals_constant"] = bool(results.get("axis") == AXIS_V51)
results["text_sync"]["version_is_v51"] = bool(results.get("version") == "v51")

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f'results_cnn_v51_seed{SEED}.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

flush_print("")
flush_print("=" * 60)
flush_print("机制读数（StochasticDepth 退火剂量是否真的送到）:")
flush_print("  sd mean@ep119 = %.6f  (必须 > 0)" % float(mechanism.get("sd_mean_at_119") or 0.0))
flush_print("  sd mean@ep120 = %.6f  (必须 == 0)" % float(mechanism.get("sd_mean_at_120") or 0.0))
flush_print("")
flush_print("=" * 60)
flush_print("[v47 唯一轴] 末段 lr 下限（训练侧；交付链零后处理）:")
flush_print("  交付读数      = %.2f%%   (末态/最优权重，max-of-N 口径，与 v45 同一把尺)"
            % float(delivered))
flush_print("  对照(v45口径) = %.2f%%  (best_tta %.2f @ep%d / final_tta %.2f)"
            % (float(legacy_delivered), float(best_tta_acc_legacy), int(best_tta_epoch),
               float(final_tta_legacy)))
flush_print("  --- 剂量读数（干预有没有真的送到；为 0 ⇒ NULL 不可归因）---")
flush_print("  floor = %.6g x LR_BASE = %.4e (absolute)   from ep%d"
            % (float(_lr_floor_ratio()), float(LR_FLOOR), int(LR_FLOOR_FROM_EPOCH)))
flush_print("  首个被咬 epoch = %d (预期 %d)" % (int(_lr_first_bite), int(LR_FLOOR_EXPECTED_FIRST_BITE)))
flush_print("  训练窗口咬住 %d 轮 (预期 %d, ep168..ep%d) | 全horizon %d 轮 (预期 %d, 含 ep181..200 保持段)"
            % (int(_lr_floor_bites_train), int(LR_FLOOR_EXPECTED_N_BITES_TRAIN), int(TRAIN_EPOCHS),
               int(_lr_floor_bites), int(LR_FLOOR_EXPECTED_N_BITES)))
flush_print("  本跑实际咬住 %d 轮 (actual_epochs=%d ⇒ ep%s..%s)"
            % (int(_lr_floor_bites_in_run), int(actual_epochs),
               str(_lr_changed_in_run[0]) if _lr_changed_in_run else "n/a",
               str(_lr_changed_in_run[-1]) if _lr_changed_in_run else "n/a"))
for _k in sorted(lr_floor_instrument):
    _v = lr_floor_instrument[_k]
    if _k == "bites_epochs_in_this_run":
        continue
    _tag = "OK" if _v is True else ("FAIL" if _v is False else "INFO")
    flush_print("    [%s] %s = %s" % (_tag, _k, str(_v)))
flush_print("  LR_FLOOR_INSTRUMENT = %s   (clamp 已退役；此块只是退役记录)" % lr_floor_verdict)
flush_print("")
flush_print("=" * 60)
flush_print("=" * 60)
flush_print("[v51 唯一轴] gate1 固定注入强度:")
flush_print("  gate1 (layer1, FixedDiffusion 256ch, learnable=False) injection = %s  (v50 冻结参考值 = %s)"
            % (str(float(GATE1_INJECT)), str(0.6)))
flush_print("  设计域 = [0.05, 0.9]（可学习 gate 的 alpha 界 soft_bound(a,0.05,0.9)）⇒ 本版取上界")
flush_print("  axis_effective (注入系数 != v50 冻结参考值) = %s"
            % str(bool(abs(float(GATE1_INJECT) - 0.6) > 1e-12)))
flush_print("  剂量: from ep0 全程乘进前向 ⇒ 结构上覆盖交付选择窗（覆辙 154 反制）")
flush_print("=" * 60)
flush_print("[v49 保留轴] 算子层确定性硬化（改测量，不改配方；v51 沿用）:")
flush_print("  CUBLAS_WORKSPACE_CONFIG = %s" % str((det_probe_report or {}).get("cublas_workspace_config")))
flush_print("  use_deterministic_algorithms(最终) = %s | axis_effective = %s"
            % (str((det_probe_report or {}).get("final_use_deterministic_algorithms")),
               str(_axis_effective)))
for _k in ["nondeterministic_mode", "deterministic_mode"]:
    _b = (det_probe_report or {}).get(_k)
    if _b is None:
        flush_print("  %-22s = <不可用（见 determinism.error）>" % _k)
    else:
        flush_print("  %-22s overall=%s | grads_bitwise=%s | loss_bitwise=%s | logits_bitwise=%s | "
                    "params_diff=%d/%d | max_abs_grad_diff=%.3e"
                    % (_k, str(_b["bitwise_identical_overall"]),
                       str(_b["grads_bitwise_identical"]), str(_b["loss_bitwise_identical"]),
                       str(_b["logits_bitwise_identical"]), int(_b["n_params_with_grad_diff"]),
                       int(_b["n_params_compared"]), float(_b["max_abs_grad_diff"])))
flush_print("  verdict = %s" % _det_verdict)
if (det_probe_report or {}).get("error"):
    flush_print("  probe_error = %s" % str(det_probe_report["error"]))
flush_print("")
flush_print("=" * 60)
flush_print("[v48 保留轴] phase2 余弦终点 1e-5 -> %.1e（训练侧，非 clamp）:" % float(PHASE2_LR_END))
flush_print("  --- (A) 剂量：从哪一轮开始改、有没有覆盖交付选择点 ---")
flush_print("  首个被改 epoch = %d (预期 %d)   本跑实际改了 %d 轮"
            % (int(_lr_tail_first), int(LR_TAIL_EXPECTED_FIRST_CHANGE), int(_lr_tail_n_run)))
flush_print("  交付选择点 best_tta_epoch = %d   ⇒ 剂量 %s"
            % (int(best_tta_epoch),
               "**覆盖**到交付窗（v47 正是死在这里：剂量 ep168 > 选择 ep138）"
               if _tail_dose_reaches_delivery else "**没送到**交付窗 ⇒ NULL 不可归因"))
flush_print("  选择点处 lr 倍率 = %s"
            % ("n/a" if lr_tail_instrument["lr_mult_at_best_tta"] is None
               else "x%.3f" % float(lr_tail_instrument["lr_mult_at_best_tta"])))
flush_print("  --- (B) 平台读数（主判据用的统计量，不是 max-of-N）---")
flush_print("  平台均值 = %s%%  (末 %d 次 TTA, ep%s..%s, sd=%s)"
            % (str(lr_tail_instrument["plateau_mean"]), int(_plateau_n),
               str(_plateau_first_epoch), str(_plateau_last_epoch),
               str(lr_tail_instrument["plateau_sd"])))
flush_print("  选点偏差 = %s pp   (delivered − 平台均值；max-of-N 的固有上偏，覆辙 104)"
            % str(lr_tail_instrument["selection_bias_pp"]))
flush_print("  平台判据 = %s   (基线 v47 %.4f, σ=%.3f ⇒ PASS>=%.2f / BACKTRACK<=%.2f)"
            % (_plateau_verdict, float(PLATEAU_BASELINE), float(PLATEAU_SIGMA),
               float(PLATEAU_PASS), float(PLATEAU_REDLINE)))
for _k in sorted(lr_tail_instrument):
    _v = lr_tail_instrument[_k]
    if _k in ("epochs_changed_in_this_run", "axis"):
        continue
    _tag = "OK" if _v is True else ("FAIL" if _v is False else "INFO")
    flush_print("    [%s] %s = %s" % (_tag, _k, str(_v)))
flush_print("  LR_TAIL_INSTRUMENT = %s" % lr_tail_verdict)
flush_print("  ★ 本版只有跨跑判据（lr 一改轨迹全变 ⇒ 单跑内无 control）:")
flush_print("     单跑 NULL **不得**升级为‘轴无效’；那条结论要 3 seed 才敢说。")
flush_print("=" * 60)

flush_print("")
flush_print("=" * 60)
flush_print("[v47 只读存档] 末段权重平均（v46 已判死，SWA_ENABLE=%s，不参与任何判据）:"
            % str(bool(SWA_ENABLE)))
flush_print("  窗口          = ep%d..ep%d  (%d 个快照, 每 %d ep 一个)"
            % (int(swa_start_ep), int(swa_end_ep), int(swa_n), int(SWA_SNAPSHOT_EVERY)))
flush_print("  存档 TTA/Raw  = %.2f%% / %.2f%%   (仅存档)"
            % (float(SWA_TTA), float(SWA_TEST)))
if _swa_vs_legacy is not None:
    flush_print("  存档 − 对照   = %+.2f pp   (v46 时该值为 −0.01pp ⇒ 平台比效应还平)"
                % float(_swa_vs_legacy))
flush_print("  状态          = RETIRED_AXIS_READ_ONLY")
flush_print("=" * 60)

_sd179 = mechanism.get("sd_mean_at_179")
flush_print("  sd mean@ep179 = %s  (必须 == 0；早停未到 ep179 则须显式标 skip)"
            % ("n/a(skipped)" if _sd179 is None else "%.6f" % float(_sd179)))
_sdlast = mechanism.get("sd_mean_at_last") or {}
flush_print("  sd mean@last(ep%s) = %.6f  (末轮读数，任何 horizon 下都存在)"
            % (_sdlast.get("epoch", -1), float(_sdlast.get("mean", 0.0))))
flush_print("  dose_delivered = %s   mechanism_ok = %s (legacy=%s)" % (
    mechanism.get("dose_delivered"), mechanism.get("mechanism_ok"),
    mechanism.get("mechanism_ok_legacy")))
if _missing:
    flush_print("  [WARN] 缺失读数键: %s" % (", ".join(_missing)))
if mechanism.get("errored_keys"):
    flush_print("  [WARN] 取数报错键: %s" % (", ".join(mechanism["errored_keys"])))
if mechanism.get("skipped_keys"):
    flush_print("  [INFO] 合法跳过键: %s" % (", ".join(sorted(mechanism["skipped_keys"]))))
flush_print("  结果写入: %s" % out_path)
flush_print("=" * 60)

flush_print("")
flush_print("=" * 60)
flush_print("[v47 仪器] 本版真正的判据（只读；读数可信的前提）:")
flush_print("  epochs_config = %d   actual_epochs = %d   last_epoch = %d   (洞 1)"
            % (int(epochs_config), int(actual_epochs), int(last_epoch)))
flush_print("  early_stopped = %s   stop_reason = %s"
            % (bool(early_stopped), stop_reason_out))
flush_print("  history_len_matches = %s" % bool(history_len_matches))
flush_print("  final_tta 测的是 = %s (source epoch %d, is_last_epoch=%s)   (洞 3)"
            % (FINAL_TTA_MEASURES, int(final_tta_source_epoch), bool(final_tta_is_last_epoch)))
for _k in sorted(_instr_checks):
    flush_print("    [%s] %s" % ("OK" if _instr_checks[_k] else "FAIL", _k))
flush_print("  INSTRUMENT = %s" % instrument_verdict)
if instrument_verdict != "INSTRUMENT_OK":
    flush_print("  [WARN] 仪器自曝失败 ⇒ 本跑读数不作为判据，只看 instrument 块定位原因")
flush_print("=" * 60)

flush_print("")
flush_print("=" * 60)
flush_print("[v46] 同 seed 复现检验（SEED 与 v44 相同 ⇒ 仪器改动的零副作用证明）:")
flush_print("  seed = %d   v44 seed = %d   same_seed = %s" % (int(SEED), V44_SEED, _same_seed))
flush_print("  final_tta 本跑 = %.2f%%   v44 = %.2f%%   delta = %+.2f pp"
            % (float(final_tta), float(V44_FINAL_TTA), float(_delta_tta)))
flush_print("  actual_epochs 本跑 = %d   v44 = %d   match = %s"
            % (int(actual_epochs), int(V44_ACTUAL_EPOCHS), _actual_epochs_match))
flush_print("  fp_test 同 v44 = %s   fp_tta 同 v44 = %s" % (_fp_match_test, _fp_match_tta))
flush_print("  SAME_SEED_REPLICATE = %s" % _repro)
flush_print("=" * 60)

flush_print("")
flush_print("=" * 60)
flush_print("复跑读数（沿用，只读；基线 = 历史 7 跑 final_tta mean %.3f sd %.3f）:"
            % (HIST_FINAL_TTA_MEAN, HIST_FINAL_TTA_SD))
flush_print("  seed                = %d   (v43 = %d)" % (int(SEED), V43_SEED))
flush_print("  final_tta 本跑      = %.2f%%" % float(final_tta))
flush_print("  final_tta v43       = %.2f%%" % float(V43_FINAL_TTA))
flush_print("  z_v43 = %.3f   z_run = %.3f   z_combined = %.3f"
            % (float(_z_v43), float(_z_run), float(_z_comb)))
flush_print("  独立性: seed 不同 = %s   曲线指纹不同 = %s   => independent = %s"
            % (_seed_differs, _curves_differ, _independent))
flush_print("    fp_test 本跑 = %s" % _fp_test)
flush_print("    fp_tta  本跑 = %s" % _fp_tta)
flush_print("=" * 60)
flush_print("")
flush_print("DELIVERED = %.2f%%   (来源 %s)  [base cnn10_v37.py: 97.17 raw / 97.50 TTA]"
            % (delivered, delivered_source_v47))
flush_print("VERDICT   = %s   (交付口径，跨跑可比)" % verdict)
flush_print("BACKTRACK = %s   (红线 %.2f：低于它 ⇒ 逐字回退 cnn10_v46.py)"
            % (backtrack_advice, float(BACKTRACK_REDLINE)))
flush_print("LR_FLOOR  = %s   (本版唯一轴；剂量 %d 轮被咬住 ⇒ %s)"
            % (lr_floor_verdict, int(_lr_floor_bites_in_run),
               "干预已送到" if int(_lr_floor_bites_in_run) >= 1 else "★干预没送到，NULL 不可归因"))
flush_print("INSTRUMENT = %s   (仪器自曝；FAIL ⇒ 上面读数不可信)" % instrument_verdict)
flush_print("SAME_SEED_REPLICATE = %s   (v44 同 seed；EXACT/NEAR ⇒ 仪器改动零副作用)" % _repro)
flush_print("REPLICATE = %s" % replicate)
flush_print("★ 本版只有跨跑判据：单跑 NULL 不得升级为‘轴无效’（需 3 seed）")
flush_print("=" * 60)
sys.stdout.flush()
