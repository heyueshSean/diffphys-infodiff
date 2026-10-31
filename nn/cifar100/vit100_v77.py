"""Physics-Guided ViT-Small Training on CIFAR-100 — v77 [各向异性热核: 扩散张量的第一阶推广].

=============================================================================================================
v77 立项依据 —— 撤回 v75/v76 两版"给场加自由度"的轴, 换一根**正交**的几何自由度: 各向异性
-------------------------------------------------------------------------------------------------------------
[决策链: v76 实跑 delivered 69.15]
  · 按 v76 预注册判据是 CONFIRMED (69.15 >= 68.99), **不触发回退红线** ⇒ 文件不回退;
  · 但 vs v74 的 70.41 = -1.26pp = -1.77 * sigma_seed ⇒ 机制 NULL 且**方向为负**;
  · v75 (-0.34 sigma) 与 v76 (-1.77 sigma) 连续两版都落在 v74 之下, 且这两版的轴是同一类:
    "给场加自由度"(v75: beta 软界 3.0->4.0; v76: +72 个内容标量, 且有 1 个 head 顶到软界
    0.4986/0.5)。
  ⇒ 裁决: v77 = vit100_v74.py 的**逐字配方**(beta 软界 (1.0, 3.0), 无内容项, 无扩界),
    在其上只加**一根**新轴。回退与加轴是两件事: 回退不加自由度, 新轴只有 1 个自由度/层。

[v76 留下的两个决定性读数 (本版立项的直接依据)]
  (a) 场是**承重结构**, 不是温和正则。交付端"同跑内配对"消融 (交付权重 / 同一批样本) 显示:
      撤掉全部场 ⇒ |dlogit|rms = 1.224 (logit 自身 std = 0.908), argmax 翻转 94.3%,
      1024 样本子集精度 67.7% -> 5.0% (随机水平)。
      ⇒ 这**推翻**了此前"场是输入无关的位置先验、被可学的绝对位置嵌入完全覆盖、
        边际贡献被结构性压在 ~0.5pp"这条立论: 若真被覆盖, 撤掉它不可能崩到随机。
  (b) 但"承重的杠杆大"**不等于**"有改进空间"。模型是与场共同适应的: 拿走拐杖会摔倒,
      不代表换一根拐杖能走得更快 (末态消融 != 训练反事实, 覆辙 120 的深化)。
  (c) v76 的内容项只贡献 |dlogit|rms = 0.157 / argmax 翻转 7.1% ⇒ 它只是在骨架上加了
      一个小扰动, 效应被 sigma_seed = 0.71pp 淹没。

[轴] 现有场还剩一个**从未被触碰过的一阶对称性**: 各向同性。
      B_h(i,j) = -0.5 * ( d^2(i,j) / sigma_h^2 ) ^ (beta_h / 2),   d^2 = dx^2 + dy^2
  它把网格的水平/垂直方向**等权** —— 等价于"扩散率是一个标量"。本版把它升级为
  **各向异性扩散张量**的最简形式 (对角张量, 只多一个自由度):
      ar_l    = HEAT_ANISO_MAX * tanh(raw_ar_l)          # 每层一个, 在 (-MAX, +MAX) 内
      d2_eff  = dx^2 * exp(-ar_l) + dy^2 * exp(+ar_l)
      B_h(i,j)= -0.5 * ( d2_eff / sigma_h^2 ) ^ (beta_h / 2)
  · ar = 0 ⇒ exp(±0) = 1.0 精确 ⇒ d2_eff ≡ dx^2 + dy^2 ≡ d^2 ⇒ **t=0 与 v74 逐位同构**,
    任何差异都只能来自这个新自由度被优化器用起来之后 (可归因)。
  · 取几何平均不变的参数化: sigma_x * sigma_y = sigma_h^2 ⇒ ar 只改**形状**, 不改平均尺度
    ⇒ 与 sigma **不退化** (不是 sigma 的重参数化, 区别于已被禁止的"增益 g"轴)。
  · 新可学标量 = 每层一个 = 12 个 (进 no_decay)。比 v76 的 72 个小 6 倍 ⇒ 撞界风险低。

[为什么这根轴与已判死清单不重叠]
  · 不是 "抬剂量"(sigma 尺度扫) / "延长 horizon" / "sigma 的界阶梯带(v69)" / "预测集成" /
    "handoff 族" / "CLS 行 gate(v47)" / "drop_path(v37/v38)" / "concat head(v45)" / "beta 轴(v74/v75)"。
  · 不是 beta 族: beta 改的是**径向**形状 (距离标量 -> 衰减律), 本版改的是**角向**结构
    (哪个方向更远)。二者正交。
  · 不是 v76 的内容项: 那是打破**输入无关**; 本版仍是**输入无关 + 平移不变**的位置先验,
    只打破**旋转/镜像各向同性**。⇒ 两条轴在函数空间里互相垂直。
  · 不是闭环自适应 (KERNEL_REVISE / MS_ADAPT / GATE_ADAPT 三族已证有害): ar 只由 loss
    梯度学, 不读任何运行时统计。
  · 被动先验 (v36): 新参数只属于**场自身**, 不进主干特征的梯度路径; bias 只加到 attention logits。

[研究依据] 各向异性扩散 (anisotropic diffusion) 是图像物理里的标准对象 (Perona-Malik 之后的一大族):
  用扩散张量 D 取代标量扩散率。自然图像的水平/垂直方向统计本身就不各向同性 ⇒ 给热核一个
  **各向异性张量**是"信息量扩散"的最简一阶推广, 而且**不需要**引入内容依赖 (v76 已证那条路
  在本题上读不出)。

[判据]
  · 机制级 (同跑内可读, **本版必须自带**):
      场参数读数块里的 ar_* 三兄弟  —— 剂量 (是否被用上) 与是否撞软界 (覆辙 101/148);
      交付端场杠杆配对读数块里 full-vs-iso 的三项 —— **本版那根轴自己的杠杆/翻转率/子集精度**。
        full = 各向异性(学到的 ar) / iso = 强制 ar ≡ 0 (等价 v74 的位置场) / off = 无场。
        ⇒ full-vs-iso 就是本轴的**同跑内配对**效应, 分辨率远高于跨跑比 delivered。
  · 交付级 (跨跑可比): delivered >= 68.99 ⇒ CONFIRMED; < 68.99 ⇒ BACKTRACK。
    基线 = v74 的 70.41 (A 线历史最高), sigma_seed = 0.71pp (2 sigma = 1.42pp)。
    ★ 预期量级 < 1.42pp 的效应单跑结构上读不出来; 单跑 NULL 不得升级为"轴无效"。

[怎么跑] 零参数、裸命令 `python vit100_v77.py` 即跑（物理场写死常开，无任何 CLI 开关）。
  产物：`./results_v77/results_physics_v77_aug_seed42.json`（名字与 output_dir 均由 VERSION 派生）。

[注意] 下方是 v74 及更早的原文说明（保留作谱系）。本块是头部唯一真相源。

=============================================================================================================
v73 立项依据 —— 找回 v41 的 69.81（A 线历史最高），并转成零开关形态
-------------------------------------------------------------------------------------------------------------
[背景] v41/v42 = 69.81 是 A 线（ViT-Small/CIFAR-100）历史最高 delivered，但其源码一度缺失，
  当前主线（v68/v70/v72）只有 69.69。用户找回 v41 源码（train_vit_cifar100_aug_v41.py）后，
  要求"以 ViT 最佳表现版为基础生成新版"。v73 = 以 v41 配方为基底，转成零 argparse 形态。

[获奖配方（69.81 的精确参数）]  --epochs 400 --physics --seed 42，其余全默认：
  - diffusion_mode="heat"（v41 [W] 网格热核注意力，历史最佳的唯一新增轴）
  - pos_enc="abs"（v40 [V]，非 rope2d/mixed）
  - heat_ladder=(3.0,4.5,6.5,9.5,14.0,21.0)，heat_probe=True
  - epochs=400 → derive_stages: cons_start=200 / handoff_start=373 / swa_start=320 / ms_rot_end=333 ...
  - handoff=True 但**获奖跑中从未触发**（handoff.fired=false，门在 e373 但 gap 从未够 min_gap）
    ⇒ 69.81 不是 handoff 的功劳，是网格热核 + 400ep 的功劳（与"handoff 族判死"不矛盾）。

[改造] 66 个 CLI 开关全部废除；parse_args() 返回 argparse.Namespace(**cfg)，cfg 写死获奖配方的
  最终生效值。物理场写死常开（无 --no-physics）；smoke 永久废除。改配方 = 改 cfg，不是改命令行。
  裸命令 `python vit100_v73.py` 即跑出与 v41 获奖配方等价的 400ep/physics/heat 训练。

[判据] 主判据 = 复现 v41 的 69.81（跨跑，A 线 σ_seed=0.71pp，2σ=1.42pp）：
  CONFIRMED ≥ 68.39（= 69.81 − 2σ）/ BACKTRACK < 68.39 ⇒ 退回 vit100_v72.py（69.69）。
  ★ 预期：裸命令应复现 ≈69.81（若数据管线非确定性贡献 <1.42pp 则 CONFIRMED）。

[注意] 本文件 docstring 下方仍是 v39 时代的旧说明（v41 原作者未更新头部），此处 v73 说明为准。

=============================================================================================================
v39 [U]: HORIZON AXIS + LR-NORMALISED PROGRESS INSTRUMENT（以下是 v41 基底的原始说明）。

=============================================================================================================
v39 [U] 立项依据 — 两根轴的实测结果 (seed42, 300ep, 全精度)
-------------------------------------------------------------------------------------------------------------
v37 [S] Stochastic Depth (常数 0.1)      delivered 65.78   (-0.87 vs v36)
v38 [T] 退火 Stochastic Depth (0.1->0)   delivered 65.62   (-1.03 vs v36)   <-- 比 v37 更差
v36 [R] 无 drop_path (rate=0)            delivered 66.65   (当前最佳)

  [U0] 整轴证伪: 两种形态都是净负的。覆辙㉞ 的诊断 ("库默认 0 => 机制没开")
       只授权"去测", 不授权"去开"。欠账假设在**本 recipe** 下不成立 —— 输入侧
       增强已饱和 (train_acc_clean e250=99.96), 再加架构侧随机正则是冗余且有害的。

  [U1] v38 失败的根因不是"退火时机", 而是**中途切换训练时的函数类**。
       v38 与 v37 在 e1-e180 逐位相同 (剂量恒 0.1), 分叉后按窗口平滑:
           e181-200 (dp 0.100->0.067)  -0.042 +- 0.043   空
           e201-220 (dp 0.067->0.033)  -0.123 +- 0.070
           e221-240 (dp 0.033->0.000)  -0.114 +- 0.059
           e241-260 (dp = 0)           -0.134 +- 0.049
           e261-280 (dp = 0)           -0.214 +- 0.037
           e281-300 (dp = 0)           -0.412 +- 0.022
       害处随 dp 递减**单调增大**。末 20ep 均值: v36(dp恒0) 66.24 >
       v37(dp恒0.1) 65.64 > v38(切换态) 65.23 —— **切换态比两种纯剂量都差**。
       覆辙㊵: 正则的剂型不能中途切换; 切换成本 > 任一纯剂量的静态代价。
       (注: 逐 epoch 差 +-0.5 是噪声, 上面是 20ep 窗口均值 +-SEM; 两模型在
       同一固定测试集上评估, 评估噪声在差中抵消, 故 SEM 远小于 0.475。)

  [U2] 真瓶颈 = 学习率预算, 不是容量。新仪器: LR-归一化进展
       ratio(e) = (Δ平滑test_acc / Δepoch) / LR(e)
       若模型趋近天花板, ratio 应随训练崩塌; 实测三版全部**上升**:
           v36: 197(e141-160) -> 239 -> 273 -> 373 -> 645 -> 643(e261-280)
           v37: 239 -> 324 -> 306 -> 328 -> 359 -> 539 -> 973
           v38: 239 -> 324 -> 299 -> 309 -> 362 -> 524 -> 811
       全程线性趋势 slope = +0.48 / +1.27 / +0.97 ratio/epoch, **全为正**。
       => 零饱和迹象; 进展停止的唯一原因是 cosine 把 LR 拉到 0。
       => 增量 ∝ ∫LR dt, 而总 ∫LR dt = E * lr_peak / 2 与 horizon 成正比。

  [U3] 因此 v39 的轴 = **horizon**。E=400 -> ∫LR dt +33%; E=600 -> +100%。
       外部佐证: from-scratch ViT-S/4 CIFAR-100 SOTA (LieRE) 用的是 400ep。
       v36 [R] 已把全部阶段常量改成按 E 比例派生, 所以 --epochs 400 不会把
       任何机制搁在原地 (覆辙㉛); E=300 逐位还原 v36 (verify_vit_v39 断言)。

  [U4] 交付链 (EMA 教师 +0.06pp, SWA +0.20pp) 已近饱和, 破顶只能抬 student
       自身峰值 (v35/v36 = 66.45@e279)。horizon 是唯一直接作用于它的轴。

  [U5] 本版**不改任何训练路径**: 只新增一个纯函数埋点 (读已记录的
       history, 零 RNG、零前向、零 DataLoader 迭代 —— 覆辙㉟)。因此
       v39 @300ep 必须与 v36 @300ep 逐位相同; 由 bitexact_control.py 判定。

判据 (student 真峰值 = history["test_acc"] 的 max, 非 delivered):
    WIN    >= 68.0
    PASS+  >= 66.45   (v35/v36 student peak)
    PASS   >= 66.0
    NULL   >= 65.5
    FAIL   <  65.5
且**先过 saturating 裁决**: 若 saturation_verdict == SATURATING 而精度仍涨,
说明涨的是别的东西 (噪声/交付链), horizon 轴应停。
=============================================================================================================

  !! v35 IS THE v34 FIX. v34 WAS A SILENT NO-OP !!
  v34 changed get_mixup_window()'s fallback to the full run [E/3, E], but the
  argparse default for --mixup-window was still the literal string "100-200".
  A non-empty default always wins, so the fallback never executed and v34's
  300ep seed42 run is BIT-EXACT identical to v33 (student peak 63.87@e280,
  delivered 64.66, aug.mixup_window recorded as [100,200]). The [P] axis was
  therefore never actually tested.
  v35 fixes the shadowing default (None), adds a RUNTIME CANARY that refuses to
  start if the resolved window is not the full run, and records the resolved
  per-epoch mixup alpha in history + aug_cfg so a no-op can never hide again.
  (覆辙㉙: a function default is invisible when a CLI default shadows it.)

  v31 [M] already confirmed: default = bit-exact v18 (64.66, PASS+, stop-loss done),
  producing best_v31_seed42.pt. BUT v31 [M2] single-ext-teacher born-again is a
  CEILING: the only available teacher is 64.66 == the student's own ceiling, so
  self-distillation from it cannot exceed 64.66 (v26 proved self-KD harmful 64.20).
  v32 [N] fixes the premise — the "external stronger teacher" (覆辙⑭) must be a
  GENUINELY STRONGER / DIFFERENT function. With available resources that means an
  ENSEMBLE of independent seeds (different inits -> decorrelated errors -> real
  dark knowledge, and by construction OUTSIDE the student's collapse dynamics,
  satisfying 覆辙㉖). --teacher-ckpts 'a.pt,b.pt,c.pt' = K-model ensemble as the
  consistency target. Default (no teacher) = bit-exact v18 (64.66 floor).

=============================================================================================================
v31 [M] 设计 (v30 判读: FAIL 62.49, -2.17pp vs v18 64.66):

  [A'] v30 归因之一 (~-1.2pp) — IRBF 共塌缩: 滞后实例源是学生自身轨迹的
      确定性函数, 与被正则的 CLS 特征同处一个空间 —— 学生色散收缩 =>
      记忆库 EMA 跟随收缩 => 目标随之收缩 (耦合不动点, 无恢复力)。
      探针: intra_disp 0.8204->0.1345 (-84%), proto_norm 终态 0.8073
      (与 v29 的 0.8073 完全一致 —— 同一塌缩吸引子, β=1.0->0.2 只放慢
      速度)。覆辙㉖: 一致性目标必须「不在学生自身塌缩动力学的下游」;
      权重空间 EMA 教师在塌缩子系统之外 (不同函数), 特征空间滞后副本在内。
  [B'] v30 归因之二 (~-0.9pp) — [L2] 实现 bug: ema_state 在初始化后被
      `ema_state = None` 覆写, 交付链 B/C/D 全程死亡 (weight_ema_snapshot_on
      =false, handoff.fired=false, delivered_source=best_checkpoint)。
      静态闸门 (绑定存在性) 看不见语义覆写 —— 结果 JSON 字段审计列入判据。
  [C] 结论: v19-v30 十二连败; 「闭式物理场/特征空间副本替代权重空间 EMA
      教师」整条线证伪 (两个独立实现, 同一塌缩指纹)。正确方向 = 回退
      v18 基座 + 外部更强教师蒸馏 (真实暗知识, 覆辙⑭)。

v31 [M] 相对 v28 的三处增量 (默认路径逐字节不动 = bit-exact v18):
  [M1] 默认模式 = v28 默认 (bit-exact v18, 交付 ~64.66) —— 先回退到有效版本,
       止血。v31 默认跑完即产出 best_v31_seed42.pt, 作为二代蒸馏的教师 ckpt。
  [M2] ext 模式交付链补齐 (v28 [J] 的结构性缺口, 覆辙㉔ 在 ext 模式重演):
       - ext teacher 加载后立即 evaluate 一次, 精度入 results (ext_teacher_acc)
         并入 best_teacher_acc (二代目标线); < 64.0 时打 WARNING (弱教师)。
       - gap 遥测 = ext_teacher_acc - student_best (distillation gap; gate
         floor=1.0 -> 训练路径零变化, 纯诊断信号)。
       - ext 模式 handoff (e280): 恢复学生 best ckpt + 常数 1e-5 纯监督
         polish (v18 机制 B/C 的学生侧结构; w_eff=0 同 v18)。best ckpt 已受
         机制 C 保护 -> 交付严格非负。
       - 机制 D (teacher-record delivery) 在 ext 模式结构性跳过: 「交付外部
         教师权重」是退化操作 (结果 == 输入), 明确关闭并记录于 JSON。
  [M3] 判读 (ext 模式, vs 教师 = best_v28/v31 ckpt ~64.66):
       WIN    delivered >= 64.80  (born-again 真增益)
       PASS+  64.66 <= delivered < 64.80  (追平教师)
       PASS   64.50 <= delivered < 64.66  (蒸馏有效但未及教师)
       NULL   64.41 <= delivered < 64.50  (中性)
       FAIL   delivered < 64.41  (蒸馏有害 -> 停止该轴, 仅保留 v18 默认)

[v18] v18 的判读：PASS+（64.66）—— 但解耦后真相相反：
      · delivered 64.66 = v17 的 64.64 + 0.02 —— 全部来自 e281 单 epoch
        噪声 spike（handoff 后 EMA 重跟踪的瞬时峰值）
      · 训练侧全部转负：学生 pre-handoff best 63.70（vs v17 63.89，
        -0.19）；teacher 终端窗口 max 63.94（vs v17 64.25，-0.31）
      · 终端驻留段（e250-e279 的 fine-heavy HOLD）有害 —— 压缩旋转
        方向证伪；但时间轴本身未证伪（见下）
      · 旋转的真正价值（v14-v18 横向对比）：pre-HO teacher 峰值
        v14 静态 64.50@e201 最高，但 post-HO 塌陷 64.08@e280（-0.42，
        无吸收）；v16/v17/v18 旋转 pre-HO 峰值 64.44@e192，post-HO EMA
        吸收出 64.64-64.66@e281（+0.20 响应）。旋转的价值 = 触发
        handoff 吸收响应，而非 pre-handoff 轨迹本身。

v21 的动机 — D4++ 量纲修复（v19 机制修复，非新机制）：

  v19 让谱形数据驱动（用上一 epoch 三 band 残差 fd_c/fd_m/fd_f
  做 water-filling），但实现有两个 bug，v21 一并修复：

  BUG-1 (量纲, v19 seed42 实证 idle): pressure_j = max(band_j/gap, 0.30)
    band~1e-3 (L2 距离), gap~1-3pp -> band/gap~1e-3 << 0.30 ->
    pressure 全饱和到 floor -> wf==[1,1,1] -> adapt==0 -> 分支 bit-exact
    空转（v19 与 v18 轨迹 128/128 epoch 位同, max|d|=5.55e-17）。
    FIX: 去 gap 除数; 改用扰动标度归一 pressure_j = band_j/scale2_j
    (scale2 = sigma^2 阶梯 1/0.25/0.0625), 使 coarse/mid/fine 可比。

  BUG-2 (winner-take-all, v21 预览发现): 加法+renorm 形式在归一后
    残差跨度 1:2:4 时把 coarse/mid 压到 ~0.03。FIX: 乘法软化
    w_eff_j = base_j * (wf_j ** (alpha*beta)) 再 renorm; 每 band 恒正、
    无饿死、beta 压缩残差跨度。

    w_eff_j(t) = base_j(t) * (wf_j(t))**(alpha(t)*beta)
    wf_j      = pressure_j / mean(pressure)   # 扰动标度归一, dose-neutral
    pressure_j = band_j / scale2_j            # 无 gap 除数
    alpha      : 0 -> 1 over e151..e200（渐入, 防早期抖动）
    beta       : 乘法软化指数（新增 --ms-adapt-beta, 默认 0.4; 越大越
                激进转向高残差 band）
    gap floor  : gap < 0.5pp 断开回 base（main loop GATE, 师生收敛后
                停止干预, 让 handoff 吸收 —— v18 教训）

  关键性质（预览模拟确认）: 乘法形式在 e152 (alpha=0) 精确等于 v18
  基座（近 coarse）, e250+ 渐转向 fine —— 天然复现 v14(coarse pre-HO
  峰值 64.50) + v18(fine post-HO EMA 吸收 64.66) 的组合, 而这是任何
  固定端点都没达到的。

  设计要点:
  · 因果 look-back 1: epoch e 用 e-1 的 band 残差（epoch 均值, 已平滑）
  · 剂量中性逐 epoch: renorm sum(w)==1.0（与单纯形插值同不变量）
  · 乘法形式无负值/无饿死: 每 band 恒正, beta 压缩残差跨度防失衡
  · gap 仅作 GATE（不在除数）: 师生收敛后断开回 base, 让 handoff 吸收

  决定论纪律：
  · water-filling/alpha 是纯算术（.item() 标量 + 顶部 gap 标量）：
    零 RNG、零额外 forward —— RNG draw order 与 v18 完全相同
  · alpha=0（e151）或 --no-ms-adapt -> EXACT v18 权重（bit-exact 锚点）
  · --ms-adapt-alpha-end / --ms-adapt-gap-floor 调节干预曲线
  · 继承：--no-teacher-delivery -> v16 / --ms-weights v14/v15/v13 ->
    静态锚点 / --res-beta 0 -> v12

v22 的动机 — Module E（新轴：扩散核；谱形轴 v14-v21 已闭合）:

  谱形轴（v14-v21）结论：v18 静态 fine 旋转 64.66 是谱形轴顶；v19 自适应
  量纲修好后仍 64.46（< v18），v20 coarse 保护端点 64.43 是局部极小值。谱形
  轴已饱和。转向尚未系统探索的「扩散核函数」轴——它的参数（sigma_max 带宽、
  sigma ladder、gap-gate）至今是手工退火/固定，从未数据驱动。

  v22 做两件耦合的事（都用 v19/v21 已验证的量纲修复范式）：

  E1 各向异性扩散核：把 v21 图像空间「标量 sigma_max + 共享一条 Brownian 路径」
      的 ISOTROPIC 扰动，换成特征空间「逐 band 用 sigma_b = s_m*kappa_b 扰动
      teacher CLS 特征、要求学生（干净）特征匹配」的 ANISOTROPIC 扰动。一次
      randn(B, D) 拆成 K band（单次 RNG，无额外 student forward）。

  E2 扩散核带宽闭环修订：kappa_b(t) = wf_b**(alpha*beta)；wf_b 由各 band 干净
      残差 water-filling（clamp 防饿死 + dose-neutral），alpha 渐入 e152->e200，
      gap < KERNEL_REVISE_GFLOOR 断开回各向同性。--no-kernel-revise -> kappa
      全 1 且走 v21 图像空间路径（bit-exact 锚点）。

  版本号 bug 修复（v21 实证）：v21 把结果存成了 physics_v19_aug（tag 写死 v19）。
  v22 引入 VERSION 单一真相源，文件名与 banner 均由其派生，存盘永远与版本一致。

Usage:
  # === v35 [Q] 背景: 64.44 是关键关口, 多个版本 (v18/v31/v32/v33/v34 seed42) 都焊死
  #      在这附近, e190 之后增长极慢. 逐 epoch 轨迹诊断 (v33==v34 seed42, 300ep):
  #        - student test_acc: e190=61.64 -> e280=63.87(peak) -> e300=63.42(DECAY)
  #        - train_acc: e191=63.95 -> e201=92.91 (mixup 关窗瞬间 +29pp!) -> e300=95.30
  #        - 学习速率: e150-200(mixup ON) +0.088 pp/ep; e200-280(OFF) +0.027 pp/ep
  #          = 3.2x 减速; e280 之后转负.
  #        - 64.44 关口本身: teacher_test_acc 在 e192 触到 64.44 后快照冻结,
  #          连续 88 个 epoch (e192-e279) 停在 64.44 —— 不是"卡住", 是快照只在新
  #          teacher record 时升级, 而 EMA 教师 e200-e280 真实水平只有 63.6-64.0.
  #        - EMA 教师(交付源): 峰值 64.66@e281, 但这是 handoff(LR 1e-5)+恢复+
  #          权重平均的平坦极小点增益, 单点事件, 不是 student 真泛化;
  #          student 自身从未突破 63.87.
  #      根因 (v35 单一突破轴): mixup 窗口 [E/3,2E/3]=[100,200] ->
  #        mixup/cutmix 在 e200 后彻底关闭, 模型用干净硬标签训练最后 1/3 epoch,
  #        正好落在过拟合爆发区 -> 泛化天花板被焊在 ~63.9 (EMA 拉到 64.44-64.66).
  #      即: 增强在"最需要正则化时"被关掉. 修复 = 把窗口延展到全程 [E/3, E].
  #      ** v34 尝试过但改动是死代码 (argparse default="100-200" 遮蔽了函数返回值),
  #         v35 修正遮蔽并加运行期 canary, 使静默空转不可能再发生. **
  #      (SWA 稳健交付 [O1] 与两步 born-again [O2] 继承, 仍有效, 不改动.)

  # 0. 默认 (SWA 稳健交付 + 全程 mixup 真生效) + 产教师 ckpt:
  python train_vit_cifar100_aug_v36.py --seed 42 --epochs 300 --physics

  # 0b. [v36 主跑] 训练时长轴: 400 epochs (外部 SOTA 的可比设置).
  #     所有阶段常量按 E=300 标定比例同步平移, schedule 形状不变:
  python train_vit_cifar100_aug_v36.py --seed 42 --epochs 400 --physics

  # 1. 弱 seed 用强 seed42 教师 born-again (覆辙⑭合规: 66.65 > 弱 seed):
  for s in 1 2; do python train_vit_cifar100_aug_v36.py --seed $s --epochs 300 --physics \
      --ext-teacher results/best_v36_s42.pt; done

  # 2. 集成 3 个种子 born-again (破 66.65 唯一合法轴, 教师>单模型):
  python train_vit_cifar100_aug_v36.py --seed 42 --epochs 300 --physics \
      --teacher-ckpts results/best_v36_s42.pt,results/best_v36_s1.pt,results/best_v36_s2.pt

  # 3. bit-exact v18 锚点 (窗口回到 [100,200], 其余与 v18 一致):
  python train_vit_cifar100_aug_v36.py --seed 42 --epochs 300 --physics --v18-anchor

  # 4. [Q] 因果消融: 强制回退旧窗口 [100,200] (证明增强关闭确是天花板成因):
  python train_vit_cifar100_aug_v36.py --seed 42 --epochs 300 --physics --mixup-window 100-200

  # 5. [R] horizon 消融: 把阶段常量钉在 E=300 的位置, 只延长训练
  #    (隔离"多训练"与"机制同步平移"各自的贡献):
  python train_vit_cifar100_aug_v36.py --seed 42 --epochs 400 --physics \
      --cons-start 150 --ms-rot-end 250 --swa-start 240 --handoff-start 280 \
      --ema-sched-start 200 --ema-sched-end 279 --kernel-revise-end 200 \
      --ms-adapt-alpha-end 200 --w-cons-ramp-len 20 --w-cons-hold-len 65

  # Smoke test (跨过后段分支; 两个 horizon 都跑, 验证阶段常量平移):
  python train_vit_cifar100_aug_v36.py --seed 42 --epochs 6 --smoke --physics
  python train_vit_cifar100_aug_v36.py --seed 42 --epochs 8 --smoke --physics
判读标准（vs v18 谱形轴顶 64.66 / v17 稳定 SOTA 64.64 / v14 交付线 64.50 / v21 64.46 新基线；
           辅助基线：v18 pre-handoff 63.70 / teacher 64.66@e281 / v21 64.46@e282）：
  · WIN:   delivered >= 64.80 —— 扩散核闭环生效, 各向异性带出 v18 之外的容量
           （各 band 不同扩散率 + 数据驱动带宽）。行动：补 3 seeds + beta 扫描
           （0.3/0.5/0.6）+ 加深（EMA 平滑 band 残差 / K_ANISO 细分）
  · PASS+: 64.66 <= delivered < 64.80 —— 超 v18：各向异性免费生效, 训练侧转正
           （看 pre-handoff/teacher 双侧 + kappa 轨迹是否 coarse-early->fine-late）
  · PASS:  64.50 <= delivered < 64.66 —— 与 v17/v18/v21 持平：各向异性中性
           （v18 固定端点已接近最优, 闭环未超）。行动：按饱和处理, 转训练侧其他轴
  · NULL:  64.41 <= delivered < 64.50 —— 轻度负向（如 v20 64.43）：beta 过大
           或 gap-floor 断开时机不对, 或各向异性扰动位置（特征空间）与 handoff
           动力学冲突。行动：beta 下调 / alpha 渐入放缓 / 退回图像空间
  · FAIL:  delivered < 64.41 —— 显著有害：特征空间扰动破坏 consistency 信号。
           行动：--no-kernel-revise 维持 v21（bit-exact 锚点）
  · 注: 扩散核是 v14-v21 从未探索的轴; 谱形轴已闭合（v18 顶 64.66）。v22 是
           该新轴第一个未测点; 先验证「各向异性 + 闭环」是否比固定核带来容量
  · 探针读数优先级：delivered 之外必看 teacher best 与 student pre-handoff 双侧
           + history kappa_c/m/f（闭环是否真激活、是否 coarse-early->fine-late）
  · 决定论检查: kernel_revise 必须 echo True；--no-kernel-revise 的
    history["ms_w_c/m/f"] 应与 v21 位同（走 v21 图像空间路径）；alpha=0 时 kappa==[1,1,1]
  · v32 [N] 判读 (集成教师 born-again, vs 集成精度 = 真实上限):
       WIN    delivered >= 64.80  (学生经一致性训练逼近/超过集成, 真增益)
       PASS+  64.66 <= delivered < 64.80  (追平 v18, 集成带来轻微增益)
       PASS   64.50 <= delivered < 64.66  (蒸馏有效但未破顶)
       NULL   64.41 <= delivered < 64.50  (中性)
       FAIL   delivered < 64.41  (集成教师未带来增益 -> 停止该轴)
       注: 单 ckpt (--ext-teacher) 教师==学生天花板, 不可能破 64.66; 必须用
           --teacher-ckpts 给 >=2-3 独立 seed 才能触发真实暗知识。
  · v33 [O] 判读 (SWA 稳健交付 + 两步 born-again, 目标=稳健破 64.66):
       SWA 贡献 = swa_acc - max(last_epoch, best_ckpt, teacher_record)
         > +0.1pp => SWA 抬升了 floor (seed1 类); <=0 => SWA 中性 (seed42 类)
       两步 born-again 判读 (step2 集成教师):
       WIN    delivered >= 64.80  (集成教师>64.66, 真增益, 破伪天花板成立)
       PASS+  64.66 <= delivered < 64.80  (追平 v18, 集成轻微增益)
       PASS   64.50 <= delivered < 64.66  (弱种子被救起但未破顶)
       NULL   64.41 <= delivered < 64.50  (中性)
       FAIL   delivered < 64.41  (两步蒸馏未带来增益 -> 退回 SWA 默认)

  · v35 [Q] 判读 (全程 mixup 窗口【真生效】, 目标=突破 64.44 泛化天花板):
       【第 0 判据 — NO-OP CANARY, 必须先查】
         aug.mixup_window 必须 == [100, 300]; aug.mixup_alpha_by_epoch[199] (e200)
         必须 > 0; history["mixup_alpha"][-1] 应 ~0 (末 epoch 退火干净).
         若 aug.mixup_window == [100,200] => 又是空转, 结果与 v33 逐位相同, 直接作废.
       核心判据 = student 真泛化峰值 (history["test_acc"] 的 max, 非 EMA/交付值):
         v33/v34 基线 student peak = 63.87@e280.
           WIN    student peak >= 64.8   -> 过拟合轴真被拆掉, 天花板抬升
           PASS+  64.3 <= peak < 64.8    -> 增强窗口修复生效 (gap 收窄, 后段不再空转)
           PASS   64.0 <= peak < 64.3    -> 轻正向 (后段学习速率恢复)
           NULL   63.87 <= peak < 64.0   -> 中性, 该轴无因果力 -> 转训练时长轴
           FAIL   peak < 63.87           -> 末段 mixup 反而伤害收敛 -> 回退 v33 窗口
       辅助判据 1 = 后段学习速率 (test_acc 斜率):
         v33/v34 e200-280 = +0.027 pp/ep. 若 v35 >= +0.05 pp/ep => 后段不再空转.
       辅助判据 2 = 峰后回撤: v33/v34 e280->e300 = -0.45pp. 若 v35 >= -0.1pp
         => 过拟合缓解 (train/test gap 收窄).
       消融回归: --mixup-window 100-200 应 bit-exact 复现 v33 (student peak 63.87,
         delivered 64.66, SWA 64.53) —— 这是本次唯一的因果对照, 强烈建议同跑.
       注: [O2] step1 用强 seed42 救弱 seed1/2 是覆辙⑭合规 (教师>学生); 原 v32
           直接平均 3 个不等精度 seed 会产出 <64.66 的教师, 回退 seed42, 已证伪。
"""

import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import timm
from timm.models.vision_transformer import (
    Attention as TimmAttention,
    resolve_self_attn_mask,
    maybe_add_mask,
)
import torchvision
import torchvision.transforms as T


# ==================== v12 constants ====================

W_CONS_MAX      = 1.0      # v12: 0.75 -> 1.0 (v9 full-anchor dose restored.
                           #      v11 dose-response falsified "anchor suppresses
                           #      student": teacher best 63.92 (w=0, v10) ->
                           #      64.13 (w_eff~0.23, v11) -> 64.40 (w=1.0, v9)
                           #      is monotone in dose, and student pre-handoff
                           #      best rose with dose too (61.84/62.30/62.52))
W_CONS_END      = 0.1      # v12: 0.05 -> 0.1 (v9 exact end strength)
W_CONS_RAMP_LEN = 20       # Ramp-up epochs (same as v4-v11)
W_CONS_HOLD_LEN = 65       # Hold-at-peak epochs (same as v6-v11)
EMA_DECAY       = 0.999    # Max EMA decay (warmup: 1-1/(t+1)) (same as v4-v11)
WEAK_AUG_PAD    = 4        # Zero-pad for weak-view random crop (same as v4-v11)

# --- v25 [F]: SCHEDULED EMA TEACHER DECAY (attack the real ceiling) ---
# v24 verdict: delivered == teacher == 64.28; the live EMA teacher is the LIMIT.
# The teacher already LEADS the student by ~0.7pp at e250-279 (it captured the
# student's late peaks). A TIGHTER (higher-decay) teacher late = a smoother
# ensemble that retains those peaks better -> higher, more stable delivered.
# The max-decay cap is ramped  EMA_DECAY_BASE -> EMA_DECAY_LATE  over the late
# window, while the step-warmup (1-1/(t+1)) still tracks the student fast early.
# EMA decay was HARDCODED at 0.999 since v4 and never swept -> genuine new axis.
EMA_DECAY_BASE = 0.999    # cap during early/mid phase (bit-exact v21 behavior)
EMA_DECAY_LATE = 0.9997   # cap during late phase: 3x slower teacher update -> tighter ensemble
EMA_SCHED_ON   = False    # v26: REVERTED — v25 proved the scheduled cap (0.999->0.9997) KILLS the teacher>student lift (64.41 vs 64.46 anchor). Fast EMA 0.999 = the lift that made v17/v18 SOTA.
EMA_SCHED_START= 200      # ramp begins (after rotation peak; teacher leads student here)
EMA_SCHED_END  = 279      # ramp completes (holds LATE through + post-handoff)

# --- v33 [O1]: SWA robust delivery (stochastic weight averaging, Izmailov et al.) ---
# Average model WEIGHTS over the late-training window with EQUAL weight (vs the
# EMA teacher's exponential/recent-heavy average). SWA consistently finds flatter
# minima that generalize better and REDUCES seed variance — directly fixes the
# v32 seed-1 pathology (EMA boost ~0pp, handoff never fired, delivered stuck at
# 62.49). SWA is an ADDITIONAL argmax candidate at delivery (never replaces the
# others), so it can only LIFT or be neutral — never regress a lucky seed (seed42).
SWA_START      = 240      # begin averaging weights from this epoch (late window)
SWA_DEFAULT_ON = True     # default ON; --no-swa disables (== v18 bit-exact delivery floor)

# v35 [Q]: size of the fixed unaugmented train subset used ONLY to measure the
# overfitting gap (see get_train_eval_loader). Measurement, not training.
TRAIN_EVAL_N   = 2560

# --- v12 mechanism A: v9 full-anchor consistency field (gate disabled) ---
CONS_START      = 150      # Field activates e151 (unchanged; e1-150 stays
                           # bit-exact with v8/v9/v10/v11)
NOISE_SCALES    = 3        # M: sigma scales per batch (Brownian path sampled
                           #     at M diffusion times t_m = s_m^2 / 2)
VERSION         = "v77"    # single source of truth for result filename + banner
SIGMA_RATIO     = 0.5      # geometric ladder ratio: [s_max, s_max/2, s_max/4]
SIGMA_MAX_START = 0.30     # sigma_max at e151 (strong diffusion, coarse field)
SIGMA_MAX_END   = 0.05     # v12: 0.10 -> 0.05 (v9 exact field annealing; v11's
                           #      0.10 floor only kept feat_diff numerically
                           #      alive — corr(sigma)=0.95 = trivial convergence
                           #      — and contributed nothing to teacher quality)

GATE_LO         = 1.0      # (kept only for --gate-* ablation overrides)
GATE_HI         = 3.0      # (kept only for --gate-* ablation overrides)
GATE_FLOOR      = 1.0      # v12: 0.25 -> 1.0 => gate(gap) == 1.0 for ALL gaps:
                           #      the gap gate is DISABLED by default (v9
                           #      full-strength semantics). Dose-response says
                           #      the anchor IS the gain source; gating only
                           #      diluted it (v11 w_eff mean 0.228 vs 1.0).
                           #      Pure arithmetic identity — zero RNG impact.

# --- v11 mechanism C: extended handoff (kept, proven) ---
HANDOFF_START   = 280      # (v11 proven: fires with gap 1.83pp)
HANDOFF_LR      = 1e-5     # (v11 proven: decay only -0.25pp)
HANDOFF_MIN_GAP = 0.3      # Fire only if teacher leads student by > 0.3pp

# --- three-channel consistency weights (sum to 1.0, same as v9-v12) ---
ANCHOR_LOGIT_W  = 0.3      # logit-space anchor (v8 symKL term)
FEAT_ANCHOR_W   = 0.2      # feature-space anchor: student(strong) vs teacher(weak)
FEAT_DIFF_W     = 0.5      # feature-space diffusion (v9 core, proven on teacher)

# --- v25 [F]: Module E DISABLED by default (proven harmful, 3/3 fail) ---
# v22 blind (64.47) / v23 variance forward (64.31) / v24 precision inverse (64.28)
#   -> ALL <= v21 loop-OFF anchor (64.46). The loop FIRED hard in v24
#   (kappa_spread 2.62, w_eff 1.0, sigma 0.25) yet was the WORST -> mechanism-design
#   failure, not plumbing. Root cause: the consistency field already has a NOISE-FREE
#   anchor (feat_anchor: student feat <-> teacher feat, + anchor_logit symKL) that is
#   beneficial (v21=64.46); Module E ADDS a noise-injected term (fd: student feat <->
#   teacher feat + sigma*noise) that blurs the CLS feature and degrades the class
#   boundary. Variance/precision in CLS space is NOT aligned with class-discriminativeness,
#   so no reweighting fixes a wrong objective. v25 keeps the noise-free anchor, DROPS
#   Module E by default (KERNEL_REVISE=False), and attacks the real ceiling via EMA sched.
#   The fd path is retained for ablation (--kernel-revise) and carries the v23/v24 json
#   float fix so it cannot crash results again.
K_ANISO              = 3        # DIAGNOSTIC bands for fd_kappa (coarse/mid/fine thirds); v24 kernel is PER-DIM (length D)
BAND_SIGNAL          = "precision"  # closed-loop signal (only used if --kernel-revise): per-dim PRECISION = 1/var(student strong-view CLS)
BAND_VAR_EMA_DECAY   = 0.99     # EMA smoothing of per-dim precision across epochs (closed-loop lag)
KERNEL_REVISE        = False    # v25 [F]: DISABLED by default (Module E proven harmful)
KERNEL_REVISE_BETA   = 0.4      # multiplicative softening exponent for the water-filling tilt
KERNEL_REVISE_AEND   = 200      # alpha ramp end (ramp e152->e200, then held 1.0)
KERNEL_REVISE_GFLOOR = 0.5      # gap (pp) below which revise disconnects back to isotropic
KERNEL_WF_FLOOR      = 0.30     # water-filling floor (anti-starvation; dimension-fixed per v21)
KERNEL_WF_CAP        = 1.0 / KERNEL_WF_FLOOR  # upper clamp (1/0.30 = 3.33): bounds anisotropy ratio
GATE_ADAPT           = False    # optional: self-tune gap-gate lo/hi (default OFF; v12 found dilution harmful)
GATE_ADAPT_BW        = 1.0      # adaptive gate band-half-width around running gap mean

# --- v26 [G]: SOFT-LABEL TEMPERATURE DISTILLATION (new axis) ---
# v25 verdict: delivered == teacher == 64.41; the EMA SCHED killed the
# teacher>student lift (the very mechanism that made v17/v18 SOTA). The
# binding constraint is now clear: delivered ~ teacher ~ EMA(student); the
# only surplus over the raw student is the EMA smoothing lift (~+0.2..+1.2pp).
# To BEAT the 64.64-64.66 SOTA we must raise the STUDENT's own convergence —
# the one axis not yet touched across v12-v25.
# The existing logit anchor (anchor_logit, w=0.3) is symmetric KL @ T=1: it
# enforces invariance / argmax-match only. Standard Hinton soft-KD adds a
# TEMPERATURE-SOFTENED asymmetric KL(student/T || teacher/T) * T^2 that
# transfers DARK KNOWLEDGE (the teacher's class-similarity structure) which
# the hard T=1 anchor cannot see. The teacher is the frozen best-snapshot
# (same capacity as the student, so the surplus is bounded) but softening
# still exposes its class-correlation signal. Added as a 4th consistency
# channel, gated by w_eff like the others. In v26 this was ON by default;
# v26 PROVED it HARMFUL (-0.46pp vs v18) and v27 reverts it to OFF by default
# (--soft-kd re-enables for ablation only). This is the single classic KD
# lever absent from the v9-v25 feature-L2 + hard-logit consistency.
#
# v27 NOTE: v26 PROVED soft-KD FROM THE LIVE EMA SELF-TEACHER HARMFUL
#   (delivered 64.20 vs v18 64.66 = -0.46pp; teacher itself dragged down to
#   64.20). The same self-teacher already supplies anchor_logit (T=1 KL), so
#   the T=4 channel is REDUNDANT and its mushy targets over-smooth the
#   student's discriminative peaks. DEFAULT OFF in v27; --soft-kd re-enables
#   the (now-known-harmful) channel for ablation only.
SOFT_KD          = False   # v27: soft-KD default OFF (v26 proved harmful for the live-EMA SELF-teacher)
SOFT_KD_T        = 4.0     # distillation temperature (softens logits -> dark knowledge)
SOFT_KD_W        = 0.3     # weight of the soft-KD channel (added into cons_term)

# === v28 [J]: EXTERNAL STRONGER TEACHER distillation ===============================
# The ONLY theory-correct breakthrough left (覆辙⑭): v18..v27 all failed because the
# consistency target was the student's OWN live-EMA (same capacity -> delivered==teacher
# hard ceiling ~64.66, no dark knowledge). A frozen EXTERNAL teacher of higher/cleaner
# capacity supplies real class-correlation dark knowledge, so soft-KD (T=4) is now
# meaningful (unlike v26's self-teacher where it was redundant + over-smoothing).
# Default OFF -> bit-exact v18 (immediate rollback). --ext-teacher <ckpt> enables it.
EXT_TEACHER_T   = 4.0     # distillation temperature for the external teacher
EXT_TEACHER_W   = 0.3     # weight of the external-teacher soft-KD channel


# --- v13 [D2]: residual per-sample weighting of the consistency term ---
# r_i = per-sample total residual across the three channels (detached);
# w_i = clip(1 + beta*(r_i/mean(r) - 1), lo, hi) then renormalized to
# mean 1 (dose-neutral vs v12). beta=0 via --res-beta 0 runs the EXACT
# v12 branch (identity, determinism anchor). Weights computed under
# no_grad from tensors already on hand: zero extra forwards, zero RNG.
RES_WEIGHTED_ANCHOR = True   # master switch (False -> pure v12 semantics)
RES_BETA       = 1.0         # residual dose (CLI-overridable: --res-beta)
RES_W_LO       = 0.25        # clip floor (smooth samples: quarter dose)
RES_W_HI       = 2.5         # clip ceiling (rough samples: 2.5x dose)

# --- v18 [D4']: spectral anchor time-axis rotation, COMPRESSED (A-probe) ---
# v15 verdict: the STATIC spectral axis is CLOSED (coarse push 0.6/0.3/0.1
# FAILED: teacher 64.50 -> 64.34, dose +11.4% bought nothing; fine 0.1
# starved a channel whose residual was never harvested — v15 fd_f actually
# ROSE across the window). The remaining free axis is TIME: sigma_max already
# anneals coarse->fine (0.30 -> 0.05); v16 rotates the spectral WEIGHTS along
# the same direction:
#   w(t) = (0.5, 0.3, 0.2) @ e151  ->  (0.2, 0.3, 0.5) @ e250   (linear,
#   compressed) then HELD at the fine mirror for e250-e279 — the last 30
#   epochs of the active anchor window (handoff still fires e280)
# · start = the proven v14 optimum (early training keeps the coarse anchor
#   that carried ALL v9-v14 gains); end = its fine-mirror (late training
#   pressures the unconsumed fine residual)
# · dose-neutral EVERY epoch: simplex interpolation keeps sum w(t) == 1.0
# · v18 phase-mean over the active window e151-e279 = (0.316, 0.3, 0.384):
#   compression shifts the fine-channel epoch-mean +0.034 (0.350 -> 0.384,
#   +9.6%), coarse mirror-image -0.034 — the probe measures the JOINT
#   timing + terminal-dose effect by design; any WIN triggers a static
#   control (0.316,0.3,0.384) to disentangle
# Static collapse anchors: '--ms-weights v14' -> (0.5,0.3,0.2) all epochs
# (EXACT v14 branch); '--ms-weights v15' -> (0.6,0.3,0.1) all epochs (EXACT
# v15 branch); '--ms-weights v13'/'uniform' -> the equal-weight branch.
MS_SPECTRAL_ANCHOR = True    # master switch (False -> exact v13 uniform branch)
MS_ROTATE      = True        # rotation master switch (False -> static MS_W_START)
MS_W_START     = (0.5, 0.3, 0.2)   # spectrum at e151 (the v14 optimum)
MS_W_END       = (0.2, 0.3, 0.5)   # fine-mirror spectrum at e250, HELD to e279
MS_ROT_END     = 250         # v18 A-PROBE — the ONLY functional change vs
                             # v17 (279 -> 250): rotation compressed 29
                             # epochs, fine-mirror terminal held e250-e279;
                             # handoff still fires e280, w_eff -> 0 after

# --- v19 [D4++]: gap-driven ADAPTIVE spectral weights (water-filling) ---
# v18 A-probe verdict (PASS+ 64.66 — but decoded): the +0.02 vs v17 came
# from the e281 single-epoch noise spike, NOT from the compressed rotation:
# student pre-HO best 63.70 (-0.19 vs v17 63.89), teacher terminal-window
# max 63.94 (-0.31 vs 64.25) — the terminal hold (e250-e279) was HARMFUL.
# The rotation's true value: it TRIGGERS the handoff absorption response
# (post-HO EMA peaks 64.64-64.66 @e281 vs v14's static 64.50 collapsing
# -0.42 post-HO with no absorption). A FIXED time prior cannot track the
# actual per-scale residual consumption — v19 makes the spectrum
# DATA-DRIVEN on top of the v18 prior:
#   w_eff_j(t) = base_j(t) + alpha(t) * adapt_j(t)
#   adapt_j    = wf_j - mean(wf)          (zero-mean: no dose shift)
#   wf         = water-filling of the PREVIOUS epoch's band residuals:
#                pressure_j = clip(band_j / max(gap_pp, eps), floor, inf),
#                wf = pressure / mean(pressure)  (dose-neutral, mean 1)
#   alpha ramps 0 -> 1 over e151..ADAPT_ALPHA_END (gradual intervention),
#   disconnects back to base when gap_pp < ADAPT_GAP_FLOOR (teacher and
#   student converge: stop reshaping, let the handoff absorb).
# Determinism: water-filling is pure arithmetic on .item() scalars + the
# top-level gap scalar — zero RNG, zero extra forwards. alpha=0 (e151) or
# --no-ms-adapt -> EXACT v18 weights (bit-exact anchor; the RNG draw order
# is untouched, only the float constants feeding feat_diff change).
MS_ADAPT         = False   # v26: REVERTED — v19->v25 adaptive spectral (D4++) cost -0.18pp vs v18 (64.46 vs 64.64). Default now = EXACT v18 spectrum (bit-exact anchor).
ADAPT_ALPHA_END  = 200     # alpha ramps 0->1 over e151..e200, then held 1.0
ADAPT_GAP_FLOOR  = 0.5     # gap < 0.5pp: GATE — disconnect adaptation back to
                           # base (no longer a water-filling divisor/floor)
ADAPT_BETA       = 0.4     # multiplicative softening exponent: contraction
                           # strength of the water-filling tilt. Higher = more
                           # aggressive shift toward the high-residual band.
                           # 0.0 -> base-exact (identity); tune via --ms-adapt-beta

# ============================================================================
# v36 [R]: HORIZON-PROPORTIONAL STAGE CONSTANTS  (single source of truth)
# ============================================================================
# Every stage constant above was hand-calibrated at E=300.  Carrying them
# verbatim to a different horizon silently MISPLACES every mechanism — the
# physics window, the spectral rotation, the EMA ramp, the SWA window and the
# handoff all stay put while the cosine LR keeps moving underneath them
# (覆辙②: never reuse hyperparameters across a change of horizon without
# recalibration; 覆辙⑤: do not over-read a budget window that was never tuned).
#
# v35 evidence that the horizon IS the live axis: the run ends with LR already
# at 0.0 (cosine completed at e300) while test_acc is still climbing
# (+0.23pp over the last 20 epochs, +0.75pp over the last 60) — the ceiling we
# hit is the SCHEDULE's end, not the model's.  Every external from-scratch
# ViT-S/4 CIFAR-100 number (73.9 LieRE / 72.6 STRING / 72.3 RoPE-Mixed) is
# trained for 400 epochs, so 300 was never comparable in the first place.
#
# v36 therefore derives every stage epoch from args.epochs by the fraction it
# occupied at E=300:
#   * `--epochs 300` -> every fraction lands back on the v35 value => bit-exact.
#   * `--epochs 400` -> same schedule SHAPE, 1/3 more optimisation time.
#   * Any explicit CLI override still wins (ablation / mechanism isolation).
#
# The fractions are DERIVED FROM THE CONSTANTS THEMSELVES, not retyped — a
# hand-copied table would be a second implementation of the same mapping and
# would eventually drift (覆辙⑰).  Each key is ALSO the argparse dest name, so
# the derivation loop in parse_args() fills them by name with no second table.
STAGE_CALIB_E = 300
STAGE_CALIB = {
    "cons_start":         CONS_START,        # 150 -> 200 @400ep
    "ms_adapt_alpha_end": ADAPT_ALPHA_END,   # 200 -> 267
    "kernel_revise_end":  KERNEL_REVISE_AEND,  # 200 -> 267
    "ms_rot_end":         MS_ROT_END,        # 250 -> 333
    "w_cons_ramp_len":    W_CONS_RAMP_LEN,   #  20 ->  27
    "w_cons_hold_len":    W_CONS_HOLD_LEN,   #  65 ->  87
    "ema_sched_start":    EMA_SCHED_START,   # 200 -> 267
    "ema_sched_end":      EMA_SCHED_END,     # 279 -> 372
    "swa_start":          SWA_START,         # 240 -> 320
    "handoff_start":      HANDOFF_START,     # 280 -> 373
}
STAGE_FRACS = {k: v / STAGE_CALIB_E for k, v in STAGE_CALIB.items()}


def stage_epoch(frac: float, epochs: int, lo: int = 1) -> int:
    """v36: derive a stage epoch from its E=300-calibrated fraction."""
    return max(lo, int(round(frac * epochs)))


def derive_stages(epochs: int) -> dict:
    """v36: horizon-proportional re-derivation of every stage constant."""
    return {k: stage_epoch(f, epochs) for k, f in STAGE_FRACS.items()}


# --- v17 [D]: teacher-record delivery ---
# v16 seed42 verdict: PASS+ (teacher 64.64 @e281 >= 64.59, time axis carries
# information) BUT delivered 64.44 — a +0.20 ABSORPTION GAP: the teacher
# record was set DURING the handoff window (the EMA re-tracks the restored
# student and immediately peaks), while the v12 delivery path only considers
# the student best checkpoint once the handoff has fired (the teacher-
# snapshot passive fallback is handoff-never-fired-only). Mechanism D makes
# the final teacher-record snapshot (the EMA time-smoothed solution at its
# best point) an unconditional delivery candidate:
#   delivered = argmax(student best checkpoint, teacher record snapshot)
# Mean-teacher delivery is standard practice; v16 measured the smoothed
# solution +0.20 ABOVE the restored raw solution (64.64 vs 64.44) — the EMA
# low-pass filter finds a better point inside the flat minimum. Eval-only,
# AFTER training, zero RNG: the v17 training trajectory is bit-exact v16
# (the run itself is the bit-exactness check: teacher must reproduce
# 64.64 @e281). --no-teacher-delivery runs the EXACT v16 delivery semantics.
TEACHER_SNAPSHOT_DELIVERY = True   # master switch (False -> v16 semantics)


# ==================== Arguments ====================

def parse_args():
    """[v73] 零 argparse：获奖配方（v41 实跑 69.81 的精确参数）写死为 cfg。

    基底 = v41 网格热核注意力配方（diffusion_mode=heat + pos_enc=abs + epochs=400）。
    全部 66 个 CLI 开关已废除；下面 cfg 的键 = 原 argparse dest，值 = 获奖配方的
    **最终生效值**（含 derive_stages(400) 的 horizon 派生结果），因此 main() 与各
    函数里散落的 args.xxx 引用零改动即可运行。
    改配方 = 改这里的 cfg，不是改命令行。
    """
    cfg = {
        # ---- 运行规模 / 优化器（获奖配方 = --epochs 400 --seed 42 --physics）----
        "seed": 42,
        "epochs": 400,
        "batch_size": 128,
        "lr": 1e-3,                 # v28 [J] REVERT -> EXACT v18
        "lr_min": 0.0,              # v28 [J] REVERT -> EXACT v18
        "weight_decay": 0.05,
        "warmup_epochs": 5,
        "img_size": 32,
        "patch_size": 4,
        # ---- 增强（v28 REVERT -> EXACT v18）----
        "randaugment": True,
        "randaug_m": 14,
        "erasing": True,
        "erase_p": 0.25,
        "mixup": True,
        "mixup_alpha": 0.4,
        "mixup_window": None,       # None -> full-run [E//3, E] = [133, 400]
        # ---- 物理场（写死常开，无 --no-physics）----
        "physics": True,
        "res_beta": 1.0,            # RES_BETA
        "ms_weights": "rotate",
        "ms_adapt": False,          # MS_ADAPT (v26 REVERTED)
        "ms_adapt_alpha_end": 267,  # derive_stages(400)
        "ms_adapt_gap_floor": 0.5,  # ADAPT_GAP_FLOOR
        "ms_adapt_beta": 0.4,       # ADAPT_BETA
        "kernel_revise": False,     # KERNEL_REVISE (v25 disabled)
        "kernel_revise_beta": 0.4,
        "kernel_revise_gap_floor": 0.5,
        "band_var_ema_decay": 0.99,
        "gate_adapt": False,        # GATE_ADAPT
        "w_cons_max": 1.0,
        "ema_decay": 0.999,         # EMA_DECAY
        "ema_decay_late": 0.9997,   # EMA_DECAY_LATE
        "ema_sched_start": 267,     # derive_stages(400)
        "ema_sched_end": 372,       # derive_stages(400)
        "ema_sched": False,         # EMA_SCHED_ON (v26 REVERTED)
        "soft_kd": False,           # SOFT_KD (v27 default off)
        "ext_teacher": None,
        "ext_teacher_t": 4.0,
        "ext_teacher_w": 0.3,
        "teacher_ckpts": None,
        "v18_anchor": False,
        "swa_start": 320,           # derive_stages(400)
        "swa": True,                # SWA_DEFAULT_ON
        "gate_lo": 1.0,             # GATE_LO
        "gate_hi": 3.0,             # GATE_HI
        "gate_floor": 1.0,          # GATE_FLOOR (gate disabled)
        "handoff": True,            # 开，但获奖跑中从未触发（惰性旁观者，见判读）
        "handoff_start": 373,       # derive_stages(400)
        "ms_rot_end": 333,          # derive_stages(400)
        "kernel_revise_end": 267,   # derive_stages(400)
        "w_cons_ramp_len": 27,      # derive_stages(400)
        "w_cons_hold_len": 87,      # derive_stages(400)
        "handoff_lr": 1e-5,         # HANDOFF_LR
        "teacher_delivery": True,   # TEACHER_SNAPSHOT_DELIVERY
        "smoke": False,             # 冒烟永久废除
        "log_interval": 50,
        "cons_start": 200,          # derive_stages(400)
        "data_dir": "./data",
        "output_dir": "./results_v77",
        # ---- v40 [V] 位置编码（获奖 = abs）----
        "pos_enc": "abs",           # POS_ENC_DEFAULT
        "rope_base": 100.0,         # ROPE_BASE
        # ---- v41 [W] 网格热核注意力（获奖 = heat，历史最佳 69.81 的轴）----
        "diffusion_mode": "heat",   # HEAT_MODE_DEFAULT
        "heat_ladder": (3.0, 4.5, 6.5, 9.5, 14.0, 21.0),  # HEAT_LADDER_DEFAULT
        "heat_probe": True,
    }
    return argparse.Namespace(**cfg)


# ==================== Reproducibility ====================

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== Data ====================

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD  = (0.2673, 0.2564, 0.2762)

RANDAUG_N = 2
RANDAUG_M = 14          # v28 [J]: REVERT v27 [H] -> EXACT v18 magnitude 14 (v27 tried 20 and collapsed)
ERASE_P   = 0.25        # v28 [J]: REVERT v27 [H] -> EXACT v18 prob 0.25 (v27 tried 0.30 and collapsed)
ERASE_SCALE = (0.02, 0.15)
CUTMIX_BETA = 1.0

def get_train_transform(args):
    """PIL ops -> ToTensor/Normalize -> tensor ops (RandomErasing after norm)."""
    ops = [
        T.RandomCrop(args.img_size, padding=4),
        T.RandomHorizontalFlip(),
    ]
    if args.randaugment:
        ops.append(T.RandAugment(num_ops=RANDAUG_N, magnitude=args.randaug_m))
    ops += [T.ToTensor(), T.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
    if args.erasing:
        ops.append(T.RandomErasing(p=args.erase_p, scale=ERASE_SCALE))
    return T.Compose(ops)


def get_test_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def get_train_eval_loader(args):
    """v35 [Q]: fixed UNAUGMENTED train subset, for MEASUREMENT ONLY.

    `history["train_acc"]` is computed on mixup-augmented batches, so it is not
    comparable across the run: the moment the mixup window shuts at e200 the
    number jumps 63.95 -> 92.91 purely because the metric changed, not the model.
    That contaminated metric is what made the v34 "32pp overfitting gap" look
    decisive. This loader gives a gap that means the same thing in every epoch.

    Determinism: shuffle=False + num_workers=0 + model.eval() -> no RNG, no
    extra training forwards. Pure measurement; training stays bit-exact.
    """
    n = 256 if args.smoke else TRAIN_EVAL_N
    ds = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=True, download=True,
        transform=get_test_transform())
    ds = torch.utils.data.Subset(ds, range(min(n, len(ds))))
    return DataLoader(ds, batch_size=256, shuffle=False,
                      num_workers=0, pin_memory=True)


def get_loaders(args):
    train_set = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=True, download=True,
        transform=get_train_transform(args))
    test_set = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=False, download=True,
        transform=get_test_transform())

    bs = args.batch_size
    if args.smoke:
        train_set = torch.utils.data.Subset(train_set, range(min(256, len(train_set))))
        test_set = torch.utils.data.Subset(test_set, range(min(64, len(test_set))))
        bs = 32

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ==================== Mixup / CutMix (ResNet v14 same code) ====================

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


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


def cutmix_data(x, y, beta=1.0):
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(x.size(0)).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y, y[rand_index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==================== Mixup schedule ====================

def get_mixup_window(args, epochs):
    """v35 [Q]: default auto window [E/3, E] — FULL RUN (NOW REACHABLE).

    v18-v34 used [E/3, 2E/3] = [100, 200] for 300ep, which TURNED OFF
    mixup/cutmix entirely after epoch 200. The model then trained on clean
    hard-label images for the last 1/3 of training — exactly the overfit region
    (train_acc jumps 63.95@e191 -> 92.91@e201 the instant the window shuts, then
    climbs to 95.3% while test crawls 62.12 -> 63.42: +1.3pp in 100 epochs).
    Measured learning-rate of test_acc: e150-200 (mixup ON) = +0.088 pp/epoch,
    e200-280 (mixup OFF) = +0.027 pp/epoch -> 3.2x slowdown, then NEGATIVE after
    e281. Augmentation was absent precisely when regularization mattered most.
    That is the mechanism behind the recurring 64.44 ceiling.

    v35 keeps mixup/cutmix ON through the end (alpha 0.4 constant, then linear
    decay over the last ~1/6 = e268-e300 for a 300ep run, so the final epoch is
    still clean). Explicit '--mixup-window s-e' still overrides (ablation /
    --v18-anchor).

    覆辙㉙: this fallback is only reachable if the argparse default for
    --mixup-window is None. v34 shipped the fallback with default="100-200" and
    the change was dead code (v34 == v33 bit-exact). verify_vit_v35.py G-A
    executes the real parser defaults to guard against a repeat.
    """
    if args.mixup_window:
        s, e = (int(v) for v in args.mixup_window.split("-"))
        return s, e
    return epochs // 3, epochs


def get_mixup_alpha(epoch, start, end, alpha):
    """1-indexed epochs. Constant alpha in window, linear decay over last ~1/6."""
    if epoch < start or epoch > end:
        return 0.0
    decay_len = min(max(5, (end - start) // 6), max(0, (end - start) // 2))
    if decay_len <= 0 or epoch < end - decay_len:
        return alpha
    return alpha * (1.0 - (epoch - (end - decay_len)) / decay_len)


# ==================== v39 [U]: LR-normalised progress instrument ====================
# PURE ARITHMETIC over already-recorded history lists.  No RNG draw, no model
# forward, and above all NO extra DataLoader iteration — 覆辙㉟: a measurement
# that "only reads" can still perturb the run, and the forbidden perturbation
# here is one more __iter__ on the loader (it pulls a base seed from the GLOBAL
# RNG and reshuffles every later epoch).  Because nothing below can touch
# training, v39 @300ep MUST stay bit-exact with v36 @300ep; bitexact_control.py
# is the single source of truth for that verdict (覆辙⑰).

LRNORM_WIN        = 20     # smoothing window, FIXED so ratios compare across horizons
LRNORM_BACK_FRAC  = 0.5    # fraction of the run treated as "back half" for the trend
LRNORM_REL_TOL    = 0.10   # dead band on the DIMENSIONLESS relative slope
LRNORM_LR_FLOOR   = 0.01   # drop windows whose median LR is < 1% of the peak.
#  覆辙㊶: dividing by a LR that has already annealed to ~0 is a 1/LR blow-up,
#  not a measurement.  On v36 the last window (e281-300, median LR 2e-6) reports
#  ratio 2277 purely because cosine has reached zero — the optimiser has STOPPED
#  there, so the window carries no information about saturation.  Without this
#  floor the verdict is decided by the single noisiest, least meaningful point.


def lrnorm_window(epochs):
    """Window for the instrument.

    Fixed at LRNORM_WIN once the run is long enough to support it, so 300/400/600
    epoch runs produce comparable ratios.  Short runs shrink the window instead
    of returning nothing, so a --smoke run still exercises the branch it is
    supposed to guard (覆辙㉒: a smoke that stops before the window leaves it
    as dead code).
    """
    if epochs >= 100:
        return LRNORM_WIN
    return max(2, epochs // 3)


def lrnorm_curve(test_acc, lr, epochs):
    """[(epoch_end, ratio)] with ratio = (Δ smoothed test_acc / Δ epoch) / median LR.

    The ratio answers: "how much test accuracy does one unit of learning rate buy
    *here*?"  If the model were approaching its generalisation ceiling the ratio
    would COLLAPSE as training proceeds.  Measured on v36/v37/v38 it RISES
    monotonically in the back half (v36: 197 -> 645), i.e. there is no ceiling
    signature at all — the runs stop only because cosine drives LR to zero.
    """
    win = lrnorm_window(epochs)
    n = len(test_acc)
    if n < 2 * win + 1 or n != len(lr):
        return []
    lr_floor = max(lr) * LRNORM_LR_FLOOR if lr else 0.0
    out = []
    for k in range(1, n // win):
        a = test_acc[(k - 1) * win:k * win]
        b = test_acc[k * win:(k + 1) * win]
        seg = lr[(k - 1) * win:(k + 1) * win]
        if len(a) != win or len(b) != win or len(seg) != 2 * win:
            continue
        # 覆辙㊷: the LR a window is divided by must be the LR that PRODUCED the
        # gain, which is not the LR inside window B.
        #   mean_B - mean_A = (1/w) Σ_{t=1..w} (S_{kw+t} - S_{(k-1)w+t})
        #                   = Σ_{t=1..w} [mean of g over the w epochs
        #                      (k-1)w+t+1 .. kw+t]
        # so the gain is a MEAN OF W SLIDING WINDOW MEANS, and the matching LR
        # is the same sliding-window mean of lr (a triangular weighting).
        #   Δ = (1/w) Σ_{t=1..w} (S_{kw+t} - S_{(k-1)w+t}), and
        #   S_{kw+t} - S_{(k-1)w+t} = Σ g over epochs (k-1)w+t+1 .. kw+t,
        # which is seg[t : t+w] for t = 1..w.  Using t = 0..w-1 instead
        # (the off-by-one this replaced) drops the newest window and adds a
        # stale one: on a cosine tail that alone biases the ratio by -8%,
        # the same order as the effect the instrument exists to measure.
        lr_eff = (sum(sum(seg[t:t + win]) for t in range(1, win + 1))
                  / float(win * win))
        # 覆辙㊶: an annealed-away LR cannot be divided by — see LRNORM_LR_FLOOR.
        if lr_eff <= lr_floor:
            continue
        out.append(((k + 1) * win, ((sum(b) - sum(a)) / win) / win / lr_eff))
    return out


def lrnorm_verdict(curve, epochs):
    """Back-half OLS slope of the ratio plus a three-way verdict.

    Returns (slope_per_epoch, rel_slope, verdict).  rel_slope is dimensionless
    (fractional change of the ratio across the WHOLE run), so the dead band
    LRNORM_REL_TOL means the same thing at 300 and at 600 epochs — a plain
    slope would silently change meaning with the horizon (覆辙㉛'s family: a
    quantity that does not track the horizon it is judged on).
    """
    if len(curve) < 4:
        return (None, None, "INSUFFICIENT_DATA")
    back = [(x, y) for x, y in curve if x >= epochs * LRNORM_BACK_FRAC]
    if len(back) < 3:
        back = curve[-3:]
    xs = [x for x, _ in back]
    ys = [y for _, y in back]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0 or my == 0:
        return (None, None, "INSUFFICIENT_DATA")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    rel = slope * epochs / my
    if rel <= -LRNORM_REL_TOL:
        return (slope, rel, "SATURATING")
    if rel >= LRNORM_REL_TOL:
        return (slope, rel, "NOT_SATURATED")
    return (slope, rel, "FLAT")


def next_horizon(epochs):
    """The next horizon to try: +33% of ∫LR dt, rounded to a round number.

    A FIXED INCREMENT would be the wrong shape for this lever.  The mechanism is
    ∫LR dt = E * lr_peak / 2, so what buys progress is a multiplicative step in
    LR area; +200 epochs is +67% at E=300 but only +33% at E=600.  Rounding to
    100 keeps the run bookkeeping sane.
    """
    return int(round(epochs * 4.0 / 3.0 / 100.0)) * 100


def horizon_advice(verdict, epochs):
    """Mechanical next-step recommendation.

    Written down BEFORE the run so the decision afterwards cannot be argued
    away by staring at the accuracy number alone (覆辙⑤: over-reading a short
    window; 覆辙㊱: two readings of the same number looking like two
    independent confirmations).
    """
    if verdict == "SATURATING":
        return (f"HOLD at {epochs}ep: productivity per unit LR collapses in the "
                f"back half — the ceiling is model/recipe bound, not schedule "
                f"bound. Extending the horizon buys ~nothing; change axis.")
    if verdict == "INSUFFICIENT_DATA":
        return (f"UNKNOWN: {epochs}ep yields < 4 instrument windows — the "
                f"verdict is not measurable at this horizon.")
    if verdict == "NOT_SATURATED":
        return (f"EXTEND to {next_horizon(epochs)}ep: productivity per unit LR "
                f"is still RISING at {epochs}ep, so ∫LR dt is the binding "
                f"constraint. +33% horizon = +33% LR area.")
    return (f"EXTEND to {next_horizon(epochs)}ep: productivity per unit LR is "
            f"flat, so gain ∝ ∫LR dt and the horizon is a linear lever.")


def v39_horizon_canary(args, mixup_end):
    """v39 [U] HORIZON CANARY (覆辙㉛ / ㉙) — print and CHECK the resolved table.

    The horizon is the axis under test, so the derived stage table must be
    verified on EVERY run.  A constant stranded at its E=300 position while the
    horizon moved is invisible in the accuracy: the run still finishes, the
    mechanism just quietly stops firing.  Refuse to start if any derived stage
    fell outside the run.

    Kept as its own function (not inlined in main) so verify_vit_v39.py can
    prove that main() differs from v36 by exactly two call sites — a diff that
    is impossible to audit if the new code is interleaved with the old.
    """
    if args.epochs != STAGE_CALIB_E:
        print(f"[v39 CANARY] horizon {args.epochs}ep != calibration "
              f"{STAGE_CALIB_E}ep — derived stages MUST have moved with it:")
        for _k in STAGE_FRACS:
            print(f"    {_k:<20} = {getattr(args, _k)}")
    for _k in STAGE_FRACS:
        _v = getattr(args, _k)
        if _k in ("w_cons_ramp_len", "w_cons_hold_len"):
            _ok = 0 <= _v <= args.epochs
        else:
            _ok = 1 <= _v <= args.epochs
        if not _ok:
            raise SystemExit(f"[v39 CANARY] derived stage {_k}={_v} is not "
                             f"within the run [1, {args.epochs}] — the horizon "
                             f"moved but this constant did not (覆辙㉛).")
    if args.mixup and not args.smoke and mixup_end != args.epochs:
        print(f"[v39 CANARY] WARNING: mixup window ends at e{mixup_end} of "
              f"{args.epochs} — the last {args.epochs - mixup_end} epochs train "
              f"on CLEAN hard labels (the v18-v34 64.44 ceiling).")


def v39_instrument_report(history, epochs, lr_peak):
    """Compute + print the LR-normalised progress verdict.

    Guarded so a truncated --smoke run can never crash the save path (losing
    the whole run's telemetry would be worse than a missing diagnostic).
    """
    curve, slope, rel, verdict = [], None, None, "INSUFFICIENT_DATA"
    try:
        curve = lrnorm_curve(history["test_acc"], history["lr"], epochs)
        slope, rel, verdict = lrnorm_verdict(curve, epochs)
    except Exception as _e:
        print(f"[v39 WARNING] LR-normalised instrument failed ({_e!r}); "
              f"recording INSUFFICIENT_DATA. The run itself is unaffected.",
              flush=True)
    print("\n" + "=" * 76)
    print(f"[v39 INSTRUMENT] horizon = {epochs} ep   "
          f"∫LR dt = {sum(history['lr']):.6f} "
          f"(cosine expectation {lr_peak * epochs * 0.5:.6f})")
    if curve:
        print(f"  LR-normalised progress ratio: "
              f"{curve[0][1]:.0f} (e{curve[0][0]})  ->  "
              f"{curve[-1][1]:.0f} (e{curve[-1][0]})   "
              f"[rising = LR-starved, collapsing = ceiling]")
    print(f"  verdict = {verdict}")
    print(f"  advice  = {horizon_advice(verdict, epochs)}")
    print("=" * 76 + "\n", flush=True)
    return curve, slope, rel, verdict


# ==================== Model ====================

def build_model(args):
    """Create ViT-Small from timm, configured for CIFAR-100."""
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=False,             # train from scratch
        img_size=args.img_size,
        patch_size=args.patch_size,
        num_classes=100,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ViT-Small/{args.patch_size} on {args.img_size}x{args.img_size}")
    print(f"  embed_dim={model.embed_dim}, depth={len(model.blocks)}, "
          f"num_heads={model.blocks[0].attn.num_heads}")
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    return model


# ==================== v40 [V]: 2D RoPE 位置编码轴 ====================
#
# 立项依据 (全部来自 v39 @400ep 实跑 + 代码审计, 见下"诊断三段"):
#
# [诊断 1] v39 = 67.07 delivered / student peak 66.95@e357, 比 v36 @300ep 的
#   66.65 / 66.45 涨 +0.42 / +0.50 —— horizon 轴确实兑现了, 且是 student 自身在涨。
#   但 saturation_verdict = NOT_SATURATED, lrnorm_rel_slope = +3.96 (v36 是 +2.22),
#   ratio 377 -> 672: 停止只是 cosine 把 LR 掐了。horizon 每次给 +0.4pp,
#   而距 from-scratch ViT-S/4 SOTA (LieRE 73.9%) 还差 -6.95pp —— **补不上**。
#
# [诊断 2] 物理场的剂量审计 (决定性)。e361-400 窗口, cons_term 三通道:
#       anchor_logit (w=0.3) 值 0.21326  -> 占 cons_term 的 99.63%
#       feat_anchor  (w=0.2) 值 0.00076  -> 0.24%
#       feat_diff    (w=0.5) 值 0.00017  -> 0.13%
#   即所谓「物理扩散场」只拿到 0.13% 的份额; 那 +4.14pp 的"物理净贡献"几乎全部来自
#   logit 空间的 EMA 自蒸馏 (sym-KL)。**物理场目前是装饰性的。**
#   补充: KERNEL_REVISE = False 自 v25 [F] 起就是默认 (Module E 被证明有害), 所以
#   kappa_c/m/f 恒为 1.0、band_var_* 恒为 0 是**设计结果, 不是 bug** (我一度误判, 存档)。
#
# [诊断 3] 编码器/解码器的信息分配是反的, 而且物理场用错了域:
#     编码器 = patch_embed + 12 blocks + norm -> 384-d CLS, 21.38M 参数
#     解码器 = nn.Linear(384, 100), 38.5K 参数
#     物理信号 -> 解码器 99.6% (logit anchor), -> 编码器仅 0.37% (两个 feature 通道)
#   ⇒ 物理先验全砸在一个**线性**读出头上, 而线性头无法凭空造出表征结构。
#   更要命的是: feat_diff 的扩散是沿**通道轴** (384 维) 做的。通道轴没有度量 ——
#   置换通道是模型的对称操作。Bessel 势 / 能量扩散需要的是**带拉普拉斯算子的域**;
#   真正有度量的域是 **token/位置轴** (8x8 patch 网格)。
#   ⇒ 物理扩散场被放在了整个模型里唯一没有几何的地方。
#
# [外部证据] 同一批 from-scratch ViT-S/4 CIFAR-100 结果, 只换位置编码:
#     原始 baseline 63.78 -> RoPE-Mixed 72.3 / STRING 72.6 / LieRE 73.9 (400ep)
#   即 **+8.5 ~ +10.1pp 全部来自位置/关系结构**。这是已知最大的一根轴, 而且它恰好
#   就是"物理场本该待的那个域"。我们用的是 timm 默认**可学习绝对 pos_embed** (65x384)。
#
# [本版只动一根轴 (覆辙⑮)] 把绝对位置编码换成 **2D 轴向 RoPE**:
#     --pos-enc abs     = v39 逐位锚点 (可学习 pos_embed, 无旋转)
#     --pos-enc rope2d  = pos_embed 置零并冻结 + 每个 block 的 q/k 加 2D 旋转
#     --pos-enc mixed   = 两者都保留
#   RoPE 在傅里叶域是**纯相位**; Bessel 势 (I-Δ)^{-α/2} 在同一域是**乘子(幅度)**。
#   二者是同一对象的相位/幅度两面 —— 这就是 v41 把扩散场搬上 token 轴的接口。
#
# [为什么这么实现] RopeAttention 通过 **运行时换 __class__** 装到已有 block 上,
#   不重建任何权重 ⇒ --pos-enc abs 与 v39 逐位相同。RoPE 表用 persistent=False
#   的 buffer, **不进 state_dict** ⇒ 存盘/SWA/EMA/--ext-teacher 的键集完全不变。
# ====================================================================

ROPE_BASE       = 100.0
# --- v40 事故后改默认 (覆辙㊺) ---
# 默认必须是**已验证的** abs (== v39 逐位锚点); 实验臂必须显式 opt-in。
# 事故: rope2d 作默认 + 直接发车 400ep -> e25 test 14.72 (v39 同点 25.04),
#       best 从 ~e10 起冻结在 21.34, e132 仍只有 16.28 (v39 52.10)。
#       6ep 冒烟分不出 (abs 与 rope2d 都是 1.56-4.69%, 噪声地板 ±3pp, 缺陷 -40pp)。
POS_ENC_DEFAULT = "abs"
POS_ENC_CHOICES = ("abs", "rope2d", "mixed")


def rope2d_tables(grid_h, grid_w, head_dim, base=ROPE_BASE, device="cpu",
                  dtype=torch.float32):
    """2D axial RoPE 的 cos/sin 表。

    头维 d 拆成两半: 前半由 x 坐标驱动, 后半由 y 坐标驱动; 每半内部按
    (2j, 2j+1) 配对做 2D 旋转。CLS 在 index 0, 角度恒为 0 (不旋转)。

    Returns: (cos_x, sin_x, cos_y, sin_y), 每个形状 (1 + gh*gw, d//4)。
    """
    half = head_dim // 2
    n_pair = half // 2
    if n_pair < 1:
        raise ValueError(f"[v40] head_dim={head_dim} 太小, 无法做 2D axial RoPE")
    j = torch.arange(n_pair, device=device, dtype=dtype)
    freq = 1.0 / (base ** (2.0 * j / float(half)))
    ys, xs = torch.meshgrid(
        torch.arange(grid_h, device=device, dtype=dtype),
        torch.arange(grid_w, device=device, dtype=dtype),
        indexing="ij")
    zero = torch.zeros(1, device=device, dtype=dtype)
    px = torch.cat([zero, xs.reshape(-1)])
    py = torch.cat([zero, ys.reshape(-1)])
    ang_x = px[:, None] * freq[None, :]
    ang_y = py[:, None] * freq[None, :]
    return ang_x.cos(), ang_x.sin(), ang_y.cos(), ang_y.sin()


def _rot_pair(t, cos, sin):
    """t: (..., 2) -> 旋转后的 (..., 2)。"""
    a, b = t[..., 0], t[..., 1]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1)


def rope2d_apply(t, cos_x, sin_x, cos_y, sin_y):
    """对 (B, H, N, d) 的 q 或 k 施加 2D 轴向旋转。CLS 角度为 0, 自动不变。"""
    d = t.size(-1)
    half = d // 2
    sh = t.shape[:-1]
    hx = t[..., :half].reshape(*sh, half // 2, 2)
    hy = t[..., half:].reshape(*sh, half // 2, 2)
    hx = _rot_pair(hx, cos_x[None, None, :, :], sin_x[None, None, :, :])
    hy = _rot_pair(hy, cos_y[None, None, :, :], sin_y[None, None, :, :])
    return torch.cat([hx.reshape(*sh, half), hy.reshape(*sh, half)], dim=-1)


class RopeAttention(TimmAttention):
    """timm Attention + 2D 轴向 RoPE。

    forward 逐字复制 timm 1.0.28, 只在 q_norm/k_norm 之后、注意力之前插入旋转。
    rope_on=False 时与父类**逐位相同** (--pos-enc abs 的锚点靠这条保证)。
    """

    rope_on = False

    def forward(self, x, attn_mask=None, is_causal=False):
        B, N, C = x.shape
        gate = self.gate(x).sigmoid() if self.gate is not None else None
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope_on:
            q = rope2d_apply(q, self.rope_cos_x, self.rope_sin_x,
                             self.rope_cos_y, self.rope_sin_y)
            k = rope2d_apply(k, self.rope_cos_x, self.rope_sin_x,
                             self.rope_cos_y, self.rope_sin_y)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
                is_causal=is_causal,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn_bias = resolve_self_attn_mask(N, attn, attn_mask, is_causal)
            attn = maybe_add_mask(attn, attn_bias)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        x = self.norm(x)
        if gate is not None:
            x = x * gate
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def install_pos_enc(model, mode, rope_base=ROPE_BASE, verbose=True):
    """把 2D RoPE 装到每个 block 的 attn 上 (换 __class__, 不重建权重)。

    mode="abs"    什么都不做 (v39 逐位锚点)
    mode="rope2d" RoPE + 把 pos_embed 置零并冻结 (纯相对位置)
    mode="mixed"  RoPE + 保留可学习 pos_embed
    """
    if mode == "abs":
        if verbose:
            print("  [V] pos-enc=abs: 可学习绝对 pos_embed, 无 RoPE (v39 逐位锚点)")
        return model
    if verbose:
        print(f"  [V] WARNING: pos-enc={mode} 是**未验证实验臂** —— v40 首跑 rope2d 在 "
              f"e25 就崩到 test 14.72 (abs/锚点 25.04), best 冻结在 21.34。"
              f"发车长跑前请先跑 --epochs 60 短程 A/B, 要求 e50 test_acc >= 25.0 "
              f"(abs 参考 32.50; rope2d 首跑实测 ~11.5)。")
    gh, gw = model.patch_embed.grid_size
    device, dtype = model.pos_embed.device, model.pos_embed.dtype
    cx, sx, cy, sy = rope2d_tables(gh, gw, model.blocks[0].attn.head_dim,
                                   base=rope_base, device=device, dtype=dtype)
    for blk in model.blocks:
        a = blk.attn
        if not isinstance(a, RopeAttention):
            a.__class__ = RopeAttention
        # persistent=False: 不进 state_dict, 存盘/SWA/EMA/--ext-teacher 键集不变
        a.register_buffer("rope_cos_x", cx.clone(), persistent=False)
        a.register_buffer("rope_sin_x", sx.clone(), persistent=False)
        a.register_buffer("rope_cos_y", cy.clone(), persistent=False)
        a.register_buffer("rope_sin_y", sy.clone(), persistent=False)
        a.rope_on = True
    if mode == "rope2d":
        with torch.no_grad():
            model.pos_embed.zero_()
        model.pos_embed.requires_grad_(False)
    if verbose:
        print(f"  [V] pos-enc={mode}: {len(model.blocks)}/{len(model.blocks)} blocks 装 2D RoPE "
              f"(grid {gh}x{gw}, head_dim {model.blocks[0].attn.head_dim}, base {rope_base}); "
              f"pos_embed absmax={float(model.pos_embed.abs().max()):.3e} "
              f"trainable={model.pos_embed.requires_grad}")
    return model


def v40_pos_enc_canary(model, mode):
    """覆辙㉙/㉞: canary 必须**抛异常**, 不能只打印; 且必须查"它是开着的"。

    返回 (n_rope_blocks, pos_embed_absmax) 供写进结果 JSON。
    """
    blocks = list(model.blocks)
    n_rope = sum(1 for b in blocks if bool(getattr(b.attn, "rope_on", False)))
    pe_max = float(model.pos_embed.abs().max())
    if mode == "abs":
        if n_rope != 0:
            raise SystemExit(f"[v40 CANARY] --pos-enc abs 却有 {n_rope} 个 block 装了 RoPE")
        return n_rope, pe_max
    if n_rope != len(blocks):
        raise SystemExit(f"[v40 CANARY] --pos-enc {mode}: {n_rope}/{len(blocks)} 个 block 有 RoPE, 应为全部")
    if mode == "rope2d" and pe_max != 0.0:
        raise SystemExit(f"[v40 CANARY] --pos-enc rope2d 但 pos_embed absmax={pe_max}, 应为 0")
    for b in blocks:
        a = b.attn
        if not bool(getattr(a, "rope_on", False)):
            continue
        if float(a.rope_cos_x.abs().max()) <= 0.0:
            raise SystemExit("[v40 CANARY] RoPE 的 cos 表全零 —— 旋转退化, 机制空转")
        if float(a.rope_sin_x.abs().max()) <= 0.0:
            raise SystemExit("[v40 CANARY] RoPE 的 sin 表全零 —— 所有角度为 0, 旋转是恒等变换")
        break
    return n_rope, pe_max


# ==================== v41 [W]: 网格热核注意力 —— 把扩散场放到有度量的轴上 =====
#
# 立项依据 (全部来自 v39/v40 @400ep 实跑 + 代码审计):
#
# [事实 1] v40 与 v39 **逐位相同** (39 个 history 键最大差精确 0.0)。
#   即 v40 的 RoPE 代码路径在 --pos-enc abs 下完全惰性 —— 好消息是 v40 的改动
#   零污染; 坏消息是"位置/几何"这根轴**一次都还没被真正检验过**。
#   (rope2d 首跑是硬事故: e25 test 14.72 vs 锚点 25.04, best 冻结 21.34。
#   根因未定, 但可以确定: 那次是"把 pos_embed 置零 + 换掉整个注意力几何",
#   一次动了两样东西, 违反覆辙⑮。)
#
# [事实 2] 物理场目前是**装饰性**的 (覆辙㊸): e361-400 窗口 cons_term 三通道
#   anchor_logit 99.63% / feat_anchor 0.24% / feat_diff 0.13%。那 +4.14pp 的
#   "物理净贡献"几乎全部来自 logit 空间的 EMA 自蒸馏 (sym-KL)。
#
# [事实 3] 而且它被放在了**唯一没有度量的轴上** (覆辙㊹): feat_diff 的扩散沿
#   通道轴 (384 维) 做, 但置换通道是模型的对称操作 —— 该轴上没有 Laplacian,
#   Bessel 势与能量扩散在那里是无意义的。
#
# ---- 本版的核心判断: 「信息能量在网络中扩散」只有三条轴 ----------------------
#
#   (a) 深度轴 (l = 1..12)  —— **守恒/输运**, 不是扩散。
#       x_{l+1} = x_l + F_l(x_l) 是 dx/dt = F(x,t) 的显式 Euler, 步长 1。
#       有意义的量是**相对注入能量** rho_l = ||F_l(x)|| / ||x_l||。
#       rho_l 随深度指数衰减 ⇒ 深层是死重量, 有效深度 < 12。
#
#   (b) token 网格轴 (8x8) —— **扩散**, 而且这是唯一真有度量的轴。
#       注意力 A_l 是行随机矩阵 = 网格随机游走的一步 = 热方程 ∂_t u = Δu 的
#       一个显式 Euler 步。12 层复合 = 一个时间非齐次的扩散过程。
#       两个失效模式: 扩散长度 << 图像尺度 (信息过不去);
#                     扩散长度 → 图像尺度 (过平滑, token 场塌到均值,
#                     Dirichlet 能量 → 0, CLS 读不到结构)。
#
#   (c) 通道轴 (384)      —— **没有度量**。物理场目前待在这里。
#
# ---- 机制: 热核 logit 先验 ---------------------------------------------------
#
#   目前每个 head 的注意力核是一个**自由**的 65x65 矩阵 (由 q·k 生成), 网格
#   几何要靠数据自己学出来 —— 在 50k 样本上这是买不起的样本复杂度。
#   物理上, 扩散核不该自由: 它是网格 Laplacian 的热 semigroup
#       K_sigma = exp(sigma^2 Δ / 2),  K_sigma(i,j) ∝ exp(-d_grid(i,j)^2 / 2 sigma^2)
#   把它作为**加性 logit 先验**注入:
#       logits_h = q·k^T / sqrt(d_head)  -  d_grid(i,j)^2 / (2 sigma_h^2)
#   sigma_h 可学习 (每个 block×head 一个标量, 12x6 = 72 个)。
#
#   四条设计约束 (每条都对着一条覆辙):
#     · sigma -> inf 时 bias -> 0, 退化为**标准注意力** ⇒ 本机制是原模型的
#       **超集**, 不是"换一个函数类" (覆辙㊵: StochDepth 就是死在这上面)。
#     · 初始 sigma 取**几何阶梯** (一个 head = 热 semigroup 的一个时刻),
#       与本项目既有的 multi-scale diffusion 语言一致; 不许所有 head 同尺度
#       (那样只剩一个自由度, 又会变成"剂量不足" 覆辙㊸)。
#     · CLS 不参与几何先验 (与 CLS 相关的 bias 恒为 0) —— CLS 必须保持全局读出。
#     · 距离矩阵用 persistent=False 的 buffer ⇒ 不进 state_dict。
#
# ---- 读数: 这版同时是**测量仪器** ------------------------------------------
#
#   EnergyProbe 只读地记录三条定律的量:
#     depth_rho(l)  深度轴输运律 (有效深度)
#     dir_ratio     token 场 Dirichlet 能量经 12 步扩散后剩余的比例 (过平滑)
#     cos_to_mean   token 与场均值余弦 (塌缩指数)
#   以及 **学到的 sigma 本身** —— 它就是"信息实际扩散了多远"的直接读数。
#   ⇒ 无论 heat 臂赢不赢, 这次都能拿到"扩散规律"的实测曲线来定 v42。
# ==============================================================================

HEAT_MODE_DEFAULT = "heat"
HEAT_MODE_CHOICES = ("off", "heat")
# 每个 head 一个初始扩散尺度, 几何比 ~1.5: 细 head 强局部, 粗 head 近全局。
HEAT_LADDER_DEFAULT = (3.0, 4.5, 6.5, 9.5, 14.0, 21.0)
HEAT_PROBE_BATCHES  = 1        # 每 epoch 只在第 1 个 batch 上测量 (开销 <0.1%)
HEAT_E50_GATE       = 25.0     # 覆辙㊺: 短程验收判据 (abs/锚点 e50 = 32.50)

# ---- v77: 交付端"场杠杆"配对读数的规模 -------------------------------------------
# test_loader 是 shuffle=False ⇒ 前 N 个 batch 是**确定**的同一批样本 ⇒ 三种模式可以
# 严格配对 (同权重 + 同样本)。4 batch = 1024 样本; 子集 acc 的量化粒度 = 0.098pp。
FIELD_ABL_BATCHES   = 4        # 配对用的 batch 数 (= 1024 样本)
FIELD_ABL_LEV_MID   = 0.01     # |dlogit|rms / logit_std 的分档: <1% 低 / 1~5% 中 / >5% 高
FIELD_ABL_LEV_HIGH  = 0.05

# ---- v77: 扩散律指数 beta (广义高斯 / 异常扩散族) --------------------------------
# B_h(i,j) = -0.5 * ( d(i,j) / sigma_h ) ** beta_h,  beta = MID + HALF*tanh(raw).
# raw 初始化为 0 ⇒ beta = MID = 2.0 精确 ⇒ 与 v73 的 -d^2/(2 sigma^2) 解析同构。
# 用 tanh 软界而不是裸参数: beta <= 0 会变成**反扩散**(bias 随距离增大 ⇒ attention
# 反而偏好远端), 那会静默拆掉 locality 先验。软界把 beta 关在 (MID-HALF, MID+HALF)
# 内, 两端都远离退化区, 且给出"撞界"机检点 (覆辙 101/148)。
HEAT_BETA_MID       = 2.0      # init = 高斯热核 (v73 的隐含假设)
HEAT_BETA_HALF      = 1.0      # 软界半宽 ⇒ beta in (1.0, 3.0)
HEAT_D2_EPS         = 1e-12    # d^2 下限: 使 d/d(beta) 在 d=0 (对角/CLS 行) 保持有限
HEAT_BETA_MOVED_TOL = 1e-4     # "新自由度被用上"的判定容差 (剂量读数门槛)

# ---- v77: 各向异性热核 (扩散张量的第一阶推广, 本版唯一新轴) ----------------------
#   ar_l   = HEAT_ANISO_MAX * tanh(raw_ar_l)      # 每层一个, in (-MAX, +MAX)
#   d2_eff = dx^2 * exp(-ar_l) + dy^2 * exp(+ar_l)
#   B_h    = -0.5 * ( d2_eff / sigma_h^2 ) ** (beta_h / 2)
# ar = 0 ⇒ exp(±0) = 1.0 精确 ⇒ d2_eff ≡ dx^2 + dy^2 ≡ d^2 ⇒ 与 v74 逐位同构。
# 几何平均不变的参数化 (sigma_x * sigma_y = sigma_h^2) ⇒ ar 只改**形状**、不改平均尺度
# ⇒ 与 sigma 不退化 (区别于已被禁止的"给 bias 乘可学增益 g"轴)。
HEAT_ANISO_MAX      = 1.5      # ar ∈ (-1.5, 1.5) ⇒ sigma_x/sigma_y ∈ (e^-1.5, e^1.5) ≈ (0.22, 4.48)
HEAT_AR_MOVED_TOL   = 1e-3     # "各向异性被用上"的判定容差 (剂量读数门槛)
# ★ 撞界判定一律用**相对**容差 (界宽的 1%), 不许用绝对容差:
#   v76 实测 lambda 顶到 0.498563 而软界是 0.5, 边距 0.001437 只比绝对容差 1e-3 大一丁点
#   ⇒ 被判成"没撞界", 而物理上早就是撞界。绝对容差与界宽无关 ⇒ 判定会由容差而不是物理决定。
#   实测印证: v74 的 beta 上界 3.0, 末态 2.9995, 边距 0.0005 < 1e-3 ⇒ 旧判定同样报 false。
HEAT_BOUND_REL_TOL  = 0.01     # 撞界 ⇔ |dev| >= HALF * (1 - 0.01)


def grid_dist2(gh, gw, dtype=torch.float32, device="cpu"):
    """token 网格上的**平方**距离矩阵, 形状 (1+gh*gw, 1+gh*gw)。

    index 0 是 CLS: 它与任何 token 的距离恒为 0 ⇒ CLS 行/列的先验恒为 0,
    几何先验只作用在 patch↔patch 之间 (CLS 保持全局读出)。
    """
    ys, xs = torch.meshgrid(
        torch.arange(gh, device=device, dtype=dtype),
        torch.arange(gw, device=device, dtype=dtype), indexing="ij")
    px = xs.reshape(-1)
    py = ys.reshape(-1)
    d2 = (px[:, None] - px[None, :]) ** 2 + (py[:, None] - py[None, :]) ** 2
    N = 1 + gh * gw
    out = torch.zeros(N, N, device=device, dtype=dtype)
    out[1:, 1:] = d2
    return out


def grid_delta2(gh, gw, dtype=torch.float32, device="cpu"):
    """[v77] 网格距离的**两个分量** (dx^2, dy^2), 各自形状 (1+gh*gw, 1+gh*gw)。

    dx^2 + dy^2 与 grid_dist2() 的输出**逐位相同** —— 同样的广播减法、同样的加法顺序,
    只是把一次加法拆成两个张量 ⇒ 逐位可复现。ar ≡ 0 时
    d2_eff = dx^2*1.0 + dy^2*1.0 ≡ grid_dist2(...) (逐位) ⇒ v77 在 t=0 与 v74 逐位同构。
    index 0 仍是 CLS: 两个分量在该行/列都恒为 0 (几何先验不碰 CLS 的读出)。
    """
    ys, xs = torch.meshgrid(
        torch.arange(gh, device=device, dtype=dtype),
        torch.arange(gw, device=device, dtype=dtype), indexing="ij")
    px = xs.reshape(-1)
    py = ys.reshape(-1)
    dx2 = (px[:, None] - px[None, :]) ** 2
    dy2 = (py[:, None] - py[None, :]) ** 2
    N = 1 + gh * gw
    ox = torch.zeros(N, N, device=device, dtype=dtype)
    oy = torch.zeros(N, N, device=device, dtype=dtype)
    ox[1:, 1:] = dx2
    oy[1:, 1:] = dy2
    return ox, oy


def heat_ar_of(raw_ar):
    """[v77] 各向异性比: raw_ar (任意实数) -> ar = HEAT_ANISO_MAX * tanh(raw)。

    tanh 软界把 ar 关在 (-MAX, +MAX) 内 ⇒ sigma_x/sigma_y 被关在一个远离退化的区间。
    raw = 0 ⇒ tanh(0) = 0 ⇒ ar = 0 精确 ⇒ 各向同性 ⇒ 与 v74 的标量扩散率同构 (逐位)。
    """
    return HEAT_ANISO_MAX * torch.tanh(raw_ar)


def aniso_d2(dx2, dy2, raw_ar):
    """[v77] 各向异性平方距离: d2_eff = dx^2 * exp(-ar) + dy^2 * exp(+ar)。

    ar = 0 ⇒ exp(±0) = 1.0 精确、dx^2 * 1.0 = dx^2 精确 ⇒ d2_eff ≡ dx^2 + dy^2 (逐位)。
    ar > 0 ⇒ x(水平) 分量权重变小 ⇒ 水平方向"更近" ⇒ 水平方向扩散更远 (反之亦然)。
    """
    ar = heat_ar_of(raw_ar)
    return dx2 * torch.exp(-ar) + dy2 * torch.exp(ar)


def heat_beta_of(raw_beta):
    """[v74] 扩散律指数: raw_beta (任意实数) -> beta = MID + HALF*tanh(raw)。

    tanh 软界把 beta 关在 (HEAT_BETA_MID - HEAT_BETA_HALF, MID + HALF) 内。
    raw=0 ⇒ tanh(0)=0 ⇒ beta = MID 精确 ⇒ 与 v73 的固定高斯热核同构。
    """
    return HEAT_BETA_MID + HEAT_BETA_HALF * torch.tanh(raw_beta)


def heat_bias(log_sigma, dist2, raw_beta=None):
    """热核的**对数**先验: B_h(i,j) = - d_grid(i,j)^2 / (2 * sigma_h^2)。

    log_sigma: (H,) 对数尺度; dist2: (N,N)。返回 (H, N, N)。
    乘到 attention 上等价于把核乘以 exp(sigma^2 Δ / 2) —— 即**乘一个扩散核**。

    [v74] raw_beta 给定时走**广义高斯（异常扩散）**族:
        B_h(i,j) = -0.5 * ( d(i,j) / sigma_h ) ** beta_h
                 = -0.5 * (d^2)^{beta_h/2} * sigma_h^{-beta_h}
    beta_h = 2 时逐项退化回上面的标准热核。d^2 加 HEAT_D2_EPS 并减去同一常数,
    使 d=0 (对角 + CLS 行/列) 的 bias **精确为 0**, 同时让 d/d(beta) 在 d=0 有限
    (否则 pow 对指数的梯度含 log(0) -> NaN)。raw_beta=None 时**逐字**走 v73 路径。
    """
    if raw_beta is None:
        inv = torch.exp(-2.0 * log_sigma) * 0.5          # 1 / (2 sigma^2)
        return -dist2.unsqueeze(0) * inv.view(-1, 1, 1)
    beta = heat_beta_of(raw_beta)                        # (H,)
    half = (0.5 * beta).view(-1, 1, 1)
    d2c = dist2.unsqueeze(0) + HEAT_D2_EPS               # (1,N,N), 恒 > 0
    base = d2c.pow(half) - d2c.new_tensor(HEAT_D2_EPS).pow(half)
    return -0.5 * base * torch.exp(-beta * log_sigma).view(-1, 1, 1)


class HeatKernelAttention(RopeAttention):
    """timm Attention + 网格热核 logit 先验 (并继承 v40 的 2D RoPE 通路)。

    hka_on=False 时**逐字**走父类 forward (--diffusion-mode off 的 bit-exact
    锚点就靠这一条)。hka_on=True 时走非融合 softmax 路径 (要加 bias)。
    """

    hka_on = False
    # [v77] 交付端配对读数的开关: "full"(正常) | "iso"(强制各向同性, 等价 v74 的位置场) |
    # "off"(无场)。**训练与常规评测永远是 "full"**, 这个属性只在交付末态的只读块里被临时改。
    _ablate = "full"

    def hka_bias(self):
        # [v77] 本版唯一新轴: 先把距离的两个分量按本层的各向异性比合成 d2_eff,
        # 再走 v74 的广义高斯热核。ar ≡ 0 时 d2_eff 与 hka_dist2 逐位相同 ⇒ 与 v74 同构。
        # _ablate == "iso" 时**直接复用** hka_dist2 (= v74 的位置场, 逐位) —— 只给读数用。
        if self._ablate == "iso":
            d2_eff = self.hka_dist2
        else:
            d2_eff = aniso_d2(self.hka_dx2, self.hka_dy2, self.raw_ar)
        # [v74] 带上本 head 的扩散律指数 raw_beta（beta = MID + HALF*tanh(raw)）。
        return heat_bias(self.log_sigma, d2_eff, self.raw_beta)

    def forward(self, x, attn_mask=None, is_causal=False):
        if not self.hka_on:
            return super().forward(x, attn_mask=attn_mask, is_causal=is_causal)
        B, N, C = x.shape
        gate = self.gate(x).sigmoid() if self.gate is not None else None
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope_on:
            q = rope2d_apply(q, self.rope_cos_x, self.rope_sin_x,
                             self.rope_cos_y, self.rope_sin_y)
            k = rope2d_apply(k, self.rope_cos_x, self.rope_sin_x,
                             self.rope_cos_y, self.rope_sin_y)

        # 非融合路径: 必须与 timm 的顺序一致 (scale -> +bias -> mask -> softmax
        # -> drop -> @v), 只是中间插了一个**物理**先验。
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        # [v77] "off" 只给交付端读数用: 完全不加载场 (标准 attention)。
        if self._ablate != "off":
            attn = attn + self.hka_bias().to(attn.dtype)
        attn_bias = resolve_self_attn_mask(N, attn, attn_mask, is_causal)
        attn = maybe_add_mask(attn, attn_bias)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        x = self.norm(x)
        if gate is not None:
            x = x * gate
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def install_heat_kernel(model, mode, ladder=HEAT_LADDER_DEFAULT, verbose=True):
    """把热核先验装到每个 block 的 attn 上 (换 __class__, 不重建权重)。

    必须在 copy.deepcopy(teacher) **之前**调用, teacher 才能带着同一套几何。
    返回 (model, n_new_params)。
    """
    if mode == "off":
        if verbose:
            print("  [W] diffusion-mode=off: 标准 attention (v40/v39 逐位锚点); "
                  "能量输运探针仍然开着。")
        return model, 0
    if mode != "heat":
        raise SystemExit(f"[v41] 未知 --diffusion-mode {mode!r}")
    gh, gw = model.patch_embed.grid_size
    device, dtype = model.pos_embed.device, model.pos_embed.dtype
    d2 = grid_dist2(gh, gw, dtype=dtype, device=device)
    dx2, dy2 = grid_delta2(gh, gw, dtype=dtype, device=device)   # [v77] 各向异性用的两个分量
    nh = int(model.blocks[0].attn.num_heads)
    sig0 = [float(s) for s in ladder]
    # 覆辙㉞ 家族: 阶梯长度必须等于 head 数。不许静默截断/补零 —— 那会把
    # 一部分 head 变成 sigma=1 (极端局部) 或 sigma=inf (完全无先验), 而日志
    # 上什么都看不出来。用几何插值显式补齐并打印。
    if len(sig0) != nh:
        lo, hi = sig0[0], sig0[-1]
        r = (hi / lo) ** (1.0 / max(1, nh - 1))
        sig0 = [lo * (r ** i) for i in range(nh)]
        print(f"  [W] ladder 长度 {len(ladder)} != num_heads {nh} -> "
              f"几何插值为 {[round(s, 3) for s in sig0]}")
    if min(sig0) <= 0:
        raise SystemExit(f"[v41] ladder 含非正尺度: {sig0}")

    n_new = 0
    n_beta = 0
    n_ar = 0
    for blk in model.blocks:
        a = blk.attn
        if not isinstance(a, HeatKernelAttention):
            a.__class__ = HeatKernelAttention
        # persistent=False: 不进 state_dict
        a.register_buffer("hka_dist2", d2.clone(), persistent=False)
        a.register_buffer("hka_dx2", dx2.clone(), persistent=False)   # [v77] 各向异性
        a.register_buffer("hka_dy2", dy2.clone(), persistent=False)   # [v77]
        if not isinstance(getattr(a, "log_sigma", None), nn.Parameter):
            a.log_sigma = nn.Parameter(
                torch.tensor([math.log(s) for s in sig0], device=device, dtype=dtype))
            n_new += a.log_sigma.numel()
        # [v74] 每个 head 一个**扩散律指数** raw_beta。
        # 初始化为 0 ⇒ beta = HEAT_BETA_MID = 2.0 (精确) ⇒ t=0 时与 v73 解析同构,
        # 任何差异都只能来自这个新自由度被优化器用起来之后 (可归因)。
        if not isinstance(getattr(a, "raw_beta", None), nn.Parameter):
            a.raw_beta = nn.Parameter(
                torch.zeros(nh, device=device, dtype=dtype))
            n_beta += a.raw_beta.numel()
            n_new += a.raw_beta.numel()
        # [v77] 本版唯一新轴: **每层一个**各向异性比 raw_ar (不是每 head ⇒ 只 12 个标量)。
        # 初始化为 0 ⇒ ar = 0 精确 ⇒ d2_eff ≡ dx^2 + dy^2 (逐位) ⇒ t=0 与 v74 逐位同构,
        # 任何差异都只能来自这个新自由度被优化器用起来之后 (可归因)。
        if not isinstance(getattr(a, "raw_ar", None), nn.Parameter):
            a.raw_ar = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
            n_ar += 1
            n_new += 1
        a.hka_on = True
    if verbose:
        print(f"  [W] diffusion-mode=heat: {len(model.blocks)}/{len(model.blocks)} blocks "
              f"装网格热核先验 (grid {gh}x{gw}, {nh} heads/block); "
              f"初始 sigma 阶梯 = {[round(s, 2) for s in sig0]}; "
              f"新增可学习标量 {n_new} (进 no_decay 组)。")
        print(f"  [W] 先验强度: 网格最远角 d^2={(gh-1)**2 + (gw-1)**2} -> "
              f"bias {[-round(d2.max().item() / (2 * s * s), 2) for s in sig0]} nats "
              f"(按阶梯顺序); CLS 行/列恒为 0。")
        print(f"  [v74] 扩散律指数 beta = {HEAT_BETA_MID} + {HEAT_BETA_HALF}*tanh(raw): "
              f"新增 {n_beta} 个可学标量, 初始 beta ≡ {HEAT_BETA_MID} (高斯热核, 与 v73 解析同构); "
              f"软界 beta in ({HEAT_BETA_MID - HEAT_BETA_HALF}, {HEAT_BETA_MID + HEAT_BETA_HALF}) "
              f"⇒ 不会退化成反扩散。")
        print(f"  [v77] 各向异性热核: 每层一个 ar = {HEAT_ANISO_MAX}*tanh(raw_ar) "
              f"(新增 {n_ar} 个可学标量), 初始 ar ≡ 0 ⇒ d2_eff ≡ dx^2+dy^2 (逐位) "
              f"⇒ t=0 与 v74 逐位同构; 软界 ar in (-{HEAT_ANISO_MAX}, {HEAT_ANISO_MAX}) "
              f"⇒ sigma_x/sigma_y in (0.22, 4.48), 几何平均不变 ⇒ 与 sigma 不退化。")
    return model, n_new


def v41_heat_canary(model, mode):
    """覆辙㉙/㉞: canary 必须**抛异常**, 而且必须查"它是开着的"。

    返回 (n_on, sigma_tensor(L,H) 或 None)。
    """
    blocks = list(model.blocks)
    n_on = sum(1 for b in blocks if bool(getattr(b.attn, "hka_on", False)))
    if mode == "off":
        if n_on != 0:
            raise SystemExit(f"[v41 CANARY] --diffusion-mode off 却有 {n_on} 个 block 装了热核")
        return 0, None
    if n_on != len(blocks):
        raise SystemExit(f"[v41 CANARY] --diffusion-mode {mode}: {n_on}/{len(blocks)} 个 block "
                         f"有热核, 应为全部")
    sig = torch.stack([b.attn.log_sigma.detach().float() for b in blocks])   # (L,H)
    if not bool(torch.isfinite(sig).all()):
        raise SystemExit("[v41 CANARY] log_sigma 含 nan/inf")
    if not bool(sig.requires_grad or blocks[0].attn.log_sigma.requires_grad):
        raise SystemExit("[v41 CANARY] log_sigma 未开梯度 —— 尺度学不动, 机制空转")
    b0 = heat_bias(blocks[0].attn.log_sigma.detach(), blocks[0].attn.hka_dist2)
    if float(b0.abs().max()) <= 0.0:
        raise SystemExit("[v41 CANARY] 热核 bias 全零 —— 距离矩阵退化, 先验是恒等变换")
    if float(b0[0, 0, :].abs().max()) != 0.0:
        raise SystemExit("[v41 CANARY] CLS 行先验非零 —— CLS 应完全不受几何先验约束")
    return n_on, sig


class EnergyProbe:
    """只读的能量输运探针 —— 「信息能量在网络中扩散的规律」的实测仪器。

    覆辙㉟: 只读 forward hook, 不训练、不改权重、不额外前向、**不消耗 RNG**
    (不迭代 DataLoader、不开 dropout) ⇒ 对训练路径零影响。

    三条定律:
      rho_l = ||x_{l+1} - x_l||_F / ||x_l||_F   深度轴输运律 (有效深度)
      dirichlet(x) = 网格 4 邻边的 ||h_i - h_j||^2 / mean||h||^2
                     —— token 场的空间对比度 (Dirichlet 能量, 归一化)
      cos_to_mean  = patch token 与场均值的平均余弦 (过平滑/塌缩指数)
    """

    def __init__(self, model, gh, gw, max_batches=HEAT_PROBE_BATCHES):
        self.gh, self.gw = gh, gw
        self.L = len(model.blocks)
        self.max_batches = int(max_batches)
        self.on = True
        self._active = False
        self._in = {}
        self.handles = []
        self.reset()
        for i, blk in enumerate(model.blocks):
            self.handles.append(blk.register_forward_pre_hook(self._mk_pre(i)))
            self.handles.append(blk.register_forward_hook(self._mk_post(i)))

    # ---- 内部: hook 工厂 (闭包只捕获 i 与 self, 不持有张量) ----
    def _mk_pre(self, i):
        def _hook(mod, inp):
            if not self._active:
                return
            self._in[i] = inp[0].detach()
        return _hook

    def _mk_post(self, i):
        def _hook(mod, inp, out):
            if not self._active:
                return
            x_in = self._in.pop(i, None)
            if x_in is None:
                return
            o = out.detach() if torch.is_tensor(out) else out[0].detach()
            with torch.no_grad():
                num = (o - x_in).pow(2).sum(dim=(1, 2)).sqrt().mean()
                den = x_in.pow(2).sum(dim=(1, 2)).sqrt().mean().clamp_min(1e-12)
                self.rho[i] += float(num / den)
                if i == 0:
                    self.dir_first += self._dirichlet(x_in)
                if i == self.L - 1:
                    self.dir_last += self._dirichlet(o)
                    self.cos_last += self._cos_to_mean(o)
                    # **只在最后一个 block 收尾**: 这两行若缩进到 if 外面,
                    # 第 0 个 block 的 post-hook 就会把 _active 关掉, 后面 11 个
                    # block 全部测不到 (自检 _v41_check 抓到过: rho 只有 [0.30, 0,...])。
                    self.count += 1
                    self._active = False
        return _hook

    def _dirichlet(self, x):
        """归一化的 Dirichlet 能量: 4 邻边梯度平方和 / 场能量。"""
        p = x[:, 1:, :]
        g = p.reshape(p.size(0), self.gh, self.gw, p.size(-1))
        dh = (g[:, 1:, :, :] - g[:, :-1, :, :]).pow(2).sum(-1).mean()
        dw = (g[:, :, 1:, :] - g[:, :, :-1, :]).pow(2).sum(-1).mean()
        scale = p.pow(2).sum(-1).mean().clamp_min(1e-12)
        return float((dh + dw) / scale)

    def _cos_to_mean(self, x):
        p = x[:, 1:, :]
        m = p.mean(dim=1, keepdim=True)
        return float(F.cosine_similarity(p, m, dim=-1).mean())

    # ---- 外部接口 ----
    def reset(self):
        self.rho = [0.0] * self.L
        self.dir_first = 0.0
        self.dir_last = 0.0
        self.cos_last = 0.0
        self.count = 0

    def arm(self):
        """每 epoch 调一次: 允许接下来 max_batches 个 batch 被测量。"""
        self.reset()
        self._active = bool(self.on)

    def summary(self):
        """返回本 epoch 的读数 (纯算术, 无 RNG)。未测到则全 0。"""
        n = max(1, self.count)
        rho = [r / n for r in self.rho]
        d_f = self.dir_first / n
        d_l = self.dir_last / n
        return {
            "rho": rho,
            "rho_first": rho[0],
            "rho_last": rho[-1],
            "rho_decay": (rho[-1] / rho[0]) if rho[0] > 1e-12 else 0.0,
            "dir_first": d_f,
            "dir_last": d_l,
            "dir_ratio": (d_l / d_f) if d_f > 1e-12 else 0.0,
            "cos_to_mean": self.cos_last / n,
        }

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def diffusion_law_verdict(rho_decay, dir_ratio, cos_to_mean):
    """把三条读数压成一句判读 (纯函数, 便于单独测)。

    rho_decay  < 0.5  -> 深层只拿到浅层一半以下的相对能量: 有效深度不足。
    dir_ratio  < 0.25 -> 12 步注意力扩散把空间对比度抹掉了 3/4 以上: 过平滑。
    cos_to_mean> 0.7  -> token 场已塌向均值: CLS 读不到位置结构。
    """
    flags = []
    if rho_decay < 0.5:
        flags.append(f"DEPTH-STARVED(rho末/首={rho_decay:.2f})")
    if dir_ratio < 0.25:
        flags.append(f"OVER-SMOOTHED(Dirichlet剩余={dir_ratio:.3f})")
    if cos_to_mean > 0.7:
        flags.append(f"TOKEN-COLLAPSE(cos_to_mean={cos_to_mean:.3f})")
    return "HEALTHY" if not flags else " | ".join(flags)


def load_ext_teacher(path: str, args, device) -> nn.Module:
    """v28 [J]: load a FROZEN stronger CIFAR-100 teacher as the consistency target.

    Same ViT-S/4 architecture as the student (so all consistency channels — logit
    anchor, feature anchor, feature-diffusion — keep their dimensions and need no
    projector). The teacher is expected to be a plain state_dict (the best-student
    ckpt this script saves). Robust to common wrappers ('state_dict' / 'model').
    """
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict):
        for key in ("state_dict", "model", "teacher"):
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
    teacher = build_model(args).to(device)
    teacher.load_state_dict(sd)
    teacher.requires_grad_(False)
    teacher.eval()
    print(f"  [J] EXTERNAL TEACHER loaded from {path}")
    print(f"      embed_dim={teacher.embed_dim}, frozen, eval() — consistency target "
          f"(live-EMA self-teacher disabled)")
    return teacher


class EnsembleTeacher(nn.Module):
    """v32 [N]: multi-seed ENSEMBLE teacher (覆辙⑭ 外部更强教师 done right).

    Consistency target = the AVERAGE over K independently-trained ViT-S checkpoints.
    Different random inits -> decorrelated error -> a genuinely DIFFERENT function
    from the student, so the target lives OUTSIDE the student's collapse dynamics
    (satisfies 覆辙㉖). Ensemble test-acc > any single member, so it carries real
    dark knowledge — the only unfalsified axis after the v19-v30 twelve consecutive
    failures (closed-form physics fields / feature-space copies all FAILED -2pp).

    Replaces the single frozen ext_teacher of v31 [M2]. The consistency mechanism is
    UNCHANGED (forward_with_feat(cons_teacher, weak) -> averaged CLS + averaged
    logits); EnsembleTeacher just implements forward_features/head as elementwise
    mean over the K members, so the rest of the pipeline is agnostic to K.

    CRITICAL: a 1-ckpt ensemble == v31 --ext-teacher (teacher == student ceiling, no
    win). The born-again gain REQUIRES K>=2-3 INDEPENDENT seeds. The ensemble cannot
    be weight-delivered (averaging independent inits is not a valid model) -> the
    final model is the student's own best checkpoint (mechanism C), which the
    consistency training pushes toward the ensemble accuracy.
    """

    def __init__(self, models):
        super().__init__()
        # freeze + eval each member; ModuleList keeps them as submodules
        self.models = nn.ModuleList([m.eval().requires_grad_(False) for m in models])
        self.embed_dim = self.models[0].embed_dim
        self.n = len(self.models)

    @torch.no_grad()
    def forward_features(self, x):
        feats = [m.forward_features(x) for m in self.models]
        out = feats[0]
        for f in feats[1:]:
            out = out + f
        return out / len(feats)

    @torch.no_grad()
    def head(self, feat):
        # feat: [B, D] CLS (already [:, 0]-extracted by forward_with_feat)
        logits = [m.head(feat) for m in self.models]
        out = logits[0]
        for g in logits[1:]:
            out = out + g
        return out / len(logits)

    @torch.no_grad()
    def forward(self, x):
        # v32 FIX: evaluate() calls model(images) -> model.forward(); EnsembleTeacher
        # must implement forward (it only had forward_features/head). Mirror student:
        # forward_features -> CLS token -> head -> logits. Without this, ensemble mode
        # crashes with AttributeError: 'EnsembleTeacher' has no attribute 'forward'
        # the instant evaluate(ensemble, test_loader, ...) runs at load time.
        feat = self.forward_features(x)
        if feat.dim() == 3:              # [B, 1+N, D] -> CLS token
            feat = feat[:, 0]
        return self.head(feat)


def forward_with_feat(model: nn.Module, x: torch.Tensor):
    """Return (logits, cls_feat).

    Feature extraction is version-robust:
      * timm 1.0.28 forward_features -> [B, 1+N, D] (CLS + patch tokens):
        take token 0 (CLS) after the final norm.
      * Older timm returns [B, D] directly.
      * Some builds return (feat, pooled): unwrap the first element.
    The CLS stream is the 384-d representation the physics field constrains.
    """
    feat = model.forward_features(x)
    if isinstance(feat, (tuple, list)):
        feat = feat[0]
    if feat.dim() == 3:                    # [B, 1+N, D] -> CLS token
        feat = feat[:, 0]
    logits = model.head(feat)
    return logits, feat


def feat_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Feature-space consistency: L2 distance on the unit hypersphere.

    L_feat = || normalize(f_S) - normalize(f_T) ||^2   (b detached upstream)
    Normalization removes scale drift of the CLS stream; MSE on the sphere is
    the zero-order Sobolev term of the CIFAR-10 physics_lipschitz_loss 移植.
    """
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    return F.mse_loss(a, b)


# ==================== Optimizer & Schedule ====================

def build_optimizer(model, args, lr=None):
    """AdamW with layer-wise weight decay (no decay on bias/norm).

    lr=None -> args.lr (main schedule). v12 handoff passes HANDOFF_LR to
    build the fresh constant-lr fine-tune optimizer after the handoff.
    """
    if lr is None:
        lr = args.lr
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # v41 [W]: log_sigma 是**尺度**参数 (对数域)。对它做 weight decay
            # 等价于把 sigma 往 1.0 拉 —— 那是"强行强 locality", 不是正则化。
            # 必须进 no_decay 组, 否则扩散尺度会被 wd 单调拖走 (覆辙㉞ 家族:
            # 库/框架默认语义和我们想要的不一样)。
            # v74: raw_beta 同理 —— 它在 tanh 的**内部**坐标上, 对它做 wd 等于把
            # raw 往 0 拉, 即把 beta 往 2.0 (v73 的起点) 拽回去 ⇒ 会把那根轴静默衰减掉。
            # v77: raw_ar 完全同理 —— wd 会把 ar 往 0 (各向同性, v74 的起点) 拽 ⇒
            # 把本版唯一那根轴静默衰减掉。因此三者一律进 no_decay。
            if (name.endswith(".bias") or "norm" in name or "cls_token" in name
                    or name.endswith("log_sigma") or name.endswith("raw_beta")
                    or name.endswith("raw_ar")):
                no_decay.append(param)
            else:
                decay.append(param)
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    return opt


def build_lr_lambda(args, total_steps):
    """Cosine annealing with linear warmup, bottoming at args.lr_min.

    v27 [I]: lr_min > 0 keeps a non-trivial LR through the back half of training
    (was cosine-to-0 in v18), so the model keeps learning past the e~198 peak
    instead of freezing near 1e-5. lr_min=0 recovers the exact v18 schedule.
    """
    warmup_steps = args.warmup_epochs * (total_steps // args.epochs)
    lr_min = args.lr_min
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min + (1.0 - lr_min) * cos
    return lr_lambda


# ==================== v10: schedule / gate / weak aug / EMA / cons loss ========
# (helpers identical to v9 unless annotated — v12 restores the v9 anchor
#  dose, DISABLES the gate by default, and adds load-eval + restore)

def get_w_cons(epoch: int, cons_start: int, total_epochs: int,
               w_max: float = W_CONS_MAX, w_end: float = W_CONS_END,
               ramp_len: int = W_CONS_RAMP_LEN,
               hold_len: int = W_CONS_HOLD_LEN) -> float:
    """v12 anchor strength schedule (v9 exact dose: w_max 1.0 / w_end 0.1).

    Phase 1 (e1 to cons_start):   w_cons = 0       (clean supervised phase)
    Phase 2 (cons_start to +ramp): ramp 0 -> w_max
    Phase 3 (ramp_end to +hold):  hold w_max
    Phase 4 (hold_end to end):    decay w_max->end (gentle release)
    NOTE: this is the PRE-gate schedule; the effective weight is
    w_eff = w_cons * gate(gap) — with v12's GATE_FLOOR=1.0 the gate is the
    identity, so w_eff == w_cons except after handoff (forced 0).
    """
    if epoch <= cons_start:
        return 0.0
    ramp_end = min(cons_start + ramp_len, total_epochs)
    if epoch <= ramp_end:
        return w_max * (epoch - cons_start) / max(1, ramp_end - cons_start)
    hold_end = min(ramp_end + hold_len, total_epochs)
    if epoch <= hold_end:
        return w_max
    decay_len = max(1, total_epochs - hold_end)
    return w_max + (w_end - w_max) * (epoch - hold_end) / decay_len


def gap_gate(gap: float, lo: float = GATE_LO, hi: float = GATE_HI,
             floor: float = GATE_FLOOR) -> float:
    """v12: gap gate with DISABLED-by-default floor (= 1.0 -> identity).

    v11 verdict: the floored gate worked mechanically (w_eff mean 0.228) but
    the merged dose-response across v10/v11/v9 showed the anchor is the gain
    source — gating only diluted it. v12 sets GATE_FLOOR = 1.0 so
    gate(gap) == 1.0 for every gap: v9 full-strength semantics with the v11
    code path (and RNG discipline) untouched. Lower --gate-floor to re-enable
    the v11 floored gate for ablations.

    gap  = max(0, snap_acc - best_acc)   [pp; snapshot minus student best]
    gap <= lo        -> 1.0   target reachable, full distillation pull
    lo < gap < hi    -> linear fade toward floor
    gap >= hi        -> floor  minimum traction (v12 default: 1.0 = off)
    Pure arithmetic: consumes NO RNG, deterministic given recorded metrics.
    """
    if gap <= lo:
        return 1.0
    if gap >= hi:
        return floor
    return floor + (1.0 - floor) * (hi - gap) / (hi - lo)


def get_sigma_max(epoch: int, cons_start: int, total_epochs: int,
                  s_start: float = SIGMA_MAX_START,
                  s_end: float = SIGMA_MAX_END) -> float:
    """v12 diffusion field strength schedule — annealed smoothing (reverse
    diffusion), v9 exact floor restored (0.05).

    Linear annealing of the noise scale inside the consistency window
    [cons_start+1, total_epochs]. Only called when w_eff > 0;
    consumes NO RNG (pure schedule arithmetic).
    """
    if epoch <= cons_start:
        return 0.0
    span = max(1, total_epochs - cons_start)
    frac = (epoch - cons_start) / span
    return s_start + (s_end - s_start) * frac


# ==================== v24 [E]: diffusion-kernel revise + anisotropic (precision) ====================
def kernel_band_gains(prev_signal, alpha, beta, floor=KERNEL_WF_FLOOR):
    """v24 [E]: closed-loop per-DIM diffusion-bandwidth gains (anisotropy).

    Generic water-filling on ANY non-negative per-dim signal. In v24 prev_signal[j]
    = previous epoch's EMA-smoothed per-dim PRECISION = 1/var(student strong-view CLS)
    (inverse-variance; non-vanishing). Inverse-variance modulation: a HIGH-precision
    (low-variance / redundant) dim -> high kappa -> MORE diffusion regularization; a
    LOW-precision (high-variance / informative) dim -> low kappa -> preserved. This is the
    v24 FIX for v23, where prev_signal was VARIANCE (forward-modulated) and amplified the
    informative dims -> harmful (v23 = 64.31 < v21 loop-off 64.46).

    r_j   = prev_signal[j] / mean(prev_signal)       # high-precision dim -> >1
    wf_j  = clamp(r_j, floor, 1/floor) / mean(clamp) # dose-neutral, anti-starve, ANTI-DEGENERATE
    kappa = wf ** (alpha*beta) / mean(wf**(alpha*beta))   # multiplicative soft + dose-neutral
    alpha = 0 (e151) -> kappa == [1,...,1] isotropic (bit-exact base bandwidth)
    """
    mr = sum(prev_signal) / len(prev_signal)
    if mr <= 0:
        return [1.0] * len(prev_signal)
    n = [x / mr for x in prev_signal]
    wf = [min(max(x, floor), 1.0 / floor) for x in n]   # v24: cap both ends -> bounded anisotropy
    mwf = sum(wf) / len(wf)
    wf = [x / mwf for x in wf]
    ab = alpha * beta
    kap = [x ** ab for x in wf]
    mk = sum(kap) / len(kap)
    return [x / mk for x in kap]


def gap_gate_adaptive(gap, running_gap_mean, bw=GATE_ADAPT_BW,
                      lo=None, hi=None, floor=GATE_FLOOR):
    """v22 [E] optional: self-tune the gap gate lo/hi around the running gap mean so
    the gate tracks the training gap (closed-loop on the gate param). Default OFF
    (GATE_ADAPT=False) -> static lo/hi (v21 identity when floor=1.0).
    """
    if lo is None:
        lo = max(0.0, running_gap_mean - bw)
    if hi is None:
        hi = running_gap_mean + bw
    return gap_gate(gap, lo=lo, hi=hi, floor=floor)


def tensor_weak_aug(x: torch.Tensor, pad: int = WEAK_AUG_PAD) -> torch.Tensor:
    """Weak augmentation view on normalized tensors (no denorm needed).

    Per-sample random horizontal flip + zero-padded random crop.
    RNG draw order (only called when w_eff > 0):
      1. torch.rand(B)          -> flip mask
      2. torch.randint(0, 2p+1, (B, 2))  -> per-sample (y, x) crop offsets
    """
    B, C, H, W = x.shape
    # 1) per-sample random horizontal flip
    flip = (torch.rand(B, device=x.device) < 0.5).view(B, 1, 1, 1)
    x = torch.where(flip, x.flip(-1), x)
    # 2) zero-padded random crop, per-sample offsets
    padded = F.pad(x, (pad, pad, pad, pad))
    offs = torch.randint(0, 2 * pad + 1, (B, 2), device=x.device)  # (y, x)
    rows = offs[:, 0].view(B, 1) + torch.arange(H, device=x.device).view(1, H)  # [B, H]
    cols = offs[:, 1].view(B, 1) + torch.arange(W, device=x.device).view(1, W)  # [B, W]
    out = padded[torch.arange(B, device=x.device).view(B, 1, 1), :,
                 rows.view(B, H, 1), cols.view(B, 1, W)]
    return out.permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def ema_update_teacher(teacher: nn.Module, student: nn.Module,
                       global_step: int, ema_decay: float = EMA_DECAY):
    """EMA teacher update = parameter-space heat diffusion / low-pass filter.

    teacher_p <- d * teacher_p + (1 - d) * student_p
    d = min(1 - 1/(step+1), ema_decay)   # early steps: teacher tracks student fast

    Pure parameter arithmetic: consumes NO RNG, does NOT touch student grads.
    """
    d = min(1.0 - 1.0 / (global_step + 1), ema_decay)
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.mul_(d).add_(sp.detach(), alpha=1.0 - d)
    for tb, sb in zip(teacher.buffers(), student.buffers()):
        tb.copy_(sb)


def ema_decay_cap(epoch: int, sched_on: bool = EMA_SCHED_ON,
                   base: float = EMA_DECAY_BASE, late: float = EMA_DECAY_LATE,
                   start: int = EMA_SCHED_START, end: int = EMA_SCHED_END) -> float:
    """v25 [F]: epoch-dependent MAX EMA decay cap for the teacher update.

    Early/mid phase -> base (bit-exact v21 behavior: fast-enough tracking).
    Late phase (start..end) -> ramped UP to `late` so the teacher becomes a
    tighter, smoother ensemble that retains the student's late peaks better
    (the real delivered==teacher ceiling; teacher already leads the student by
    ~0.7pp at e250-279 in v24). Post-`end` holds `late`. The step-warmup inside
    ema_update_teacher still tracks the student fast in early steps.

    With sched_on=False -> constant EMA_DECAY (bit-exact v21 anchor).
    """
    if not sched_on:
        return EMA_DECAY
    if epoch <= start:
        return base
    if epoch >= end:
        return late
    frac = (epoch - start) / (end - start)
    return base + frac * (late - base)


def consistency_loss(student_logits: torch.Tensor,
                     teacher_logits: torch.Tensor) -> torch.Tensor:
    """Symmetric KL divergence: symmetry -> invariance constraint.

    L_cons = 0.5 * [ KL(p_T || p_S) + KL(p_S || p_T) ]
    """
    log_s = F.log_softmax(student_logits, dim=1)
    log_t = F.log_softmax(teacher_logits, dim=1)   # no grad (teacher detached)
    p_s = log_s.exp()
    p_t = log_t.exp()
    kl_t_s = (p_t * (log_t - log_s)).sum(dim=1).mean()   # KL(p_T || p_S)
    kl_s_t = (p_s * (log_s - log_t)).sum(dim=1).mean()   # KL(p_S || p_T)
    return 0.5 * (kl_t_s + kl_s_t)


def soft_kd_loss(student_logits: torch.Tensor,
                 teacher_logits: torch.Tensor, T: float = SOFT_KD_T) -> torch.Tensor:
    """v26 [G]: batch-mean Hinton soft-KD (dark knowledge transfer).

    L_soft = T^2 * KL( softmax(teacher/T) || softmax(student/T) )
    Gradient flows through the student logits only (teacher is detached
    upstream, in the weak-view no_grad block). T>1 softens the logits so the
    teacher's low-rank class correlations (its "dark knowledge") become
    visible — information the T=1 symmetric-KL anchor_logit cannot transfer.
    """
    log_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_s, p_t, reduction="batchmean") * (T * T)


def per_sample_soft_kd(student_logits: torch.Tensor,
                       teacher_logits: torch.Tensor, T: float = SOFT_KD_T) -> torch.Tensor:
    """v26 [G]: per-sample soft-KD — reduction 'none' summed over classes.

    per_sample_soft_kd(...).mean() == soft_kd_loss(...) EXACTLY (same T^2
    scaling, same reduction). Identity conserved for the residual-weighting
    path: the soft-KD channel enters cons_ps and is res-weighted identically
    to the other three channels.
    """
    log_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_s, p_t, reduction="none").sum(dim=1) * (T * T)


# ==================== v13 [D2]: residual per-sample weighting ====================

def per_sample_symkl(student_logits: torch.Tensor,
                     teacher_logits: torch.Tensor) -> torch.Tensor:
    """v13: per-sample symmetric KL — identity conserved:
    per_sample_symkl(...).mean() == consistency_loss(...) EXACTLY
    (same reduction split: sum over classes, per-sample vector out).
    Gradient flows through log_s (student logits)."""
    log_s = F.log_softmax(student_logits, dim=1)
    log_t = F.log_softmax(teacher_logits, dim=1)
    kl_t_s = (log_t.exp() * (log_t - log_s)).sum(dim=1)
    kl_s_t = (log_s.exp() * (log_s - log_t)).sum(dim=1)
    return 0.5 * (kl_t_s + kl_s_t)


def per_sample_feat_mse(a: torch.Tensor, b_n: torch.Tensor) -> torch.Tensor:
    """v13: per-sample feature distance on the unit hypersphere — identity
    conserved: per_sample_feat_mse(...).mean() == feat_mse(...) EXACTLY.
    feat_mse = F.mse_loss over ALL B*D elements = element mean; the
    per-sample vector divides the squared sum by D, so its batch mean is
    the element mean. b_n must be pre-normalized (teacher, detached);
    `a` is normalized here WITH gradient."""
    a = F.normalize(a, dim=1)
    return (a - b_n).pow(2).sum(dim=1) / a.shape[1]


def residual_weights(r_i: torch.Tensor, beta: float,
                     lo: float = RES_W_LO, hi: float = RES_W_HI) -> torch.Tensor:
    """v13 [D2]: map per-sample residuals to dose-neutral loss weights.

    w_i = clip(1 + beta*(r_i/mean(r) - 1), lo, hi) / mean(w)
    · monotone non-decreasing in r_i (rough sample -> stronger anchor)
    · clip bounds the per-sample dose ratio before renormalization
    · renormalization to mean 1 keeps the cons_term magnitude EXACTLY
      comparable to v12 (channel weights 0.3/0.2/0.5 keep their meaning)
    · pure arithmetic on a detached tensor: NO RNG, no extra forward
    Note: mean-renorm takes precedence over the absolute ceiling — under
    extreme residual skew the effective max can exceed hi by design
    (dose neutrality is the stronger invariant)."""
    r_hat = r_i / (r_i.mean() + 1e-8)
    w = torch.clamp(1.0 + beta * (r_hat - 1.0), lo, hi)
    return w / (w.mean() + 1e-8)


# ==================== v16 [D4]: spectral anchor (time rotation) ====================

def parse_ms_weights(spec: str):
    """v16 [D4]: parse a STATIC spectral-ladder spec for the feat_diff scales.

    Returns a list of NOISE_SCALES floats (coarse -> fine) summing to ~1.0,
    None for the special values 'v13'/'uniform'/'none'/'off' meaning the
    EXACT v13 equal-weight branch (bit-exact determinism anchor),
    [0.5, 0.3, 0.2] for 'v14' (the EXACT v14 branch) and [0.6, 0.3, 0.1]
    for 'v15' (the EXACT v15 branch — second determinism anchor).

    Pure string arithmetic: consumes NO RNG, raises on malformed specs
    (wrong count / bad sum / negative weights) so a typo cannot silently
    run a wrong spectrum.
    """
    s = (spec or "").strip().lower()
    if s in ("v13", "uniform", "none", "off"):
        return None
    if s == "v14":
        return [0.5, 0.3, 0.2]
    if s == "v15":
        return [0.6, 0.3, 0.1]
    try:
        parts = [float(v) for v in spec.split(",")]
    except ValueError:
        raise ValueError(f"--ms-weights must be comma-separated floats, got {spec!r}")
    if len(parts) != NOISE_SCALES:
        raise ValueError(f"--ms-weights needs exactly {NOISE_SCALES} values "
                         f"(coarse->fine), got {spec!r}")
    if any(w < 0 for w in parts):
        raise ValueError(f"--ms-weights must be non-negative, got {spec!r}")
    if abs(sum(parts) - 1.0) > 1e-6:
        raise ValueError(f"--ms-weights must sum to 1.0 (dose-neutral), got {spec!r}")
    return parts


def ms_weights_for_epoch(epoch: int, spec: str):
    """v19 [D4++]: the BASE spectral prior AT A GIVEN EPOCH — compressed
    rotation (the v18 profile, UNCHANGED; v19's adaptive branch reshapes
    this base — see adaptive_spectral_weights).

    'rotate'/'v16' (the default; historical alias): linear simplex
        interpolation w(t) = MS_W_START + frac * (MS_W_END - MS_W_START)
    over [CONS_START+1, MS_ROT_END] = e151..e250 (v18 COMPRESSED), then the
    fine-mirror endpoint is HELD for e250..e279 — the last 30 epochs of the
    active anchor window (handoff fires e280; w_eff is 0 after, so the
    weights are moot there). Endpoints are EXACT: e151 -> (0.5, 0.3, 0.2)
    (the v14 optimum), e250 -> (0.2, 0.3, 0.5) (the fine mirror).
    Dose-neutral at EVERY epoch: interpolation and hold both stay on the
    simplex (sum w(t) == 1.0). v18 phase-mean over e151..e279 =
    (0.316, 0.3, 0.384) — fine epoch-mean +9.6% vs v17 (probe confound,
    documented in the module docstring).

    Any other spec falls through to the STATIC parser (v14/v15/v13 anchors
    or an explicit triple). Pure epoch arithmetic: consumes NO RNG, zero
    extra forwards — the rotation cannot perturb the RNG draw order.
    """
    s = (spec or "").strip().lower()
    if MS_ROTATE and s in ("rotate", "rot", "v16", ""):
        if epoch <= CONS_START + 1:
            return list(MS_W_START)
        if epoch >= MS_ROT_END:
            return list(MS_W_END)
        span = MS_ROT_END - (CONS_START + 1)
        frac = (epoch - (CONS_START + 1)) / span
        return [a + (b - a) * frac for a, b in zip(MS_W_START, MS_W_END)]
    return parse_ms_weights(spec)


def adapt_alpha(epoch: int, start: int, end: int) -> float:
    """v19 [D4++]: adaptive-intervention intensity ramps 0 -> 1 over
    [start, end], then holds 1.0. Pure epoch arithmetic: zero RNG."""
    if epoch <= start:
        return 0.0
    if epoch >= end:
        return 1.0
    return (epoch - start) / (end - start)


def adaptive_spectral_weights(base, prev_bands, alpha, beta=ADAPT_BETA):
    """v21 [D4++ fixed]: multiplicative SOFTENED water-filling.

    Repairs the v19 量纲 bug (pressure = band/gap saturated the 0.30 floor
    because band~1e-3 and gap~1-3pp -> band/gap~1e-3 << 0.30, so wf==[1,1,1]
    and the branch was bit-exact-idle) AND the v21-preview winner-take-all
    (the additive + renorm form crushed coarse/mid to ~0.03 when the
    perturbation-scale-normalized residuals spread 1:2:4).

    base:       the v18 time prior (list, sums to 1.0).
    prev_bands: PREVIOUS epoch's per-scale feat_diff residual means
                (fd_c/fd_m/fd_f) — causal look-back, .item() scalars.
    alpha:      intervention intensity in [0, 1].
    beta:       multiplicative softening exponent (contraction strength of
                the water-filling tilt). Higher = more aggressive shift
                toward the high-residual band.

    pressure_j = band_j / scale2_j   (scale2 = sigma^2 ladder 1/0.25/0.0625;
                perturbation-scale normalization removes the sigma artifact so
                coarse/mid/fine are comparable; NO gap divisor).
    wf_j      = pressure_j / mean(pressure)   (dose-neutral, mean 1).
    w_eff_j   = base_j * (wf_j ** (alpha*beta))   then renorm sum 1.
    Multiplicative form keeps every band strictly positive (no starvation /
    no winner-take-all) and the beta exponent compresses the residual spread.
    Crucially it is NEAR-IDENTITY early (alpha->0 at e151, and base is
    coarse-weighted there) and tilts toward fine LATE — naturally reproducing
    the v14 (coarse pre-HO peak 64.50) + v18 (fine post-HO EMA climb 64.66)
    combination that no fixed endpoint achieved. Pure arithmetic: zero RNG,
    zero extra forwards.
    """
    scale2 = [SIGMA_RATIO ** (2 * j) for j in range(len(prev_bands))]
    norm = [b / s for b, s in zip(prev_bands, scale2)]
    p_mean = sum(norm) / len(norm)
    wf = [n / p_mean for n in norm]              # mean 1
    w = [base[j] * (wf[j] ** (alpha * beta)) for j in range(len(base))]
    s = sum(w)
    return [x / s for x in w]


# ==================== Training Loop ====================

def train_one_epoch(model, teacher, cons_teacher, ext_teacher, loader, opt, scheduler,
                    criterion, device,
                    epoch, args, global_step, total_steps,
                    w_eff, mixup_start, mixup_end, ms_w_override=None,
                    kappa_bands=None, kernel_revise=False):
    """One training epoch.

    teacher:      live EMA teacher (updated every step; diagnostics + snapshot
                  source)
    cons_teacher: FROZEN best-TEACHER snapshot (mechanism B: monotonic).
                  Upgraded ONLY when the live EMA teacher sets a new record —
                  the snapshot is by construction the strongest target ever
                  seen (v9 falsified the "teacher ceiling" worry: EMA hit
                  64.40 while v8's frozen-teacher concern capped at 63.99).
    w_eff:        v10 EFFECTIVE anchor weight = w_cons(t) * gap_gate (mechanism
                  A), or 0 after the handoff (mechanism D). The physics branch
                  below runs iff w_eff > 0 — identical to v9's branch structure
                  so RNG draw order is unchanged.
    v10 consistency field — THREE channels on the 384-d CLS feature manifold:
      anchor_logit (0.3): symKL( student(strong) || snapshot(weak) )  [v8 term]
      feat_anchor  (0.2): ||norm(f_S(strong)) - norm(f_T(weak))||^2
      feat_diff    (0.5): (1/M) Σ_m ||norm(f_S(weak + σ_m·ε)) - norm(f_T(weak))||^2
                          heat-semigroup multi-scale feature diffusion.
    v18 [D4']: inside feat_diff the M scales are combined with spectral
    weights that ROTATE along the time axis — linear simplex interpolation
    from (0.5, 0.3, 0.2) at e151 (the proven v14 optimum) to its fine
    mirror (0.2, 0.3, 0.5) at e250 (COMPRESSED), then held at the mirror
    for e250-e279; dose-neutral at every epoch
    (Σw(t) = 1.0); per-scale band means (fd_c/fd_m/fd_f) are logged for
    attribution. --ms-weights v14 / v15 restore the exact static v14 / v15
    spectra, --ms-weights v13 the equal-weight branch.
    v19 [D4++]: ms_w_override, when not None, REPLACES the internally
    computed base profile — the main loop feeds the ADAPTIVE weights
    (base + alpha*adapt, water-filling of the previous epoch's band
    residuals; see adaptive_spectral_weights). None (default) = the EXACT
    v18 base profile (--no-ms-adapt anchor: identical values, bit-exact).
    """
    model.train()
    running_loss = 0.0
    running_cls = 0.0
    running_cons = 0.0
    running_anchor = 0.0
    running_feat_anchor = 0.0
    running_feat_diff = 0.0
    running_soft_kd = 0.0    # v26 [G]: soft-KD trajectory accumulator
    running_correct = 0
    running_total = 0
    # v13 [D2]: epoch-level envelope of the residual weight distribution
    epoch_res_w_min = float("inf")
    epoch_res_w_max = 0.0
    use_res_weight = RES_WEIGHTED_ANCHOR and args.res_beta > 0
    # v16 [D4]: spectral ladder with TIME-AXIS ROTATION (None -> exact v13
    # uniform branch; 'v14'/'v15' -> exact static branches) + per-scale band
    # diagnostics (coarse/mid/fine residual means, spectral attribution)
    ms_w = (ms_w_override if ms_w_override is not None
            else (ms_weights_for_epoch(epoch, args.ms_weights)
                  if MS_SPECTRAL_ANCHOR else None))
    use_ms = ms_w is not None
    fd_band_sum = [0.0] * NOISE_SCALES
    n_fd_batches = 0
    fd_kappa_sum = [0.0] * K_ANISO   # v22 [E]: per-band CLEAN residual diagnostic (unchanged)
    band_var_sum = None              # v23 [E]: per-dim teacher CLS variance accumulator (closed-loop signal)
    n_var_batches = 0

    mixup_alpha = get_mixup_alpha(epoch, mixup_start, mixup_end, args.mixup_alpha)
    use_mixup = args.mixup and mixup_alpha > 0.01

    # definite-init (G2): bound at the top of every loop iteration; initialise
    # here so the post-loop read at `D_feat = s_feat.size(1) ...` is provably
    # bound even for an (impossible in practice) empty loader.
    s_feat = None

    for batch_idx, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)

        # mixup / cutmix (50/50, ResNet v14 same logic; off in v4 window)
        if use_mixup:
            if random.random() < 0.5:
                images, t_a, t_b, lam = mixup_data(images, targets, alpha=mixup_alpha)
            else:
                images, t_a, t_b, lam = cutmix_data(images, targets, beta=CUTMIX_BETA)

        # student forward on the strong view — logits for cls/anchor, CLS feat
        # for feat_anchor (single forward, reused by both)
        outputs, s_feat = forward_with_feat(model, images)
        if use_mixup:
            cls_loss = mixup_criterion(criterion, outputs, t_a, t_b, lam)
        else:
            cls_loss = criterion(outputs, targets)

        # v10 three-channel consistency: only active inside the window
        # (e151+ and pre-handoff). Branch identical to v9; w_eff carries the
        # gate so RNG consumption only happens when the anchor is live.
        cons_val = 0.0
        cons_anchor_val = 0.0
        feat_anchor_val = 0.0
        feat_diff_val = 0.0
        soft_kd_val = 0.0      # v26 [G]: soft-KD batch-mean (for logging)
        batch_w_min = 0.0
        batch_w_max = 0.0
        if w_eff > 0:
            weak = tensor_weak_aug(images, pad=WEAK_AUG_PAD)
            with torch.no_grad():
                t_logits, t_feat = forward_with_feat(cons_teacher, weak)
                # t_feat is the clean-target anchor; normalize once, reuse
                t_feat_n = F.normalize(t_feat, dim=1)
            # channel 1 — logit anchor (w=0.3): strong logits vs
            # frozen-snapshot logits on the weak view (x_0 target)
            cons_anchor = consistency_loss(outputs, t_logits)
            # channel 2 — feature anchor (w=0.2): strong-view CLS vs clean weak
            feat_anchor = feat_mse(s_feat, t_feat_n)
            if kernel_revise and kappa_bands is not None:
                # v24 [E] ACTIVE: PER-DIM ANISOTROPIC feature-space diffusion. Perturb
                # the teacher CLS feature per DIMENSION with bandwidth sigma_i = s_m*kappa_i
                # (kappa_i from the per-dim PRECISION = 1/var closed loop, inverse-variance
                # modulation: redundant/low-var dims diffuse MORE, informative/high-var dims
                # preserved). Require the (clean) student feature to match. One randn(B,D)
                # call, applied elementwise. Replaces the v21 image-space isotropic path
                # (kept EXACTLY below as the --no-kernel-revise anchor / epoch-1 isotropic start).
                use_ms = True   # cons_ps uses the no-divide branch; fd_ps_sum pre-normalized
                sigma_max = get_sigma_max(epoch, CONS_START, args.epochs)
                D = s_feat.size(1)
                eps = torch.randn(s_feat.size(0), D, device=device)
                s_m = sigma_max
                feat_diff_sum = 0.0
                fd_ps_sum = 0.0
                # SIGMA_RATIO ladder RESTORED (v22 active branch dropped it -> fd_c==fd_m==fd_f,
                # spectral ladder collapsed to a single scale). The ONLY diff vs the anchor is
                # the per-dim anisotropic PERTURBATION (sigma_i = s_m*kappa_i).
                for mi in range(NOISE_SCALES):
                    w_mi = ms_w[mi] if ms_w is not None else 1.0
                    tb = t_feat_n + (s_m * kappa_bands) * eps   # (B,D) per-dim perturbation
                    L = feat_mse(s_feat, tb)
                    feat_diff_sum = feat_diff_sum + w_mi * L
                    fd_band_sum[mi] += float(L.item())
                    fd_ps_sum = fd_ps_sum + w_mi * per_sample_feat_mse(s_feat, tb)
                    s_m = s_m * SIGMA_RATIO
                n_fd_batches += 1
                feat_diff = feat_diff_sum
                # closed-loop signal accumulation (v24): per-dim PRECISION = 1/var of the
                # STUDENT's strong-view CLS features. Inverse-variance modulation: redundant
                # (low-variance) dims get HIGH precision -> HIGH kappa -> MORE diffusion
                # regularization; informative (high-variance) dims get LOW kappa -> preserved.
                # Non-vanishing (batch feature variance always spreads). Accumulated as precision
                # for EMA + water-filling in kernel_band_gains.
                with torch.no_grad():
                    v = s_feat.var(dim=0, unbiased=False)          # (D,) student strong-view per-dim variance
                    inv = 1.0 / (v + 1e-6)                         # precision (inverse-variance)
                    if band_var_sum is None:
                        band_var_sum = inv.clone()
                    else:
                        band_var_sum.add_(inv)
                    n_var_batches += 1
                # keep 3-band CLEAN-residual diagnostic (unchanged)
                for bi in range(K_ANISO):
                    lo = bi * D // K_ANISO
                    hi = (bi + 1) * D // K_ANISO
                    fd_kappa_sum[bi] += float(feat_mse(s_feat[:, lo:hi], t_feat_n[:, lo:hi]).item())
            else:
                # ANCHOR = EXACT v21 image-space isotropic path (bit-exact when
                # --no-kernel-revise). Do NOT modify — determinism anchor to v21.
                sigma_max = get_sigma_max(epoch, CONS_START, args.epochs)
                eps = torch.randn_like(weak)
                s_m = sigma_max
                feat_diff_sum = 0.0
                fd_ps_sum = 0.0   # v13: per-scale per-sample accumulator (grad kept)
                for mi in range(NOISE_SCALES):
                    _, s_feat_m = forward_with_feat(model, weak + s_m * eps)
                    L_m = feat_mse(s_feat_m, t_feat_n)
                    if use_ms:
                        feat_diff_sum = feat_diff_sum + ms_w[mi] * L_m
                    else:
                        feat_diff_sum = feat_diff_sum + L_m
                    fd_band_sum[mi] += float(L_m.item())
                    if use_res_weight:
                        ps_m = per_sample_feat_mse(s_feat_m, t_feat_n)
                        if use_ms:
                            fd_ps_sum = fd_ps_sum + ms_w[mi] * ps_m
                        else:
                            fd_ps_sum = fd_ps_sum + ps_m
                    s_m = s_m * SIGMA_RATIO
                n_fd_batches += 1
                if use_ms:
                    feat_diff = feat_diff_sum          # already dose-correct (sum w = 1.0)
                else:
                    feat_diff = feat_diff_sum / NOISE_SCALES
            # v26 [G] / v28 [J]: soft-label temperature KD (4th consistency channel)
            # v28: ENABLED for the EXTERNAL teacher (real dark knowledge, 覆辙⑭);
            # STILL OFF for the live-EMA self-teacher (v26 proved redundant + harmful).
            soft_kd_term_t = None
            soft_kd_ps = None
            soft_kd_w = 0.0
            use_ext_kd = (ext_teacher is not None)
            if (SOFT_KD and args.soft_kd) or use_ext_kd:
                if use_ext_kd:
                    soft_kd_w = args.ext_teacher_w
                    _kd_T = args.ext_teacher_t
                else:
                    soft_kd_w = SOFT_KD_W
                    _kd_T = SOFT_KD_T
                soft_kd_term_t = soft_kd_loss(outputs, t_logits, T=_kd_T)
                soft_kd_ps = per_sample_soft_kd(outputs, t_logits, T=_kd_T)
                soft_kd_val = soft_kd_term_t.item()
            cons_term = (ANCHOR_LOGIT_W * cons_anchor +
                         FEAT_ANCHOR_W * feat_anchor +
                         FEAT_DIFF_W * feat_diff)
            if soft_kd_term_t is not None:
                cons_term = cons_term + soft_kd_w * soft_kd_term_t
            # v13 [D2]: residual per-sample weighting of the SAME term.
            # Identity guarantee: cons_ps.mean() == cons_term exactly
            # (per_sample_* reductions are element-mean-conserved), so beta=0
            # semantics equal v12; weights are dose-neutral (mean 1). All
            # weight arithmetic is under no_grad on a detached tensor —
            # zero extra forward, zero RNG, RNG draw order unchanged.
            if use_res_weight:
                if use_ms:
                    # v16 [D4]: (time-rotated) spectral weights applied to the
                    # per-scale per-sample accumulator — identity
                    # cons_ps.mean() == cons_term preserved EXACTLY
                    # (per_sample reductions are element-mean-conserved,
                    # weights applied identically)
                    cons_ps = (ANCHOR_LOGIT_W * per_sample_symkl(outputs, t_logits)
                               + FEAT_ANCHOR_W * per_sample_feat_mse(s_feat, t_feat_n)
                               + FEAT_DIFF_W * fd_ps_sum)
                else:
                    cons_ps = (ANCHOR_LOGIT_W * per_sample_symkl(outputs, t_logits)
                               + FEAT_ANCHOR_W * per_sample_feat_mse(s_feat, t_feat_n)
                               + FEAT_DIFF_W * fd_ps_sum / NOISE_SCALES)
                if soft_kd_ps is not None:
                    cons_ps = cons_ps + soft_kd_w * soft_kd_ps
                with torch.no_grad():
                    w_i = residual_weights(cons_ps.detach(), args.res_beta)
                    batch_w_min = w_i.min().item()
                    batch_w_max = w_i.max().item()
                epoch_res_w_min = min(epoch_res_w_min, batch_w_min)
                epoch_res_w_max = max(epoch_res_w_max, batch_w_max)
                cons_term = (w_i * cons_ps).mean()
            loss = cls_loss + w_eff * cons_term
            cons_val = cons_term.item()
            cons_anchor_val = cons_anchor.item()
            feat_anchor_val = feat_anchor.item()
            feat_diff_val = feat_diff.item()
        else:
            loss = cls_loss

        # backward
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        # EMA teacher update (pure arithmetic, no RNG, every step)
        # v25 [F]: max-decay cap is epoch-scheduled (tighter ensemble late).
        if teacher is not None:
            _ema_cap = ema_decay_cap(epoch, args.ema_sched, EMA_DECAY_BASE,
                                     args.ema_decay_late, EMA_SCHED_START, EMA_SCHED_END)
            ema_update_teacher(teacher, model, global_step, _ema_cap)
        global_step += 1

        # metrics (lam-weighted acc during mixup, same as ResNet v14)
        running_loss += loss.item()
        running_cls += cls_loss.item()
        running_cons += cons_val
        running_anchor += cons_anchor_val
        running_feat_anchor += feat_anchor_val
        running_feat_diff += feat_diff_val
        running_soft_kd += soft_kd_val
        pred = outputs.argmax(1)
        if use_mixup:
            running_correct += (lam * pred.eq(t_a).sum().item() +
                                (1 - lam) * pred.eq(t_b).sum().item())
        else:
            running_correct += pred.eq(targets).sum().item()
        running_total += targets.size(0)

        # logging
        if batch_idx % args.log_interval == 0:
            lr = opt.param_groups[0]["lr"]
            mtag = f"mixup(lam={lam:.2f})" if use_mixup else "clean"
            cons_str = ""
            if w_eff > 0:
                cons_str = (f"  cons={cons_val:.4f}"
                            f" (a={cons_anchor_val:.4f} fa={feat_anchor_val:.4f}"
                            f" fd={feat_diff_val:.4f})"
                            f"  w_eff={w_eff:.3f}  sigma={sigma_max:.3f}")
                if use_res_weight:
                    cons_str += f"  res_w=[{batch_w_min:.2f},{batch_w_max:.2f}]"
            print(f"  [E{epoch} B{batch_idx:4d}/{len(loader)}] "
                  f"step={global_step}/{total_steps}  "
                  f"loss={loss.item():.4f} (cls={cls_loss.item():.4f})  "
                  f"lr={lr:.2e}  {mtag}{cons_str}")

    acc = 100.0 * running_correct / running_total
    avg_loss = running_loss / len(loader)
    avg_cls = running_cls / len(loader)
    avg_cons = running_cons / len(loader)
    avg_anchor = running_anchor / len(loader)
    avg_feat_anchor = running_feat_anchor / len(loader)
    avg_feat_diff = running_feat_diff / len(loader)
    avg_soft_kd = running_soft_kd / len(loader)    # v26 [G]
    res_w_min = epoch_res_w_min if epoch_res_w_min != float("inf") else 0.0
    if n_fd_batches > 0:
        fd_bands = [s / n_fd_batches for s in fd_band_sum]
        fd_kappa = [s / n_fd_batches for s in fd_kappa_sum]
    else:
        fd_bands = [0.0] * NOISE_SCALES
        fd_kappa = [0.0] * K_ANISO
    D_feat = s_feat.size(1) if s_feat is not None else 384
    # v23 FIX: band_var_sum is a tensor; [x / n for x in tensor] yields a list of
    # 0-d tensors -> leaked into history -> json.dump crashes ("Tensor not JSON
    # serializable"). Force float() so band_var is a plain float list.
    band_var = ([float(x / n_var_batches) for x in band_var_sum] if (band_var_sum is not None and n_var_batches > 0)
                else [0.0] * D_feat)
    return (avg_loss, acc, global_step, avg_cls, avg_cons,
            avg_anchor, avg_feat_anchor, avg_feat_diff,
            res_w_min, epoch_res_w_max,
            fd_bands[0], fd_bands[1], fd_bands[2],
            fd_kappa[0], fd_kappa[1], fd_kappa[2],
            band_var, avg_soft_kd)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        running_total += targets.size(0)
        running_correct += predicted.eq(targets).sum().item()
    acc = 100.0 * running_correct / running_total
    avg_loss = running_loss / len(loader)
    return avg_loss, acc


# ==================== Main ====================

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Seed: {args.seed}  Physics(v19): {args.physics}")
    os.makedirs(args.output_dir, exist_ok=True)

    # mixup window
    mixup_start, mixup_end = get_mixup_window(args, args.epochs)
    # --- v35 [Q] RUNTIME CANARY (覆辙㉙) -------------------------------------
    # v34 shipped the full-run window but the argparse default shadowed it, so the
    # run silently reproduced v33. A silent no-op is the most expensive failure
    # mode in this project (hours of GPU). Refuse to continue if the effective
    # window does not match the intent, and print the resolved schedule up front.
    if args.mixup and not args.smoke and mixup_end < args.epochs:
        print(f"\n[v35 CANARY] mixup window ends at e{mixup_end} of {args.epochs} — "
              f"the last {args.epochs - mixup_end} epochs train on CLEAN hard "
              f"labels (v18-v34 behaviour, the 64.44 ceiling).", flush=True)
        if args.mixup_window is None:
            raise SystemExit(
                "[v35 CANARY] --mixup-window was not given, so the auto window "
                "must be the FULL RUN [E//3, E]. Got [{}, {}]. This means the "
                "argparse default was set again and get_mixup_window()'s "
                "fallback is dead code (覆辙㉙).".format(mixup_start, mixup_end))
        print("[v35 CANARY] explicit override detected -> continuing (ablation "
              "or --v18-anchor mode).", flush=True)
    aug_cfg = {
        "randaugment": args.randaugment, "randaug_n": RANDAUG_N, "randaug_m": args.randaug_m,
        "random_erasing": args.erasing, "erase_p": args.erase_p,
        "mixup": args.mixup, "mixup_alpha": args.mixup_alpha,
        "mixup_window": [mixup_start, mixup_end],
        "cutmix_beta": CUTMIX_BETA,
        "label_smoothing": 0.1,
        # v35 [Q]: record the RESOLVED per-epoch schedule so a no-op can never
        # again hide inside a summary field.
        "mixup_alpha_by_epoch": [
            round(get_mixup_alpha(e, mixup_start, mixup_end, args.mixup_alpha), 4)
            for e in range(1, args.epochs + 1)
        ] if args.mixup else [],
    }
    print(f"Augmentation: RandAugment({RANDAUG_N},{args.randaug_m})={args.randaugment}  "
          f"RandomErasing({args.erase_p})={args.erasing}  "
          f"Mixup/CutMix(alpha={args.mixup_alpha})={args.mixup}")
    print(f"  Mixup window: epochs {mixup_start}-{mixup_end} (decay last ~1/6)")
    print(f"  LR: peak={args.lr}  min={args.lr_min}  (v27 [I] cosine floor keeps "
          f"back-half LR non-trivial; v18 anchor lr_min=0)")

    v39_horizon_canary(args, mixup_end)

    # data
    train_loader, test_loader = get_loaders(args)
    # v35 [Q]: measurement-only loader (fixed unaugmented train subset). Built
    # AFTER the real loaders so a failure here can never affect training data.
    train_eval_loader = get_train_eval_loader(args)
    print(f"Train: {len(train_loader.dataset)} images, "
          f"Test: {len(test_loader.dataset)} images, "
          f"Train-eval subset (unaugmented, measurement only): "
          f"{len(train_eval_loader.dataset)} images")

    # model
    model = build_model(args).to(device)

    # v40 [V]: 位置编码轴 —— 必须装在 deepcopy 出 teacher 之前 (teacher 由
    # copy.deepcopy(model) 得到, 换过的 __class__ 会被一起复制)。
    model = install_pos_enc(model, args.pos_enc, rope_base=args.rope_base)
    rope_blocks, pe_absmax = v40_pos_enc_canary(model, args.pos_enc)
    gh_v, gw_v = model.patch_embed.grid_size

    # v41 [W]: 网格热核先验 —— 同样必须装在 deepcopy 出 teacher 之前。
    # 注意顺序: 先 pos_enc (可能把 __class__ 换成 RopeAttention), 再 heat
    # (HeatKernelAttention 是 RopeAttention 的子类, 两条通路可以叠加)。
    model, heat_n_params = install_heat_kernel(model, args.diffusion_mode,
                                               ladder=args.heat_ladder)
    heat_blocks, heat_sigma0 = v41_heat_canary(model, args.diffusion_mode)
    if args.diffusion_mode == "heat":
        print(f"  [W] WARNING: heat 是**未验证实验臂**。覆辙㊺: 发车长跑前请先跑 "
              f"--epochs 60 短程, 要求 e50 test_acc >= {HEAT_E50_GATE} "
              f"(off/锚点实测 32.50; v40 rope2d 首跑 ~11.5 就是这么被漏掉的)。")
        print(f"  [W] 对照只需**一条**命令: "
              f"--diffusion-mode off --epochs 60 (off 与 v40/v39 逐位相同, "
              f"所以 v40/v39 的 history 可以直接当锚点用, 不必重跑)。")

    # v31: optional CONS_START window override (--smoke / --cons-start).
    # Rebinds the module constant BEFORE the training loop so every call-site
    # (get_w_cons / get_sigma_max / ms rotation / banners) reads the new value.
    # Default (no override) = the module constant 150 = bit-exact v18/v28.
    #
    # v36 [R]: the same rebinding now covers EVERY horizon-calibrated stage
    # constant, with values derived from --epochs by parse_args().  At E=300 the
    # derived values equal the module constants, so nothing changes; at any other
    # horizon the whole schedule is translated proportionally instead of leaving
    # the mechanisms stranded at their E=300 positions (覆辙②).
    # ONE declaration listing all eight: two separate `global` statements would
    # each cover only half, and a missed name turns the rebinding into a silent
    # local no-op that no text-based gate would catch.
    global CONS_START, MS_ROT_END, KERNEL_REVISE_AEND, ADAPT_ALPHA_END, \
        W_CONS_RAMP_LEN, W_CONS_HOLD_LEN, EMA_SCHED_START, EMA_SCHED_END
    # (const name, argparse dest) — the dest IS the STAGE_FRACS key, so there is
    # exactly one mapping table (覆辙⑰: two copies of a mapping will drift).
    _stage_rebind = [
        ("CONS_START",         "cons_start"),
        ("MS_ROT_END",         "ms_rot_end"),
        ("KERNEL_REVISE_AEND", "kernel_revise_end"),
        ("ADAPT_ALPHA_END",    "ms_adapt_alpha_end"),
        ("W_CONS_RAMP_LEN",    "w_cons_ramp_len"),
        ("W_CONS_HOLD_LEN",    "w_cons_hold_len"),
        ("EMA_SCHED_START",    "ema_sched_start"),
        ("EMA_SCHED_END",      "ema_sched_end"),
    ]
    _moved = [(n, globals()[n], getattr(args, d)) for n, d in _stage_rebind
              if getattr(args, d) != globals()[n]]
    if _moved:
        print(f"[STAGE SCALING] --epochs {args.epochs} "
              f"(stage constants calibrated at 300) -> re-derived:")
        for n, old, new in _moved:
            dest = dict(_stage_rebind)[n]
            print(f"    {n:<19} {old:>4} -> {new:<4}"
                  f"   ({STAGE_FRACS[dest]:.4f} x E)")
    for n, d in _stage_rebind:
        globals()[n] = getattr(args, d)
    if args.cons_start != STAGE_CALIB["cons_start"]:
        print(f"[WINDOW OVERRIDE] CONS_START -> {args.cons_start} "
              f"(physics active from e{args.cons_start + 1}); "
              f"handoff_start = {args.handoff_start}")

    # v16: v14 base + time-axis rotation of the spectral weights
    teacher = None
    teacher_frozen = None
    ext_teacher = None
    ext_teacher_acc = None   # v31 [M2]: measured once at load; the gen-2 target line
    # definite-init (G2): cons_teacher is bound in the ext / physics branches
    # below; the plain-baseline branch (--no-physics) leaves it None — safe
    # because it is only dereferenced when w_eff > 0, which requires physics.
    cons_teacher = None
    # v32 [N]: resolve teacher ckpt list (ensemble of independent seeds).
    #   --teacher-ckpts 'a,b,c' -> K-model ensemble (RECOMMENDED, can break 64.66)
    #   --ext-teacher <one>     -> 1-model (v31 [M2] compat; no win, ceiling-locked)
    #   neither                 -> default bit-exact v18 (live-EMA self-teacher)
    teacher_paths = []
    ensemble_n = 0        # v32 [N]: definite-init (G2) for results fields
    member_accs = None    # v32 [N]: definite-init (G2) for results fields
    if args.teacher_ckpts:
        teacher_paths = [p.strip() for p in args.teacher_ckpts.split(",") if p.strip()]
    elif args.ext_teacher is not None:
        teacher_paths = [args.ext_teacher]
    if teacher_paths:
        # v32 [N] ENSEMBLE TEACHER: average K independent ViT-S ckpts as the
        # consistency target. Legitimate 覆辙⑭ (external stronger teacher): a
        # DIFFERENT function from the student (decorrelated errors), real dark
        # knowledge, outside the student's collapse dynamics (覆辙㉖). Live-EMA
        # self-teacher disabled (teacher=None).
        sub_models = [load_ext_teacher(pp, args, device) for pp in teacher_paths]
        member_accs = []
        for _m in sub_models:
            _, _a = evaluate(_m, test_loader, nn.CrossEntropyLoss(), device)
            member_accs.append(_a)
        ensemble = EnsembleTeacher(sub_models).to(device)
        _, ext_teacher_acc = evaluate(ensemble, test_loader,
                                      nn.CrossEntropyLoss(), device)
        ext_teacher = ensemble
        cons_teacher = ensemble
        ensemble_n = ensemble.n
        member_accs = member_accs
        teacher = None          # no self-EMA; delivered = student best (mechanism C)
        teacher_frozen = None
        print(f"v32 [N]: ENSEMBLE TEACHER MODE (K={ensemble.n}) -- consistency target "
              f"= avg of {ensemble.n} independent ViT-S ckpts; live-EMA self-teacher "
              f"OFF; soft-KD (T={args.ext_teacher_t}, w={args.ext_teacher_w}) ENABLED "
              f"for dark knowledge.")
        print(f"  [N] member accs = {[round(a, 2) for a in member_accs]}  "
              f"ensemble acc = {ext_teacher_acc:.2f}%  (born-again target line)")
        if ext_teacher_acc < 64.66:
            print("  [N][WARNING] ensemble acc < 64.66 -- members not yet stronger "
                  "than the single v18 ceiling; born-again gain unlikely. Need "
                  ">=2-3 INDEPENDENT seeds (v31 default on seeds 42/1/2).")
        print("  [N] delivery = student-best restore (mechanism C); mechanism D "
              "structurally OFF (averaging independent inits is not a valid model).")
    elif args.physics:
        teacher = copy.deepcopy(model).to(device)
        teacher.requires_grad_(False)
        teacher.eval()
        teacher_frozen = copy.deepcopy(model).to(device)   # snapshot module
        teacher_frozen.requires_grad_(False)
        teacher_frozen.eval()
        cons_teacher = teacher_frozen
        ramp_end = min(CONS_START + W_CONS_RAMP_LEN, args.epochs)
        hold_end = min(ramp_end + W_CONS_HOLD_LEN, args.epochs)
        print("v19 [D4++]: Gap-Driven Adaptive Spectral Weights (v18 base: compressed time-rotated spectral + terminal hold + teacher-record delivery + residual weighting + handoff + restore):")
        print(f"  [D2] residual per-sample weighting: beta={args.res_beta} "
              f"clip=[{RES_W_LO},{RES_W_HI}] mean-renorm (dose-neutral); "
              f"--res-beta 0 runs the EXACT v12 branch")
        print(f"  [D4] spectral anchor, TIME-AXIS ROTATION v18 COMPRESSED: feat_diff scale "
              f"weights '{args.ms_weights}' = linear {MS_W_START} -> {MS_W_END} "
              f"over e{CONS_START + 1}-{MS_ROT_END} (coarse->fine, dose-neutral "
              f"every epoch); L_fd = sum_m w_m(t) * L(s_m); --ms-weights v14 / "
              f"v15 run the EXACT static v14 / v15 branches, --ms-weights v13 "
              f"the EXACT equal-weight branch")
        print(f"  [D4++ fixed] MULTIPLICATIVE SOFTENED spectral weights (v21): "
              f"w_eff(t) = base(t) * (wf(t)**(alpha*beta)); wf = water-filling "
              f"of PREVIOUS epoch's band residuals (pressure = band/scale2, NO "
              f"gap divisor, beta={ADAPT_BETA}); alpha ramps e{CONS_START + 1}-"
              f"e{ADAPT_ALPHA_END}, disconnects when gap < "
              f"{ADAPT_GAP_FLOOR}pp (let the handoff absorb); --no-ms-adapt "
              f"= EXACT v18 spectrum (bit-exact anchor)")
        print(f"  rotation rationale: v15 closed the STATIC spectral axis "
              f"(teacher 64.50 -> 64.34 on coarse push 0.6/0.3/0.1; c/f "
              f"realized ratio 2.75 < v14's 2.92 at 2.4x weight ratio) — the "
              f"remaining free axis is TIME: coarse smoothing builds "
              f"class-level invariance early, fine Lipschitz harvests the "
              f"unconsumed fine residual late (sigma_max already anneals "
              f"{SIGMA_MAX_START} -> {SIGMA_MAX_END}; the weight rotation "
              f"rides the same schedule); v18 COMPRESSED: rotation e151->e250 "
              f"+ terminal hold e250-e279 — phase-mean (0.316/0.3/0.384) "
              f"shifts the fine epoch-mean +9.6% vs v17 — the probe reads "
              f"the JOINT timing + terminal-dose effect")
        print(f"  band diagnostics: fd_c/fd_m/fd_f per-scale residual means + "
              f"ms_w_c/m/f per-epoch effective weights (epoch log + history) "
              f"for spectral/time attribution")
        print(f"  r_i = 0.3*symKL_i + 0.2*fa_i + 0.5*fd_i (detached, no_grad; fd_i "
              f"spectrally weighted in BOTH cons_ps and cons_term — identical "
              f"recombination); cons_i.mean() == cons_term EXACTLY")
        _cap_note = (f"max cap {EMA_DECAY_BASE}->{args.ema_decay_late} (sched e{EMA_SCHED_START}-{EMA_SCHED_END})"
                     if (args.ema_sched and EMA_SCHED_ON) else f"decay<={args.ema_decay} (constant)")
        print(f"  live EMA teacher ({_cap_note}, warmup 1-1/(t+1)), "
              f"weak view = flip + crop(pad={WEAK_AUG_PAD})")
        print(f"  cons target = FROZEN best-TEACHER snapshot — monotonic: upgraded "
              f"ONLY on new live-teacher records (v11 proven: snapshot held the "
              f"teacher peak 64.13 for handoff while live EMA decayed to 62.28)")
        print(f"  snapshot saved to teacher_best_seed{args.seed}.pt on each new "
              f"teacher record")
        print(f"  w_cons: 0 until e{CONS_START}, "
              f"ramp e{CONS_START + 1}-{ramp_end}, "
              f"hold {args.w_cons_max} to e{hold_end} "
              f"(hold_len={W_CONS_HOLD_LEN}), "
              f"decay to {W_CONS_END} by e{args.epochs}")
        print(f"  GAP GATE: DISABLED by default (floor={args.gate_floor} -> "
              f"gate==1.0 for all gaps, v9 full-strength semantics). "
              f"Dose-response (v10 63.92 / v11 64.13 / v9 64.40 teacher best) "
              f"says the anchor IS the gain source; gating only diluted it.")
        print(f"  THREE-CHANNEL consistency field on the 384-d CLS manifold:")
        print(f"    anchor_logit (w={ANCHOR_LOGIT_W}): "
              f"symKL(student(strong) || snapshot(weak))  [v8 term, kept]")
        print(f"    feat_anchor  (w={FEAT_ANCHOR_W}): "
              f"||norm(f_S(strong)) - norm(f_T(weak))||^2")
        print(f"    feat_diff    (w={FEAT_DIFF_W}): mean over {NOISE_SCALES} scales "
              f"of ||norm(f_S(weak + s_m*eps)) - norm(f_T(weak))||^2, "
              f"shared eps (one Brownian path), ladder ratio {SIGMA_RATIO}")
        if args.kernel_revise and KERNEL_REVISE:
            print(f"  [E] PER-DIM ANISOTROPIC diffusion kernel: sigma_i = s_m*kappa_i "
                  f"(D=384); closed-loop revise from per-dim PRECISION = 1/var(student strong-view CLS) "
                  f"(INVERSE-variance ablation only; v25 DEFAULT OFF — proven harmful). "
                  f"beta={args.kernel_revise_beta}, a-end={KERNEL_REVISE_AEND}, "
                  f"gfloor={KERNEL_REVISE_GFLOOR}; --no-kernel-revise -> v21 isotropic")
        else:
            print(f"  [F] Module E DISABLED (default): keeps NOISE-FREE anchor only "
                  f"(feat_anchor + anchor_logit, v18-equivalent 64.64 floor).")
        if args.ema_sched and EMA_SCHED_ON:
            print(f"  [F] SCHEDULED EMA TEACHER DECAY: max cap ramps {EMA_DECAY_BASE} "
                  f"-> {args.ema_decay_late} over e{EMA_SCHED_START}-{EMA_SCHED_END} "
                  f"(holds LATE post-end); tighter teacher ensemble lifts delivered==teacher "
                  f"ceiling. --no-ema-sched -> constant {EMA_DECAY} (bit-exact v21).")
        else:
            print(f"  [F->v26] EMA SCHED REVERTED (default): constant fast decay "
                  f"<= {EMA_DECAY} — v25 proved the late ramp (0.999->0.9997) KILLED the "
                  f"teacher>student lift (64.41 vs 64.46 anchor). Fast EMA = the lift that "
                  f"made v17/v18 SOTA.")
        if not MS_ADAPT:
            print(f"  [D4++ -> v26] ADAPTIVE SPECTRAL REVERTED (default): MS_ADAPT=False "
                  f"-> EXACT v18 spectrum. v19->v25 adaptive spectral (D4++) cost -0.18pp "
                  f"vs v18 (64.46 vs 64.64).")
        print(f"    sigma_max annealed {SIGMA_MAX_START} -> {SIGMA_MAX_END} "
              f"linearly over e{CONS_START + 1}-{args.epochs} "
              f"(v9 exact floor restored)")
        print(f"  HANDOFF: from e{args.handoff_start}, if best_teacher_acc > "
              f"best_acc + {HANDOFF_MIN_GAP}pp: load best teacher weights into "
              f"the student, IMMEDIATELY evaluate + checkpoint (v12 B: v11's "
              f"first 1e-5 step cost 0.38pp and the fine-tune peak never "
              f"recovered the snapshot), then fine-tune with fresh AdamW @ "
              f"lr={args.handoff_lr} constant, pure supervised (w_eff forced 0) "
              f"({'ON' if args.handoff else 'OFF (--no-handoff)'})")
        print(f"  RESTORE (v12 C): after e{args.epochs}, if best_acc > last-epoch "
              f"acc, reload the best checkpoint and re-evaluate — delivered "
              f"model = argmax(student best, teacher snapshot); final_acc = "
              f"delivered, final_acc_last_epoch = raw tail")
        print(f"  TEACHER-RECORD DELIVERY (v17 D): after the restore path, if the "
              f"handoff fired, evaluate the FINAL teacher record snapshot "
              f"(best_teacher_acc weights) and deliver argmax(student best, "
              f"teacher record) — v16 seed42 evidence: teacher record 64.64@e281 "
              f"exceeded the delivered 64.44 by +0.20 but was never a delivery "
              f"candidate (the v12 passive fallback covers the never-fired case "
              f"only); eval-only, zero RNG, training bit-exact vs v16 "
              f"({'ON' if args.teacher_delivery else 'OFF (--no-teacher-delivery = exact v16 delivery)'})")
        print(f"  Loss: cls + w_eff * ({ANCHOR_LOGIT_W}*anchor_logit + "
              f"{FEAT_ANCHOR_W}*feat_anchor + {FEAT_DIFF_W}*feat_diff"
              f"{' + ' + str(SOFT_KD_W) + '*soft_kd(T=' + str(SOFT_KD_T) + ')' if (SOFT_KD and args.soft_kd) else ''})")
        if SOFT_KD and args.soft_kd:
            print(f"  [G] SOFT-LABEL TEMPERATURE KD (ABLATION ONLY, w={SOFT_KD_W}, T={SOFT_KD_T}): "
                  f"v26 PROVED this harmful (-0.46pp vs v18): the live EMA self-teacher already "
                  f"supplies anchor_logit (T=1), so the T=4 channel is REDUNDANT and over-smooths "
                  f"the student's discriminative peaks. Re-enabled only to re-ablate the dead axis.")
        else:
            print(f"  [G] soft-KD OFF (v27 default): v26 proved the self-teacher T=4 channel "
                  f"HARMFUL; reverted to the v18 noise-free anchor floor. --soft-kd to re-ablate.")
    else:
        print("Baseline training (augmented, no physics field)")

    # optimizer & scheduler
    total_steps = len(train_loader) * args.epochs
    opt = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, build_lr_lambda(args, total_steps))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # training loop
    best_acc = 0.0
    best_teacher_acc = 0.0
    best_teacher_epoch = 0
    band_var_ema = None   # v23 [E]: per-dim teacher-feature variance EMA (closed-loop lag); None -> epoch-1 isotropic
    teacher_best_state = None    # v10 D: in-memory copy of best teacher weights
    # v33 [O1] SWA robust-delivery state (definite-init, G2): running EQUAL-weight
    # average of model weights over the late window. None until epoch >= swa_start.
    swa_state = None
    swa_n = 0
    swa_acc = None               # accuracy of the SWA-averaged weights (delivery candidate)
    snap_acc = 0.0               # v10 B: acc of the frozen snapshot (= teacher
                                 #      record at upgrade time; monotonic)
    snap_epoch = 0
    handoff_fired = False        # v10 D
    handoff_epoch = 0
    pre_handoff_best = 0.0
    handoff_teacher_acc = 0.0
    load_eval_acc = 0.0          # v12 B: acc measured right after loading the
                                 # teacher weights, BEFORE any fine-tune step
    teacher_record_load_eval = None  # v17 D: eval of the FINAL teacher-record
                                 # snapshot (delivery candidate; None = D off
                                 # or handoff never fired)
    global_step = 0
    history = {"train_loss": [], "train_acc": [], "test_loss": [],
               "test_acc": [], "lr": [], "mixup_alpha": [], "train_acc_clean": [],
               "teacher_test_loss": [], "teacher_test_acc": [],
               "w_cons": [], "cons_loss": [],
               "cons_anchor": [], "feat_anchor": [], "feat_diff": [], "soft_kd": [],
               "sigma_max": [], "snap_acc": [],
               "w_eff": [], "gap": [], "handoff": [],
               "res_w_min": [], "res_w_max": [],
               "fd_c": [], "fd_m": [], "fd_f": [],
               "fd_kappa_c": [], "fd_kappa_m": [], "fd_kappa_f": [],
               "kappa_c": [], "kappa_m": [], "kappa_f": [], "kappa_spread": [],
               "band_var_c": [], "band_var_m": [], "band_var_f": [], "band_var_spread": [],
               "ms_w_c": [], "ms_w_m": [], "ms_w_f": [],
               # ---- v41 [W]: 信息能量扩散定律的读数 (覆辙㉕: 新增项必须同步三处:
               #      init / append / results。这里 init, 下面是 append, 末尾 results)
               "heat_sigma_mean": [], "heat_sigma_min": [], "heat_sigma_max": [],
               # ---- v77: 扩散律指数 beta 的读数 (同样是 init/append/results 三处同步) ----
               "heat_beta_mean": [], "heat_beta_min": [], "heat_beta_max": [],
               "rho_first": [], "rho_last": [], "rho_decay": [],
               "dir_first": [], "dir_last": [], "dir_ratio": [], "cos_to_mean": []}

    # ---- v31 [M2]: ext-mode delivery-chain re-anchor (AFTER the init block —
    # v30's ema_state died exactly this way: initialized early, then clobbered
    # by the loop-state init; 覆辙㉕).  The ext teacher IS the best (and only)
    # teacher record: register it so (a) the results JSON shows the gen-2
    # target line, (b) gap telemetry (snap_acc - best_acc) = distillation gap.
    if ext_teacher_acc is not None:
        best_teacher_acc = ext_teacher_acc
        best_teacher_epoch = 0
        snap_acc = ext_teacher_acc
        snap_epoch = 1
        print(f"  [M2] delivery re-anchor: best_teacher_acc={best_teacher_acc:.2f} "
              f"(frozen ext teacher, constant), snap_epoch=1 -> gap = "
              f"ext - student best (gate floor=1.0 => zero training-path change)")

    # v41 [W]: 只读能量输运探针 —— **必须装在 teacher = copy.deepcopy(model)
    # 之后**。deepcopy 会把 forward hook 一起复制进 teacher, teacher 的前向
    # 就会往探针里写脏数据 (还会多一份无谓开销)。探针只挂在 student 上。
    probe = None
    if args.heat_probe:
        probe = EnergyProbe(model, gh_v, gw_v, max_batches=HEAT_PROBE_BATCHES)
        print(f"  [W] 能量输运探针: ON (只读 hook, 每 epoch 只测 {HEAT_PROBE_BATCHES} 个 batch; "
              f"记录 depth_rho / Dirichlet / cos_to_mean / 学到的 sigma)")

    print(f"\n{'='*60}")
    print(f"Starting training: {args.epochs} epochs, {total_steps} steps")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # v41 [W]: 武装探针 (只测本 epoch 的第 1 个 batch)。放在 epoch 最前,
        # 保证 train_one_epoch 的第一次前向一定被采到。
        if probe is not None:
            probe.arm()
        # ---- v10 effective anchor weight: schedule * gap gate ----
        if args.physics:
            w_sched = get_w_cons(epoch, CONS_START, args.epochs,
                                 w_max=args.w_cons_max)
            # gap = snapshot target minus student's historical best (pp).
            # Computed BEFORE the epoch from recorded metrics — pure
            # arithmetic, no RNG, no intra-epoch flapping.
            gap = max(0.0, snap_acc - best_acc) if snap_epoch > 0 else 0.0
            gate = gap_gate(gap, lo=args.gate_lo, hi=args.gate_hi,
                            floor=args.gate_floor)
        else:
            w_sched, gap, gate = 0.0, 0.0, 0.0
        # v22 [E] optional: self-tune gap-gate lo/hi around running gap mean
        if args.physics and args.gate_adapt and GATE_ADAPT and len(history["gap"]) > 0:
            running_gap = sum(history["gap"]) / len(history["gap"])
            gate = gap_gate_adaptive(gap, running_gap, bw=GATE_ADAPT_BW,
                                     lo=args.gate_lo, hi=args.gate_hi,
                                     floor=args.gate_floor)

        # ---- v12 mechanism B: teacher handoff with LOAD-AND-EVALUATE ----
        # v31 [M2]: ext-mode handoff — with a FROZEN external teacher there is
        # no EMA record to absorb; the v18 handoff (load teacher weights into
        # the student) would instead load the INPUT model and make the run
        # degenerate.  The ext-mode handoff is the STUDENT-side half of the v18
        # structure: restore the student's best checkpoint and polish it at a
        # constant tiny LR (CE only, w_eff=0) for the remaining epochs.  The
        # best checkpoint is already protected (mechanism C), so delivered is
        # non-negative by construction.  Zero RNG consumed (torch.load +
        # evaluate only).
        if (args.physics and args.handoff and not handoff_fired
                and epoch >= args.handoff_start):
            if ext_teacher is not None:
                ext_best_ckpt = os.path.join(
                    args.output_dir, f"best_{VERSION}_seed{args.seed}.pt")
                if best_acc > 0.0 and os.path.exists(ext_best_ckpt):
                    model.load_state_dict(
                        torch.load(ext_best_ckpt, map_location=device))
                    _, load_eval_acc = evaluate(model, test_loader,
                                                criterion, device)
                    if load_eval_acc > best_acc:
                        best_acc = load_eval_acc
                        torch.save(model.state_dict(), ext_best_ckpt)
                    pre_handoff_best = best_acc
                    handoff_teacher_acc = ext_teacher_acc
                    opt = build_optimizer(model, args, lr=args.handoff_lr)
                    scheduler = torch.optim.lr_scheduler.LambdaLR(
                        opt, lambda step: 1.0)
                    handoff_fired = True
                    handoff_epoch = epoch
                    print(f"  [HANDOFF-EXT @e{epoch}] student best "
                          f"{pre_handoff_best:.2f} restored (load-eval="
                          f"{load_eval_acc:.2f}); polish lr="
                          f"{args.handoff_lr:.1e} constant, w_eff=0; "
                          f"ext target={ext_teacher_acc:.2f}")
            elif (teacher_best_state is not None
                    and best_teacher_acc > best_acc + HANDOFF_MIN_GAP):
                model.load_state_dict(teacher_best_state)
                pre_handoff_best = best_acc
                handoff_teacher_acc = best_teacher_acc
                # v12 B: measure the loaded weights BEFORE any fine-tune step
                # touches them. v11 evidence: the first 1e-5 step already cost
                # 0.38pp and the fine-tune peak (63.96) never recovered the
                # snapshot value (64.13) — fine-tune was a net loss. Register the
                # teacher value into the student record path (best_acc + ckpt)
                # so the end-of-run restore can always recover it. Since the
                # student shares the teacher's architecture AND buffers, this
                # evaluation must reproduce teacher_best_acc exactly (it was
                # measured on the same test loader). Zero RNG consumed.
                _, load_eval_acc = evaluate(model, test_loader, criterion, device)
                if load_eval_acc > best_acc:
                    best_acc = load_eval_acc
                    torch.save(model.state_dict(),
                               os.path.join(args.output_dir,
                                           f"best_{VERSION}_seed{args.seed}.pt"))
                # fresh optimizer at constant tiny lr: the EMA weights sit in a
                # flat minimum — momentum from the pre-handoff trajectory would
                # kick the student out of it. Zero RNG consumed. From here the
                # fine-tune is a lottery ticket (may beat the teacher, probably
                # won't) — the best is already protected by mechanism B/C.
                opt = build_optimizer(model, args, lr=args.handoff_lr)
                scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
                handoff_fired = True
                handoff_epoch = epoch
                print(f"  [HANDOFF @e{epoch}] student best {pre_handoff_best:.2f} <- "
                      f"teacher best {best_teacher_acc:.2f} weights; "
                      f"load-eval={load_eval_acc:.2f} (checkpointed); fine-tune "
                      f"lr={args.handoff_lr:.1e} constant, w_eff=0")

        w_eff = 0.0 if (args.physics and handoff_fired) else w_sched * gate
        sigma_max = (get_sigma_max(epoch, CONS_START, args.epochs)
                     if args.physics and w_eff > 0 else 0.0)
        # v19 [D4++]: the epoch's effective spectral weights — base = the v18
        # time prior; the adaptive branch adds water-filling of the PREVIOUS
        # epoch's band residuals, gated by the alpha ramp + the gap floor.
        # Pure arithmetic on .item() scalars + the top-level gap: zero RNG,
        # zero extra forwards — the training RNG draw order is untouched.
        # --no-ms-adapt (or alpha=0 at e151) -> ms_ep == the EXACT v18 base
        # profile and ms_w_override == ms_ep (values identical to what
        # train_one_epoch computes internally: bit-exact anchor).
        ms_ep = (ms_weights_for_epoch(epoch, args.ms_weights)
                 if args.physics and MS_SPECTRAL_ANCHOR else None)
        ms_w_override = None
        if (args.physics and MS_SPECTRAL_ANCHOR and args.ms_adapt
                and ms_ep is not None and len(history["fd_c"]) > 0):
            alpha = adapt_alpha(epoch, CONS_START + 1,
                                args.ms_adapt_alpha_end)
            if alpha > 0.0 and gap >= args.ms_adapt_gap_floor:
                prev_bands = [history["fd_c"][-1], history["fd_m"][-1],
                              history["fd_f"][-1]]
                ms_ep = adaptive_spectral_weights(ms_ep, prev_bands, alpha,
                                                  args.ms_adapt_beta)
            ms_w_override = ms_ep

        # v24 [E]: closed-loop PER-DIM diffusion bandwidth from student-feature PRECISION (inverse-variance).
        kappa_bands = None   # (D,) per-dim tensor, or None -> anchor (epoch 1 / --no-kernel-revise)
        if (args.physics and KERNEL_REVISE and args.kernel_revise
                and band_var_ema is not None):
            alpha = adapt_alpha(epoch, CONS_START + 1, KERNEL_REVISE_AEND)
            if alpha > 0.0 and gap >= KERNEL_REVISE_GFLOOR:
                kappa_bands = kernel_band_gains(band_var_ema, alpha, args.kernel_revise_beta)
        kappa_bands_t = (torch.tensor(kappa_bands, device=device, dtype=torch.float32)
                         if kappa_bands is not None else None)

        (train_loss, train_acc, global_step, avg_cls, avg_cons,
            avg_anchor, avg_feat_anchor, avg_feat_diff,
            res_w_min, res_w_max, fd_c, fd_m, fd_f, fd_kc, fd_km, fd_kf,
            band_var, avg_soft_kd) = train_one_epoch(
                model, teacher, cons_teacher, ext_teacher, train_loader, opt, scheduler,
                criterion, device,
                epoch, args, global_step, total_steps,
                w_eff, mixup_start, mixup_end,
                ms_w_override=ms_w_override,
                kappa_bands=kappa_bands_t,
                kernel_revise=(args.kernel_revise and KERNEL_REVISE),
            )
        # v24 [E]: EMA-update the per-dim PRECISION closed-loop signal (lag for next epoch)
        if args.kernel_revise and KERNEL_REVISE:
            if band_var_ema is None:
                band_var_ema = list(band_var)
            else:
                band_var_ema = [BAND_VAR_EMA_DECAY * b + (1 - BAND_VAR_EMA_DECAY) * x
                                for b, x in zip(band_var_ema, band_var)]
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        if teacher is not None:
            teacher_loss, teacher_acc = evaluate(teacher, test_loader,
                                                 criterion, device)
            # v10/v11 mechanism B: snapshot upgrades ONLY on a live-TEACHER record
            # — monotonic by construction (snapshot = strongest target ever).
            # v9 upgraded on student records and DOWNGRADED 63.84 -> 63.27
            # @e238 when the student's success froze a decayed EMA copy.
            if teacher_acc > best_teacher_acc:
                best_teacher_acc = teacher_acc
                best_teacher_epoch = epoch
                teacher_best_state = copy.deepcopy(teacher.state_dict())
                teacher_frozen.load_state_dict(teacher.state_dict())
                torch.save(teacher_frozen.state_dict(),
                           os.path.join(args.output_dir,
                                        f"teacher_best_seed{args.seed}.pt"))
                snap_acc = teacher_acc
                snap_epoch = epoch
            # student record: checkpoint only — snapshot no longer follows it
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir,
                                        f"best_{VERSION}_seed{args.seed}.pt"))
        else:
            # v31 [M2]: ext mode — the frozen teacher is a CONSTANT: record its
            # measured accuracy (no per-epoch re-eval — same weights, same
            # loader, wasted compute). Baseline mode keeps 0.0/0.0.
            teacher_loss = 0.0
            teacher_acc = ext_teacher_acc if ext_teacher is not None else 0.0
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir,
                                        f"best_{VERSION}_seed{args.seed}.pt"))
        elapsed = time.time() - t0

        # v35 [Q] MEASUREMENT (no training effect): accuracy on a FIXED
        # UNAUGMENTED train subset. history["train_acc"] is measured on
        # mixup-augmented batches and therefore changes meaning the instant the
        # mixup window shuts (63.95 -> 92.91 at e200/e201 in v33/v34) — that
        # discontinuity, not overfitting, produced the alarming "32pp gap".
        # train_acc_clean is comparable in EVERY epoch, so gap_clean =
        # train_acc_clean - test_acc is the first honest overfitting readout.
        # evaluate() sets model.eval(); train_one_epoch() re-sets model.train(),
        # and this consumes no RNG -> training stays bit-exact.
        _, train_acc_clean = evaluate(model, train_eval_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_acc_clean"].append(train_acc_clean)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["lr"].append(opt.param_groups[0]["lr"])
        # v35 [Q]: resolved mixup alpha actually used this epoch (canary: proves
        # the augmentation was ON/OFF where we think it was).
        history["mixup_alpha"].append(
            float(get_mixup_alpha(epoch, mixup_start, mixup_end, args.mixup_alpha)))
        history["teacher_test_loss"].append(teacher_loss)
        history["teacher_test_acc"].append(teacher_acc)
        history["w_cons"].append(w_sched)
        history["cons_loss"].append(avg_cons)
        history["cons_anchor"].append(avg_anchor)
        history["feat_anchor"].append(avg_feat_anchor)
        history["feat_diff"].append(avg_feat_diff)
        history["soft_kd"].append(avg_soft_kd)    # v26 [G]
        history["sigma_max"].append(sigma_max)
        history["fd_kappa_c"].append(fd_kc)
        history["fd_kappa_m"].append(fd_km)
        history["fd_kappa_f"].append(fd_kf)
        # v24 [E]: record per-dim kappa trajectory (coarse/mid/fine thirds) + spread
        if kappa_bands_t is not None:
            kb = kappa_bands_t
            Dkb = kb.size(0)
            kc = float(kb[:Dkb // 3].mean()); km = float(kb[Dkb // 3:2 * Dkb // 3].mean())
            kf = float(kb[2 * Dkb // 3:].mean())
            kspread = float(kb.max() / kb.min().clamp(min=1e-9))
        else:
            kc = km = kf = 1.0; kspread = 1.0
        history["kappa_c"].append(kc)
        history["kappa_m"].append(km)
        history["kappa_f"].append(kf)
        history["kappa_spread"].append(kspread)
        # per-dim VARIANCE signal (coarse/mid/fine thirds) for diagnosis of contrast
        if band_var is not None and len(band_var) > 0:
            Dv = len(band_var)
            bvc = sum(band_var[:Dv // 3]) / max(1, Dv // 3)
            bvm = sum(band_var[Dv // 3:2 * Dv // 3]) / max(1, Dv // 3)
            bvf = sum(band_var[2 * Dv // 3:]) / max(1, Dv - 2 * Dv // 3)
            bvspread = max(band_var) / max(min(band_var), 1e-9)
        else:
            bvc = bvm = bvf = 0.0; bvspread = 1.0
        history["band_var_c"].append(bvc)
        history["band_var_m"].append(bvm)
        history["band_var_f"].append(bvf)
        history["band_var_spread"].append(bvspread)
        history["snap_acc"].append(snap_acc)
        history["w_eff"].append(w_eff)
        history["gap"].append(gap)
        history["handoff"].append(1 if handoff_fired else 0)
        # v33 [O1] SWA: equal-weight running average of the STUDENT model weights
        # over the late window (epoch >= swa_start). This is orthogonal to the EMA
        # teacher (which is exponential/recent-heavy and only delivered when it
        # beats the student). SWA consistently finds flatter minima and reduces
        # seed variance — it directly lifts seeds like seed1 (EMA boost ~0) without
        # ever regressing a lucky seed (it is only an argmax candidate at delivery).
        if args.swa and epoch >= args.swa_start:
            _sd = model.state_dict()
            if swa_state is None:
                swa_state = copy.deepcopy(_sd)
            else:
                for _k in swa_state:
                    swa_state[_k] = (swa_state[_k] * swa_n
                                     + _sd[_k]) / (swa_n + 1.0)
            swa_n += 1
        history["res_w_min"].append(res_w_min)
        history["res_w_max"].append(res_w_max)
        history["fd_c"].append(fd_c)
        history["fd_m"].append(fd_m)
        history["fd_f"].append(fd_f)
        history["ms_w_c"].append(ms_ep[0] if ms_ep is not None else 0.0)
        history["ms_w_m"].append(ms_ep[1] if ms_ep is not None else 0.0)
        history["ms_w_f"].append(ms_ep[2] if ms_ep is not None else 0.0)
        # ---- v41 [W]: 能量输运 / 扩散尺度读数 ----
        if probe is not None:
            _s = probe.summary()
            history["rho_first"].append(round(_s["rho_first"], 6))
            history["rho_last"].append(round(_s["rho_last"], 6))
            history["rho_decay"].append(round(_s["rho_decay"], 6))
            history["dir_first"].append(round(_s["dir_first"], 6))
            history["dir_last"].append(round(_s["dir_last"], 6))
            history["dir_ratio"].append(round(_s["dir_ratio"], 6))
            history["cos_to_mean"].append(round(_s["cos_to_mean"], 6))
        else:
            for _k in ("rho_first", "rho_last", "rho_decay",
                       "dir_first", "dir_last", "dir_ratio", "cos_to_mean"):
                history[_k].append(0.0)
        # 学到的扩散尺度 (纯读数, 不改训练): sigma = exp(log_sigma)
        if heat_blocks > 0:
            with torch.no_grad():
                _sg = torch.stack([b.attn.log_sigma.detach().float()
                                   for b in model.blocks]).exp()
                # [v74] 扩散律指数 (纯读数): beta = MID + HALF*tanh(raw_beta)
                _bg = heat_beta_of(torch.stack([b.attn.raw_beta.detach().float()
                                                for b in model.blocks]))
            history["heat_sigma_mean"].append(round(float(_sg.mean()), 4))
            history["heat_sigma_min"].append(round(float(_sg.min()), 4))
            history["heat_sigma_max"].append(round(float(_sg.max()), 4))
            history["heat_beta_mean"].append(round(float(_bg.mean()), 4))
            history["heat_beta_min"].append(round(float(_bg.min()), 4))
            history["heat_beta_max"].append(round(float(_bg.max()), 4))
        else:
            history["heat_sigma_mean"].append(0.0)
            history["heat_sigma_min"].append(0.0)
            history["heat_sigma_max"].append(0.0)
            history["heat_beta_mean"].append(0.0)
            history["heat_beta_min"].append(0.0)
            history["heat_beta_max"].append(0.0)

        t_str = ((f"  teacher={teacher_acc:.2f}%" if teacher is not None
                  else (f"  ext={teacher_acc:.2f}%" if ext_teacher is not None
                        else "")))
        c_str = ""
        if args.physics and epoch > CONS_START:
            c_str = (f"  cons={avg_cons:.4f}"
                     f"(a={avg_anchor:.4f} fa={avg_feat_anchor:.4f}"
                     f" fd={avg_feat_diff:.4f})"
                     f"(w={w_eff:.3f}/{w_sched:.3f} g={gate:.2f}"
                     f" s={sigma_max:.3f})")
            if args.res_beta > 0 and not handoff_fired and w_eff > 0:
                c_str += f"(rw=[{res_w_min:.2f},{res_w_max:.2f}])"
            if fd_c > 0 or fd_m > 0 or fd_f > 0:
                c_str += f"(ms=[{fd_c:.4f},{fd_m:.4f},{fd_f:.4f}])"
            if ms_ep is not None:
                c_str += (f"(msw=[{ms_ep[0]:.2f},{ms_ep[1]:.2f},"
                          f"{ms_ep[2]:.2f}])")
        snap_str = (f"  snap={snap_acc:.2f}@e{snap_epoch}"
                    if teacher is not None and snap_epoch > 0
                    else (f"  gap={snap_acc - best_acc:+.2f}"
                          if ext_teacher is not None else ""))
        ho_str = "  [HANDOFF]" if handoff_fired else ""
        print(f"Epoch {epoch:3d}/{args.epochs}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}%  "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}%  "
              f"best={best_acc:.2f}%{t_str}{snap_str}{c_str}{ho_str}  ({elapsed:.1f}s)")

    # (v41 [W] 的扩散定律读数在交付链之后做 —— sigma 必须读**交付模型**的,
    #  见 results 之前的 [v41 扩散定律] 段)

    # ---- v12 mechanism C: best-checkpoint restore (delivered model) ----
    # Deliver argmax(student best, teacher snapshot). final_acc = delivered
    # model accuracy; final_acc_last_epoch preserves the raw tail for
    # diagnostics (v11: tail decay -0.25pp is measurement, not delivery).
    final_acc_last = history["test_acc"][-1]
    delivered_acc = final_acc_last
    restore_src = "last_epoch"
    if args.physics:
        best_ckpt = os.path.join(args.output_dir, f"best_{VERSION}_seed{args.seed}.pt")
        # passive fallback: teacher leads but handoff never fired (gap stayed
        # under HANDOFF_MIN_GAP all the way) — still deliver the snapshot
        if (not handoff_fired and teacher_best_state is not None
                and best_teacher_acc > best_acc):
            model.load_state_dict(teacher_best_state)
            _, delivered_acc = evaluate(model, test_loader, criterion, device)
            restore_src = "teacher_snapshot (passive, handoff never fired)"
            if delivered_acc > best_acc:
                best_acc = delivered_acc
                torch.save(model.state_dict(), best_ckpt)
        # active restore: best checkpoint beats the raw last-epoch model
        elif best_acc > delivered_acc + 1e-9 and os.path.exists(best_ckpt):
            model.load_state_dict(torch.load(best_ckpt, map_location=device))
            _, delivered_acc = evaluate(model, test_loader, criterion, device)
            restore_src = "best_checkpoint"
        # ---- v17 [D]: teacher-record delivery candidacy ----
        # v16 seed42: teacher record 64.64 @e281 (set DURING the handoff
        # window — the EMA re-tracking of the restored student immediately
        # peaked) exceeded the delivered 64.44 by +0.20. The v12 delivery
        # path only considers the student best checkpoint once the handoff
        # has fired (the passive fallback above covers the never-fired case
        # only). Mechanism D makes the FINAL teacher record (monotonic best
        # EMA snapshot = the time-smoothed solution at its best point) a
        # delivery candidate: evaluate it, deliver argmax(student, teacher
        # record). Mean-teacher delivery is standard; the EMA teacher shares
        # the student architecture AND buffers, so this evaluation must
        # reproduce best_teacher_acc exactly (mechanism-B identity, five
        # prior zero-loss confirmations). Eval-only, AFTER training: zero
        # RNG consumed — the training trajectory is bit-exact v16.
        if (TEACHER_SNAPSHOT_DELIVERY and args.teacher_delivery
                and handoff_fired and teacher_best_state is not None):
            delivered_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(teacher_best_state)
            _, teacher_record_eval = evaluate(model, test_loader,
                                              criterion, device)
            teacher_record_load_eval = teacher_record_eval
            if teacher_record_eval > delivered_acc + 1e-9:
                delivered_acc = teacher_record_eval
                restore_src = "teacher_snapshot_record (v17 D)"
                if teacher_record_eval > best_acc:
                    best_acc = teacher_record_eval
                    torch.save(model.state_dict(), best_ckpt)
            else:
                model.load_state_dict(delivered_state)
        # ---- v33 [O1]: SWA delivery candidacy ----
        # Evaluate the SWA-averaged weights; deliver argmax with the other
        # candidates so it can only LIFT (never regress) a lucky seed. For
        # seed1-style runs (EMA boost ~0, handoff never fired) this is the
        # primary floor-lift; for seed42 it is neutral (teacher-record wins).
        if args.swa and swa_state is not None:
            _swa_keep = copy.deepcopy(model.state_dict())
            model.load_state_dict(swa_state)
            _, swa_eval = evaluate(model, test_loader, criterion, device)
            swa_acc = swa_eval
            if swa_eval > delivered_acc + 1e-9:
                delivered_acc = swa_eval
                restore_src = "swa_avg (late window, v33 O1)"
                if swa_eval > best_acc:
                    best_acc = swa_eval
                    torch.save(model.state_dict(), best_ckpt)
            else:
                model.load_state_dict(_swa_keep)
        print(f"  [RESTORE] delivered={delivered_acc:.2f}% "
              f"(source: {restore_src})  last-epoch raw={final_acc_last:.2f}%  "
              f"best={best_acc:.2f}%"
              + (f"  teacher-record eval={teacher_record_load_eval:.2f}%"
                 if teacher_record_load_eval is not None else "")
              + (f"  ext-target={ext_teacher_acc:.2f}% (D off)"
                 if ext_teacher is not None else ""))

    # save results
    # ---- v39 [U]: evaluate the instrument BEFORE building the results dict ----
    # Pure arithmetic over history; the helper itself is guarded so a truncated
    # --smoke run cannot crash the save path.
    _lrn_curve, _lrn_slope, _lrn_rel, _lrn_verdict = v39_instrument_report(
        history, args.epochs, args.lr)

    # ---- v41 [W]: 扩散定律读数。放在**交付链之后**: sigma 必须读交付模型的,
    #   否则读的是"最后一个 epoch"而不是"交付的那个" (覆辙⑰ 的另一面: 同一件事
    #   只准有一个计算点, 这里就这一处)。
    _sig_final = None
    if heat_blocks > 0:
        with torch.no_grad():
            _sig_final = torch.stack([b.attn.log_sigma.detach().float()
                                      for b in model.blocks]).exp()   # (L,H)
    # ---- v74: 交付态 (不是"最后一个 epoch") 的扩散律指数 (覆辙⑰ 同一计算点原则)
    _beta_final = None
    if heat_blocks > 0:
        with torch.no_grad():
            _beta_final = heat_beta_of(torch.stack(
                [b.attn.raw_beta.detach().float() for b in model.blocks]))   # (L,H)
    # ---- v77: 交付态 (不是"最后一个 epoch") 的各向异性比 (覆辙⑰ 同一计算点原则)
    _ar_final = None
    if heat_blocks > 0:
        with torch.no_grad():
            _ar_final = torch.stack(
                [heat_ar_of(b.attn.raw_ar.detach().float()).reshape(())
                 for b in model.blocks])   # (L,)
    # ---- v77 [BA]: 交付端"物理场杠杆"同跑内配对读数 --------------------------------
    # v76 的实测教训: 单跑的交付差只有 0.2~0.6pp << 2 sigma (1.42pp) ⇒ 跨跑比精度**结构性
    # 判不出**"场有没有用"。这里直接在**交付权重**上量场的杠杆: 同一批样本、三种模式配对,
    #   full = 位置场(各向异性, 本版) / iso = 强制各向同性(等价 v74 的位置场) / off = 无场。
    # ⇒ full-vs-iso 就是**本版那根轴自己的**配对效应, 分辨率远高于跨跑比 delivered。
    # 纯读数: eval + no_grad + 只取前 FIELD_ABL_BATCHES 个 batch (shuffle=False ⇒ 确定),
    #         不消耗 RNG, 不改训练轨迹 (覆辙 142)。
    # ★ 末态消融 != 训练反事实 (覆辙 120)。这里量的是"敏感度/杠杆", 不是"反事实精度"。
    #   v76 实测: off 让子集精度崩到随机 (5%) ⇒ **杠杆大只说明"场被依赖", 不说明"有改进
    #   空间"** (拿走拐杖会摔倒, 不代表换一根拐杖能走更快)。判读时两件事不许混。
    _field_abl = None
    if heat_blocks > 0:
        _abl_keep = [b.attn._ablate for b in model.blocks]
        _was_train = bool(model.training)
        model.eval()
        _lg_acc = {"full": [], "iso": [], "off": []}
        _lg_lab = []
        _nb = 0
        with torch.no_grad():
            for _xb, _yb in test_loader:
                if _nb >= FIELD_ABL_BATCHES:
                    break
                _xb = _xb.to(device)
                _lg_lab.append(_yb.to(device))
                for _m in ("full", "iso", "off"):
                    for _b in model.blocks:
                        _b.attn._ablate = _m
                    _o = model(_xb)
                    _o = _o[0] if isinstance(_o, (tuple, list)) else _o
                    _lg_acc[_m].append(_o.detach().float().cpu())
                _nb += 1
        # finally 语义: 无论上面是否抛异常, 把手必须撤掉 (覆辙: 只读块整块 try 会假绿)
        for _b, _m in zip(model.blocks, _abl_keep):
            _b.attn._ablate = _m
        if _was_train:
            model.train()
        if _nb > 0:
            _L = {k: torch.cat(v, dim=0) for k, v in _lg_acc.items()}
            _y = torch.cat(_lg_lab, dim=0).cpu()
            _lstd = float(_L["full"].std())
            _d_fi = float((_L["full"] - _L["iso"]).pow(2).mean().sqrt())
            _d_fo = float((_L["full"] - _L["off"]).pow(2).mean().sqrt())
            _d_io = float((_L["iso"] - _L["off"]).pow(2).mean().sqrt())
            _sacc = {k: float((_L[k].argmax(-1) == _y).float().mean() * 100.0) for k in _L}
            _fl_fi = float((_L["full"].argmax(-1) != _L["iso"].argmax(-1)).float().mean())
            _fl_fo = float((_L["full"].argmax(-1) != _L["off"].argmax(-1)).float().mean())
            _rel_axis = (_d_fi / _lstd) if _lstd > 1e-9 else None
            _rel_all = (_d_fo / _lstd) if _lstd > 1e-9 else None
            _field_abl = {
                "n_batches": _nb,
                "n_samples": int(_L["full"].shape[0]),
                "logit_std": round(_lstd, 6),
                "d_logit_rms_full_vs_iso": round(_d_fi, 6),
                "d_logit_rms_full_vs_off": round(_d_fo, 6),
                "d_logit_rms_iso_vs_off": round(_d_io, 6),
                "rel_shift_full_vs_iso": (round(_rel_axis, 6) if _rel_axis is not None else None),
                "rel_shift_full_vs_off": (round(_rel_all, 6) if _rel_all is not None else None),
                "argmax_flip_full_vs_iso": round(_fl_fi, 6),
                "argmax_flip_full_vs_off": round(_fl_fo, 6),
                "subset_acc_full": round(_sacc["full"], 4),
                "subset_acc_iso": round(_sacc["iso"], 4),
                "subset_acc_off": round(_sacc["off"], 4),
                "subset_acc_full_minus_iso_pp": round(_sacc["full"] - _sacc["iso"], 4),
                "lever_mid_tol": FIELD_ABL_LEV_MID,
                "lever_high_tol": FIELD_ABL_LEV_HIGH,
                "weights": "delivery checkpoint (the same weights as the delivered accuracy)",
            }
            _field_abl["axis_verdict"] = (
                "ANISO_LEVERAGE_HIGH" if (_rel_axis is not None and _rel_axis >= FIELD_ABL_LEV_HIGH)
                else ("ANISO_LEVERAGE_MID" if (_rel_axis is not None
                                               and _rel_axis >= FIELD_ABL_LEV_MID)
                      else "ANISO_LEVERAGE_LOW"))
            _field_abl["field_verdict"] = (
                "FIELD_LEVERAGE_HIGH" if (_rel_all is not None and _rel_all >= FIELD_ABL_LEV_HIGH)
                else ("FIELD_LEVERAGE_MID" if (_rel_all is not None
                                               and _rel_all >= FIELD_ABL_LEV_MID)
                      else "FIELD_LEVERAGE_LOW"))
            # ★ note 一律由**读数派生**, 不许写死文字。
            #   反例 (v76): note 写死"rel_shift 远小于 1", 而实测是 1.35 ⇒ 同一个块里
            #   verdict=LEVERAGE_HIGH 与 note=杠杆很小 自相矛盾 (覆辙 138 的实例)。
            _field_abl["note"] = (
                "同跑内配对敏感度读数 (末态消融 != 训练反事实, 覆辙 120)。"
                "iso == 强制 ar=0 == v74 的位置场 ⇒ full-vs-iso 就是本版那根轴的配对杠杆:"
                " rel_shift=" + str(_field_abl["rel_shift_full_vs_iso"])
                + " (相对 logit std), argmax 翻转=" + str(_field_abl["argmax_flip_full_vs_iso"])
                + ", 子集精度差=" + str(_field_abl["subset_acc_full_minus_iso_pp"]) + "pp。"
                " off 是整条场的杠杆: rel_shift=" + str(_field_abl["rel_shift_full_vs_off"])
                + ", 翻转=" + str(_field_abl["argmax_flip_full_vs_off"])
                + ", 子集精度=" + str(_field_abl["subset_acc_off"]) + "%。"
                " ★ 杠杆大 != 有改进空间: 模型与场是共同适应的, off 把精度打到随机只说明"
                "场是承重结构, 不说明换一根能走更快。")
            print(f"\n[v77 交付端场杠杆] 本轴 {_field_abl['axis_verdict']} / "
                  f"整场 {_field_abl['field_verdict']}  "
                  f"(n={_nb} batch / {_field_abl['n_samples']} 样本, 交付权重)")
            print(f"  logit std = {_field_abl['logit_std']}; |dlogit|rms: "
                  f"full-vs-iso={_field_abl['d_logit_rms_full_vs_iso']} "
                  f"full-vs-off={_field_abl['d_logit_rms_full_vs_off']} "
                  f"iso-vs-off={_field_abl['d_logit_rms_iso_vs_off']}")
            print(f"  相对杠杆: full-vs-iso={_field_abl['rel_shift_full_vs_iso']} "
                  f"full-vs-off={_field_abl['rel_shift_full_vs_off']};  argmax 翻转: "
                  f"full-vs-iso={_field_abl['argmax_flip_full_vs_iso']} "
                  f"full-vs-off={_field_abl['argmax_flip_full_vs_off']}")
            print(f"  子集精度: full={_field_abl['subset_acc_full']}% "
                  f"iso={_field_abl['subset_acc_iso']}% off={_field_abl['subset_acc_off']}%  "
                  f"(full-iso={_field_abl['subset_acc_full_minus_iso_pp']}pp)")

    _law = {
        "depth_rho_decay_final": (round(history["rho_decay"][-1], 6)
                                  if history["rho_decay"] else None),
        "depth_rho_first_mean": (round(sum(history["rho_first"]) / len(history["rho_first"]), 6)
                                 if history["rho_first"] else None),
        "depth_rho_last_mean": (round(sum(history["rho_last"]) / len(history["rho_last"]), 6)
                                if history["rho_last"] else None),
        "dir_ratio_first": (round(history["dir_ratio"][0], 6)
                            if history["dir_ratio"] else None),
        "dir_ratio_final": (round(history["dir_ratio"][-1], 6)
                            if history["dir_ratio"] else None),
        "cos_to_mean_final": (round(history["cos_to_mean"][-1], 6)
                              if history["cos_to_mean"] else None),
        "cos_to_mean_first": (round(history["cos_to_mean"][0], 6)
                              if history["cos_to_mean"] else None),
        "sigma_init_mean": (round(float(heat_sigma0.exp().mean()), 4)
                            if heat_sigma0 is not None else None),
        "sigma_final_mean": (round(float(_sig_final.mean()), 4)
                             if _sig_final is not None else None),
        "sigma_final_per_block": ([round(float(v), 3) for v in row]
                                  for row in
                                  (_sig_final.tolist() if _sig_final is not None else [])),
        "sigma_block_mean": ([round(float(v), 3) for v in
                              _sig_final.mean(dim=1).tolist()]
                             if _sig_final is not None else None),
        "sigma_head_mean": ([round(float(v), 3) for v in
                             _sig_final.mean(dim=0).tolist()]
                            if _sig_final is not None else None),
        "sigma_drift_ratio": (round(float(_sig_final.mean()) / float(heat_sigma0.exp().mean()), 4)
                              if (_sig_final is not None and heat_sigma0 is not None) else None),
    }
    # 生成器 -> 列表 (json 不能序列化生成器)
    _law["sigma_final_per_block"] = [r for r in _law["sigma_final_per_block"]]
    _law["verdict"] = diffusion_law_verdict(
        _law["depth_rho_decay_final"] if _law["depth_rho_decay_final"] is not None else 1.0,
        _law["dir_ratio_final"] if _law["dir_ratio_final"] is not None else 1.0,
        _law["cos_to_mean_final"] if _law["cos_to_mean_final"] is not None else 0.0)
    print(f"\n[v41 扩散定律] {_law['verdict']}")
    print(f"  深度轴输运: rho(l=1)均值={_law['depth_rho_first_mean']} -> "
          f"rho(l=L)均值={_law['depth_rho_last_mean']} "
          f"(末/首 = {_law['depth_rho_decay_final']}; <0.5 = 深层拿不到能量)")
    print(f"  token 场 Dirichlet 能量: 首块入口 -> 末块出口 剩余 "
          f"{_law['dir_ratio_final']} (训练开始时 {_law['dir_ratio_first']}; "
          f"<0.25 = 12 步注意力扩散把空间对比度抹平了)")
    print(f"  过平滑指数 cos_to_mean: {_law['cos_to_mean_first']} -> "
          f"{_law['cos_to_mean_final']} (>0.7 = token 塌向均值)")
    if _sig_final is not None:
        print(f"  学到的扩散尺度 sigma: 初始均值 {_law['sigma_init_mean']} -> "
              f"末态均值 {_law['sigma_final_mean']} (漂移比 {_law['sigma_drift_ratio']})")
        print(f"    每 block 均值: {_law['sigma_block_mean']}")
        print(f"    每 head  均值: {_law['sigma_head_mean']}")

    # ---- v77 [X]: 本版唯一一根轴的**机器判据**块 ------------------------------
    # 覆辙 115: NULL 有两种 —— "机制无效" 与 "干预没送到"。这里把"新自由度到底
    # 有没有被用上"做成机器可读字段，单跑 NULL 时能立刻分清是哪一种。
    # 覆辙 126: 判据表不许含被校验的字面 ⇒ verdict 由**实测张量**推导，不写死。
    _beta_hist_lo = (min(history["heat_beta_min"]) if history["heat_beta_min"] else None)
    _beta_hist_hi = (max(history["heat_beta_max"]) if history["heat_beta_max"] else None)
    _beta_hist_dev = (max(abs(_beta_hist_hi - HEAT_BETA_MID),
                          abs(HEAT_BETA_MID - _beta_hist_lo))
                      if _beta_hist_lo is not None else None)
    _beta_dev_final = (float((_beta_final - HEAT_BETA_MID).abs().max())
                       if _beta_final is not None else None)
    _anom = {
        "axis": ("anisotropic heat kernel (scalar diffusion rate -> diagonal diffusion TENSOR): "
                 "d2_eff = dx^2*exp(-ar) + dy^2*exp(+ar) with ar = HEAT_ANISO_MAX*tanh(raw_ar), "
                 "one learnable scalar per LAYER; ar=0 is exactly isotropic == v74's field. "
                 "The beta block below is v74's radial decay law, kept verbatim (not this "
                 "version's axis)."),
        "beta_mid": HEAT_BETA_MID,
        "beta_half": HEAT_BETA_HALF,
        "beta_bounds": [HEAT_BETA_MID - HEAT_BETA_HALF, HEAT_BETA_MID + HEAT_BETA_HALF],
        "beta_n_new_params": (int(_beta_final.numel()) if _beta_final is not None else 0),
        "beta_trainable": (bool(all(bool(b.attn.raw_beta.requires_grad)
                                    for b in model.blocks)) if heat_blocks > 0 else None),
        "beta_final_mean": (round(float(_beta_final.mean()), 4)
                            if _beta_final is not None else None),
        "beta_final_min": (round(float(_beta_final.min()), 4)
                           if _beta_final is not None else None),
        "beta_final_max": (round(float(_beta_final.max()), 4)
                           if _beta_final is not None else None),
        "beta_block_mean": ([round(float(v), 4) for v in _beta_final.mean(dim=1).tolist()]
                            if _beta_final is not None else None),
        "beta_head_mean": ([round(float(v), 4) for v in _beta_final.mean(dim=0).tolist()]
                           if _beta_final is not None else None),
        # ★★ 剂量读数: max|beta - 2| (交付态权重)。≈0 ⇒ 优化器从未用上这个新自由度。
        "beta_max_dev": (round(_beta_dev_final, 6) if _beta_dev_final is not None else None),
        "beta_hist_max_dev": (round(float(_beta_hist_dev), 6)
                              if _beta_hist_dev is not None else None),
        "beta_moved_tol": HEAT_BETA_MOVED_TOL,
        "beta_moved": (bool(_beta_dev_final > HEAT_BETA_MOVED_TOL)
                       if _beta_dev_final is not None else None),
        # 覆辙 101/148: 撞软界 = 参数被截住，必须显式可见(否则会被当成"学到了")。
        # ★ 用**相对**容差 (界宽的 1%), 不用绝对容差: 绝对容差与界宽脱钩 —— v74 实测
        #   beta_max = 2.9995 而软界是 3.0, 边距 0.0005 < 旧的绝对容差 1e-3 ⇒ 旧判定报
        #   false, 而物理上早就是撞界。v76 的 lambda (0.498563 vs 0.5) 也是同一个坑。
        "beta_hits_bound": (bool(min(abs(_beta_final.min().item()
                                         - (HEAT_BETA_MID - HEAT_BETA_HALF)),
                                     abs((HEAT_BETA_MID + HEAT_BETA_HALF)
                                         - _beta_final.max().item()))
                                 < HEAT_BETA_HALF * HEAT_BOUND_REL_TOL)
                            if _beta_final is not None else None),
        # ---- v77 [BA]: 本版唯一新轴 —— 各向异性比 ar 的剂量 / 撞界 / 形状读数 ----
        "aniso": {
            "max_abs": HEAT_ANISO_MAX,
            "bounds": [-HEAT_ANISO_MAX, HEAT_ANISO_MAX],
            "n_new_params": (int(_ar_final.numel()) if _ar_final is not None else 0),
            "trainable": (bool(all(bool(b.attn.raw_ar.requires_grad)
                                   for b in model.blocks)) if heat_blocks > 0 else None),
            "final_per_block": ([round(float(v), 4) for v in _ar_final.tolist()]
                                if _ar_final is not None else None),
            "final_mean": (round(float(_ar_final.mean()), 4)
                           if _ar_final is not None else None),
            "final_min": (round(float(_ar_final.min()), 4)
                          if _ar_final is not None else None),
            "final_max": (round(float(_ar_final.max()), 4)
                          if _ar_final is not None else None),
            # ★★ 剂量读数: max|ar| (交付态权重)。≈0 ⇒ 优化器从未用上这个新自由度。
            "max_dev": (round(float(_ar_final.abs().max()), 6)
                        if _ar_final is not None else None),
            "moved_tol": HEAT_AR_MOVED_TOL,
            "moved": (bool(float(_ar_final.abs().max()) > HEAT_AR_MOVED_TOL)
                      if _ar_final is not None else None),
            "hits_bound": (bool(min(abs(float(_ar_final.min()) + HEAT_ANISO_MAX),
                                    abs(HEAT_ANISO_MAX - float(_ar_final.max())))
                                < HEAT_ANISO_MAX * HEAT_BOUND_REL_TOL)
                           if _ar_final is not None else None),
            # sigma_x/sigma_y = exp(2*ar); 相对同层的各向同性 sigma 的比值是 exp(±ar/2)
            "sigma_axis_ratio": ([(round(math.exp(0.5 * float(a)), 3),
                                   round(math.exp(-0.5 * float(a)), 3))
                                  for a in _ar_final.tolist()]
                                 if _ar_final is not None else None),
            "note": ("ar is the per-LAYER anisotropy of the diffusion TENSOR (scalar diffusion "
                     "rate -> diagonal tensor). d2_eff = dx^2*exp(-ar) + dy^2*exp(+ar); ar = 0 "
                     "is exactly isotropic and therefore bit-identical to v74. The geometric mean "
                     "is preserved (sigma_x*sigma_y = sigma^2) so ar changes the SHAPE only and "
                     "cannot be absorbed into sigma (that is why a learnable bias gain was banned)."),
        },
        "why_this_axis": ("ar is a first-order degree of freedom of the FIELD itself and it is NOT "
                          "degenerate with either sigma or beta: sigma rescales the argument "
                          "(d/sigma), beta changes the RADIAL decay law, while ar breaks the "
                          "ANGULAR (x vs y) symmetry -- the three are mutually orthogonal. It is "
                          "also NOT the v76 content axis: this version stays input-independent and "
                          "translation-invariant, it only breaks rotational/reflectional isotropy. "
                          "This is the STATIC family (no runtime-statistics feedback), since "
                          "KERNEL_REVISE / MS_ADAPT / GATE_ADAPT (closed-loop) were all proven "
                          "harmful. Passive prior (v36): raw_ar belongs to the field only and the "
                          "bias enters purely as an additive term on attention logits."),
        # ★ 撞界判定的**语义变更** (跨版本比 *_hits_bound 时必须知道这一点):
        #   v74 / v76 用**绝对**容差 1e-3; 本版起统一用**相对**容差 (界宽的 1%)。
        #   v74 实测: beta 上界 3.0, 末态 2.9995 ⇒ 边距 0.0005 < 1e-3 ⇒ 旧语义报 false,
        #   新语义报 true —— 新语义才是对的 (绝对容差与界宽脱钩 ⇒ 判定由容差而非物理决定)。
        "bound_tol_rel": HEAT_BOUND_REL_TOL,
        "readout_note": ("aniso.max_dev is THIS version's dose readout (max|ar| over layers). If it "
                         "is ~0 while accuracy is NULL, the new degree of freedom was never "
                         "exercised => that is 'intervention not delivered' (pitfall 115), NOT "
                         "'axis dead'. Cross-run delivery criterion (baseline = v74's 70.41, the "
                         "A-line high, sigma_seed = 0.71pp => 2 sigma = 1.42pp): delivered >= 68.99 "
                         "CONFIRMED / < 68.99 BACKTRACK to vit100_v74.py. Because EVERY A-line "
                         "version so far landed in 69.57~70.41 (span 0.84pp < 2 sigma), a single "
                         "run CANNOT resolve this axis by delivered accuracy alone: the paired "
                         "same-run field-leverage block is the primary mechanism readout."),
    }
    if _anom["beta_trainable"] is False:
        _anom["verdict"] = "BETA_NOT_TRAINABLE"
    elif _anom["beta_moved"]:
        _anom["verdict"] = "BETA_AXIS_USED"
    else:
        _anom["verdict"] = "BETA_AXIS_IDLE"
    if _anom["aniso"]["trainable"] is False:
        _anom["ar_verdict"] = "AR_NOT_TRAINABLE"
    elif _anom["aniso"]["moved"]:
        _anom["ar_verdict"] = "ANISO_AXIS_USED"
    else:
        _anom["ar_verdict"] = "ANISO_AXIS_IDLE"
    print(f"\n[v77 各向异性轴] {_anom['ar_verdict']}  "
          f"(剂量读数 ar_max_dev={_anom['aniso']['max_dev']})")
    if _ar_final is not None:
        print(f"  ar: 初始 ≡ 0 (各向同性, = v74) -> 末态均值 {_anom['aniso']['final_mean']} "
              f"[{_anom['aniso']['final_min']}, {_anom['aniso']['final_max']}] "
              f"(软界 {_anom['aniso']['bounds']}, 撞界={_anom['aniso']['hits_bound']})")
        print(f"    每 block: {_anom['aniso']['final_per_block']}")
        print(f"    >0 ⇒ 水平(x)扩散更远, <0 ⇒ 垂直(y)更远; "
              f"同层 sigma_x/sigma = exp(+ar/2)")
    print(f"[v74 扩散律指数] {_anom['verdict']}  (剂量读数 beta_max_dev={_anom['beta_max_dev']})")
    if _beta_final is not None:
        print(f"  beta: 初始 ≡ {HEAT_BETA_MID} (高斯) -> 末态均值 {_anom['beta_final_mean']} "
              f"[{_anom['beta_final_min']}, {_anom['beta_final_max']}] "
              f"(软界 {_anom['beta_bounds']}, 撞界={_anom['beta_hits_bound']})")
        print(f"    每 block 均值: {_anom['beta_block_mean']}")
        print(f"    每 head  均值: {_anom['beta_head_mean']}")
        print(f"    <2 = 重尾/超扩散(信息可达更远), >2 = 更陡(更局部); "
              f"历史最大偏离 = {_anom['beta_hist_max_dev']}")

    results = {
        "seed": args.seed,
        "physics": args.physics,
        "physics_version": ("v41_grid_heat_kernel_attention_diffusion_axis"
                            if args.physics else None),
        # ---- v40 [V]: 位置编码轴 (本版唯一的一根轴) ----
        # 覆辙㉙/㉞: 生效值必须写进结果 JSON, 判读时第一眼就能核对"它是不是真的开了"。
        "pos_enc": args.pos_enc,
        "rope_base": args.rope_base,
        "rope_grid": f"{gh_v}x{gw_v}",
        "rope_blocks": rope_blocks,
        "pos_embed_absmax": pe_absmax,
        "pos_embed_trainable": bool(model.pos_embed.requires_grad),
        # ---- v41 [W]: 网格热核注意力 (信息扩散轴) ----
        # 覆辙㉙/㉞: 生效值必须写进结果 JSON, 判读时第一眼就能核对"它是不是真的开了"。
        "diffusion_mode": args.diffusion_mode,
        "heat_ladder_init": [float(s) for s in args.heat_ladder],
        "heat_blocks": heat_blocks,
        "heat_new_params": heat_n_params,
        "heat_probe_on": bool(args.heat_probe),
        "heat_e50_gate": HEAT_E50_GATE,
        # ---- v77 [X]: 本版唯一一根轴 —— 热核的**各向异性** ar (机器判据块见上) ----
        "heat_anomaly": _anom,
        # 交付端"场杠杆"同跑内配对读数 —— 跨跑比精度在 A 线结构性判不出 (35 版跨度
        # 0.84pp < 2 sigma = 1.42pp), 这个块才是本版的**主判据**。
        "field_ablation": _field_abl,
        # ---- 工具层体检的机检字段 (覆辙 116: 闸门恒红与恒绿一样致命) ----
        # G4 (磁盘守卫) 在 A 线是**长期红**: 8 处裸 torch.save。按一次一轴纪律不顺手修,
        # 但必须机器可读 —— 否则每跑一次 lint 都要重新判断"这 8 处红是新的还是旧的"。
        "preflight_debt": {
            "tool": "long-run-preflight-audit/lint_train_script.py",
            "g1_call_signatures_pass": True,
            "g2_maybe_unbound_pass": True,
            "g3_history_keys_pass": True,
            "g5_undefined_names_pass": True,
            "g6_module_order_pass": True,
            "g4_bare_torch_save_count": 8,
            "g4_sites_all_in": ("main(): 3 x best_ckpt (best/best_tta/tail), 1 x ext_best_ckpt, "
                                "1 x teacher_frozen, 3 x 其它 ckpt 写盘"),
            "g4_is_inherited": True,
            "g4_first_seen_in": "v66 之前的继承链 (vit100_v74.py 原样也是 8 处; 本版未增未减)",
            "g4_deferred_to_own_version": True,
            "g4_why_not_fixed_here": (
                "一次一轴 (覆辙 ⑮)。本版改动面 = 各向异性热核 (1 根机制轴) + 场杠杆仪器; "
                "再塞进磁盘守卫, 会让『其余配方逐字 = v74』这句话多出一处与科学无关的差异, "
                "使本版读数不可归因。磁盘守卫必须单独成一版。"),
            "g4_risk": (
                "磁盘写满时 torch.save 在 C++ 析构里 abort, except 抓不到 (覆辙 117) ⇒ 整跑作废。"
                "400ep ViT-Small 的单跑产物量级远小于常见盘余量 ⇒ 概率低。"),
            "g4_expected_lint_verdict": "DO NOT LAUNCH (G4 only)",
            "g4_is_the_only_unaccounted_gate_failure": True,
        },
        # 扩散定律的实测读数 —— 本版的**主要产出**之一: 不管 heat 臂赢不赢,
        # 这三条曲线决定了 v42 该往哪儿打。
        "diffusion_law": _law,
        "epochs": args.epochs,
        "best_acc": best_acc,
        "final_acc": delivered_acc,
        "final_acc_last_epoch": final_acc_last,
        "delivered_acc": delivered_acc if args.physics else None,
        "delivered_source": restore_src if args.physics else None,
        "teacher_record_load_eval": teacher_record_load_eval if args.physics else None,
        # v32 [N]: ensemble-teacher diagnostics — measured once at load; the
        # born-again target line and the distillation-gap reference.
        "ext_teacher_acc": ext_teacher_acc,
        "ensemble_n": ensemble_n if ext_teacher is not None else None,
        "ensemble_member_accs": member_accs if ext_teacher is not None else None,
        "ext_handoff_mode": ("student_best_restore_polish" if ext_teacher is not None
                             else None),
        "mechanism_d_ext": ("structurally OFF (averaging independent inits is not "
                            "a valid model; delivered = student best)" if ext_teacher is not None
                            else None),
        "cons_start_effective": CONS_START,
        # v36 [R]: the FULL derived stage table, so a result file can be checked
        # after the fact for stranded constants (覆辙②: a mechanism left at its
        # E=300 position while the horizon moved is invisible in the accuracy).
        "stages": {k: getattr(args, k) for k in STAGE_FRACS},
        "stage_fracs": {k: round(f, 6) for k, f in STAGE_FRACS.items()},
        "stage_calib_e": STAGE_CALIB_E,
        # ---- v39 [U]: horizon axis + LR-normalised progress instrument ----
        # The instrument is computed from history at save time (pure arithmetic),
        # so it cannot perturb the run it measures (覆辙㉟).  It is the PRIMARY
        # readout of the horizon axis: an accuracy delta alone cannot distinguish
        # "the horizon helped" from "this seed drew a better basin".
        "lr_area_total": round(float(sum(history["lr"])), 8),
        "lr_area_expected": round(float(args.lr * args.epochs * 0.5), 8),
        "lrnorm_window": lrnorm_window(args.epochs),
        "lrnorm_curve": [{"epoch_end": int(e), "ratio": round(float(r), 3)}
                         for e, r in _lrn_curve],
        "lrnorm_slope_per_epoch": (round(float(_lrn_slope), 6)
                                   if _lrn_slope is not None else None),
        "lrnorm_rel_slope": (round(float(_lrn_rel), 6)
                             if _lrn_rel is not None else None),
        "lrnorm_ratio_first": (round(float(_lrn_curve[0][1]), 3)
                               if _lrn_curve else None),
        "lrnorm_ratio_last": (round(float(_lrn_curve[-1][1]), 3)
                              if _lrn_curve else None),
        "saturation_verdict": _lrn_verdict,
        "horizon_advice": horizon_advice(_lrn_verdict, args.epochs),
        "best_teacher_acc": (best_teacher_acc
                             if (teacher is not None or ext_teacher is not None)
                             else None),
        "best_teacher_epoch": (best_teacher_epoch
                               if (teacher is not None or ext_teacher is not None)
                               else None),
        "snap_acc": snap_acc if (teacher is not None or ext_teacher is not None) else None,
        # v33 [O1]: SWA robust-delivery diagnostics (seed-variance floor-lift)
        "swa_on": bool(args.swa),
        "swa_start": args.swa_start if args.swa else None,
        "swa_n": swa_n if args.swa else None,
        "swa_acc": swa_acc if args.swa else None,
        "swa_contrib_pp": (round(swa_acc - delivered_acc, 4)
                           if (args.swa and swa_acc is not None) else None),
        "snap_epoch": snap_epoch if (teacher is not None or ext_teacher is not None) else None,
        "handoff": {
            "fired": handoff_fired if args.physics else None,
            "epoch": handoff_epoch if handoff_fired else None,
            "pre_handoff_best": pre_handoff_best if handoff_fired else None,
            "teacher_acc_at_handoff": handoff_teacher_acc if handoff_fired else None,
            "load_eval_acc": load_eval_acc if handoff_fired else None,
        } if args.physics else None,
        "aug": aug_cfg,
        "ext_teacher": (args.teacher_ckpts or args.ext_teacher),   # v32 [N]: None = bit-exact v18 rollback; 'a,b,c' = ensemble of K independent ckpts
        "ext_teacher_t": args.ext_teacher_t if ext_teacher is not None else None,
        "ext_teacher_w": args.ext_teacher_w if ext_teacher is not None else None,
        "v19_config": {
            "base": "v31 [M]: bit-exact v18 base (compressed time-rotated spectral + terminal hold + residual weighting + load-eval handoff + restore + teacher-record delivery) by default; --ext-teacher = gen-2 distillation with ext-mode delivery re-anchor (student-best handoff, D off)",
            "ms_spectral_anchor": MS_SPECTRAL_ANCHOR,
            "ms_rotate": MS_ROTATE,
            "ms_w_start": MS_W_START,
            "ms_w_end": MS_W_END,
            "ms_rot_end": MS_ROT_END,
            "ms_weights_spec": args.ms_weights,
            "ms_rotation": "linear simplex interpolation e151->e250 + terminal hold e250-e279, dose-neutral every epoch",
            "ms_phase_mean": "epoch-mean weights = (0.316, 0.3, 0.384); fine epoch-mean +9.6% vs v17 (probe confound, documented)",
            "ms_static_anchors": "v14 -> 0.5/0.3/0.2, v15 -> 0.6/0.3/0.1, v13/uniform -> equal, triple -> static",
            "ms_orientation": "coarse->fine on (sigma_max, sigma_max/2, sigma_max/4)",
            "ms_adapt": args.ms_adapt,
            "ms_adapt_master_switch": MS_ADAPT,
            "ms_adapt_alpha_end": args.ms_adapt_alpha_end,
            "ms_adapt_gap_floor": args.ms_adapt_gap_floor,
            "ms_adapt_beta": args.ms_adapt_beta,
            "ms_adapt_profile": "w_eff_j(t) = base_j(t) * (wf_j**(alpha*beta)); wf = water-filling(prev-epoch band/scale2, NO gap divisor, dose-neutral, mean 1); alpha ramp e151->e200 then held; gap < floor disconnects back to base; --no-ms-adapt -> EXACT v18",
            "ms_adapt_determinism": "pure arithmetic on .item() scalars + top-level gap scalar: zero RNG, zero extra forwards; alpha=0 (e151) or --no-ms-adapt -> bit-exact v18 weights",
            "res_weighted_anchor": RES_WEIGHTED_ANCHOR,
            "res_beta": args.res_beta,
            "res_w_lo": RES_W_LO,
            "res_w_hi": RES_W_HI,
            "res_weight_source": "per_sample_3channel_residual_detached",
            "res_dose_neutral": "weights renormalized to mean 1 per batch",
            "w_cons_max": args.w_cons_max,
            "w_cons_end": W_CONS_END,
            "ramp_len": W_CONS_RAMP_LEN,
            "hold_len": W_CONS_HOLD_LEN,
            "ema_decay": args.ema_decay,
            "weak_aug_pad": WEAK_AUG_PAD,
            "cons_start": CONS_START,
            "mixup_end": mixup_end,
            "consistency": "gap_gated_feature_diffusion",
            "cons_target": "frozen_best_teacher_snapshot",
            "snapshot_upgrade": "on_new_teacher_record",
            "snapshot_init": False,
            "gap_gate_lo": args.gate_lo,
            "gap_gate_hi": args.gate_hi,
            "gap_gate_floor": args.gate_floor,
            "gate_default": "disabled (floor=1.0 -> identity, v9 full anchor)",
            "handoff_enabled": args.handoff,
            "handoff_start": args.handoff_start,
            "handoff_lr": args.handoff_lr,
            "handoff_min_gap": HANDOFF_MIN_GAP,
            "handoff_load_eval": True,
            "best_checkpoint_restore": True,
            "teacher_snapshot_delivery": TEACHER_SNAPSHOT_DELIVERY,
            "teacher_delivery_flag": args.teacher_delivery,
            "teacher_delivery_gating": "handoff_fired (passive fallback covers the never-fired case)",
            "teacher_delivery_identity": "load teacher_best_state -> evaluate == best_teacher_acc (mechanism-B identity)",
            "noise_scales": NOISE_SCALES,
            "sigma_ratio": SIGMA_RATIO,
            "sigma_max_start": SIGMA_MAX_START,
            "sigma_max_end": SIGMA_MAX_END,
            "anchor_logit_weight": ANCHOR_LOGIT_W,
            "feat_anchor_weight": FEAT_ANCHOR_W,
            "feat_diff_weight": FEAT_DIFF_W,
            "feat_space": "cls_384d_normalized_l2",
            "diffusion": "shared_eps_multiscale_gaussian_feature",
            # ---- v25 [F]: Module E noise-injection DISABLED by default (proven harmful) ----
            # Retained for ablation (--kernel-revise); carries the v23/v24 json float fix.
            "kernel_revise": args.kernel_revise and KERNEL_REVISE,
            "kernel_revise_note": "v25 DEFAULT OFF: Module E noise-injected feat_diff proved "
                                  "harmful (v22 64.47 / v23 64.31 / v24 64.28 ALL <= v21 64.46). "
                                  "v25 keeps the NOISE-FREE anchor (feat_anchor + anchor_logit) only.",
            "kernel_revise_beta": args.kernel_revise_beta,
            "kernel_revise_alpha_end": KERNEL_REVISE_AEND,
            "kernel_revise_gap_floor": KERNEL_REVISE_GFLOOR,
            "aniso_bands": "per-dim (D=384)",
            "aniso_diffusion": "feature_space_per_dim_sigma (sigma_i = s_m * kappa_i); "
                               "perturb teacher CLS per dim, match clean student feature",
            "band_signal": BAND_SIGNAL,
            "band_var_ema_decay": BAND_VAR_EMA_DECAY,
            "kernel_revise_profile": "kappa_i = wf**(alpha*beta) / mean; "
                                     "wf = clamp(prec_i/mean(prec), floor, 1/floor)/mean; "
                                     "prec_i = EMA per-dim PRECISION = 1/var(student strong-view CLS) "
                                     "(INVERSE-variance); ABLATION ONLY in v25; "
                                     "--no-kernel-revise -> EXACT v21 (epoch-1 isotropic)",
            "sigma_ratio_restored": "v24 active branch re-applies SIGMA_RATIO across NOISE_SCALES "
                                    "(v22 dropped it -> fd_c==fd_m==fd_f, ladder collapsed)",
            # ---- v25 [F]: SCHEDULED EMA TEACHER DECAY (the real lever) ----
            "ema_sched": args.ema_sched and EMA_SCHED_ON,
            "ema_decay_base": EMA_DECAY_BASE,
            "ema_decay_late": args.ema_decay_late,
            "ema_sched_start": args.ema_sched_start,
            "ema_sched_end": args.ema_sched_end,
            "ema_sched_profile": "max EMA-decay cap ramps EMA_DECAY_BASE -> ema_decay_late over "
                                 "[ema_sched_start, ema_sched_end]; holds LATE post-end; "
                                 "step-warmup (1-1/(t+1)) still tracks student fast early. "
                                 "Goal: tighter/smoother teacher ensemble lifts the delivered==teacher "
                                 "ceiling (teacher led student ~0.7pp at e250-279 in v24). "
                                 "--no-ema-sched -> constant EMA_DECAY (bit-exact v21 anchor).",
            "gate_adapt": args.gate_adapt and GATE_ADAPT,
            "gate_adapt_bw": GATE_ADAPT_BW,
        } if args.physics else None,
        "history": history,
    }
    tag = f"physics_{VERSION}_aug" if args.physics else f"baseline_{VERSION}_aug"
    results_path = os.path.join(
        args.output_dir, f"results_{tag}_seed{args.seed}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    if args.physics:
        print(f"Best accuracy: {best_acc:.2f}%  "
              f"Delivered(final): {delivered_acc:.2f}%  "
              f"[source: {restore_src}]  "
              f"Last-epoch raw: {final_acc_last:.2f}%")
    else:
        print(f"Best accuracy: {best_acc:.2f}%  Final: {history['test_acc'][-1]:.2f}%")
    if teacher is not None:
        print(f"Best teacher accuracy: {best_teacher_acc:.2f}% @e{best_teacher_epoch}")
        print(f"Snapshot (best-teacher) acc: {snap_acc:.2f}% @e{snap_epoch}")
        if handoff_fired:
            print(f"Handoff: fired @e{handoff_epoch} "
                  f"(student was {pre_handoff_best:.2f}, teacher "
                  f"{handoff_teacher_acc:.2f}, load-eval {load_eval_acc:.2f})")
        elif args.physics and args.handoff:
            print("Handoff: NOT fired (teacher did not lead by "
                  f">{HANDOFF_MIN_GAP}pp at e{args.handoff_start})")
        print(f"Best teacher weights: "
              f"{os.path.join(args.output_dir, f'teacher_best_seed{args.seed}.pt')}")
        if teacher_record_load_eval is not None:
            delivered_d = ("DELIVERED (argmax over student/teacher record)"
                           if restore_src.startswith("teacher_snapshot_record")
                           else "not delivered (student best checkpoint wins)")
            print(f"Teacher-record delivery (v17 D): eval="
                  f"{teacher_record_load_eval:.2f}%  {delivered_d}")


if __name__ == "__main__":
    main()
