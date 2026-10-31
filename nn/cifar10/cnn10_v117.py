"""
Physics-Guided ResNet-50 Training - v110
================================================================================

## v110 = cnn10_v103.py 的训练数学 + 唯一一根轴：
##     **Sobolev 度量 SAM（以场自己的 Bessel 势度量替换 SAM 的欧氏信任域）**
##
##   立项依据（v109 实跑裁定，2026-10-23：用户口头读数"效果没怎么变化"= NULL）：
##     (1) 三个作用面全部试完且全部 NULL/负：特征注入 v103(参考) / v104(-0.06) /
##         v105(97.40) / 优化器标量 v106(97.36) / v107(97.42) / v108(97.28) /
##         数据侧 v109(NULL)。六版读数全落在 sigma_diff=0.169 内或更差。
##     (2) ★ 六版共有的**结构性缺陷**（本轮诊断的核心）：控制信号 q = smooth 自己
##         就在损失里（`loss = cls + w_phys*smooth + PINN_W*pinn_reg + reg_loss`），
##         w_phys 峰值 0.15 全程把它往下压 ⇒ q 单调漂移 ⇒ 任何"相对运行均值的 z"
##         都退化成常数偏置。铁证：v107 scale_mean=1.0998（抬高 10% 却零增益）、
##         v108 mu_std=0.011 + mu_mean=0.9186（单向偏置，判词 MOM_FLAT）、
##         v109 零变化。⇒ **信号本身无可用高频**，换作用面救不了信号，
##         再在这条信号上做第四个点没有意义（P0.5 预注册的"根本性复盘"触发）。
##     (3) 第二个根因：v107 早期 -0.45pp、v108 早期 -1.60pp 证明这些调制**有力**，
##         但力都作用在早期高 lr 相位上 ∩ 末段地板 lr 区对剂量不敏感
##         ⇒ 任何"控制 schedule 的额外时变项" = 给一份已调好的 schedule 加扰动。
##     (4) ⇒ 本轴两个办法都避开：**不改任何 schedule 标量**（lr/momentum/mixup 强度
##         一个都不动），改的是**更新几何**：SAM 找平坦极小
##         eps = rho*g/||g||，在 w+eps 处取梯度更新 w。SAM 是 ISR armed 的、干净
##         test acc 提升 +0.3~1.0pp 的少数稳健机制之一 ⇒ 预期效应 **远大于**
##         sigma_diff=0.169（唯一能让我们量到东西的选项）。
##     (5) 物理嵌入点（本版的轴本体）：把欧氏度量 ||.||_2 换成 Bessel 势诱导的
##         Sobolev 度量：K(k) = (1 + (lam/nu)|k|^2)^{-s}，它正是 (I - ell^2 Laplacian)^{-s}
##         的谱表示（Bessel 势 = H^{-s} 的对偶核）。⇒ 在这个度量下"最陡上升"只取
##         梯度的**场许可的平滑分量**： trader 的精细方向（纹理级、脆的方向）被排除
##         在信任域之外。"平坦性"不再是全平面的平坦，而是**沿扩散场所关心的方向**
##         的平坦。lam/nu 取场自己的 running 统计（相关长度平方的倒数，pooled），
##         不是外来的超参 ⇒ 零新参数、零新方向的 RNG。
##     (6) ★ 技术前提（本版花 GPU 前必须先证的坑 —— 教训 331）：现役 `_dft_axis`
##         只在**偶数** N 上等于真实 DFT bin 频率；卷积核绝大多数是 **3x3（奇数）**，
##         奇数下 (-pi + 2*pi*i/N) 与 fftshift 后的真值错半格（N=3：[-pi,-pi/3,pi/3]
##         vs 真值 [-2pi/3, 0, 2pi/3]）⇒ 常数核被衰减到 0.313（-68%），高度沦为高通。
##         ⇒ 本轴自带 `_v110_freq_axis`（fftfreq + fftshift，奇偶通用），并把
##         "常数核必须原样通过"写进自测（case5）——它能精确咬住这个 bug。
##
##   代价与纯度（必须 disclose）：每步 2 次 fwd+bwd ⇒ 训练墙钟 ~1.8-2x；第二轮 forward
##     会多抽一次 StochasticDepth 掩码 ⇒ **RNG 流与 v103 不再逐位可比**（配对尺度仍用
##     sigma_diff=0.169，它是跨版本 RMS，含此类差异）。
##
##   判据预注册：五档 SAM_SOBOLEV_DELIVERED（applied_frac>=0.9 且
##     ||K*g||/||g|| 均值 ∈ (0.05, 0.95)）/ SAM_NOT_APPLIED / SAM_METRIC_DEGENERATE /
##     SAM_SELFTEST_FAILED / SAM_READOUT_ERROR。
##     归因预注册：若本版 **涨过 sigma_diff**，下一版必须是**裸 SAM（K≡1，同一 rho）**
##     作对照跑，才能把"SAM 的功劳"与"Sobolev 度量的功劳"拆开；不许直接宣称
##     是本轴的度量赢了。
## v96 = cnn10_v51.py **逐字回退** + 唯一一根轴：**DFT 频率网格修正**
##
## v97 = v96 **逐字不动**（训练配方 / 物理场 / 判据 / randomness 全同）
##     + 唯一一根轴：**磁盘写盘三防线**（守卫，不是机制）
##
##   起因（本行所记是真事，不是假想的风险）：v96 实跑到 ep129（180 的 72%）时，
##   训练机磁盘写满，ENS 存 ckpt 的 torch.save 中途 OSError [Errno 28]；
##   该异常在 torch serialization 内部被吞后 return，离开 with 时二次抛
##   RuntimeError(unexpected pos)，且是从 PyTorchStreamWriter **析构函数**抛的
##   C++ c10::Error ⇒ Python try/except 拦不住 ⇒ core dump ⇒ 129 epoch 归零、无 JSON。
##
##   ★ 本版最关键的一条认识：**防线必须在写之前** —— C++ abort 不受 Python 异常
##     机制管辖，事后的 try/except 救不了。所以三道防线的可靠性排序是
##       (1) 写前空间预检 => 不够就不写（真防线）
##       (2) 原子写 .tmp + os.replace => 盘上不留半截文件
##       (3) 异常兜底 + 计数进 JSON + 主 JSON 失败时整份打 stdout（双通道）
##
##   另外修掉同一处的一个静默 bug：save_ckpt 淘汰旧 ckpt 时 `except OSError: pass`
##   ⇒ 删失败被吞掉 ⇒ 被淘汰的 ckpt 变成幽灵文件继续占盘。现在计数并打印。
##
##   ⚠ 本版**不加** resume（用户 2026-10-16 决定）。磁盘三防线只保证"不会因为
##     写盘失败而归零"，不保证"中断后能续跑"。
##
##   ⚠ 注意产物文件名：best 权重仍是硬编码的 `best_physics_resnet_v51.pth` /
##     `best_tta_resnet_v51.pth`（继承自 v51，本版不改以免改变对照语义）。
##     重跑 v97 会**覆盖**上一次跑到一半留下的那两份权重 —— 如果要留作证据，
##     请先手动改名备份。

## v100 = v99 **逐字不动**（训练数学零改动）+ 唯一一根轴：**判据/仪器修正**
##     ------------------------------------------------------------------
##     本版不改任何训练数学（无参数、无调度、无 RNG 抽取、无损失项、无物理场常量变化）
##     ⇒ train()/test_tta() 与整条物理链和 v99 逐位相同。改的是**尺子**，且尺子错在五处：
##
##     (1) σ_diff 0.108 只用了 **一对**（v45/v46）。归档定案里有**四对**同口径数据：
##           |Δdelivered| = v45/v46 0.120 · v48/v49 0.040 · v44/v54 0.140 · v51/v55 0.280
##           ⇒ RMS = sqrt((0.120^2+0.040^2+0.140^2+0.280^2)/4) = sqrt(0.0285) = **0.169pp**
##             （⚠ 中间值是 0.0285：sqrt(0.0117)=0.108 **恰是旧的被低估值**，
##              写错中间值会把人引回旧尺子 —— 覆辙 297，注释里的演算必须自己也算对）
##         ⇒ 旧常数低估噪声底 **1.565 倍**，所有旧判据线集体偏松。
##         ⇒ 教训：**一对样本估不出方差**（覆辙 113 的测量版）。
##         ⚠ v99 自己的 L3760 注释**已经写着** 0.169/0.338，而常量还是 0.108
##           ⇒ 文件内部自相矛盾且没人看（覆辙 303 的又一形态）。
##     (2) 四条判据线全部改为**由常量派生**（v99 的 `BACKTRACK_REDLINE = 97.27` 与
##         `if delivered >= 97.61:` 仍是硬编码字面量，且 97.27 用的 σ=0.06 与
##         PLATEAU_SIGMA 不是同一把尺）。
##     (3) lr_tail 剂量门**拆成两门**：v99 用一把 `dose_delivered` 判两个通道，
##         而 v99 自己的数据说明这是错的 —— 剂量 ep121-156 真跑 36 轮、平台窗
##         ep135-156 完整落在剂量窗内，但交付选择点是 ep120（TTA 在 ep120/ep126
##         并列 97.44，first-wins 选中前者）⇒ 交付通道的"盲"是**选点规则的产物**。
##     (4) argmax 并列升级为**机器字段**（v99 的并列是读曲线才发现的）。
##     (5) 修掉被自己的数据证伪的 `expected_delivered_vs_v98: "identical (97.42)"`
##         （v99 实跑 97.44）⇒ 以后只写"统计不可区分"，不写"逐位相同"。
##
##     ⚠ 诚实标注：σ 修正后平台 PASS 线 = 97.4625 + 0.338 = **97.80**，
##       **高于**本线历史最好平台 97.4625 ⇒ 单跑平台判词在算术上预先决定为 NULL。
##       这是 v47/v48 之错的**镜像**（那次线太低、这次线太高）⇒ 本版的交付物是
##       "尺子修准了"，不是一次精度主张。字段：`plateau_pass_line_reachable`。

## v101 = v100 **逐字不动**的部分 + 唯一一根轴：**各向异性 Matérn 核**（v90 轴重开）
##     ------------------------------------------------------------------
##     训练数学：只有 MaternDiffusion.forward 的核从径向 |k|^2 = kx^2+ky^2 改为
##       k_sq_g = kx^2 + a_g*ky^2，a_g = 1.0 + ANISO_RANGE*tanh(aniso_param_g) ∈ (0, 2]
##       （phys2 64 组 + phys3 8 组；FixedDiffusion 无方向自由度，不动）。
##     归因纯度：aniso_param = nn.Parameter(torch.zeros(groups))
##       · tanh(0)=0 ⇒ a_g 精确 1.0 ⇒ 第 0 步核与 v100 逐位相同（1.0*t 精确返回 t）
##       · torch.zeros 不耗 RNG ⇒ 初始化序列与 v100 逐位相同
##       · 无交叉项 ⇒ flip 等变（两轴）⇒ flip-TTA 交付通道受保护
##     v90 的判词 NULL + ANISO_SELFTEST_FAILED 两条红灯已裁定为**自测假红**：
##       · f32 tanh(10)=1.0 精确饱和 ⇒ a=2.0 精确（旧自测期望"严格小于 2"⇒ 假红）
##       · 旧 DFT 网格上连各向同性核也不等变（5.7e-2@8x8）⇒ 旧期望不成立
##     ⇒ v90 的机制读数有效（max|a-1|=0.283 ⇒ 优化器真的推动了）；精度 NULL 是当时的
##       尺子问题（σ=0.108 且网格错一档），不是"轴无效"。
##     ⚠ 这是 **recipe 域改动** ⇒ 只有跨跑判据，单跑 NULL 不得升级为"轴无效"（需 3 seed）。

## v102 = v101 + 纯仪器修复（**训练数学逐位相同，无新轴**）
##     ------------------------------------------------------------------
##     起因（v101 实跑，2026-10-18）：开机自测 `_aniso_selftest(device)` 在 CUDA 上
##       崩了，日志却照样打印"7/7 ALL PASS"。两个 bug 叠出一个假绿（覆铣 #312）：
##       (1) device 半吊子：自测收了 device 参数却只用来建**输入张量**，
##           7 个被构造的 nn.Module 从不 .to(dev) ⇒ case2 起
##           RuntimeError: Input type (cuda) and weight type (CPU)，15 条只跑 7 条。
##           修法：每个模块构造过 `_on_dev(m, dev)`；**构造仍在 CPU**（在 CUDA 上
##           直接构造会消耗 CUDA RNG，而 _det_probe_snapshot_rng 只快照 CPU RNG），
##           .to(dev) 是纯拷贝不耗 RNG ⇒ RNG 消耗量与 v101 逐位相同。
##       (2) all_pass 不防崩：旧公式分母是"崩溃前已跑条数" ⇒ 崩了恒真绿。
##           修法：rep 增 completed 标志 + n_expected（预期总条数常量
##           ANISO_SELFTEST_CASES=15 / DFT_SELFTEST_CASES=8），all_pass 改
##           `completed and n_pass==n_total`；判词增 selftest-error 档。
##           _dft_grid_selftest（v96 仪器）同款公式一并修（它的旧公式在 try 内末尾、
##           崩了会 early-return 侥幸安全，但分母语义同样错 ⇒ 统一）。
##     v101 实跑裁定：provenance 内容级可信（横幅 sha1 精确匹配）；163 轮早停；
##       RNG 还原 case 在机器上通过 ⇒ 训练轨迹未被污染；**精度 NULL**
##       （delivered 97.48，全部差值落在 σ_diff=0.169 内 ⇒ DO_NOT_BACKTRACK，
##        不得升级"轴无效"）；**机制 ANISO_DELIVERED**（phys2 max|a-1|=0.474、
##       phys3=0.180、162/163 epochs 有推动）——但那是带着坏尺子算出来的。
##     ⚠ 本版是**仪器域**改动 ⇒ 无跨跑判据；验收看"自测 15/15 且 completed、
##       判词七档可达、训练数学与 v101 逐位相同"这三件事，不主张任何精度结论。
## v103 = cnn10_v102.py **逐字不动**的训练数学 + 唯一一根轴：**换 seed（1234 -> 42）**
##     ------------------------------------------------------------------
##     立项依据（全部来自归档定案与 v102 实跑裁定，不是本轮猜测）：
##       (1) 预注册协议明写"单跑 NULL 不得升级为'轴无效'，需 3 seed"，而 v90/v101/v102
##           三次各向异性观测**全是 seed 1234** ⇒ 按同 seed 噪声本质是**一个**观测；
##           本跑是当前代码栈上的第 2 个 seed（协议 mandated）。
##       (2) v102 实跑 provenance **内容级破损**（B 线第一次）：实跑文件名 loss.mmm1.py、
##           self_sha1=ad097731... != 交付片 c12256ab... ⇒ 用交付片重跑，开机横幅
##           sha1 必须等于交付值，断链在本版修复。
##       (3) v102 实跑裁定 = 精度 NULL（delivered 97.42，−0.83σ_diff）+ 机制
##           ANISO_DELIVERED（尺子已真绿：15/15 且 completed=True）⇒ 轴值得在独立
##           seed 上确认，且这是唯一能推进协议的跑法（一跑 7.5h）。
##     纯度：唯一改动是 SEED 行的整数。seed 的全部消费者（random/np/torch.manual_seed/
##       torch.cuda.manual_seed_all、worker_init_fn 的 SEED+worker_id、CKPT_DIR/OUT_DIR/
##       JSON 名）都从 SEED 派生 ⇒ 换 seed 完备；遗留的 1234 字面只有历史记录与三个
##       **独立探针 seed**（12345 / 2026096 / 2026092，跨版可比必须固定不动）。
##     ⚠ v43 也是 seed 42 ⇒ 本跑与 v43 同 seed 不同代码（z 不可合并，覆铣 106）；
##       与 v102 同代码不同 seed ⇒ 构成**第一对同代码跨 seed 配对**（v103_seed_pair）。
##     判据预注册：预期与 v102 的 97.42 落在 σ_diff=0.169 内；本跑目的是**量当前
##       代码的 seed 方差**，不主张精度结论（单跑 NULL 不得升级，需 3 seed）。

================================================================================
### 〇 为什么逐字回退到 v51（v95 的裁定）

  v95 delivered = **97.33**，vs v51 的 97.56 = **−0.23pp = −2.13σ_diff**（σ=0.108）。
  落在 NULL 区间 (97.27, 97.68) ⇒ **未触红线，不回退**。机制侧拆开看：
    · 轴**送达**：配对通道秩 64 vs 8（phys2/phys3 都 rank_ratio 8.0）✓
    · 谱**没塌**：std_lam .0429/.0063、`n_lam_at_lower_bound = 0` ⇒ 优化器没拒绝 ✓
    · 但精度是 **NULL 偏负** ⇒ 没有正面证据支持把 groups=64 留在下一版。
  ⇒ 按「回退优先 + 归因纯度」，逐字回到 cnn10_v51.py（sha1 3516a363…），
     在其上只叠一根新轴。groups 轴的结论单独归档，不带进本版。

### 一、本版唯一一根轴：DFT 频率网格（**修 bug**，不是加机制）

  生产路径（MaternDiffusion / FixedDiffusion 各 2 处）从 v9 起一直是：
      ky = torch.linspace(-math.pi, math.pi, H)     # step = 2*pi/(N-1)，**含双端点**
  而 fftshift 之后第 i 个 bin 的真实角频率是 -pi + 2*pi*i/N ⇒ step = **2*pi/N**，半开区间。

  后果一（覆辙 273）：整条频率轴"错一档"，除两端外每个 bin 拿到的 |k| 都偏大；
          且 -pi 与 +pi 是**同一个** bin 却被放了两次 ⇒ Nyquist 重复。
  后果二（覆辙 253）：DC（k=0）根本**不在网格上** ⇒ 常数场不能原样通过。

  实测数量级（v95 的 physgroup 自测；v93 本机同款算子标定过一次）：
      flip 等变   6.379e-02  →  2.384e-07
      DC 增益偏离 2.918e-01  →  0

  ★ 这**直接**打在交付口径上：`delivered = max-of-N TTA`，而 TTA 就是
      outputs = (net(x) + net(flip(x))) / 2      ← **唯一依赖 flip 等变性**
    本文件 L388-391 的历史记录：flip 增益 ep<=120 平均 +0.333 → ep>120 平均 +0.199。
    v95 实测 TTA 增益 ≈ +0.21pp ⇒ 正好卡在衰减后的低档 ⇒ 修好等变有一档可回升。

  ★★ 覆辙 275：torch.linspace **没有** endpoint 参数（那是 numpy.linspace 的 API）。
      v92 写 `torch.linspace(-pi, pi, N, endpoint=False)` ⇒ 训练机第 0 ep 前就崩。
      ⇒ 本版用 `arange(N) * 2*pi/N - pi` 表达同一数学对象。

  ⚠ 这是 **recipe 域改动**（|k| 步长变了 ⇒ 核的实际衰减率变了 ⇒ 第 0 步就不同）
    ⇒ 只有**跨跑**判据，单跑 NULL 不得升级为"轴无效"（需 3 seed）。
  ⚠ 这条修正在 B 线**从未被真正判定过**：v92 发车即崩 / v93 实跑 97.52 但 JSON
    崩丢（覆辙 279）/ v94 换成 KITI 轴判死（覆辙 281）。本版是它第一次拿到干净判据。

### 二、判据（用户选：2σ 边距）

  基线 v51 = 97.56，σ_diff = 0.108（**实测**：v45/v46 一对零改动跑）
      PASS      >= 97.78     (97.56 + 2σ)
      NULL      (97.34, 97.78)
      BACKTRACK <= 97.34     (97.56 − 2σ) ⇒ 逐字回退 cnn10_v51.py，不做分析
  依据：v48 的 noise_floor/note 明写「v47/v48 的 PASS 边距 0.10pp 只有 0.93σ
        ⇒ 那两个 NULL 是算术上被预先决定的。以后 B 线 PASS 边距不得小于 2σ_diff」。

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
# [v103] 本版唯一一根改动轴：换 seed（1234 -> 42）。
#   v102 = seed 1234（delivered 97.42）；本跑 = seed 42，协议 mandated 的第 2 个 seed
#   （v90/v101/v102 三次各向异性观测同为 seed 1234 ⇒ 本质是一个观测，见文件头）。
#   顺带修复 provenance：v102 实跑文件名 loss.mmm1.py、self_sha1 与交付片不符
#   ⇒ 本版必须以 cnn10_v103.py 之名跑，开机横幅 sha1 须等于交付值。
#   除此之外与 cnn10_v102.py 逐字相同（含 v102 的全部仪器修复）。
SEED = 42
VERSION         = "v117"   # [v110] 五处同步的唯一真相源：CKPT_DIR / OUT_DIR / JSON 名 / results / banner
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)   # [v49] 唯一一根轴（探针失败时自动降级并如实记录）

# ========== [v100] 判据常量（定义在**最前面**：banner 与收尾都要打印它们）==========
# ★★★ σ 第二次修正（v100 的第一根轴）：0.108 只用了 v45/v46 **一对**。
#   归档定案里其实有**四对**同口径数据，把四对的 |Δdelivered| 求 RMS：
#       v45/v46 = 0.120    v48/v49 = 0.040    v44/v54 = 0.140    v51/v55 = 0.280
#       RMS = sqrt((0.120^2 + 0.040^2 + 0.140^2 + 0.280^2) / 4) = sqrt(0.0285) = **0.169pp**
#   ⚠ 中间值是 0.0285 不是 0.0117：sqrt(0.0117) = 0.108 **恰是旧的被低估值**，
#     写错中间值会把人引回旧尺子（覆辙 297：注释里的演算必须自己也算对）。
#   ⇒ σ_diff 0.108 → 0.169（低估 1.565×），2σ 0.216 → 0.338。
#   ⇒ 教训：**一对样本估不出方差**。v49 量到的是"那一次的运气"，不是噪声底的宽度
#     （覆辙 113 的测量版：连尺子本身也要用足够的样本去量）。
PLATEAU_BASELINE = 97.4625   # [v49] v48 的末 8 次 TTA 均值（同 seed、同口径，= 本版直接前身）
PLATEAU_SIGMA_MEASURED = 0.169   # [v100] 四对零改动跑 |Δdelivered| 的 RMS（来源见 PLATEAU_BASELINE_SOURCE）
PLATEAU_SIGMA = PLATEAU_SIGMA_MEASURED
# ★★ 边距**由 σ 算出**，不写死：σ 一改，判据线自动跟随，不会静默脱钩（覆辙 113/145）。
PLATEAU_PASS_MARGIN = round(2.0 * float(PLATEAU_SIGMA_MEASURED), 3)   # -> 0.338 = 2σ
PLATEAU_PASS = round(PLATEAU_BASELINE + PLATEAU_PASS_MARGIN, 2)      # -> 97.80
PLATEAU_REDLINE = round(PLATEAU_BASELINE - PLATEAU_PASS_MARGIN, 2)   # -> 97.12
PLATEAU_WINDOW = 8           # 末 8 次 TTA 读数
PLATEAU_BEST_EVER_PP = 97.4625   # [v100] 本线历史最好平台（v48，末 8 次 TTA 均值）
# 两条守卫（会被塞进 lr_tail_instrument 参与门禁 ⇒ 任一条为 False 即 INSTRUMENT_FAIL）
PLATEAU_MARGIN_OK = bool(float(PLATEAU_PASS_MARGIN) >= 2.0 * float(PLATEAU_SIGMA) - 1e-12)
# [v100] 从"容差 0.005 的近似等式"升级为**精确等式**：边距既由 σ 算出，
#   PASS/红线就必须分毫不差地等于 baseline ± 边距 —— 容差会把 0.004pp 的漂移判成合格。
PLATEAU_DERIVED_OK = bool(float(PLATEAU_PASS) == round(float(PLATEAU_BASELINE) + float(PLATEAU_PASS_MARGIN), 2)
                          and float(PLATEAU_REDLINE) == round(float(PLATEAU_BASELINE) - float(PLATEAU_PASS_MARGIN), 2))
# ★★★ [v100] 判据**可达性**的诚实读数（是事实字段，不是守卫）：
#   σ 修正后 PASS 线 97.80 **高于**本线历史最好平台 97.4625 ⇒ 单跑平台判词
#   在算术上预先决定为 NULL。这是 v47/v48 之错的**镜像**（那次线太低，这次线太高）
#   ⇒ 本版的交付物是"尺子修准了"，不是一次精度主张。记成字段，不让它藏在注释里。
PLATEAU_PASS_REACHABLE = bool(float(PLATEAU_PASS) <= float(PLATEAU_BEST_EVER_PP))
PLATEAU_BASELINE_SOURCE = ("v48 实跑 97.4625（末 8 次 TTA 均值）；σ_diff 来源 = 归档定案的**四对**"
                           "同口径跑 |Δdelivered| 的 RMS：v45/v46=0.120、v48/v49=0.040、"
                           "v44/v54=0.140、v51/v55=0.280 ⇒ RMS = 0.169pp。"
                           "★ 该 σ 是**实测**，不是 n=3 估计，也不是一对的运气。")


# ========== [v101] 各向异性 Matérn 核轴（v90 轴在修正标尺上重开）==========
# ★ 立项依据（三条，全部来自档案定案，不是本轮猜测）：
#   (1) v90（2026-10-13）第一次开这根轴，判词 NULL + ANISO_SELFTEST_FAILED，
#       **但两条红灯后来被本机 float64 复算裁定为自测假红**：
#         · case2：f32 tanh(10) 饱和 ⇒ a_hi == 2.0 精确，旧自测期望"严格小于 2" ⇒ 假红；
#         · case3：旧 DFT 网格（linspace 含双端点）上**连各向同性核也不等变**
#           （5.7e-2@8x8）⇒ 期望本身不成立 ⇒ 假红。
#   (2) v90 机制读数有效：max|a_g-1| = 0.283、mean 0.082（≫1e-4）⇒ 轴送到了、
#       优化器真的推动了。精度 NULL 是尺度问题（当时 σ=0.108 且网格错一档），
#       **不是"轴无效"**。
#   (3) 现在两根尺子都修好了：DFT 网格已是真实频率（_dft_axis，v96 落地）、
#       σ_diff 已是四对 RMS 0.169（v100 落地）⇒ 各向异性轴**从未被公正地判过**。
#
# ★ 轴的内容（一句话）：MaternDiffusion 的核从径向 |k|^2 = kx^2 + ky^2 改为
#   k_sq_g = kx^2 + a_g*ky^2，a_g = 1.0 + ANISO_RANGE*tanh(aniso_param_g) ∈ (0, 2]。
#   a_g < 1 ⇒ ky 方向频率衰减更快（竖向平滑）；a_g > 1 ⇒ 横向衰减更快。
#   **只开 Matern 一支**（phys2/phys3）；FixedDiffusion（phys1）无方向自由度，不动。
#
# ★★ 归因纯度（本版能不能成立，全靠这一节）：
#   aniso_param = nn.Parameter(torch.zeros(groups))：
#     · zeros 初始化 ⇒ tanh(0) = 0 ⇒ a_g **精确等于 1.0** ⇒ 第 0 步核与 v100 逐位相同
#       （IEEE-754：1.0*t 精确返回 t；kx_sq + 1.0*ky_sq 与 kx_sq + ky_sq 操作数相同，
#         生成前已实测 torch.equal 为 True）；
#     · torch.zeros **不消耗 RNG**（生成前已实测 RNG state 逐位不变）⇒ 模型初始化
#       序列与 v100 逐位相同，跨跑只差"这一处"；
#     · 无交叉项（无 kx*ky）⇒ 核对 kx / ky 各自偶 ⇒ flip 等变（两轴）
#       ⇒ flip-TTA 交付通道受保护（v88 教训）；
#     · ∂kernel/∂a (a=1) = -s*lam*ky^2/nu*(1+lam*|k|^2/nu)^(-s-1) < 0（lam>0）
#       ⇒ 梯度非零，优化器推得动（v57 形态的防线）。
#   ★ 反面（本版明确不做）：ModeController 的 mode_weights 死计算含两个 Conv2d，
#     初始化耗 RNG ⇒ 删它等于换 seed，**不是**可归因的轴。本版不动它。
#
# ★ 代价（诚实列出，不藏）：+groups 个标量参数/模块（phys2 64 + phys3 8 = 72 个），
#   每前向多 2×groups 次 tanh + 同数量乘加 ⇒ ~0 FLOPs；aniso_trace 每 epoch 多两次
#   .item() 同步（与 sd_trace 同阶，不碰 42% 的那条线）。
ANISO_RANGE = 1.0            # a_g = 1.0 + ANISO_RANGE*tanh(param_g) ⇒ a ∈ (0, 2]
ANISO_DELIVER_MIN = 1e-3     # 末态 max|a_g-1| 下限：低于此 ⇒ ANISO_NOT_DELIVERED（轴没送到）
ANISO_FROZEN_MAX = 1e-6      # 全程 |param| 运行最大值上限：低于此 ⇒ ANISO_PARAM_FROZEN
                             #   （优化器从未推动，v57 形态；与 DELIVER_MIN 的大小关系由
                             #     _aniso_selftest case7 守着：FROZEN_MAX 必须 < DELIVER_MIN）
ANISO_MODULES_EXPECTED = 2   # phys2 + phys3 两个 MaternDiffusion 带 aniso_param；
                             #   FixedDiffusion 不许有（轴只开 Matern 一支）
ANISO_EQUIV_TOL = 1e-5       # flip 等变容差：修正网格实测 f32 ~2.4e-7 ⇒ 留 ~40x 余量
                             #   （覆铣 301/272：阈值不许贴着噪声底）
ANISO_TANH_SAT_PARAM = 10.0  # f32 下 tanh(10)=1.0 **精确饱和** ⇒ a=2.0 精确：
                             #   文档化的预期，不是失败（v90 case2 假红之源就是旧自测
                             #   期望"a 严格小于 2"；生成前已实测 tanh(9) 不饱和而 tanh(10) 精确=1）

# ---- [v102] 自测跑批的**预期总条数**（覆铣 #312：all_pass 的分母必须是它，
#      不能是"崩溃前已跑条数"—— v101 实跑在 CUDA 上 15 条只跑 7 条，
#      旧公式分母=7 ⇒ n_pass==n_total 恒真 ⇒ 假绿"7/7 ALL PASS"）----
ANISO_SELFTEST_CASES = 15      # _aniso_selftest 的 chk() 站数（部署自证 J4 静态核对）
DFT_SELFTEST_CASES = 8         # _dft_grid_selftest 的 chk() 站数（部署自证 J5 静态核对）

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
CKPT_DIR = f'./ckpts_{VERSION}_s{SEED}'  # [v96] 版本号由 VERSION 派生（五处同步）
OUT_DIR = f'./out_{VERSION}_s{SEED}'     # [v96] 版本号由 VERSION 派生（五处同步）
# ★★ [v115] P0-C：**OUT_DIR 开机即建**（v114 实跑踩的第二个坑）。
#   v114 的判词预落盘在 L4889，而 `os.makedirs(OUT_DIR)` 在 L5940 ⇒ 目录当时
#   不存在 ⇒ 预落盘 FileNotFoundError ⇒ prewrite_ok=false；同一根因也让磁盘
#   守卫 statvfs 报 STATVFS_ERROR（dirname 指向不存在的目录）。
#   ⇒ 这里立刻建（exist_ok=True 幂等；原处那次保留，不影响任何读数）。
try:
    os.makedirs(OUT_DIR, exist_ok=True)
except Exception as _exc_dir:
    pass

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

# ############################################################################
# ★★★★★ [v96] 本版唯一一根轴：DFT 频率网格修正（修 bug，不是加机制）
# ############################################################################
#
# 〇 旧写法（v9 起，v51/v95 都在用）
#     ky = torch.linspace(-math.pi, math.pi, H)     # step = 2*pi/(N-1)，含双端点
#   与 fftshift 之后的真实 bin 频率 -pi + 2*pi*i/N（step = 2*pi/N，半开区间）
#   **不是同一个网格** ⇒ 除两端外每个 bin 拿到的 |k| 都偏大（覆辙 273）；
#   且 -pi 与 +pi 是同一个 bin 却被放两次；DC（k=0）不在网格上（覆辙 253）。
#
# 一 新写法
#     _dft_axis(N) = arange(N) * (2*pi/N) - pi
#   ★ torch.linspace **没有** endpoint 参数（覆辙 275，那是 numpy 的 API）
#     ⇒ v92 写 endpoint=False 发车即崩 ⇒ 这里用 arange 表达同一数学对象。
#
# 二 为什么值得修（打在交付口径上）
#   delivered = max-of-N TTA，而 TTA = (net(x) + net(flip(x)))/2 —— 唯一依赖
#   **flip 等变性**，而旧网格把它破坏了（实测 6.379e-02）。
#
# 三 ⚠ recipe 域改动：|k| 步长 2*pi/(N-1) → 2*pi/N ⇒ 核的实际衰减率变了
#   ⇒ 第 0 步就与 v51 不同 ⇒ 只有**跨跑**判据。
# ############################################################################

# [v96] 判据常量（2σ 边距 —— 依据 v48 noise_floor/note 立的规矩）
# [v100] σ 修正为四对 RMS 0.169pp ⇒ 两条线**由公式派生**（v96 时还是写死的 97.78 / 97.34）
BASELINE_V51_ACC = 97.56     # 逐字回退基线：cnn10_v51.py 实跑 delivered
SIGMA_DIFF_PP    = 0.169     # [v100] **实测**噪声底（四对同口径跑 |Δdelivered| 的 RMS）
PASS_LINE_96     = round(BASELINE_V51_ACC + 2.0 * SIGMA_DIFF_PP, 2)   # -> 97.90
REDLINE_96       = round(BASELINE_V51_ACC - 2.0 * SIGMA_DIFF_PP, 2)   # -> 97.22
# [v100] 容差 0.011 → **精确等式**（派生线必须分毫不差）
DFT_GRID_2SIGMA_MARGIN_OK = bool(
    float(PASS_LINE_96) == round(float(BASELINE_V51_ACC) + 2.0 * float(SIGMA_DIFF_PP), 2)
    and float(REDLINE_96) == round(float(BASELINE_V51_ACC) - 2.0 * float(SIGMA_DIFF_PP), 2))

# [v96] 自测容差（全部按 v93 本机 float32 实测标定 —— 覆辙 272：容差不许拍脑袋）
DFT_FLIP_TOL     = 1e-5      # 新网格 flip/转置等变差 <= 2.4e-7
DFT_DC_TOL       = 1e-4      # 新网格 DC 增益偏离 <= 0
DFT_NEGCTRL_MIN  = 1e-3      # ★ 负向对照下限：旧网格必须**至少**这么红
                             #   （旧网格 flip 差 6.97e-02、DC 偏离 0.29 —— 都远超此线）
                             #   若旧网格不红 ⇒ 上面两条判据是同义反复（覆辙 267）

AXIS_V96 = ("v96 axis = DFT FREQUENCY-GRID CORRECTION on the physics field. "
            "The production grid was torch.linspace(-pi, pi, N) whose step is 2*pi/(N-1) "
            "(closed interval; -pi and +pi are the same bin listed twice), while the true "
            "fftshift-ed DFT bin spacing is 2*pi/N (half-open) with DC at index N//2. "
            "Replaced by _dft_axis(N) = arange(N)*2*pi/N - pi, which matches 2*pi*fftfreq. "
            "This is a v51 INHERITED BUG (pitfall 273/253), not a new mechanism: it restores "
            "FLIP EQUIVARIANCE -- which flip-TTA, i.e. the delivery metric, depends on "
            "exclusively -- and makes the DC gain exactly 1. torch.linspace has NO endpoint "
            "kwarg (numpy API, pitfall 275) so arange is used. All the rest is verbatim "
            "cnn10_v51.py; groups stays 8. Recipe-domain change: step 0 already differs "
            "from v51, so this is a cross-run criterion only.")


def _v96_file_name_selfcheck():
    """[v96] 版本自检必须至少有一条**对齐身外事实**（覆辙 266）。

    VERSION 是文件**内部**的名字：它写错了，文件自己不会痛，跑出来的结果
    照样写着 "v96"。文件名是**身外事实** —— 训练机上如果跑的是一个改名/复制的
    副本（历史上真的出现过 `loss.mmm1.py` 这种部署名），这条就会转红。
    """
    try:
        _base = os.path.basename(os.path.abspath(__file__))
    except Exception as _e:                                   # pragma: no cover
        return {"file_name": None, "version": VERSION, "ok": False,
                "error": "%s: %s" % (type(_e).__name__, _e)}
    return {"file_name": _base, "version": VERSION,
            "ok": bool(VERSION in _base), "error": None}


_V96_FILE_NAME_CHECK = _v96_file_name_selfcheck()


# =============================================================================
# [v98] gate2 通道粒度轴  ——  本版**唯一**一根轴（配方域改动）
#
# ★ 立项依据（三条，全部来自档案里的既有定案，不是本轮的猜测）：
#   (1) v56 实测的 gate 杠杆分解：gate1 5.3% < gate3 8.5% < **gate2 11.7%**。
#       ⇒ gate2 是三支 gate 里杠杆最大的一支。
#   (2) 但 v52 / v53 / v54 三版的轴**全做在杠杆最小的 gate1 上**（gate1 是固定注入
#       常量、改它不会被优化器补偿，所以历史上一直被当成"安全"的旋钮）。
#       ⇒ 杠杆最大的那支，自 v9 起**从未被单独动过**。
#   (3) v97 实跑把磁盘事故与 DFT 网格继承 bug 都结清了（跑满 180 轮、
#       自测 8/8 全过含两条负向对照），读数链第一次是干净的，现在才轮得到开新轴。
#
# ★ 轴的内容（一句话）：把 phys2（MaternDiffusion，512ch，@16x16，接在 layer2 后）
#   的 groups 从 8 提到 64 —— 扩散核的**通道粒度**从"每 64 通道一套核"细化到
#   "每 8 通道一套核"。
#
#   机制上到底发生了什么（读 MaternDiffusion.forward 的代码得出，不是猜）：
#       x_g = x.reshape(B, groups, cpg, H, W).mean(dim=2)      # 组内通道求平均
#       逐 group: kernel_g = (1 + lam_g*|k|^2/nu_g)^(-s_g)      # 每组一套 (s,lam,nu)
#       phys = cat(filtered_g).repeat_interleave(cpg, dim=1)    # 广播回 512 通道
#   ⇒ groups 8 -> 64：
#     · 可学习的扩散参数 8x3=24 个 -> 64x3=192 个；
#     · 组内做平均的通道数 64 -> 8 ⇒ 场保留更多通道特异性结构；
#     · phys 里互不相同的通道块模式数 8 -> 64。
#
# ★★ 归因纯度（本版能不能成立，全靠这一节）：
#   MaternDiffusion 的 s_param / lam_param / nu_param 全是
#   nn.Parameter(torch.zeros(self.groups))，running_* 是 torch.full(...) ⇒
#   **改变 groups 不消耗任何 RNG**。Gate(512) 的 proj/bn/alpha/scale 逐字不动，
#   phys1 / phys3 的 groups 保持 8。
#   ⇒ 模型初始化序列与 v97 **逐位相同**，跨跑只差"这一处"。
#   ★ 反面（本版明确不做）：MaternDiffusion 里的
#     `mode_weights = self.mode_net(x)` 算完**从未被使用**（死计算），
#     但 ModeController 含两个 nn.Conv2d，**初始化要耗 RNG**
#     ⇒ 删它等于等换乘 seed，**不是**可归因的轴。本版不动它。
#
# ★ 代价（诚实列出，不藏）：phys2 的 FFT 调用 8 次/前向 -> 64 次/前向
#   （每组 fft2 + ifft2 + 2 次 fftshift，作用在 (B,1,16,16) 小张量上）。
#   估算 +2.2 ms/前向 ⇒ 整跑 +~80 秒。可接受，故**不改**成批量 FFT：
#   批量 FFT 数学上等价但**不保证逐位相同**，那会悄悄引入第二根轴。
#
# ★ 判据分两层（不要混读）：
#   · 机制级（本版特有，单跑即可判，与精度无关）→ _v98_granularity_verdict；
#   · 精度级沿用 v96/v97 的平台口径：PASS >= 97.68 / 红线 <= 97.2425，
#     σ_diff 实测 0.169pp（2σ = 0.338pp）⇒ 任何 <0.22pp 的效应单跑都读不出来。
#     **单跑 NULL 不得升级为"轴无效"，需 3 seed**（档案已定案：13 版 delivered
#     极差 0.280pp < 2σ，跨跑比精度已不可判）。
# =============================================================================
PHYS2_GROUPS = 64            # [v98] 唯一一根轴：phys2 的通道粒度（v97 及之前恒为 8）
PHYS1_GROUPS = 8             # 对照组：必须与 v97 逐字相同（保证"只动了一处"可被机检）
PHYS3_GROUPS = 8             # 对照组：同上
GRAN_SEPARATION_MIN = 1e-3   # ★ 64 组的 (s,lam,nu) **参数**跨组离散度下限：低于此 ⇒
                             #   64 组收敛成了同一个核 ⇒ 粒度提升是空转。
                             #   负向对照（执行级自证实测 C5/C6）：
                             #   · s_param/lam_param/nu_param 是 torch.zeros ⇒ std **精确为 0.0**；
                             #   · running_* 是 f32 的 torch.full（64 个同一个值），
                             #     但 std 自己在 f32 下做两遍归约 ⇒ 实测 **5.96e-08**，
                             #     不是 0（覆辙 272 新形态：拿 1e-12 去卡 f32 必然假红）。
                             #   ⇒ 判据**只用 s/lam/nu 三个 param 的 std**，不用 running_*；
                             #     且阈值 1e-3 比那点噪声大 4 个数量级。
                             #   ⇒ 这条判据能红，不是同义反复（覆辙 267）。
GRAN_ALPHA_BOUND_EPS = 1e-3  # gate2.alpha 撞界判定容差（soft_bound(a, 0.05, 0.9)）

AXIS_V98 = ("v98 axis = GATE2 CHANNEL GRANULARITY. phys2 = MaternDiffusion(512ch, @16x16) "
            "goes from groups=8 to groups=64: the per-group Matern kernel "
            "(1+lam_g|k|^2/nu_g)^(-s_g) is applied to the mean of cpg=channels/groups "
            "channels, so 8->64 shrinks the block from 64 channels to 8 and yields 64 "
            "independent (s,lam,nu) triples instead of 8. Rationale from archive v56: the "
            "measured gate leverage ranking is gate1 5.3% < gate3 8.5% < gate2 11.7%, yet "
            "v52/v53/v54 all worked on gate1 (the lowest-leverage one) and gate2 has never "
            "been touched alone since v9. ATTRIBUTION PURITY: s/lam/nu params are "
            "nn.Parameter(torch.zeros(groups)) and running_* are torch.full => changing "
            "groups consumes NO RNG, so the init sequence is bit-identical to v97 and the "
            "cross-run diff is this one thing. phys1/phys3 stay at groups=8; Gate(512) is "
            "verbatim. COST: phys2 FFT calls per forward go 8 -> 64 (accepted, ~+80s per "
            "run; batched FFT is deliberately NOT used because it is not guaranteed "
            "bitwise-equal, which would smuggle in a second axis). Recipe-domain change "
            "=> cross-run criterion only; single-run NULL must not be upgraded to "
            "'axis invalid' (needs 3 seeds).")


AXIS_V99 = ("v99 = NO NEW AXIS. This version exists solely to fix a P0 crash that "
            "destroyed v98's entire JSON: `_v98_gran` was referenced INSIDE the "
            "`results = {...}` literal (L3644) but only bound at L3704 - a module-level "
            "USED_BEFORE_BOUND, the exact shape of v93's `json_dict` (pitfall 279). "
            "Training math is bit-identical to v98: no parameter, no schedule, no RNG "
            "draw, no loss term changed. Expected delivered = 97.42 (exact repeat of "
            "v98 under seed 1234); a deviation is evidence of NON-DETERMINISM, not of "
            "an axis effect. The gate2-granularity axis of v98 is RECORDED AS UNJUDGED "
            "(its mechanism readout died with the JSON) by user decision - v99 does not "
            "spend 7.5h re-running to recover it. Structural fix: the results literal no "
            "longer references any post-hoc name; readouts are computed after "
            "construction and attached by assignment, each wrapped in _safe_readout; the "
            "main results are written to *_core.json BEFORE any post-hoc readout runs.")


AXIS_V100 = ("v100 axis = INSTRUMENT / CRITERION CORRECTION ONLY - ZERO training-math "
            "change. No parameter, no schedule, no RNG draw, no loss term, no physics "
            "constant differs from v99; train()/test_tta() and the whole physics chain are "
            "bit-identical. What changes is the RULER, and the ruler was provably wrong in "
            "five places. (1) sigma_diff was 0.108pp estimated from a SINGLE pair (v45/v46) "
            "while the archive holds FOUR same-recipe pairs - |delta delivered| = 0.120 "
            "(v45/v46), 0.040 (v48/v49), 0.140 (v44/v54), 0.280 (v51/v55) - whose RMS is "
            "0.169pp, i.e. the old constant UNDERSTATED the noise floor by 1.565x and made "
            "every old line too loose. sigma_diff := 0.169. (2) All derived lines are now "
            "computed from constants (PLATEAU_PASS/REDLINE, PASS_LINE_96/REDLINE_96, "
            "BACKTRACK_REDLINE) instead of hardcoded literals - v99 still had 97.27 and "
            "97.61 as raw literals, and 97.27 was derived from a sigma of 0.06 that is not "
            "the same ruler as PLATEAU_SIGMA. (3) The lr_tail dose gate is split in two: "
            "v99 used ONE gate (`dose_delivered`) for two different channels, and v99's own "
            "data shows why that is wrong - the dose DID run for 36 epochs (ep121-156) and "
            "the plateau window (ep135-156) lies FULLY inside it, but the delivery "
            "selection landed on ep120 because TTA ties at ep120 and ep126 (both 97.44) "
            "and the rule is first-wins. So the delivery-channel blindness is an artifact "
            "of the tie-break rule, not a missing dose. (4) The argmax tie is now a machine "
            "field (best_tta_argmax_epochs / _n / tie_break_rule). (5) The delivery claim "
            "that v99's own data falsified (`expected_delivered_vs_v98: identical 97.42`, "
            "actually 97.44) is corrected in place, with the original text preserved under "
            "a *_AS_WRITTEN_AT_DELIVERY key. HONESTY NOTE: with sigma_diff = 0.169 the "
            "plateau PASS line becomes 97.80, which is ABOVE the best plateau this line has "
            "ever produced (97.4625, v48) - so a single-run plateau verdict is "
            "arithmetically pre-decided to NULL. That is the mirror image of the v47/v48 "
            "error (line too low); the deliverable of this version is the corrected ruler, "
            "not an accuracy claim. Recipe-domain change: NO. Cross-run criterion only; "
            "single-run NULL must NOT be upgraded to 'axis invalid' (needs 3 seeds).")

AXIS_V101 = ("v102 = v101 **逐字不动**的训练数学 + 纯仪器修复（覆铣 #312）；"
            "the axis itself is UNCHANGED from v101: ANISOTROPIC MATERN KERNEL "
            "(re-opening the v90 axis on corrected instruments). MaternDiffusion's "
            "radial kernel (1+lam*|k|^2/nu)^(-s) with |k|^2 = kx^2+ky^2 becomes "
            "k_sq_g = kx^2 + a_g*ky^2, a_g = 1.0 + 1.0*tanh(aniso_param_g) in (0,2], "
            "applied ONLY to the two MaternDiffusion modules (phys2 512ch groups=64, "
            "phys3 1024ch groups=8); FixedDiffusion (phys1) has no directional DOF "
            "and is untouched. Attribution purity: aniso_param is "
            "nn.Parameter(torch.zeros(groups)) - zeros init consumes NO RNG "
            "(measured: the RNG state is bitwise unchanged) so the model init "
            "sequence is bit-identical to v100, and tanh(0)=0 exactly so a_g == 1.0 "
            "exactly and the step-0 kernel is bit-identical to v100's kx^2+ky^2 "
            "(measured: 1.0*t returns t exactly in IEEE-754). No cross term (no "
            "kx*ky) => the kernel is even in kx and ky separately => flip-equivariant "
            "on both axes => the flip-TTA delivery channel is protected. v90's two "
            "red lights were adjudicated as SELF-TEST false reds (float32 tanh(10) "
            "saturates to exactly 1.0 so a == 2.0 exactly; on the OLD dft grid even "
            "the isotropic kernel was not equivariant, 5.7e-2@8x8) - both are now "
            "CALIBRATED in _aniso_selftest (saturation is documented expected "
            "behaviour at param=ANISO_TANH_SAT_PARAM; equivariance is measured on "
            "the corrected _dft_axis grid with a 40x margin over the measured "
            "2.4e-7 f32 noise). v90's mechanism readout was valid (max|a-1|=0.283 "
            "=> the optimizer really pushed). v101's real run (2026-10-18) was "
            "adjudicated: accuracy NULL (delivered 97.48; every delta inside "
            "sigma_diff=0.169 => DO_NOT_BACKTRACK, and a single NULL must NOT be "
            "upgraded to 'axis invalid' - that needs 3 seeds) and mechanism "
            "ANISO_DELIVERED (phys2 max|a-1|=0.474, phys3=0.180, 162/163 epochs "
            "moved, no saturation) - BUT that verdict was computed with a BROKEN "
            "instrument: _aniso_selftest took a device argument that applied only "
            "to its INPUT tensors while every constructed module stayed on CPU => "
            "on CUDA the selftest crashed at case2 after 7 of 15 cases, and its "
            "all_pass formula (denominator = cases run before the crash) still "
            "printed '7/7 ALL PASS' (pitfall #312). v102 fixes exactly that and "
            "NOTHING else: every module construction goes through _on_dev(m, dev), "
            "the rep carries a completed flag with the denominator pinned to the "
            "expected case count (ANISO_SELFTEST_CASES / DFT_SELFTEST_CASES), and "
            "the verdict gains a selftest-error tier. Training math is bitwise "
            "identical to v101 (measured: MaternDiffusion class md5 unchanged, "
            "RNG-site multiset unchanged, .to(dev) consumes no RNG). NOT done "
            "this version: removing ModeController's dead mode_weights computation "
            "(its two Conv2d inits consume RNG => that is a seed swap, not an "
            "attributable axis).")


AXIS_V103 = ("v103 = cnn10_v102.py **逐字不动**的训练数学 + 唯一一根轴：换 seed "
             "1234 -> 42 (seed swap; the protocol-mandated SECOND seed). No parameter, "
             "no schedule, no loss term, no physics constant, no RNG site changed: the "
             "ONLY edit is the integer on the SEED line. Everything the seed touches "
             "derives from it - random.seed / np.random.seed / torch.manual_seed / "
             "torch.cuda.manual_seed_all / worker_init_fn's np.random.seed(SEED+id) / "
             "CKPT_DIR / OUT_DIR / the results JSON name - so the swap is complete by "
             "construction (statically verified: the only remaining '1234' literals "
             "are historical records plus THREE INDEPENDENT probe seeds 12345 / "
             "2026096 / 2026092 which must stay fixed for cross-version probe "
             "comparability). Why seed 42: the archive holds only two seeds - v43=42 "
             "and everything from v44 onward=1234 - and the pre-registered protocol "
             "says a single NULL run must NOT be upgraded to 'axis invalid' (that "
             "needs 3 seeds); v90 / v101 / v102 are three readings of ONE seed (1234) "
             "so they count as one. This run is the second seed on the CURRENT code "
             "stack; the v101 anisotropic Matern axis is INHERITED unchanged (not "
             "re-opened, not edited). Expected reading: statistically "
             "indistinguishable from v102's 97.42 at sigma_diff=0.169pp (same code, "
             "different seed) - the purpose of this run is to MEASURE the seed spread "
             "of the current code, not to claim a number. Provenance repair: v102's "
             "actual run was on loss.mmm1.py with self_sha1 ad097731... != the "
             "delivered cnn10_v102.py c12256ab... (the B line's first content-level "
             "provenance break) - this version must be run AS cnn10_v103.py so the "
             "startup banner sha1 equals the delivered value. Paired readout: v103 "
             "vs v102 is the FIRST same-code cross-seed pair on the current stack "
             "(see results['v103_seed_pair']); v103 vs v43 shares seed 42 but not "
             "code, so their z-scores must not be merged (pitfall 106).")


def _v98_granularity_readout():
    """[v98] gate2 通道粒度轴的**机制读数**（只读；不跑前向，不碰梯度路径）。

    ★ 为什么不跑一次前向去量 phys 的通道模式数：
      MaternDiffusion.forward 会推进 running_s / running_lam / running_nu /
      update_count（覆辙 56/221：前向**非纯**），为了一个读数去动训练态不划算。
      参数空间（参数 + running 缓冲 + gate 标量）已经够判，而且零风险。

    ★ 每个键独立 try（覆辙 130）：一个键挂掉不许吞掉整块读数，也不许留下假绿字段。
    """
    def _f(name, fn):
        try:
            return {"value": fn(), "error": None}
        except Exception as _e:
            return {"value": None, "error": "%s: %s" % (type(_e).__name__, _e)}

    def _std(t):
        _t = t.detach().reshape(-1).float().cpu()
        return float(_t.std(unbiased=False)) if _t.numel() > 1 else 0.0

    def _mean(t):
        _t = t.detach().reshape(-1).float().cpu()
        return float(_t.mean()) if _t.numel() > 0 else None

    p1, p2, p3 = raw_model.phys1, raw_model.phys2, raw_model.phys3
    g2, g3 = raw_model.gate2, raw_model.gate3

    a2 = soft_bound(g2.alpha, 0.05, 0.9)
    a3 = soft_bound(g3.alpha, 0.05, 0.9)

    return {
        # ---- 轴本体 ----
        "phys2_groups": _f("phys2_groups", lambda: int(p2.groups)),
        "phys2_cpg": _f("phys2_cpg", lambda: int(p2.cpg)),
        "phys2_channels": _f("phys2_channels", lambda: int(p2.channels)),
        "phys2_groups_times_cpg": _f("phys2_groups_times_cpg",
                                     lambda: int(p2.groups) * int(p2.cpg)),
        "phys2_fft_calls_per_forward": _f("phys2_fft_calls_per_forward",
                                          lambda: int(p2.groups)),
        "s_param_shape": _f("s_param_shape", lambda: list(p2.s_param.shape)),
        # ---- 对照组：另外两支必须仍是 8（证明"只动了一处"）----
        "phys1_groups": _f("phys1_groups", lambda: int(p1.groups)),
        "phys3_groups": _f("phys3_groups", lambda: int(p3.groups)),
        # ---- 粒度是否真的被用上：64 组的核参数有没有分化 ----
        "s_param_std": _f("s_param_std", lambda: _std(p2.s_param)),
        "lam_param_std": _f("lam_param_std", lambda: _std(p2.lam_param)),
        "nu_param_std": _f("nu_param_std", lambda: _std(p2.nu_param)),
        "running_s_std": _f("running_s_std", lambda: _std(p2.running_s)),
        "running_lam_std": _f("running_lam_std", lambda: _std(p2.running_lam)),
        "running_nu_std": _f("running_nu_std", lambda: _std(p2.running_nu)),
        "running_s_mean": _f("running_s_mean", lambda: _mean(p2.running_s)),
        "running_lam_mean": _f("running_lam_mean", lambda: _mean(p2.running_lam)),
        "running_nu_mean": _f("running_nu_mean", lambda: _mean(p2.running_nu)),
        # ---- gate2 / gate3 的注入剂量（v97 完全没落 ⇒ 选下一根轴时是盲区）----
        "gate2_alpha": _f("gate2_alpha", lambda: float(a2)),
        "gate2_alpha_raw": _f("gate2_alpha_raw", lambda: float(g2.alpha)),
        "gate2_scale": _f("gate2_scale", lambda: float(g2.scale)),
        "gate2_inject_effective": _f("gate2_inject_effective",
                                     lambda: float(g2.scale) * float(a2)),
        "gate3_alpha": _f("gate3_alpha", lambda: float(a3)),
        "gate3_scale": _f("gate3_scale", lambda: float(g3.scale)),
        "gate3_inject_effective": _f("gate3_inject_effective",
                                     lambda: float(g3.scale) * float(a3)),
        "gate1_inject_fixed": _f("gate1_inject_fixed", lambda: float(GATE1_INJECT)),
    }


def _v98_granularity_verdict(rd):
    """[v98] gate2 通道粒度轴的判词（**必须是函数** —— 覆辙 263）。

    阶梯（从最严重往下，先修尺子再看结论）：
      GRAN_READOUT_ERROR      —— 读数块本身坏了 ⇒ 尺子坏，任何结论都不可信
      GRAN_GROUPS_NOT_APPLIED —— 常量没落到模型上（phys2.groups != PHYS2_GROUPS）
      GRAN_OFF_TARGET         —— 动到了别处（phys1 / phys3 不再是 8）
      GRAN_NO_SEPARATION      —— 64 组没分化 ⇒ 粒度提升是空转（等价于只有一组）
      GRAN_GATE2_SATURATED    —— gate2.alpha 撞在 soft_bound 的设计域边界上
      GRAN_DELIVERED          —— 机制已送达（**与精度无关**；精度另走跨跑判据）
    """
    try:
        if not isinstance(rd, dict) or not rd:
            return "GRAN_READOUT_ERROR"

        def _v(k):
            _c = rd.get(k)
            return None if not isinstance(_c, dict) else _c.get("value")

        g2 = _v("phys2_groups")
        g1 = _v("phys1_groups")
        g3 = _v("phys3_groups")
        if g2 is None or g1 is None or g3 is None:
            return "GRAN_READOUT_ERROR"
        if int(g2) != int(PHYS2_GROUPS):
            return "GRAN_GROUPS_NOT_APPLIED"
        if int(g1) != int(PHYS1_GROUPS) or int(g3) != int(PHYS3_GROUPS):
            return "GRAN_OFF_TARGET"

        stds = [_v("s_param_std"), _v("lam_param_std"), _v("nu_param_std")]
        stds = [float(s) for s in stds if s is not None]
        if not stds:
            return "GRAN_READOUT_ERROR"
        if max(stds) < float(GRAN_SEPARATION_MIN):
            return "GRAN_NO_SEPARATION"

        a2 = _v("gate2_alpha")
        if a2 is not None:
            if (float(a2) >= 0.9 - float(GRAN_ALPHA_BOUND_EPS) or
                    float(a2) <= 0.05 + float(GRAN_ALPHA_BOUND_EPS)):
                return "GRAN_GATE2_SATURATED"
        return "GRAN_DELIVERED"
    except Exception:
        return "GRAN_READOUT_ERROR"




def _v101_aniso_readout():
    """[v101] 各向异性轴的**机制读数**（只读；不跑前向，不碰梯度路径）。

    ★ 只读 master 侧**参数**（aniso_param 由优化器更新）：
      DataParallel 下 forward 跑在 replica 上，写在 buffer 里的量不回流 master
      ⇒ 这里绝不读任何"forward 顺手存下来"的量（v98 的 running_* 读数就是那个形态）。
    ★ a 的统计在读数时从末态参数现算（a = 1 + ANISO_RANGE*tanh(param)），
      不依赖训练过程中的任何缓存 ⇒ 不存在"缓存过期"这一失效模式。
    ★ 每个键独立 try（覆铣 130）：一个键挂掉不许吞掉整块读数。
      ⚠ 边界说清楚（覆铣 297：注释不许过度承诺）：**共享前置区**
      （aniso_param 提取 + aniso_trace 归约，下面 5 行）不在 per-key try 里。
      它坏掉 ⇒ 整块读数抛异常 ⇒ 由调用点的 `_safe_readout(..., {})` 装配守卫
      接住 ⇒ 读数回落 {}、判词落 ANISO_READOUT_ERROR（诚实红，不假绿）。
      前置区在生产路径上不可能坏：phys2/phys3 恒为 MaternDiffusion（缺轴由开机
      自测 case4b 咬成 ANISO_SELFTEST_FAILED，比读数错更严重、且在阶梯更前面），
      aniso_trace 由唯一 append 站点写 3 元组（部署检查 C9 守着）。
    """
    def _f(name, fn):
        try:
            return {"value": fn(), "error": None}
        except Exception as _e:
            return {"value": None, "error": "%s: %s" % (type(_e).__name__, _e)}

    def _a_stats(p):
        _t = p.detach().reshape(-1).float().cpu()
        _a = 1.0 + ANISO_RANGE * torch.tanh(_t)
        return _t, _a

    p1, p2, p3 = raw_model.phys1, raw_model.phys2, raw_model.phys3
    _t2, _a2 = _a_stats(p2.aniso_param)
    _t3, _a3 = _a_stats(p3.aniso_param)
    _run2 = max([m for (_, m, _) in aniso_trace], default=0.0)
    _run3 = max([m for (_, _, m) in aniso_trace], default=0.0)
    return {
        # ---- 轴常量（读数自曝：判据门槛随 JSON 落盘，不靠读代码）----
        "aniso_range": _f("aniso_range", lambda: float(ANISO_RANGE)),
        # ---- 对照组：phys1（FixedDiffusion）不许带方向自由度 ----
        "phys1_has_aniso_param": _f("phys1_has_aniso_param",
                                    lambda: bool(hasattr(p1, "aniso_param"))),
        # ---- phys2（512ch, groups=64）----
        "phys2_aniso_numel": _f("phys2_aniso_numel", lambda: int(_t2.numel())),
        "phys2_aniso_abs_max_final": _f("phys2_aniso_abs_max_final",
                                        lambda: float(_t2.abs().max())),
        "phys2_a_min": _f("phys2_a_min", lambda: float(_a2.min())),
        "phys2_a_max": _f("phys2_a_max", lambda: float(_a2.max())),
        "phys2_max_abs_a_minus_1": _f("phys2_max_abs_a_minus_1",
                                      lambda: float((_a2 - 1.0).abs().max())),
        "phys2_mean_abs_a_minus_1": _f("phys2_mean_abs_a_minus_1",
                                       lambda: float((_a2 - 1.0).abs().mean())),
        "phys2_aniso_abs_max_running": _f("phys2_aniso_abs_max_running",
                                          lambda: float(_run2)),
        "phys2_aniso_saturated_groups": _f("phys2_aniso_saturated_groups",
                                           lambda: int((_a2 >= 2.0).sum())),
        # ---- phys3（1024ch, groups=8）----
        "phys3_aniso_numel": _f("phys3_aniso_numel", lambda: int(_t3.numel())),
        "phys3_aniso_abs_max_final": _f("phys3_aniso_abs_max_final",
                                        lambda: float(_t3.abs().max())),
        "phys3_a_min": _f("phys3_a_min", lambda: float(_a3.min())),
        "phys3_a_max": _f("phys3_a_max", lambda: float(_a3.max())),
        "phys3_max_abs_a_minus_1": _f("phys3_max_abs_a_minus_1",
                                      lambda: float((_a3 - 1.0).abs().max())),
        "phys3_mean_abs_a_minus_1": _f("phys3_mean_abs_a_minus_1",
                                       lambda: float((_a3 - 1.0).abs().mean())),
        "phys3_aniso_abs_max_running": _f("phys3_aniso_abs_max_running",
                                          lambda: float(_run3)),
        "phys3_aniso_saturated_groups": _f("phys3_aniso_saturated_groups",
                                           lambda: int((_a3 >= 2.0).sum())),
        # ---- per-epoch 轨迹（区分"从未推动"与"没跑"的唯一依据）----
        "aniso_trace_len": _f("aniso_trace_len", lambda: int(len(aniso_trace))),
        "aniso_epochs_with_movement": _f(
            "aniso_epochs_with_movement",
            lambda: int(sum(1 for (_, m2, m3) in aniso_trace
                            if max(m2, m3) > float(ANISO_FROZEN_MAX)))),
        "aniso_trace_head": _f("aniso_trace_head",
                               lambda: [list(t) for t in aniso_trace[:3]]),
        "aniso_trace_tail": _f("aniso_trace_tail",
                               lambda: [list(t) for t in aniso_trace[-3:]]),
    }


def _v101_aniso_verdict(rd, st):
    """[v101] 各向异性轴的判词（**必须是函数** —— 覆铣 263）。

    阶梯（从最严重往下，先修尺子再看结论）：
      ANISO_SELFTEST_ERROR     —— 自测本身崩了/没跑/没跑完（error 非空或
                                   completed=False）⇒ 尺子坏，任何结论都不可信
      ANISO_SELFTEST_FAILED    —— 自测有案例红（含两条负向对照）
      ANISO_READOUT_ERROR      —— 读数块坏了（含 aniso_trace 为空：没有 per-epoch
                                   轨迹就无法区分"从未推动"与"根本没跑"）
      ANISO_MODULE_COUNT_MISMATCH —— phys1 带了 aniso（轴跑出 Matern 之外）
      ANISO_PARAM_FROZEN       —— 全程 |param| 运行最大值 < ANISO_FROZEN_MAX
                                   （优化器从未推动，v57 形态）
      ANISO_NOT_DELIVERED      —— 末态 max|a-1| < ANISO_DELIVER_MIN（轴没送到）
      ANISO_DELIVERED          —— 机制已送达（**与精度无关**；精度另走跨跑判据）
    """
    try:
        if not isinstance(st, dict) or not st.get("ran"):
            return "ANISO_SELFTEST_ERROR"
        # [v102] 覆铣 #312：崩了/没跑完 ⇒ SELFTEST_ERROR（不是 FAILED）。
        #   v101 实跑：CUDA 上 selftest 在 case2 崩、15 条只跑 7 条，
        #   旧 all_pass 照样为真 ⇒ 判词拿到的是"尺子验过了"的假前提。
        if st.get("error") is not None or not st.get("completed"):
            return "ANISO_SELFTEST_ERROR"
        if not st.get("all_pass"):
            return "ANISO_SELFTEST_FAILED"
        if not isinstance(rd, dict) or not rd:
            return "ANISO_READOUT_ERROR"

        def _v(k):
            _c = rd.get(k)
            return None if not isinstance(_c, dict) else _c.get("value")

        _p1 = _v("phys1_has_aniso_param")
        if _p1 is None:
            return "ANISO_READOUT_ERROR"
        if bool(_p1):
            return "ANISO_MODULE_COUNT_MISMATCH"
        for _k in ("phys2_aniso_numel", "phys3_aniso_numel",
                   "phys2_aniso_abs_max_running", "phys3_aniso_abs_max_running",
                   "phys2_max_abs_a_minus_1", "phys3_max_abs_a_minus_1",
                   "aniso_trace_len"):
            if _v(_k) is None:
                return "ANISO_READOUT_ERROR"
        if int(_v("aniso_trace_len")) == 0:
            return "ANISO_READOUT_ERROR"

        if max(float(_v("phys2_aniso_abs_max_running")),
               float(_v("phys3_aniso_abs_max_running"))) < float(ANISO_FROZEN_MAX):
            return "ANISO_PARAM_FROZEN"
        if max(float(_v("phys2_max_abs_a_minus_1")),
               float(_v("phys3_max_abs_a_minus_1"))) < float(ANISO_DELIVER_MIN):
            return "ANISO_NOT_DELIVERED"
        return "ANISO_DELIVERED"
    except Exception:
        return "ANISO_READOUT_ERROR"


# =============================================================================
# [v97] 磁盘写盘三防线  ——  本版唯一一根轴（守卫，不是机制；训练配方零改动）
#
# ★ 起因（2026-10-16 v96 实跑事故，跑了 129/180 epoch 后 core dump）：
#     训练机磁盘写满，ENS 存 ckpt 时 torch.save 中途 `OSError: [Errno 28]`。
#     torch/serialization.py:379 的 except 捕获它，然后 L380 `return`；
#     离开 with 时 __exit__ 调 write_end_of_file()，二次抛
#       RuntimeError: [enforce fail at inline_container.cc:319]
#                     unexpected pos 33527424 vs 33527320   (差 104 B)
#     该异常从 PyTorchStreamWriter 的**析构函数**抛出，是 C++ 层 c10::Error，
#     Python 的 try/except **根本拦不住** ⇒ terminate ⇒ Aborted (core dumped)
#     ⇒ 整跑归零，**连 JSON 都没有**。
#
# ★★★ 由此得到本版最关键的一条认识：**真正的防线必须在写之前。**
#     写盘失败后再兜底是没有用的 —— C++ abort 不受 Python 异常机制管辖。
#     所以三道防线按可靠性排序，防线 1 才是真防线，2/3 处理的是"拦得住的那部分"。
#     不把 try/except 说成能拦 core dump，那是假的安全感（覆辙 267）。
#
#   防线 1（真防线）：用 os.statvfs 预检可用空间，低于"需要 + 水位"就**不写**，
#                     跳过并计数。磁盘紧张时宁可少几个 ENS ckpt，也不让整跑归零。
#   防线 2：原子写 —— 先写 "<path>.tmp"，再 os.replace（同 fs 内 rename 是原子的）。
#           保证目标文件要么不存在、要么是完整的，不留半截 zip。
#   防线 3：Python 层异常兜底 + 计数落 JSON + stdout 打印。
#           以及：主结果 JSON 写失败时把完整内容打到 stdout（覆辙 279 的双通道）。
# =============================================================================
DISK_MIN_FREE_BYTES = int(1.5 * (1024 ** 3))   # 警戒水位：可用空间低于此就不再写
DISK_HEADROOM       = 1.25                     # state_dict raw bytes -> 实际占盘估算系数
DISK_MAX_EVENTS     = 20                       # JSON 里最多保留的事件条数

DISK_GUARD = {
    "writes_ok": 0,        # 成功写盘次数
    "writes_skipped": 0,   # 因空间不足主动跳过次数（★ 这是"防线 1 生效"的直接证据）
    "writes_failed": 0,    # 写了但失败次数
    "remove_failed": 0,    # 淘汰旧 ckpt 时删除失败次数（幽灵文件计数）
    "min_free_gb": float("inf"),   # 全程见过的最低可用空间
    "events": [],
}


def _disk_free_bytes(path):
    """目标路径所在文件系统的**非特权可用**字节数。

    取 f_bavail（非 root 可用）而不是 f_bfree —— 后者会把 root 保留块算进来，
    训练机通常以普通用户跑，f_bfree 会给出一个拿不到的乐观数字。
    """
    try:
        d = path if os.path.isdir(path) else (os.path.dirname(path) or ".")
        st = os.statvfs(d)
        return int(st.f_bavail) * int(st.f_frsize)
    except Exception as _e:
        DISK_GUARD["events"].append("STATVFS_ERROR %s: %s" % (type(_e).__name__, _e))
        return -1


def _approx_state_bytes(sd):
    """state_dict 落盘后大约占多少字节（raw bytes * 余量系数）。"""
    try:
        return int(sum(int(v.numel()) * int(v.element_size()) for v in sd.values()) * DISK_HEADROOM)
    except Exception:
        return 0


def _disk_note_free(free_b):
    """记录全程最低水位读数。"""
    if free_b is None or free_b < 0:
        return
    gb = free_b / float(1024 ** 3)
    if gb < DISK_GUARD["min_free_gb"]:
        DISK_GUARD["min_free_gb"] = round(gb, 3)


def _disk_push_event(msg):
    DISK_GUARD["events"].append(msg)
    if len(DISK_GUARD["events"]) > DISK_MAX_EVENTS:
        DISK_GUARD["events"] = DISK_GUARD["events"][-DISK_MAX_EVENTS:]


def _atomic_torch_save(obj, path):
    """防线 2：先写 .tmp 再 os.replace（同一文件系统内 rename 是原子的）。

    注意：这个函数**不能**防止 core dump —— 若真在磁盘写满的瞬间调用 torch.save，
    abort 仍会发生。它的职责是更弱但真实的一件事：让目标路径永远不出现半截文件。
    """
    tmp = path + ".tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
        return True, None
    except Exception as _e:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)      # 清掉半截 tmp；清不掉也不致命（下次会覆盖）
        except Exception:
            pass
        return False, "%s: %s" % (type(_e).__name__, _e)


def _safe_save(state, path, tag):
    """三防线合一的**唯一写盘入口**。返回 True/False；**永不抛出**（覆辙 206
    的推广：守卫自己也不许炸掉训练）。

    注意 ckpt 语义：调用方必须依据本函数的返回值决定是否把它记进 ckpt_pool ——
    没写成的东西不许进 pool，否则 ensemble 阶段会去 load 一个不存在的文件。
    """
    try:
        free_b = _disk_free_bytes(path)
        _disk_note_free(free_b)
        need_b = _approx_state_bytes(state)
        if free_b >= 0 and free_b < (need_b + DISK_MIN_FREE_BYTES):
            # ★ 防线 1：这里才是真正拦住本次事故的那一道。
            DISK_GUARD["writes_skipped"] += 1
            _disk_push_event("SKIP %s need=%.1fMB free=%.2fGB water=%.2fGB"
                             % (tag, need_b / 1048576.0, free_b / float(1024 ** 3),
                                DISK_MIN_FREE_BYTES / float(1024 ** 3)))
            flush_print("  [disk] 跳过写盘 %s：可用 %.2fGB < "
                        "需要 %.1fMB + 水位 %.2fGB"
                        % (tag, free_b / float(1024 ** 3), need_b / 1048576.0,
                           DISK_MIN_FREE_BYTES / float(1024 ** 3)))
            return False
        ok, err = _atomic_torch_save(state, path)
        if ok:
            DISK_GUARD["writes_ok"] += 1
            return True
        DISK_GUARD["writes_failed"] += 1
        _disk_push_event("FAIL %s %s" % (tag, err))
        flush_print("  [disk] 写盘失败但训练继续: %s -> %s" % (tag, err))
        return False
    except Exception as _e:
        DISK_GUARD["writes_failed"] += 1
        _disk_push_event("GUARD_ERROR %s %s: %s" % (tag, type(_e).__name__, _e))
        return False


def _safe_readout(fn, fallback):
    """[v99] 后置读数的**装配守卫**（覆辙 279 的第 3 件，v98 就是栽在这件上）。

    ★ 为什么需要它：`_safe_write_json` 只防**磁盘**失败，防不住"读数装配自己抛异常"。
      v98 跑满 180 轮之后，收尾的一个 NameError 把整份 JSON 带走 ⇒ 7.5h 机时归零。
      ⇒ 后置读数一律走这里：装配挂掉 ⇒ 打印 + 用 fallback 顶上，**主结果不受影响**。

    ★ fallback 语义：它必须是"诚实表示没读到"的值（{} / None / ERROR 判词），
      不能是一个看起来正常的假读数（覆辙 267：假绿比没有更危险）。
    """
    try:
        return fn()
    except Exception as _e:
        try:
            flush_print("  [v99] ⚠ 读数装配失败（主结果不受影响）: %s: %s"
                        % (type(_e).__name__, _e))
        except Exception:
            pass
        return fallback


def _safe_write_json(results_obj, path):
    """★ 覆辙 279 的磁盘版：**主结果不许被写盘失败毁掉**。

    与普通 ckpt 的区别：JSON 是本次交付的唯一读数载体，写不出来就等于这一跑白跑。
    所以这里除了同样的预检 + 原子写，还多一条 ——
    失败时把**完整的 JSON 内容打到 stdout**（双通道），至少留得下读数。
    """
    try:
        payload = json.dumps(results_obj, ensure_ascii=False, indent=2)
    except Exception as _e:
        flush_print("  [disk] JSON 序列化失败(%s: %s)，"
                    "主结果无法落盘。" % (type(_e).__name__, _e))
        return False
    need_b = int(len(payload.encode("utf-8")) * 2 + 1048576)
    try:
        free_b = _disk_free_bytes(path)
        _disk_note_free(free_b)
        if free_b >= 0 and free_b < need_b:
            DISK_GUARD["writes_skipped"] += 1
            _disk_push_event("SKIP json need=%.1fMB free=%.2fGB"
                             % (need_b / 1048576.0, free_b / float(1024 ** 3)))
            flush_print("  [disk] ★ 主结果 JSON 未落盘"
                        "（空间不足），"
                        "完整内容打在下方 stdout：")
            flush_print(payload)
            return False
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, path)
        DISK_GUARD["writes_ok"] += 1
        return True
    except Exception as _e:
        DISK_GUARD["writes_failed"] += 1
        _disk_push_event("FAIL json %s: %s" % (type(_e).__name__, _e))
        flush_print("  [disk] ★ 主结果 JSON 写盘失败，"
                    "完整内容打在下方 stdout：")
        flush_print(payload)
        return False


def _disk_verdict_now():
    """判词必须是函数（覆辙 263），不许在收尾处写字面量。

    DISK_ALL_WRITES_OK          —— 全部写盘成功，这跑的 ckpt/JSON 是完整的
    DISK_WRITE_SKIPPED_DEGRADED —— 有过主动跳过：跑完了，但 ckpt 不全（判读须知）
    DISK_WRITE_FAILED_DEGRADED  —— 有过真失败：更严重，但至少没 core dump
    """
    n_skip = int(DISK_GUARD.get("writes_skipped") or 0)
    n_fail = int(DISK_GUARD.get("writes_failed") or 0)
    if n_skip == 0 and n_fail == 0:
        return "DISK_ALL_WRITES_OK"
    if n_fail > 0:
        return "DISK_WRITE_FAILED_DEGRADED"
    return "DISK_WRITE_SKIPPED_DEGRADED"


def _disk_initial_report():
    """开机汇报一次可用空间 —— 这类读数的价值在于**崩之前**就能看见趋势。"""
    try:
        _target = OUT_DIR if "OUT_DIR" in globals() else "."
        _fb = _disk_free_bytes(_target)
        _disk_note_free(_fb)
        if _fb >= 0:
            flush_print("[v97 disk] 开跑时可用空间 = %.2f GB"
                        "  (警戒水位 %.2f GB)"
                        % (_fb / float(1024 ** 3), DISK_MIN_FREE_BYTES / float(1024 ** 3)))
            if _fb < DISK_MIN_FREE_BYTES * 4:
                flush_print("[v97 disk] ★★ WARNING: 可用空间"
                            "已低于水位的 4 倍，"
                            "跑到一半很可能触发"
                            "跳过/失败 —— 建议先清盘。")
    except Exception as _e:
        flush_print("[v97 disk] 初始空间读数失败"
                    "（不影响训练）: %s" % (type(_e).__name__,))



def _v96_self_sha1():
    """[v96] 运行时算自身 sha1（覆辙 248：版本出处不许由文件自己声明）。"""
    try:
        _h = hashlib.sha1()
        with open(os.path.abspath(__file__), "rb") as _f:
            for _chunk in iter(lambda: _f.read(1 << 20), b""):
                _h.update(_chunk)
        return _h.hexdigest()
    except Exception as _e:                                   # pragma: no cover
        return "SELF_SHA1_ERROR: %s: %s" % (type(_e).__name__, _e)


SELF_SHA1 = _v96_self_sha1()


def _dft_axis(N, device=None, dtype=None):
    """[v96] fftshift 之后第 i 个 bin 的真实角频率：-pi + 2*pi*i/N，i = 0..N-1。

    ★★ 覆辙 275：torch.linspace **没有** endpoint 参数（那是 numpy.linspace 的 API）。
       v92 写 `torch.linspace(-pi, pi, N, endpoint=False)` ⇒ 训练机在第 0 个 epoch
       之前就崩（TypeError: linspace() got an unexpected keyword argument 'endpoint'）。
       ⇒ 用 arange + 仿射变换表达同一数学对象（覆辙 275 的修法，已在 v93 验证）。

    ★★ 覆辙 273：旧写法 `torch.linspace(-pi, pi, N)` 的 step 是 2*pi/(N-1)，
       而 N 个 DFT bin 均分 2*pi ⇒ 真实步长是 2*pi/N（半开区间）。
       除两端外，**每个 bin 拿到的 |k| 都偏大**，核被系统性高估衰减。
       且 -pi 与 +pi 是同一个 bin 却出现两次 ⇒ Nyquist 重复。

    ★★ 覆辙 253：DC（k=0）在新网格上**精确**落在 index N//2；旧网格上该处
       是 ±pi/(N-1) 量级的非零值 ⇒ 常数场不能原样通过（DC 增益 ≠ 1）。
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    _step = 2.0 * math.pi / float(N)
    _idx = torch.arange(int(N), device=device, dtype=dtype)
    return _idx * _step - math.pi


def _dft_grid_new(N, device=None, dtype=None):
    """[v96] 生产网格表达式（**与 forward 逐字同式**），供自测复用。

    ★ 自测必须测**生产代码本身**：生产用 `_dft_axis(...)`，自测就调同一个函数
      （覆辙 260：单一真相源 —— 不许在自测里另写一份"看起来一样"的表达式）。
    """
    return _dft_axis(int(N), device, dtype)


def _dft_grid_old(N, device=None, dtype=None):
    """[v96] 旧网格（v9–v95 的写法）。**只**用于负向对照与核差读数，**不进生产**。

    部署检查器会断言：全文件 `torch.linspace(-math.pi` 只出现这一次。
    """
    return torch.linspace(-math.pi, math.pi, int(N), device=device, dtype=dtype)


def _on_dev(m, dev):
    """[v102] 把模块搬到 dev（覆铣 #312：自测收了 device 参数就必须对**模块**也生效）。

    ★ 为什么必须"构造在 CPU、再 .to(dev)"，而不是直接在 dev 上构造：
      参数初始化（Conv2d 的 kaiming_uniform_ 等）消耗的是**张量所在设备**的
      RNG。在 CUDA 上直接构造会消耗 CUDA RNG，而 _det_probe_snapshot_rng
      只快照 CPU RNG ⇒ 还原覆盖不到 ⇒ 训练数据流被污染（跨跑不可比）。
      .to(dev) 本身是纯拷贝、**不消耗任何 RNG** ⇒ CPU 上构造再搬家，
      RNG 消耗量与 v101 逐位相同（部署自证 I 组守着站数多重集）。
    ★ dev=None 时原样返回 ⇒ v101 的 CPU 路径行为逐字不变。
    """
    return m.to(dev) if dev is not None else m


def _aniso_selftest(device=None):
    """[v101] 各向异性轴的算子级自测（开工前判，与精度无关）。

    ★ v90 的两条假红在本函数里被**校准**（不是被绕过）：
      · case1b 的 f32 饱和：tanh(ANISO_TANH_SAT_PARAM=10) 在 f32 下**精确**等于 1.0
        ⇒ a 精确等于 2.0。旧自测期望"a_hi 严格小于 2" ⇒ 必然假红（生成前已实测：
        tanh(9)=0.99999994 不饱和、tanh(10)=1.0 精确饱和）。新期望分两句：
        param=+4 时 a < 2（不饱和区间），param=+10 时 a == 2.0 **精确**（饱和是预期）。
      · case2 的等变：旧 DFT 网格上各向同性核也不等变（5.7e-2@8x8）⇒ 旧期望不成立。
        新期望在**修正网格**（_dft_axis）上测，容差 1e-5 = 实测 f32 噪声 2.4e-7 的 ~40x。
    ★ 不耗全局 RNG：构造 MaternDiffusion 会建 ModeController 的 Conv2d（耗 RNG）
      ⇒ 全程包在 _det_probe_snapshot_rng / _det_probe_restore_rng 里，
      case5 验证还原逐位相同（否则训练数据流与 v100 不同，跨跑不可比）。
    ★ 探针必须能红（覆铣 227/259）：case0d / case2c 是两条负向对照。
    ★ forward **非纯**（推进 running_* 缓冲，覆铣 56/221）⇒ case2 每次调用前把
      named_buffers 快照还原，测的是"同一状态下的函数映射"，不是状态演化。
    ★ [v102] device 一致性（覆铣 #312）：本函数的 `device` 参数对**模块也生效**
      —— 所有 nn.Module 构造都过 `_on_dev(m, dev)`。v101 实跑在 CUDA 上，
      输入张量用 dev 建、模块留在 CPU ⇒ case2 起
      `RuntimeError: Input type (torch.cuda.FloatTensor) and weight type
      (torch.FloatTensor)`，15 条只跑 7 条。
    ★ [v102] all_pass 同防「case 红」与「提前崩」（覆铣 #312）：
      分母是**预期总条数** ANISO_SELFTEST_CASES，不是"崩溃前已跑条数"；
      另有 completed 标志 ⇒ 少跑一条即红，崩溃不再被吞成"7/7 ALL PASS"。
    """
    rep = {"ran": False, "error": None, "n_pass": 0, "n_total": 0,
           "checks": {}, "all_pass": False,
           "completed": False, "n_expected": int(ANISO_SELFTEST_CASES)}

    def chk(name, cond, msg=""):
        rep["n_total"] += 1
        ok = bool(cond)
        if ok:
            rep["n_pass"] += 1
        rep["checks"][name] = {"pass": ok, "msg": msg}
        return ok

    def _rng_equal(a, b):
        if a is None or b is None:
            return a is b
        if not torch.equal(a[0], b[0]):
            return False
        if a[1][0] != b[1][0] or not np.array_equal(a[1][1], b[1][1]) or a[1][2:] != b[1][2:]:
            return False
        if a[2] != b[2]:
            return False
        if (a[3] is None) != (b[3] is None):
            return False
        if a[3] is not None:
            return all(torch.equal(x, y) for x, y in zip(a[3], b[3]))
        return True

    _st = None
    try:
        _st = _det_probe_snapshot_rng()
        rep["ran"] = True
        dev = device

        # ---- case0 第 0 步核与 v100 逐位相同（严格推广）----
        #   (a) 表达式层面：a=1.0 精确时 kx_sq + 1.0*ky_sq 与 kx_sq + ky_sq 同操作数
        _ok_expr, _msg_expr = True, []
        for N in (8, 16, 32):
            _ky = _dft_axis(N, dev)
            _kx = _dft_axis(N, dev)
            _ky, _kx = torch.meshgrid(_ky, _kx, indexing="ij")
            _kxs, _kys = _kx**2, _ky**2
            _same = torch.equal(_kxs + 1.0 * _kys, _kxs + _kys)
            _ok_expr = _ok_expr and _same
            _msg_expr.append("N%d:%s" % (N, _same))
        chk("case0a_step0_expression_identical", _ok_expr, " ".join(_msg_expr))

        #   (b) 模块层面：zeros 初始化后 aniso_param 精确为 0 ⇒ a 精确为 1
        _m0 = _on_dev(MaternDiffusion(64, groups=8), dev)
        _nz = int(torch.count_nonzero(_m0.aniso_param.detach()).item())
        chk("case0b_aniso_init_exact_zero", _nz == 0,
            "aniso_param 非零元素数 = %d" % _nz)
        _a0 = 1.0 + ANISO_RANGE * torch.tanh(_m0.aniso_param.detach())
        chk("case0c_a_is_exactly_one_at_init",
            bool(torch.equal(_a0, torch.ones_like(_a0))),
            "a 与 1.0 的 max|diff| = %.3e" % float((_a0 - 1.0).abs().max()))

        #   (c) 负向对照：param 非零 ⇒ (b) 的等式必须被打破（否则 case0 是同义反复）
        with torch.no_grad():
            _m0.aniso_param.fill_(0.5)
        _a1 = 1.0 + ANISO_RANGE * torch.tanh(_m0.aniso_param.detach())
        chk("case0d_negctrl_nonzero_breaks_identity",
            bool(not torch.equal(_a1, torch.ones_like(_a1))),
            "param=0.5 ⇒ a=%.6f（必须 != 1）" % float(_a1[0]))

        # ---- case1 a 的值域 (0, 2] + f32 饱和文档化（v90 case2 假红的校准）----
        _m1 = _on_dev(MaternDiffusion(64, groups=8), dev)
        with torch.no_grad():
            _m1.aniso_param.fill_(4.0)
            _a_hi = 1.0 + ANISO_RANGE * torch.tanh(_m1.aniso_param)
            _m1.aniso_param.fill_(-4.0)
            _a_lo = 1.0 + ANISO_RANGE * torch.tanh(_m1.aniso_param)
            _m1.aniso_param.fill_(ANISO_TANH_SAT_PARAM)
            _a_sat = 1.0 + ANISO_RANGE * torch.tanh(_m1.aniso_param)
        chk("case1a_a_range_open_closed",
            bool(float(_a_lo.min()) > 0.0 and float(_a_hi.max()) < 2.0),
            "a(-4)=%.6f > 0, a(+4)=%.6f < 2" % (float(_a_lo.min()), float(_a_hi.max())))
        chk("case1b_f32_saturation_is_exact_two",
            bool(float(_a_sat.max()) == 2.0),
            "tanh(%.0f) 在 f32 下精确饱和 ⇒ a == 2.0 精确（%.17g）——这是**预期**；"
            "v90 旧自测期望 a 严格小于 2，恰好被它咬成假红"
            % (float(ANISO_TANH_SAT_PARAM), float(_a_sat.max())))

        # ---- case2 flip 等变（修正网格；每次调用前还原缓冲，见 docstring）----
        #   ★ 输入必须 **64 通道**：MaternDiffusion(64, groups=8) 的 forward 有两处
        #     硬性要求 C = groups*cpg = 64 —— mode_net 的 Conv2d(64→8) 与
        #     x.reshape(B, groups, cpg, H, W)。v90 的旧自测测的是**裸核公式**
        #     （_aniso_kernel，无模块 ⇒ 1 通道足够），v101 改成测**模块本身**，
        #     若沿用 1 通道输入会在开机自测第 7 条后崩
        #     （RuntimeError: expected input to have 64 channels）。
        #     该崩相由执行级自证 _cnn10_v101_exec_selftest.py 的 B 组抓到并修掉。
        _m2 = _on_dev(MaternDiffusion(64, groups=8), dev)
        _m2.eval()
        _buf = {n: b.detach().clone() for n, b in _m2.named_buffers()}
        _x = torch.arange(2 * 64 * 16 * 16, dtype=torch.float32,
                          device=dev).reshape(2, 64, 16, 16)
        _x = (_x / float(_x.max()) - 0.5) * 2.0     # 固定输入（arange 不耗 RNG）

        def _run(t):
            with torch.no_grad():
                for n, b in _m2.named_buffers():
                    b.copy_(_buf[n])
                return _m2(t, t, 0)[0]

        _p = _run(_x)
        _pf_w = _run(torch.flip(_x, dims=[3]))
        _pf_h = _run(torch.flip(_x, dims=[2]))
        _dw = float((_pf_w - torch.flip(_p, dims=[3])).abs().max())
        _dh = float((_pf_h - torch.flip(_p, dims=[2])).abs().max())
        chk("case2_flip_equivariant_both_axes",
            _dw < float(ANISO_EQUIV_TOL) and _dh < float(ANISO_EQUIV_TOL),
            "max|diff| W=%.3e H=%.3e (tol=%.0e)" % (_dw, _dh, float(ANISO_EQUIV_TOL)))

        # 各向异性打开后仍须等变（核只含平方项 ⇒ 与 a 的取值无关）
        with torch.no_grad():
            _m2.aniso_param.copy_(torch.linspace(-0.7, 0.7, _m2.groups))
        _pa = _run(_x)
        _paf_w = _run(torch.flip(_x, dims=[3]))
        _daw = float((_paf_w - torch.flip(_pa, dims=[3])).abs().max())
        chk("case2b_equivariant_with_anisotropy_on",
            _daw < float(ANISO_EQUIV_TOL),
            "aniso 打开后 max|diff| W=%.3e" % _daw)

        # 负向对照：拿**错轴**的翻转做期望 ⇒ 必须 > 容差（输出全零也能"通过"等变，
        #   这条对照就是堵那个洞 —— 覆铣 267）
        _dwrong = float((_pf_w - torch.flip(_p, dims=[2])).abs().max())
        chk("case2c_negctrl_wrong_flip_fails",
            _dwrong > float(ANISO_EQUIV_TOL),
            "W-翻转的输出 vs H-翻转的期望：max|diff| = %.3e（必须 > tol）" % _dwrong)

        # ---- case3 梯度能回流（优化器推得动；v57 形态的防线）----
        #   同样必须 64 通道（理由见 case2 上方的注释）。
        _m3 = _on_dev(MaternDiffusion(64, groups=8), dev)
        _x3 = torch.arange(1 * 64 * 16 * 16, dtype=torch.float32,
                           device=dev).reshape(1, 64, 16, 16)
        _x3 = (_x3 / float(_x3.max()) - 0.5) * 2.0     # 固定输入（arange 不耗 RNG）
        _p3, _sm3, _, _, _ = _m3(_x3, _x3, 0)
        (_p3.sum() + _sm3).backward()
        _g = _m3.aniso_param.grad
        chk("case3_grad_reaches_aniso_param",
            _g is not None and float(_g.abs().max()) > 0.0,
            "grad max = %s" % ("None" if _g is None else "%.3e" % float(_g.abs().max())))

        # ---- case4 轴只开 Matern 一支：FixedDiffusion 没有 aniso_param ----
        _f4 = _on_dev(FixedDiffusion(64, groups=8), dev)
        chk("case4_fixed_diffusion_has_no_aniso",
            not hasattr(_f4, "aniso_param"),
            "FixedDiffusion 不带方向自由度（本轴只开 Matern 两支）")
        _m4a = _on_dev(MaternDiffusion(512, groups=64), dev)   # phys2 形态
        _m4b = _on_dev(MaternDiffusion(1024, groups=8), dev)   # phys3 形态
        chk("case4b_both_matern_shapes_have_aniso",
            hasattr(_m4a, "aniso_param") and hasattr(_m4b, "aniso_param")
            and int(_m4a.aniso_param.numel()) == 64
            and int(_m4b.aniso_param.numel()) == 8,
            "phys2 形态 64 个 / phys3 形态 8 个")

        # ---- case6 torch.zeros 不推进 RNG（跨版初始化序列逐位相同的直接证据）----
        torch.manual_seed(12345)
        _before = torch.get_rng_state().clone()
        _ = torch.zeros(64)
        chk("case6_zeros_does_not_advance_rng",
            bool(torch.equal(_before, torch.get_rng_state())),
            "RNG state before/after torch.zeros(64) 逐位相同")

        # ---- case7 判据阶梯的阈值次序（FROZEN_MAX 必须 < DELIVER_MIN）----
        chk("case7_ladder_thresholds_ordered",
            bool(float(ANISO_FROZEN_MAX) < float(ANISO_DELIVER_MIN)),
            "FROZEN_MAX=%.0e < DELIVER_MIN=%.0e" % (float(ANISO_FROZEN_MAX),
                                                    float(ANISO_DELIVER_MIN)))
    except Exception as e:
        rep["error"] = "%s: %s" % (type(e).__name__, e)
    finally:
        try:
            if _st is not None:
                _det_probe_restore_rng(_st)
        except Exception as e:
            rep["error"] = ((rep.get("error") or "") + " | restore failed: %s: %s"
                            % (type(e).__name__, e))

    # ---- case5 RNG 还原验证（放在 finally 之后：验证 restore 真的生效了）----
    try:
        _st2 = _det_probe_snapshot_rng()
        chk("case5_rng_restored_bitwise", _rng_equal(_st, _st2),
            "进入自测时的 RNG 状态 vs 还原后：逐位相同")
    except Exception as e:
        chk("case5_rng_restored_bitwise", False, "case5 本身崩了: %s" % e)

    # [v102] 覆铣 #312：completed/all_pass 的分母是**预期总条数**，不是崩溃前条数。
    #   v101 实跑：CUDA 上 15 条只跑 7 条，旧公式 ran and n_total>0 and
    #   n_pass==n_total 照样报"7/7 ALL PASS" ⇒ 假绿。新公式少跑一条即红。
    rep["completed"] = bool(rep["ran"] and rep["error"] is None
                            and rep["n_total"] == int(ANISO_SELFTEST_CASES))
    rep["all_pass"] = bool(rep["completed"] and rep["n_pass"] == rep["n_total"])
    return rep


def _dft_grid_selftest(device=None):
    """[v96] 频率标尺的算子级自测（8 条，其中 **2 条是负向对照**）。

    ★ 判据方向：flip 等变 + DC 增益是**开工前就能判**的（与精度无关）；
      精度判据是跨跑的（recipe 域改动，第 0 步就不同）。
    ★ 全程不构造 nn.Module、**不消耗全局 RNG**（用自己的 generator，case6 验证）
      ⇒ RNG 序列与 v51 共享 ⇒ 跨跑仍可比。
    ★ [v102] 覆铣 #312（与 _aniso_selftest 同款）：rep 带 completed 标志，
      分母是预期总条数 DFT_SELFTEST_CASES ⇒ 崩了/少跑一条即红。
    """
    rep = {"ran": False, "error": None, "n_pass": 0, "n_total": 0,
           "checks": {}, "all_pass": False,
           "completed": False, "n_expected": int(DFT_SELFTEST_CASES)}

    def chk(name, cond, msg=""):
        rep["n_total"] += 1
        ok = bool(cond)
        if ok:
            rep["n_pass"] += 1
        rep["checks"][name] = {"pass": ok, "msg": msg}
        return ok

    try:
        rep["ran"] = True
        dev = device

        # ---- case0 新网格与 2*pi*fftfreq 是**同一频率集合**（半开区间，Nyquist 只一次）----
        ok0, m0 = True, []
        for N in (8, 16, 32):
            g = _dft_grid_new(N, dev)
            f = torch.sort(2.0 * math.pi
                           * torch.fft.fftfreq(int(N), device=g.device,
                                               dtype=g.dtype)).values
            d0 = float((torch.sort(g).values - f).abs().max().item())
            m0.append("N%d:%.2e" % (N, d0))
            ok0 = ok0 and (d0 < 1e-5)
        chk("case0_matches_dft_frequencies", ok0, " ".join(m0))

        # ---- case1 DC 精确落在 index N//2 且值为 0（旧网格该处不是 0）----
        ok1, m1 = True, []
        for N in (8, 16, 32):
            v_new = float(_dft_grid_new(N, dev)[int(N) // 2].abs().item())
            v_old = float(_dft_grid_old(N, dev)[int(N) // 2].abs().item())
            m1.append("N%d new=%.3e old=%.3e" % (N, v_new, v_old))
            ok1 = ok1 and (v_new < 1e-6)
        chk("case1_dc_bin_is_zero", ok1, " ".join(m1))

        # ---- 生产同式的二维核 + 滤波管线（与 MaternDiffusion.forward 逐字同式）----
        def _k2(gy, gx, lam, nu, s):
            KY, KX = torch.meshgrid(gy, gx, indexing="ij")
            return (1.0 + lam * (KX ** 2 + KY ** 2) / nu) ** (-s)

        def _filt(x, k_sq):
            # ★ 与 v51 生产路径**逐字同式**：fftshift 未指定 dim（v51 形状是 (B,1,H,W)）
            f = fftshift(fft2(x))
            return torch.real(ifft2(ifftshift(f * k_sq)))

        Hh = Ww = 16
        s_v, l_v, n_v = 0.8, 2.6, 1.7
        K_new = _k2(_dft_grid_new(Hh, dev), _dft_grid_new(Ww, dev), l_v, n_v, s_v)
        K_old = _k2(_dft_grid_old(Hh, dev), _dft_grid_old(Ww, dev), l_v, n_v, s_v)

        # ★ 用自己的 generator：绝不碰全局 RNG（否则跨跑 RNG 序列与 v51 不再共享）
        _g = torch.Generator()
        _g.manual_seed(2026096)
        xg = torch.randn(2, 1, Hh, Ww, generator=_g).to(dev)
        ones_in = torch.ones(2, 1, Hh, Ww, device=dev)

        # ---- case2/case3 翻转 / 转置等变（核是 |k|^2 的各向同性函数 ⇒ 严格成立）----
        with torch.no_grad():
            y = _filt(xg, K_new)
            yf = _filt(torch.flip(xg, dims=[-1]), K_new)
            d2 = float((torch.flip(y, dims=[-1]) - yf).abs().max().item())
            yt = _filt(xg.transpose(-1, -2), K_new)
            d3 = float((y.transpose(-1, -2) - yt).abs().max().item())
        chk("case2_flip_equivariant", d2 < DFT_FLIP_TOL,
            "max|diff|=%.3e (阈 %.0e)" % (d2, DFT_FLIP_TOL))
        chk("case3_transpose_equivariant", d3 < DFT_FLIP_TOL,
            "max|diff|=%.3e (阈 %.0e)" % (d3, DFT_FLIP_TOL))

        # ---- case4 DC 增益 = 1（核在 k=0 处为 1 ⇒ 常数场原样通过）----
        with torch.no_grad():
            d4 = float((_filt(ones_in, K_new) - 1.0).abs().max().item())
        chk("case4_dc_gain_is_one", d4 < DFT_DC_TOL,
            "max|dev from 1|=%.3e (阈 %.0e)" % (d4, DFT_DC_TOL))

        # ---- case5/case6 ★★ 负向对照：上面两条判据必须**有能力变红** ----
        #   在**旧网格**上重跑同一条判据 ⇒ 必须 >= DFT_NEGCTRL_MIN。
        #   否则 case2/case4 就是同义反复（覆辙 267：永远绿的检查比没有更危险）。
        with torch.no_grad():
            yo = _filt(xg, K_old)
            yof = _filt(torch.flip(xg, dims=[-1]), K_old)
            d5 = float((torch.flip(yo, dims=[-1]) - yof).abs().max().item())
            d6 = float((_filt(ones_in, K_old) - 1.0).abs().max().item())
        chk("NEGCTRL_case5_old_grid_breaks_flip", d5 >= DFT_NEGCTRL_MIN,
            "旧网格 max|diff|=%.3e (须 >= %.0e)" % (d5, DFT_NEGCTRL_MIN))
        chk("NEGCTRL_case6_old_grid_breaks_dc", d6 >= DFT_NEGCTRL_MIN,
            "旧网格 max|dev from 1|=%.3e (须 >= %.0e)" % (d6, DFT_NEGCTRL_MIN))

        # ---- case7 自测不消耗全局 RNG ----
        _s0 = torch.get_rng_state().clone()
        _dft_grid_new(16, dev)
        _dft_grid_old(16, dev)
        chk("case7_no_rng_consumed", bool(torch.equal(_s0, torch.get_rng_state())),
            "arange / 独立 generator ⇒ 全局 RNG 不变 ⇒ 跨跑序列与 v51 共享")

        # ---- 读数：recipe 域改动的量级（落盘可查）----
        rep["kernel_max_abs_diff"] = float((K_new - K_old).abs().max().item())
        rep["grid_step_new"] = 2.0 * math.pi / float(Hh)
        rep["grid_step_old"] = 2.0 * math.pi / float(Hh - 1)
        rep["flip_diff_new"] = d2
        rep["flip_diff_old"] = d5
        rep["dc_dev_new"] = d4
        rep["dc_dev_old"] = d6
    except Exception as _e:                                   # pragma: no cover
        rep["error"] = "%s: %s" % (type(_e).__name__, _e)
        return rep
    # [v102] 覆铣 #312：completed/all_pass 移到 try 之外算。
    #   旧公式在 try 内末尾 ⇒ 崩了会 early-return 侥幸安全；但它的分母仍是
    #   "崩溃前已跑条数" ⇒ 与 aniso 自测同款假绿风险 ⇒ 统一改成预期总条数。
    rep["completed"] = bool(rep["ran"] and rep["error"] is None
                            and rep["n_total"] == int(DFT_SELFTEST_CASES))
    rep["all_pass"] = bool(rep["completed"] and rep["n_pass"] == rep["n_total"])
    return rep


def _dft_grid_verdict_now(selftest):
    """[v96] 本轴判词（必须是函数 —— 覆辙 263）。

    ★ 方向要说清：自测 8 条全绿 = 频率标尺已正确（flip 等变 + DC 增益），
      这是**开工前就能判**的，与精度无关。精度判据是跨跑的（recipe 域改动）。
    """
    st = selftest or {}
    if not st.get("ran"):
        return "DFT_SELFTEST_ERROR"
    if st.get("error"):
        return "DFT_SELFTEST_ERROR"
    if not st.get("all_pass"):
        return "DFT_SELFTEST_FAILED"
    return "DFT_DELIVERED"


def _v96_print_fingerprint():
    """[v96] 把 provenance 打在"按下回车之前就能看见"的地方（覆辙 279 的正面配套）。"""
    flush_print("=" * 60)
    _pfx = "[%s provenance]" % VERSION
    flush_print("%s file    = %s" % (_pfx, _V96_FILE_NAME_CHECK.get("file_name")))
    flush_print("%s sha1    = %s" % (_pfx, SELF_SHA1))
    flush_print("%s VERSION = %s" % (_pfx, VERSION))
    if not _V96_FILE_NAME_CHECK.get("ok"):
        flush_print("[%s] ★★ WARNING: 文件名不含 '%s' —— 跑的很可能是改名/复制的副本，"
                    "provenance 不可信（覆辙 266）。" % (VERSION, VERSION))
    _disk_initial_report()
    flush_print("=" * 60)


def _v96_banner_phase2_derived():
    """[v96] ★★ 覆辙 267：v51 的 `banner_phase2_end_derived` 是**写死的 True**。

    一个永远绿的检查比没有更危险 —— 它伪装成证据。这里换成真检查：
    读自己的源码，要求 (a) 至少有 1 行 flush_print 用 `{PHASE2_LR_END...}`
    插值（而非字面量），(b) 没有任何 flush_print 行把终点写成硬编码的
    '3.0e-5' / '3e-05' 之类字样。
    """
    try:
        with open(os.path.abspath(__file__), "r", encoding="utf-8") as _f:
            _lines = _f.readlines()
    except Exception as _e:                                   # pragma: no cover
        return {"ok": False, "n_interpolated": 0, "hardcoded_lines": [],
                "error": "%s: %s" % (type(_e).__name__, _e)}
    _interp = [(_i + 1) for _i, _l in enumerate(_lines)
               if ("flush_print" in _l) and ("PHASE2_LR_END" in _l) and ("{" in _l)]
    _hard = [(_i + 1, _l.rstrip()[:120]) for _i, _l in enumerate(_lines)
             if ("flush_print" in _l) and ("PHASE2_LR_END" not in _l)
             and (("e-05" in _l.lower()) or ("e-5" in _l.lower()))]
    return {"ok": bool(len(_interp) >= 1 and not _hard),
            "n_interpolated": int(len(_interp)), "hardcoded_lines": _hard,
            "error": None}


_V96_BANNER_PHASE2 = _v96_banner_phase2_derived()


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


# ====================== [v113] 约束通道：物理约束项 pinn 的作用面（本版唯一一根轴） ======================
#   一句话：**把 `pinn`（物理约束项）的作用点从「layer4 输出（含物理注入）」**
#       **移到「layer3 输出（物理注入之前）」** ⇒ 它的梯度**结构性地到不了物理参数**。
#
#   ★★ 立项依据（v111 + v112 实跑 + 读代码，2026-10-26）：
#     ① 【代码事实】`GradientStablePhysResNetV3.forward` 的 pinn 计算点：
#          x = layer3(x); ph3 = phys3(x, x, epoch); x = gate3(x, ph3)
#          x = layer4(x);  deep = x;  ...  pinn = mean(dx²) + mean(dy²)   ← deep 在 gate3 **之后**
#        而 `Gate.forward` 是 `feat + scale * a * bn(proj(phys))`
#        ⇒ **pinn 的梯度路径 = deep → gate3 → ph3 → phys3 的 s/lam/nu/aniso 参数**。
#     ② 【机制推论】于是 `pinn` 把"**注入之后**的最终特征要平滑"这个要求
#        **回传给了物理参数** ⇒ 优化器抬 l（场更平滑/更 DC）比让骨干变平滑更省
#        ⇒ **物理参数被自己的正则压成常数**（自强化回路）。
#     ③ 【v111 实测印证】band_frac_mean = 0.0968（场输出 **90% 能量仍在 DC**）、
#        raw_lambda_c_px = 7.627、raw_ratio_mean = 1.4736 ⇒ 场确实被压成了近常数。
#        ⇒ v111 判词 `BAND_DELIVERED` 而精度 NULL，并据此**归档加性注入通道**
#          （"场输出是骨干特征的滤波摹本 ⇒ 信息冗余"）。
#     ④ 【v113 的可证伪点】v111 的归档有一个**隐含前提**：场之所以是"摹本"，
#        是因为**物理参数被数据/损失持续塑形**（lam/nu 被推到与骨干特征匹配的位置）。
#        ★ 若切断 pinn→物理参数这条塑形路径，场就**不再**是摹本 ⇒
#          **注入通道可能复活** ⇒ v111 的归档结论就被证伪。
#        ⇒ 本版因此**不只**是一个机制改动，而是**对已归档结论的证伪性检验**。
#     ⑤ 【v112 实测，2026-10-26】视图群判负（10 视图 −0.18pp，角位比中心位 −0.4575pp）
#        ⇒ 读数侧视图族归档 ⇒ v113 回退到 v110 的训练态再换轴（回退优先）。
#
#   ★ 为什么"改作用点"而不是"改权重"：
#     改 PINN_WEIGHT 是**标量调整**（v47 lr clamp / v57 role_mix 一类，已整族判死），
#     且**切不断**梯度路径（场参数的梯度占比只按比例缩小，不归零）。
#     改**作用点**是**拓扑切断**：pinn 的计算图里根本不含 ph3 ⇒ 场参数梯度占比 ≡ 0。
#     ⇒ 这是同一族里唯一**结构上**能让"物理只作被动先验"真正成立的改法。
#
#   ★ 前向逐位不变：pinn 只进 loss（`PINN_WEIGHT * pinn_reg`），不进前向数值
#     ⇒ 同一权重下 logits **逐位相同** ⇒ 本版改的**只有梯度路径**。
#
#   ★ 读数自带"旧对照"：探针里同时算 `pinn` 的**新位置**（layer3 输出）
#     与**旧位置**（layer4 输出）的量级 ⇒ 口径变化自曝（不靠注释承诺）。
V113_PINN_FACE_GRAD_FRAC_MAX = 1e-6        # 场参数梯度占比上限（切断成功的判据）
V113_BAND_FRAC_MIN = 0.20                  # 场"带内能量占比"下限（场是否恢复内容）
V113_BASE_DELIVERED = 97.50                # 基线 = v110 delivered（同 seed 42，跑满 180）
V113_SIGMA_DIFF = 0.169                    # σ_diff（四对零改动跑 RMS）
V113_PASS_LINE = 97.838                    # = 97.50 + 2σ（纪律：B 线 PASS 边距 ≥ 2σ）
V113_REDLINE = 97.04                       # 回退红线（沿用 v110）
V113_BASELINE_BAND_FRAC = 0.0968           # v111 实测：场 90% 能量在 DC
V113_BASELINE_LAMBDA_C_PX = 7.627          # v111 实测：raw_lambda_c_px


def _v113_roughness(x):
    """[v113] 特征图的粗糙度（与生产 pinn 的定义逐字同式）：mean(dx²) + mean(dy²)。

    ★ 与生产同式 ⇒ 读数与 loss 项**同一把尺**（不是另造一个近似指标）。
    ★ 生产原式（v110 逐字）：
        dx = deep[..., 1:] - deep[..., :-1]        ← 沿最后一维（W）求差
        dy = deep[..., 1:, :] - deep[..., :-1, :]    ← 沿倒数第二维（H）求差
      ⇒ dy 的两个切片**最后一维必须相同** ⇒ 写成 [..., 1:, :] 与 [..., :-1, :]
      （★ 写成 [..., 1:, :-1] 会把最后一维也切掉 ⇒ 形状不匹配 ⇒ 自测直接炸）
    """
    dx = x[..., 1:] - x[..., :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return torch.mean(dx ** 2) + torch.mean(dy ** 2)


def _v113_band_frac(phys):
    """[v113] 场的"带内能量占比"（1 − DC 占比）：常数场 ⇒ 0；棋盘格 ⇒ 1。

    ★ 与 v111 的 band_frac 同一口径（DC 分量 = 空间均值那一项）。
      v111 实测 raw band_frac_mean = 0.0968 ⇒ 场 90% 能量在 DC（近常数）。
    """
    _f = phys.reshape(phys.shape[0], -1).float()
    _dc = _f.mean(dim=1)
    _tot = _f.pow(2).sum(dim=1) + 1e-12
    _dcp = _dc.pow(2) * _f.shape[1]
    _frac = 1.0 - (_dcp / _tot)
    return float(torch.clamp(_frac, 0.0, 1.0).mean().item())


def _v113_grad_split_probe(x3, ph3, scale):
    """[v113] **梯度分流探针**（纯张量，零模型零 RNG）：pinn 换作用点后，
    场参数还收得到梯度吗？

    模拟生产拓扑（**注入在特征流上**）：
      deep = x3 + scale * ph3        （= gate 的注入：feat + scale*a*p 的形状）
      pinn_old = roughness(deep)      （旧作用点：gate3 之后 ⇒ 能到 ph3）
      pinn_new = roughness(x3)        （新作用点：gate2 之前 ⇒ 到不了 ph3）

    返回 (frac_new, frac_old)：场张量 ph3 在两条路径上的梯度占比
    （占 ph3 自身梯度范数的比例；★ 用**占位比例**而不是绝对值 ⇒ 与 scale 无关）。

    ★ 这是本轴**机制交付的直接证据**：frac_new 应当 ≡ 0（拓扑切断），
      而 frac_old 应当 > 0（旧路径确实有梯度 ⇒ 探针没有瞎）。
    ★ ★ 真模型上还有**更细**的一层（实测发现）：作用点在 layer3 输出处时，
      梯度会经 layer3→layer2→gate2→ph2 **漏回** phys2 ⇒ 切断不彻底；
      故生产侧取 layer2 输出（gate2 之前）⇒ 两个场都断开。
    """
    _s = float(scale)
    deep = x3 + _s * ph3
    _pn_new = _v113_roughness(x3)
    _pn_old = _v113_roughness(deep)
    _g_new = torch.autograd.grad(_pn_new, ph3, retain_graph=True, allow_unused=True)[0]
    _g_old = torch.autograd.grad(_pn_old, ph3, retain_graph=True, allow_unused=True)[0]
    _den = float(torch.sqrt((ph3.detach() ** 2).sum()).item()) + 1e-12

    def _frac(g):
        if g is None:
            return 0.0
        return float(torch.sqrt((g ** 2).sum()).item()) / _den

    return _frac(_g_new), _frac(_g_old)


def _v113_pinface_verdict(grad_frac_new, grad_frac_old, band_frac, delivered,
                          selftest_ok=True, readout_ok=True):
    """[v113] 判词（五档）。门槛全部来自 V113_* 常量（预注册，运行时不得放宽）。

    ★ 分层判词：**机制层**与**精度层**分开 ⇒ 精度 NULL 时仍能区分
      "作用面没切掉" vs "切掉了但没用"（纪律 304：每档要有证据）。
    """
    if not selftest_ok:
        return "PINN_FACE_SELFTEST_FAILED"
    if not readout_ok:
        return "PINN_FACE_READOUT_ERROR"
    try:
        gn = float(grad_frac_new)
        go = float(grad_frac_old)
        bf = float(band_frac)
        dl = float(delivered)
    except Exception:
        return "PINN_FACE_READOUT_ERROR"
    # --- 机制层 ---
    if go <= 0.0:
        # 旧路径本来就没有梯度 ⇒ 探针瞎了（或拓扑本就不通）⇒ 轴空转
        return "PINN_FACE_NOT_BINDING"
    if gn > float(V113_PINN_FACE_GRAD_FRAC_MAX):
        # 切不掉 ⇒ 本版改动没生效
        return "PINN_FACE_NOT_BINDING"
    _cut = True
    _content = (bf >= float(V113_BAND_FRAC_MIN))
    # --- 精度层 ---
    if dl >= float(V113_PASS_LINE):
        return "PINN_FACE_DELIVERED"
    if dl <= float(V113_REDLINE):
        return "PINN_FACE_BACKTRACK"
    if _cut and _content:
        # 机制交付（切断成功 + 场恢复内容）但精度未到 PASS 线
        return "PINN_FACE_MECHANISM_ONLY"
    if _cut and not _content:
        # 切断成功但场仍是近常数 ⇒ 场被压成 DC 的主因不是 pinn 这条路
        return "PINN_FACE_CUT_NO_CONTENT"
    return "PINN_FACE_READOUT_ERROR"


def _v113_pinface_selftest(device=None):
    """[v113] 执行级自测（构造性、**零 RNG 消耗**、秒级、纯 CPU）。

    ★ 零 RNG 是硬要求：自测若消耗 RNG，训练轨迹就偏了。
    ★ 全程在 **CPU** 上做（不占 GPU、不碰 CUDA RNG、不碰全局 RNG）。
    ★ 分母固定：_NAMES 预先登记全部用例名 ⇒ n_expected 恒定（纪律 326）。
    """
    _NAMES = ("shape_ok", "grad_new_is_zero", "grad_old_is_nonzero",
              "grad_split_monotone", "roughness_const_is_zero",
              "roughness_matches_prod_formula", "band_frac_const_is_zero",
              "band_frac_checkerboard_is_one", "band_frac_monotone_in_content",
              "verdict_delivered", "verdict_mechanism_only",
              "verdict_cut_no_content", "verdict_not_binding", "verdict_probe_blind",
              "verdict_backtrack", "verdict_selftest_failed",
              "verdict_readout_error", "selftest_consumes_no_rng")
    _res = {}

    def _chk(name, ok, msg=""):
        _res[name] = {"name": name, "pass": bool(ok), "msg": str(msg)}

    st_before = torch.get_rng_state()
    _err = None
    try:
        # ---- 梯度分流（核心机制断言）----
        #  ★ 全部用**确定性**张量（linspace 网格 ⇒ 不消耗 RNG）⇒ 自测零 RNG 消耗
        #    （★ 用 torch.randn 会消耗全局 RNG ⇒ 既让自测必红，又污染训练轨迹）
        _g = torch.linspace(-1.0, 1.0, 8)
        _gy, _gx = torch.meshgrid(_g, _g, indexing="ij")
        _base = torch.stack([_gx, _gy], dim=0).unsqueeze(0).repeat(2, 1, 1, 1)
        _x3 = _base.clone().requires_grad_(True)
        _ph = (_base * 0.5).clone().requires_grad_(True)
        _fn, _fo = _v113_grad_split_probe(_x3, _ph, 0.9)
        _chk("shape_ok", isinstance(_fn, float) and isinstance(_fo, float),
             "fn=%r fo=%r" % (_fn, _fo))
        _chk("grad_new_is_zero", _fn <= float(V113_PINN_FACE_GRAD_FRAC_MAX),
             "新作用点（gate3 之前）对场参数的梯度占比必须 ≡ 0，实测 %.3e" % _fn)
        _chk("grad_old_is_nonzero", _fo > 1e-3,
             "旧作用点（gate3 之后）必须仍有梯度 ⇒ 探针没瞎，实测 %.6f" % _fo)
        _chk("grad_split_monotone", _fn < _fo,
             "新路径梯度占比必须小于旧路径：%.3e vs %.6f" % (_fn, _fo))
        # ---- roughness 与生产公式逐字同式 ----
        _z = torch.zeros(1, 3, 8, 8)
        _chk("roughness_const_is_zero", float(_v113_roughness(_z).item()) == 0.0,
             "常数场粗糙度必须为 0")
        _r = _v113_roughness(_x3)
        _dx = _x3[..., 1:] - _x3[..., :-1]
        _dy = _x3[..., 1:, :] - _x3[..., :-1, :]
        _ref = torch.mean(_dx ** 2) + torch.mean(_dy ** 2)
        _chk("roughness_matches_prod_formula",
             bool(torch.allclose(_r.detach(), _ref.detach(), atol=1e-8)),
             "roughness 必须与生产公式逐位同：%.8g vs %.8g"
             % (float(_r.detach()), float(_ref.detach())))
        # ---- band_frac 口径 ----
        _c = torch.ones(1, 4, 8, 8)
        _chk("band_frac_const_is_zero", abs(_v113_band_frac(_c)) < 1e-6,
             "常数场 band_frac 必须为 0，实测 %.6g" % _v113_band_frac(_c))
        # 棋盘格：**全局零均值**的高频场（每个通道 DC 必须为 0 ⇒ 带内占比 = 1）
        _cb = torch.zeros(1, 4, 8, 8)
        _cb[:, :, ::2, ::2] = 1.0
        _cb[:, :, 1::2, 1::2] = -1.0
        _chk("band_frac_checkerboard_is_one", abs(_v113_band_frac(_cb) - 1.0) < 1e-6,
             "零均值棋盘格 band_frac 必须为 1，实测 %.6g" % _v113_band_frac(_cb))
        # 内容量单调：DC 分量相同、带内分量递增 ⇒ band_frac 递增
        _mono = []
        for _k in (0, 1, 2, 3):
            _t = torch.zeros(1, 4, 8, 8)
            _t[:, :, ::2, ::2] = 1.0
            _t[:, :, 1::2, 1::2] = -1.0
            _mix = 3.0 + _t * float(_k)
            _mono.append(_v113_band_frac(_mix))
        _chk("band_frac_monotone_in_content",
             _mono[0] <= _mono[-1] + 1e-9,
             "内容越多 band_frac 越大：%r" % (_mono,))
        # ---- 判词五档 ----
        _chk("verdict_delivered",
             _v113_pinface_verdict(0.0, 0.5, 0.5, 97.90) == "PINN_FACE_DELIVERED",
             "切断成功 + 场有内容 + 精度过线 => DELIVERED")
        _chk("verdict_mechanism_only",
             _v113_pinface_verdict(0.0, 0.5, 0.5, 97.60) == "PINN_FACE_MECHANISM_ONLY",
             "切断成功 + 场有内容 + 精度未过线 => MECHANISM_ONLY")
        _chk("verdict_cut_no_content",
             _v113_pinface_verdict(0.0, 0.5, 0.05, 97.60) == "PINN_FACE_CUT_NO_CONTENT",
             "切断成功但场仍是近常数 => CUT_NO_CONTENT")
        _chk("verdict_not_binding",
             _v113_pinface_verdict(0.3, 0.5, 0.5, 97.90) == "PINN_FACE_NOT_BINDING",
             "切不掉 => NOT_BINDING")
        _chk("verdict_probe_blind",
             _v113_pinface_verdict(0.0, 0.0, 0.5, 97.90) == "PINN_FACE_NOT_BINDING",
             "旧路径也没梯度 ⇒ 探针瞎了 ⇒ NOT_BINDING（这条守住『负向对照』不能被删）")
        _chk("verdict_backtrack",
             _v113_pinface_verdict(0.0, 0.5, 0.5, 97.00) == "PINN_FACE_BACKTRACK",
             "精度触红线 => BACKTRACK")
        _chk("verdict_selftest_failed",
             _v113_pinface_verdict(0.0, 0.5, 0.5, 97.90, selftest_ok=False)
             == "PINN_FACE_SELFTEST_FAILED", "自测不过 => 读数不可信")
        _chk("verdict_readout_error",
             _v113_pinface_verdict(0.0, 0.5, 0.5, 97.90, readout_ok=False)
             == "PINN_FACE_READOUT_ERROR", "读数崩了 => 降级")
    except Exception as _e:
        _err = "%s: %s" % (type(_e).__name__, _e)
        for _n in _NAMES:
            _res.setdefault(_n, {"pass": False, "msg": "not reached"})
        _res[_NAMES[0]] = {"pass": False, "msg": "EXCEPTION " + _err}
    _chk("selftest_consumes_no_rng", bool(torch.equal(st_before, torch.get_rng_state())),
         "自测不得消耗全局 RNG（否则训练轨迹偏）")
    checks = []
    for _n in _NAMES:
        _r = _res.get(_n, {"pass": False, "msg": "not reached"})
        checks.append({"name": _n, "pass": bool(_r["pass"]), "msg": str(_r.get("msg", ""))})
    n_pass = 0
    for _c in checks:
        if _c["pass"]:
            n_pass += 1
    return {"n_pass": n_pass, "n_expected": len(_NAMES),
            "all_pass": bool(n_pass == len(_NAMES)),
            "completed": True, "error": _err, "checks": checks}


def _v113_pinface_selftest_cases_count():
    """[v113] 自测条数的**独立真值**：字面数源码里的 _chk( 调用点（纪律 326）。"""
    try:
        with open(os.path.abspath(__file__), "r", encoding="utf-8") as _f:
            _txt = _f.read()
    except Exception:
        return -1
    _i = _txt.find("def _v113_pinface_selftest(")
    _j = _txt.find("def _v113_pinface_selftest_cases_count(")
    if _i < 0 or _j < 0:
        return -1
    _seg = _txt[_i:_j]
    _n = 0
    for _ln in _seg.split(chr(10)):
        if "_chk(" in _ln and "def _chk" not in _ln:
            _n += 1
    return _n




# ====================== [v116] 探针读数修复（仪器层；训练数学逐字未改）======================
#   ★★ 本版**不改任何训练数学**：父本 cnn10_v115.py 的顶层函数/类 md5 零改动零删除，
#      仅新增下面这几个顶层函数（全是仪器）。
#
#   ★★ 背景：v113/v114/v115 **三次**都没测到「约束通道 · pinn 作用面」这根轴，
#      三次都是**探针自己崩**（顺序 bug / device 不匹配 / math domain error）。
#      v115 把探针**外部**的三处 P0（device / BN / OUT_DIR）全修好了（实测已确认：
#      probe_device=cuda:0、probe_align_ok=true、bn_restore_ok=true、prewrite_ok=true），
#      但探针**内部**还有 5 处缺陷（见文件头 D1~D5）⇒ 本版专修探针内部。

V116_READOUT_FIX_NOTE = (
    "v116 = v115 + 探针内部 5 处缺陷修复（D1 负数开方 / D2 levels 键假判负 / "
    "D3 grad_leaks 从未写入 / D4 sqrt+除法无保护 / D5 raw_ratio 口径漂移）。"
    "训练数学与全部计算**逐字零改动**（父本顶层函数/类 md5 零改动零删除）。"
    "★ 探针改为**分节降级**：三个读数各自独立 try ⇒ 任一段崩，其余照常交付。"
    "★ 探针**双写** levels 子键 + 顶层平铺 ⇒ 两种读法都对（根除 v114 引入的假判负回归）。"
    "★ 训练**开始前**跑一次探针自检 ⇒ 探针若崩立即暴露，不等 180ep。"
    "⇒ 预期 delivered 仍 ≈ 97.4（NULL，训练数学未动）；本版价值在**机制读数**。"
)

V116_V115_HISTORY = {
    "version": "v115",
    "delivered": 97.43,
    "precision_vs_v110": "-0.07pp = -0.41σ ⇒ NULL（非断崖）",
    "verdict": "PINN_FACE_READOUT_ERROR",
    "probe_ok": False,
    "probe_error": "ValueError: math domain error",
    "probe_device": "cuda:0",
    "probe_align_ok": True,
    "bn_restore_ok": True,
    "prewrite_ok": True,
    "disk_verdict": "DISK_ALL_WRITES_OK",
    "note": ("v115 的**外部**三处 P0 全部修好（device 对齐 / BN 还原 / OUT_DIR 早建），"
             "但探针**内部** L2751 的 math.sqrt(负数) 崩 ⇒ 机制读数**第三次**全丢。"
             "⇒ 那根轴**至今未被测到** ⇒ 未测 ≠ 判负 ⇒ v116 第四次重测同一根轴。"),
}


def _v116_probe_safe_sqrt(x):
    """[v116] 非负开方（根除 D4）。负数/NaN ⇒ 返回 None（**不抛**）。

    ★ 背景：v115 实跑崩在 `math.sqrt(raw_ratio_mean)`（D1）⇒ `ValueError: math domain error`
      ⇒ 崩在**梯度分流段之前** ⇒ 三个判据全 null。
    ★ 本函数是**探针内部**的唯一开方入口：负数 ⇒ None（并由调用方记 NaN 标记），
      **绝不抛** ⇒ 单点数值异常不再吃掉整轴读数。
    """
    try:
        _v = float(x)
    except Exception:
        return None
    if not (_v == _v):  # NaN
        return None
    if _v < 0.0:
        return None
    try:
        return math.sqrt(_v)
    except Exception:
        return None


def _v116_probe_safe_ratio(num, den, eps=1e-12):
    """[v116] 安全除法（根除 D4 的除零/NaN）。分母近零 ⇒ 返回 None。

    ★ 返回 (value_or_None, ok_flag)：调用点据此打 NaN 标记，**不抛**。
    """
    try:
        _n = float(num)
        _d = float(den)
    except Exception:
        return None, False
    if not (_n == _n) or not (_d == _d):
        return None, False
    _d2 = _d + eps
    if abs(_d2) <= eps:
        return None, False
    try:
        return (_n / _d2), True
    except Exception:
        return None, False


def _v116_probe_collect_leaks(grads, names, tol=0.0):
    """[v116] 收集**新作用点**下梯度非零的物理参数名（根除 D3）。

    ★ 背景：`grad_leaks` 在 v113/v114/v115 里**只有读、没有写** ⇒
      「新作用点漏到哪些物理参数」这条**直接证据恒为空** ⇒ 形同虚设。
    ★ 本函数逐参数判 `grad is not None and grad.abs().max() > tol` ⇒
      真写出来。★ 判据**不变**：grad_frac_new <= 1e-6 是主判据，
      grad_leaks 是**并列的直接证据**（两条必须一致，否则报 INCONSISTENT）。
    """
    _leaks = []
    try:
        for _i, _g in enumerate(grads):
            if _g is None:
                continue
            _nm = names[_i] if _i < len(names) else ("idx%d" % _i)
            try:
                _mx = float(_g.detach().abs().max().item())
            except Exception:
                continue
            if not (_mx == _mx):
                continue
            if _mx > float(tol):
                _leaks.append(_nm)
    except Exception:
        return _leaks
    return _leaks


def _v116_verify_runnable():
    """[v116] **可运行性静态自检**（秒级；回答『这份脚本能不能直接跑』）。

    ★ 为什么需要它：本项目的三次白跑（v113/v114/v115）都是**发车后才崩**：
      v113 是模块级 use-before-def、v114/v115 是探针自身崩 ⇒ 全部在
      「训练跑完之后」才暴露。⇒ 本函数把**「这个模块能不能跑起来」**
      变成**开机 1 秒内**就能回答的问题。

    ★ 检查项（全部只读，不改任何状态）：
      1. 关键全局量是否都已绑定（VERSION / SEED / TRAIN_EPOCHS / device /
         OUT_DIR / CKPT_DIR / model / net / trainloader / testloader）。
      2. 探针相关的 5 个仪器函数是否都已定义（可调用）。
      3. 探针调用点用的函数是否存在（名字拼写）。
      4. 预注册判据常量是否齐备（判词要用的每一个）。
    ★ 返回 dict：{"ok":bool, "missing":[...], "notes":[...]}。
    ★ **绝不 exit**（纪律：不许在主路径里抛 SystemExit）⇒ 只返回读数。
    """
    _res = {"ok": True, "missing": [], "notes": []}
    _g = globals()
    _need_globals = ("VERSION", "SEED", "TRAIN_EPOCHS", "device", "OUT_DIR",
                     "CKPT_DIR", "model", "net", "trainloader", "testloader",
                     "BATCH_SIZE", "PINN_WEIGHT", "TOTAL_EPOCHS")
    for _k in _need_globals:
        if _k not in _g or _g.get(_k) is None:
            _res["missing"].append("global:%s" % _k)
    _need_funcs = ("_v116_probe_safe_sqrt", "_v116_probe_safe_ratio",
                   "_v116_probe_collect_leaks", "_v116_pinface_probe",
                   "_v116_pinface_precheck", "_v113_roughness",
                   "_v113_band_frac", "_v113_pinface_verdict",
                   "_v115_align_probe_inputs", "_safe_readout")
    for _k in _need_funcs:
        if not callable(_g.get(_k)):
            _res["missing"].append("func:%s" % _k)
    _need_consts = ("V113_PINN_FACE_GRAD_FRAC_MAX", "V113_BAND_FRAC_MIN",
                     "V113_BASE_DELIVERED", "V113_SIGMA_DIFF",
                     "V113_PASS_LINE", "V113_REDLINE")
    for _k in _need_consts:
        if _g.get(_k) is None:
            _res["missing"].append("const:%s" % _k)
    _res["ok"] = (len(_res["missing"]) == 0)
    return _res
# ====================== [v117] 发车路径三修（仪器/守卫层；训练数学逐字未改）======================
#   ★★ v116 实跑：**发车就死**（探针自检 grad=False ⇒ 紧接着 run_determinism_probe
#      在 DataParallel 的第一层 conv 抛 CUDNN_STATUS_NOT_INITIALIZED ⇒ 整跑当场死，
#      180ep 一次没跑）。⇒ v117 只修**发车路径**，轴与判据**一个字都不动**。
#
#   ★ 本版**不修改**任何 v116 顶层函数（父本逐字保留），只**新增**下面这些函数。
# =====================================================================================

V117_LAUNCH_FIX_NOTE = (
    "v117 = v116 + 发车路径三修（F1 探针小批 / F2 CUDA 清扫 / F3 确定性探针不许杀主流程"
    " / F4 逐段打印 section_errors 原文 / F5 CPU 兜底）。"
    "★ v116 实跑：训练前自检 sections_ok.grad=False，紧接着 run_determinism_probe() 抛"
    "cuDNN error: CUDNN_STATUS_NOT_INITIALIZED ⇒ **整跑当场死在训练开始前**。"
    "训练数学与全部计算**逐字零改动** ⇒ 预期 delivered 仍 ≈ 97.4（NULL）；"
    "本版价值 = ① 跑得完 ② 拿得到机制读数。"
)

V117_V116_HISTORY = {
    "version": "v116",
    "ran_epochs": 0,
    "crash_stage": "module-level run_determinism_probe() (before training)",
    "precheck_sections_ok": {"forward": True, "roughness": True,
                             "band_frac": True, "ratios": True, "grad": False},
    "precheck_error": "incomplete: sections_ok={...grad: False}",
    "crash": "RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED "
             "(DataParallel replica 0 / backbone.conv1)",
    "root_cause_chain": (
        "探针梯度分流段用 testloader **整批 256 张**跑带图前向 + 两次 "
        "torch.autograd.grad(retain_graph=True) ⇒ 显存峰值最高 ⇒ 极可能 CUDA OOM "
        "⇒ 之后 cuDNN 拿不到 workspace ⇒ 任何 conv 都报 NOT_INITIALIZED "
        "⇒ run_determinism_probe() 是模块级裸调用 ⇒ 整跑当场死。"),
    "note": ("★ 两次教训：(a) 只读仪器**绝不许**是模块级裸调用——它崩了主流程就死；"
             "(b) 自检只打概要（sections_ok）不打原文 ⇒ 真错被吞 ⇒ 下一轮仍然瞎。"),
}

# ★ 探针小批量：8 张图。整批 256 张的带图前反向是 v116 的死因。
V117_PROBE_N_SMALL = 8
# ★ 二次降级批量（小批也 OOM 时再试一次）
V117_PROBE_N_FALLBACK = 4


def _v117_probe_slice(inputs, n):
    """[v117] 把探针输入切成小批（**显存是本脚本的死因**）。

    ★ 返回 (sliced_inputs, actual_n)。输入不是张量 / 取不到 shape ⇒ 原样返回。
    ★ 用 `.contiguous()`：切片后仍要保证内存连续（conv 对非连续输入有额外拷贝）。
    """
    try:
        _n = int(n)
    except Exception:
        return inputs, None
    try:
        _cur = int(inputs.shape[0])
    except Exception:
        return inputs, None
    if _n <= 0 or _cur <= _n:
        return inputs, int(_cur)
    try:
        return inputs[:_n].contiguous(), int(_n)
    except Exception:
        return inputs, int(_cur)


def _v117_cuda_sweep():
    """[v117] CUDA 现场清扫：gc.collect + empty_cache + synchronize。

    ★ 为什么必须有它：v116 的崩溃链 = 探针梯度段（极可能 OOM）⇒ 之后
      DataParallel 第一层 conv 抛 **CUDNN_STATUS_NOT_INITIALIZED** ⇒ 整跑当场死。
      OOM 之后 PyTorch 缓存分配器仍占着块 ⇒ cuDNN 拿不到 workspace ⇒
      NOT_INITIALIZED。⇒ 探针跑完**立刻**把缓存还回去，切断这条次生链。
    ★ 全 try 包住：清扫失败也**不许**杀主流程。
    """
    _r = {"gc": False, "empty_cache": False, "synchronize": False}
    try:
        _gc = __import__("gc")
        _gc.collect()
        _r["gc"] = True
    except Exception:
        pass
    try:
        if bool(torch.cuda.is_available()):
            torch.cuda.empty_cache()
            _r["empty_cache"] = True
    except Exception:
        pass
    try:
        if bool(torch.cuda.is_available()):
            torch.cuda.synchronize()
            _r["synchronize"] = True
    except Exception:
        pass
    return _r


def _v117_is_oom(msg):
    """[v117] 判定一条错误信息是不是"显存/cuDNN 现场"类（OOM 及其次生错误）。"""
    try:
        _s = str(msg).lower()
    except Exception:
        return False
    return (("out of memory" in _s)
            or ("cudnn_status_not_initialized" in _s)
            or ("cudnn_status_alloc_failed" in _s)
            or ("cuda error" in _s))


def _v117_precheck2(model, inputs, epoch):
    """[v117] **训练开始前**的探针自检（小批 + 清扫 + CPU 兜底 + 错误可见）。

    ★ 与 `_v116_pinface_precheck` 的区别（父本**保留不动**）：
      ① 固定**小批** `V117_PROBE_N_SMALL` ⇒ 显存峰值降 ~30×（v116 死于此）；
      ② 跑完立刻 `_v117_cuda_sweep()`；
      ③ GPU 版不 ok ⇒ 在 **CPU 深拷贝**上再跑一次 ⇒ 把「逻辑通不通」与
         「CUDA 显存够不够」解耦（两者都能单独致盲）；
      ④ 返回体里带 `probe["section_errors"]`（**逐段原文**，由调用方打印）。
    ★ 仍然**只碰副本** + 跑完校验原模型 buffer 逐位未变（v115 的教训：
      `update_count` 会决定 momentum ⇒ 训练前污染 = 改训练起点）。
    ★ 绝不 exit（纪律）：只返回读数。
    """
    _res = {"ok": False, "error": None, "attempts": [], "probe_n": None,
            "used_copy": False, "model_state_unchanged": None,
            "oom_suspected": False, "cpu_fallback": False}
    # ---- 原模型 buffer 快照（**全部** buffer；v115 的名单漏掉 update_count）----
    _snap = {}
    try:
        for _bn, _bv in model.named_buffers():
            if torch.is_tensor(_bv):
                _snap[_bn] = _bv.detach().clone()
    except Exception:
        _snap = {}
    _copy_mod = None
    try:
        _copy_mod = __import__("copy")
    except Exception:
        _copy_mod = None
    _in_s, _n_s = _v117_probe_slice(inputs, V117_PROBE_N_SMALL)
    _res["probe_n"] = _n_s
    # ---- ① GPU 副本 + 小批 ----
    try:
        _m = _copy_mod.deepcopy(model) if _copy_mod is not None else model
        _res["used_copy"] = bool(_copy_mod is not None)
        try:
            _pr = _v116_pinface_probe(_m, _in_s, epoch)
            _res["probe"] = _pr
            _res["ok"] = bool(isinstance(_pr, dict) and _pr.get("ok"))
            if not _res["ok"]:
                _res["error"] = (_pr.get("error") if isinstance(_pr, dict)
                                 else "no probe dict")
        finally:
            try:
                del _m
            except Exception:
                pass
    except Exception as _e:
        _res["error"] = "%s: %s" % (type(_e).__name__, _e)
    _res["attempts"].append({"where": "gpu_copy_small", "n": _n_s,
                             "ok": bool(_res["ok"]), "error": _res.get("error")})
    _res["oom_suspected"] = _v117_is_oom(_res.get("error"))
    _v117_cuda_sweep()
    # ---- ② CPU 兜底（**完全不碰 CUDA**）----
    if not _res["ok"]:
        try:
            _m2 = _copy_mod.deepcopy(model) if _copy_mod is not None else model
            _res["used_copy"] = bool(_copy_mod is not None)
            try:
                try:
                    _m2 = _m2.cpu()
                except Exception:
                    pass
                _in2, _n2 = _v117_probe_slice(inputs, V117_PROBE_N_SMALL)
                try:
                    _in2 = _in2.cpu()
                except Exception:
                    pass
                _pr2 = _v116_pinface_probe(_m2, _in2, epoch)
                _res["cpu_probe"] = _pr2
                _ok2 = bool(isinstance(_pr2, dict) and _pr2.get("ok"))
                _res["cpu_fallback"] = True
                _res["attempts"].append(
                    {"where": "cpu_copy_small", "n": _n2, "ok": _ok2,
                     "error": (None if _ok2 else (isinstance(_pr2, dict)
                                                  and _pr2.get("error")))})
                if _ok2:
                    _res["probe"] = _pr2
                    _res["ok"] = True
                    _res["error"] = None
                    _res["probe_n"] = _n2
            finally:
                try:
                    del _m2
                except Exception:
                    pass
        except Exception as _e2:
            _res["attempts"].append(
                {"where": "cpu_copy_small", "n": None, "ok": False,
                 "error": "%s: %s" % (type(_e2).__name__, _e2)})
        _v117_cuda_sweep()
    # ---- 校验：原模型 buffer 必须**逐位未变** ----
    try:
        _same = True
        for _bn3, _bv3 in model.named_buffers():
            if _bn3 in _snap and torch.is_tensor(_bv3):
                if not torch.equal(_snap[_bn3], _bv3.detach()):
                    _same = False
                    break
        _res["model_state_unchanged"] = bool(_same)
        _res["n_buffers_checked"] = int(len(_snap))
    except Exception:
        _res["model_state_unchanged"] = None
    return _res


def _v117_main_probe(model, inputs, epoch):
    """[v117] 训练后的**主探针**：整批优先 + 小批自动降级。

    ★ 返回值**就是 probe dict 本体**（附加 `probe_n` / `small_batch_fallback` /
      `oom_suspected` / `attempts`）⇒ 下游 `_v113_probe.get(...)` 口径**不变**。
    ★ 为什么要降级：训练后显存碎片化，整批 256 张的带图前反向 + 两次
      `autograd.grad(retain_graph=True)` 极易 OOM（v116 的 precheck 就是这么死的）
      ⇒ 整批失败就清显存改小批，并在读数里**自曝**改了批量。
    """
    _att = []

    def _merge(_pr, _where, _n):
        _d = dict(_pr) if isinstance(_pr, dict) else {"ok": False,
                                                      "error": "no probe dict"}
        _d["probe_n"] = _n
        _d["attempts"] = _att
        _att.append({"where": _where, "n": _n, "ok": bool(_d.get("ok")),
                     "error": _d.get("error")})
        return _d

    try:
        _n0 = int(inputs.shape[0])
    except Exception:
        _n0 = None
    try:
        _res = _merge(_v116_pinface_probe(model, inputs, epoch), "full", _n0)
        _res["small_batch_fallback"] = False
        _res["oom_suspected"] = False
    except Exception as _e:
        _res = {"ok": False, "error": "%s: %s" % (type(_e).__name__, _e),
                "probe_n": _n0, "small_batch_fallback": False,
                "oom_suspected": True, "attempts": _att,
                "sections_ok": {}, "levels": None}
        _att.append({"where": "full", "n": _n0, "ok": False,
                     "error": _res["error"]})
    if bool(_res.get("ok")):
        _v117_cuda_sweep()
        return _res
    _oom = _v117_is_oom(_res.get("error"))
    _res["oom_suspected"] = bool(_oom)
    _v117_cuda_sweep()
    for _nn in (int(V117_PROBE_N_SMALL), int(V117_PROBE_N_FALLBACK)):
        _in_s, _ns = _v117_probe_slice(inputs, _nn)
        try:
            _r2 = _merge(_v116_pinface_probe(model, _in_s, epoch), "small", _ns)
            _r2["small_batch_fallback"] = True
            _r2["oom_suspected"] = bool(_oom)
            _res = _r2
            if bool(_res.get("ok")):
                break
        except Exception as _e2:
            _res["ok"] = False
            _res["error"] = "%s: %s" % (type(_e2).__name__, _e2)
            _res["attempts"] = _att
            _att.append({"where": "small", "n": _ns, "ok": False,
                         "error": _res["error"]})
        _v117_cuda_sweep()
    _res["attempts"] = _att
    return _res


def _v117_cuda_canary():
    """[v117] CUDA 健康探针：训练开始前用一次**极小 conv** 确认 cuDNN 还活着。

    ★ 为什么需要它：v116 的崩溃是**次生**的（探针 OOM ⇒ 之后任何 conv 都报
      CUDNN_STATUS_NOT_INITIALIZED）。⇒ 单看探针自己的报错**判断不了**"CUDA 现场
      到底还健不健康"。本探针在**训练前最后一刻**用 2×3×8×8 的一次 conv 直接问
      cuDNN：活着就走，不活着就**大声报出来**（并再清扫重试一次）。
    ★ 绝不 exit（纪律）：只报事实。若训练首步也崩，日志里会有这条**独立证据**
      ⇒ 下一轮不必再猜"是不是探针把 CUDA 搞坏了"。
    ★ 无 GPU ⇒ 直接返回 ran=False（不假绿）。
    """
    _r = {"ran": False, "ok": False, "error": None, "attempts": []}
    try:
        if not bool(torch.cuda.is_available()):
            return _r
    except Exception:
        return _r
    for _i in range(2):
        _c = None
        try:
            _dev = torch.device("cuda:0")
            _c = torch.nn.Conv2d(3, 4, 3, padding=1).to(_dev)
            _x = torch.randn(2, 3, 8, 8, device=_dev)
            _y = _c(_x)
            _r["ran"] = True
            _r["ok"] = True
            _r["attempts"].append({"i": int(_i), "ok": True})
            break
        except Exception as _e:
            _r["ran"] = True
            _r["error"] = "%s: %s" % (type(_e).__name__, _e)
            _r["attempts"].append({"i": int(_i), "ok": False,
                                   "error": _r["error"]})
            _v117_cuda_sweep()
        finally:
            try:
                del _c
            except Exception:
                pass
    return _r


def _v117_guarded_determinism_probe():
    """[v117] 确定性探针的**守卫壳**：只读仪器**绝不许杀主流程**。

    ★★ v116 的死法：模块级裸调 `run_determinism_probe()` ⇒ 它一抛
      （CUDNN_STATUS_NOT_INITIALIZED）⇒ **整跑当场死** ⇒ 180ep 一次没跑。
    ★ 本守卫做三件事：
      ① 探针**任何**异常都只记 error，**不外抛**；
      ② `finally` 里**无条件**关掉 `use_deterministic_algorithms` 并还原
         cudnn.deterministic / benchmark（否则崩在"已开确定性"之后 ⇒
         后面 180ep 会带着 deterministic=True 跑：更慢且可能再崩）；
      ③ **保证 `det_probe_report` 一定是 dict**（下游大量 `.get()` ⇒
         若为 None 会 AttributeError **二次崩**）。
    ★ 额外：崩了也要做一次 CUDA 清扫（OOM 的次生链要切断）。
    """
    global det_probe_report
    _r = {"ok": False, "error": None, "verdict": None, "restored": False}
    try:
        _d0 = bool(torch.backends.cudnn.deterministic)
        _b0 = bool(torch.backends.cudnn.benchmark)
    except Exception:
        _d0, _b0 = False, False
    try:
        _rep = run_determinism_probe()
        _r["ok"] = True
        _r["verdict"] = (_rep or {}).get("verdict")
        _r["error"] = (_rep or {}).get("error")
    except Exception as _e:
        _err = "%s: %s" % (type(_e).__name__, _e)
        _r["error"] = _err
        try:
            det_probe_report = {
                "axis": ("operator-level determinism hardening "
                         "(cuBLAS workspace + deterministic algorithms)"),
                "probe_batch": int(globals().get("DET_PROBE_BATCH", 16)),
                "probe_seed": int(globals().get("DET_PROBE_SEED", 0)),
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "error": _err[:600],
                "fallback_applied": True,
                "nondeterministic_mode": None,
                "deterministic_mode": None,
                "final_use_deterministic_algorithms": False,
                "axis_effective": False,
                "elapsed_s": 0.0,
                "verdict": "DETERMINISM_PROBE_CRASHED",
            }
        except Exception:
            det_probe_report = {"verdict": "DETERMINISM_PROBE_CRASHED",
                                "axis_effective": False, "error": _err[:600]}
    finally:
        try:
            _uda = getattr(torch, "use_deterministic_algorithms", None)
            if callable(_uda):
                _uda(False)
            torch.backends.cudnn.deterministic = _d0
            torch.backends.cudnn.benchmark = _b0
            _r["restored"] = True
        except Exception as _e2:
            _r["restore_error"] = "%s: %s" % (type(_e2).__name__, _e2)
        try:
            _r["cuda_sweep"] = _v117_cuda_sweep()
        except Exception:
            pass
    if not isinstance(det_probe_report, dict):
        det_probe_report = {
            "verdict": "DETERMINISM_PROBE_CRASHED", "axis_effective": False,
            "error": _r.get("error"), "nondeterministic_mode": None,
            "deterministic_mode": None,
            "final_use_deterministic_algorithms": False,
            "cudnn_deterministic": _d0, "cudnn_benchmark": _b0,
        }
    return _r


def _v117_verify_runnable():
    """[v117] 可运行性自检 = v116 的检查项 **+ v117 新增仪器函数/常量**。

    ★ 不修改父本的 `_v116_verify_runnable`（逐字保留）⇒ 这里做**包装**。
    """
    try:
        _r = _v116_verify_runnable()
    except Exception as _e:
        _r = {"ok": False, "missing": ["_v116_verify_runnable raised: %s" % _e],
              "notes": []}
    if not isinstance(_r, dict):
        _r = {"ok": False, "missing": ["_v116_verify_runnable:bad_return"],
              "notes": []}
    _r.setdefault("missing", [])
    _r.setdefault("notes", [])
    _g = globals()
    for _k in ("_v117_probe_slice", "_v117_cuda_sweep", "_v117_is_oom",
               "_v117_precheck2", "_v117_main_probe",
               "_v117_guarded_determinism_probe"):
        if not callable(_g.get(_k)):
            _r["missing"].append("func:%s" % _k)
    for _k in ("V117_PROBE_N_SMALL", "V117_PROBE_N_FALLBACK"):
        if _g.get(_k) is None:
            _r["missing"].append("const:%s" % _k)
    _r["ok"] = (len(_r["missing"]) == 0)
    return _r


def _v116_pinface_probe(model, inputs, epoch):
    """[v116] 机制探针（**v115 探针的加固版**；只读，不改任何训练态）。

    ① `pinn` 新旧作用点的**量级**（口径自曝：作用点换了 ⇒ 量级可能变）。
    ② **梯度分流**：pinn（在新位置）对**物理参数**的梯度占比（应 ≡ 0）
       vs pinn（旧位置）的占比（应 > 0 ⇒ 探针没瞎）⇒ **机制交付的直接证据**。
    ③ 场的**带内能量占比** band_frac（v111 实测 0.0968 ⇒ 场是否恢复内容）。

    ★★ [v116] 与 v115 探针的**唯一区别 = 内部加固**（训练数学无关）：
      · **分节降级**：三段各自独立 try ⇒ 任一段崩，其余两段**照常交付**
        （v115 是**一个大 try 包全部** ⇒ 单点 math domain error 吃掉整轴读数）。
      · **双写**：`levels` 子键 + 顶层平铺都写 ⇒ 两种读法都对
        （v114 起判词读 `levels`，而 v113 探针只平铺 ⇒ band_frac 恒 0 ⇒ 假判负）。
      · **真写 `grad_leaks`**（逐参数判梯度非零）⇒ 这条直接证据不再恒空。
      · sqrt/除法全部走安全包装 ⇒ 负数/NaN/除零**不抛**（只标 NaN）。
      · `raw_ratio_mean` 同时给**原始参数**与 **soft_bound 口径**两个值
        （生产 forward 用的是 soft_bound 后的 nu_vals ⇒ 口径必须对齐）。

    ★ 梯度用 `torch.autograd.grad`（**不**用 .backward()）⇒ **不污染 .grad**
      ⇒ 不影响任何后续读数。
    ★ 场参数清单：phys2/phys3 的 s_param / lam_param / nu_param / aniso_param。
    """
    _out = {"ok": False, "error": None, "sections_ok": {}}
    _lvl = {}
    _phys_params = []
    try:
        for _pn in ("phys2", "phys3"):
            _p = getattr(model, _pn, None)
            if _p is None:
                continue
            for _nm in ("s_param", "lam_param", "nu_param", "aniso_param"):
                _q = getattr(_p, _nm, None)
                if _q is not None:
                    _phys_params.append((_pn + "." + _nm, _q))
    except Exception as _e:
        _out["error"] = "phys_params: %s: %s" % (type(_e).__name__, _e)
        return _out
    if not _phys_params:
        _out["error"] = "no physics params found"
        return _out

    _net = model
    if hasattr(_net, "module"):
        _net = _net.module
    _names = [_n for (_n, _q) in _phys_params]

    # ================= §1 前向读数（量级 + band_frac）=================
    #   ★ v115 在这一段崩（math.sqrt 负数）⇒ 本段**独立 try**：崩了不影响 §2/§3。
    #   ★★ [v116] 再细分：§1a 前向 / §1b 粗糙度 / §1c band_frac / §1d 比值
    #      ⇒ **band_frac_mean 是关键判据**，绝不能被粗糙度计算之类的旁支拖死。
    _ph2 = None
    _ph3 = None
    _h2 = None
    _deep = None
    try:
        _net.eval()
        with torch.no_grad():
            _h = _net.backbone.conv1(inputs)
            _h = _net.backbone.bn1(_h)
            _h = torch.relu(_h)
            _h = _net.backbone.layer1(_h)
            _h, _ = _net.gate1(_h, _net.phys1(_h))
            _h = _net.backbone.layer2(_h)
            _h2 = _h
            _ph2, _sm2, _s2, _l2, _nu2 = _net.phys2(_h, _h, epoch)
            _h, _a2 = _net.gate2(_h, _ph2)
            _h = _net.backbone.layer3(_h)
            _ph3, _sm3, _s3, _l3, _nu3 = _net.phys3(_h, _h, epoch)
            _h, _a3 = _net.gate3(_h, _ph3)
            _h = _net.backbone.layer4(_h)
            _deep = _h
        _out["sections_ok"]["forward"] = True
    except Exception as _e:
        _out["sections_ok"]["forward"] = False
        _out.setdefault("section_errors", {})["forward"] = (
            "%s: %s" % (type(_e).__name__, _e))

    # --- §1b 粗糙度（pinn 量级）---
    try:
        _lvl["pinn_level_new"] = float(_v113_roughness(_h2).item())
        _lvl["pinn_level_old"] = float(_v113_roughness(_deep).item())
        _lvl["pinn_level_ratio"] = (_lvl["pinn_level_new"]
                                     / (_lvl["pinn_level_old"] + 1e-12))
        _out["sections_ok"]["roughness"] = True
    except Exception as _e:
        _out["sections_ok"]["roughness"] = False
        _out.setdefault("section_errors", {})["roughness"] = (
            "%s: %s" % (type(_e).__name__, _e))

    # --- §1c band_frac（**关键判据**：独立于粗糙度）---
    try:
        _lvl["band_frac_mean"] = _v113_band_frac(_ph3)
        _lvl["band_frac_mean_phys2"] = _v113_band_frac(_ph2)
        _out["sections_ok"]["band_frac"] = True
    except Exception as _e:
        _out["sections_ok"]["band_frac"] = False
        _out.setdefault("section_errors", {})["band_frac"] = (
            "%s: %s" % (type(_e).__name__, _e))

    # --- §1d 比值 / lambda_c（D5：原始参数口径 + soft_bound 口径）---
    try:
        _lam3_raw = _net.phys3.lam_param.detach().float().mean().item()
        _nu3_raw = _net.phys3.nu_param.detach().float().mean().item()
        _lvl["lam_mean_raw"] = float(_lam3_raw)
        _lvl["nu_mean_raw"] = float(_nu3_raw)
        _rr_raw, _rr_ok = _v116_probe_safe_ratio(_lam3_raw, _nu3_raw)
        _lvl["raw_ratio_mean"] = _rr_raw
        _lvl["raw_ratio_mean_ok"] = bool(_rr_ok)
        # ★ 生产 forward 用的是 soft_bound 后的 nu_vals（≥0.5）⇒ 这个口径才可比。
        _lvl["nu_mean_softbound"] = float(_nu3)
        _lvl["lam_mean_softbound"] = float(_l3)
        _rr_sb, _rr_sb_ok = _v116_probe_safe_ratio(_l3, _nu3)
        _lvl["raw_ratio_mean_softbound"] = _rr_sb
        _lvl["raw_ratio_mean_softbound_ok"] = bool(_rr_sb_ok)
        # ★★ D1 根除：开方走安全包装 ⇒ 负数不再抛 math domain error。
        _lam_c_raw = _v116_probe_safe_sqrt(_rr_raw)
        _lvl["raw_lambda_c_px"] = (None if _lam_c_raw is None
                                     else float(2.0 * math.pi * _lam_c_raw))
        _lvl["raw_lambda_c_px_ok"] = bool(_lam_c_raw is not None)
        _lam_c_sb = _v116_probe_safe_sqrt(_rr_sb)
        _lvl["raw_lambda_c_px_softbound"] = (None if _lam_c_sb is None
                                               else float(2.0 * math.pi * _lam_c_sb))
        _lvl["raw_lambda_c_px_softbound_ok"] = bool(_lam_c_sb is not None)
        _lvl["raw_ratio_negative"] = bool((_rr_raw is not None) and (_rr_raw < 0.0))
        _out["sections_ok"]["ratios"] = True
    except Exception as _e:
        _out["sections_ok"]["ratios"] = False
        _out.setdefault("section_errors", {})["ratios"] = (
            "%s: %s" % (type(_e).__name__, _e))

    # ================= §2 梯度分流（带梯度的一次前向）=================
    #   ★ 独立 try：即便 §1 崩了，这里照跑 ⇒ 至少拿到最关键的直接证据。
    _grad_frac_new = None
    _grad_frac_old = None
    _leaks = []
    try:
        _net.train()
        try:
            _h = _net.backbone.conv1(inputs)
            _h = _net.backbone.bn1(_h)
            _h = torch.relu(_h)
            _h = _net.backbone.layer1(_h)
            _h, _ = _net.gate1(_h, _net.phys1(_h))
            _h = _net.backbone.layer2(_h)
            _h2g = _h
            _ph2b, _sm2b, _s2b, _l2b, _nub = _net.phys2(_h, _h, epoch)
            _h, _a2 = _net.gate2(_h, _ph2b)
            _h = _net.backbone.layer3(_h)
            _ph3g, _sm3b, _s3b, _l3b, _nub2 = _net.phys3(_h, _h, epoch)
            _h, _a3 = _net.gate3(_h, _ph3g)
            _h = _net.backbone.layer4(_h)
            _deepg = _h
            _pnew = _v113_roughness(_h2g)
            _pold = _v113_roughness(_deepg)
            _tgt = [q for (_n, q) in _phys_params]
            _gn = torch.autograd.grad(_pnew, _tgt, retain_graph=True, allow_unused=True)
            _go = torch.autograd.grad(_pold, _tgt, retain_graph=True, allow_unused=True)

            def _frac_of(gs):
                _num = 0.0
                _den = 0.0
                for _g, (_n, q) in zip(gs, _phys_params):
                    if _g is None:
                        continue
                    _num += float((_g ** 2).sum().item())
                    _den += float((q.detach() ** 2).sum().item())
                # ★ D4 根除：NaN / 分母近零**不抛**、不编造。
                #   ★ 语义与 v115 原式**逐字一致**（`0/1e-12 = 0.0`）：
                #     分子为 0（**无梯度**）⇒ 占比恒 0.0（这正是「切断成立」的证据，
                #     **不能**因分母为 0 而丢成 None）；
                #     分子 > 0 而分母近零 ⇒ 异常 ⇒ None（由调用方标 NaN）。
                if not (_num == _num) or not (_den == _den):
                    return None
                if _den <= 1e-24:
                    return 0.0 if _num <= 0.0 else None
                return (_num ** 0.5) / ((_den ** 0.5) + 1e-12)

            _grad_frac_new = _frac_of(_gn)
            _grad_frac_old = _frac_of(_go)
            # ★ D3 根除：**真写** grad_leaks（逐参数判非零）。
            _leaks = _v116_probe_collect_leaks(_gn, _names)
            _lvl["grad_leaks"] = list(_leaks)
            _lvl["n_leaks"] = len(_leaks)
        finally:
            _net.eval()
        _out["sections_ok"]["grad"] = True
    except Exception as _e:
        _out["sections_ok"]["grad"] = False
        _out.setdefault("section_errors", {})["grad"] = (
            "%s: %s" % (type(_e).__name__, _e))

    # ================= §3 装配 =================
    #   ★ **双写**：`levels` 子键 + 顶层平铺（D2 根除 ⇒ 两种读法都对）。
    try:
        _out.update(_lvl)
        _out["levels"] = dict(_lvl)
        _out["grad_leaks"] = list(_leaks)
        if _grad_frac_new is not None:
            _out["grad_frac_new"] = float(_grad_frac_new)
        if _grad_frac_old is not None:
            _out["grad_frac_old"] = float(_grad_frac_old)
        _out["n_phys_params"] = len(_phys_params)
        _out["phys_param_names"] = list(_names)
        # ★ ok 的判据：**关键判据齐备**（前向 + band_frac + 梯度分流）。
        #   ★ 与 v115 同口径（不放宽）：v115 要求『没抛异常』，这里要求
        #   **三个判据读数都拿到** ⇒ 更严，不会假绿。
        _out["ok"] = bool(_out["sections_ok"].get("forward", False)
                            and _out["sections_ok"].get("band_frac", False)
                            and _out["sections_ok"].get("grad", False)
                            and (_grad_frac_new is not None)
                            and (_grad_frac_old is not None))
        if not _out["ok"] and _out.get("error") is None:
            _out["error"] = "incomplete: sections_ok=%s" % (_out["sections_ok"],)
    except Exception as _e:
        _out["error"] = "assemble: %s: %s" % (type(_e).__name__, _e)
    return _out


def _v116_pinface_precheck(model, inputs, epoch):
    """[v116] **训练开始前**的探针自检（这是三次白跑的共同教训）。

    ★ 为什么必须有它：v113/v114/v115 都是在**训练跑完之后**才发现探针崩了
      ⇒ 每次赔上一整轮 180ep 机时。⇒ 本函数在**训练开始前**用同一批输入、
      同一份权重跑一遍探针 ⇒ 探针若崩，**立刻**暴露。

    ★★★ **在副本上跑**（`copy.deepcopy`）⇒ 原模型**零污染**：
      探针会前向 phys2/phys3 ⇒ 改写 `running_s/running_lam/running_nu` 与
      **`update_count`**（后者决定 momentum ⇒ 影响场的输出 ⇒ 影响训练轨迹）。
      ★ v115 的探针在**训练结束后**跑 ⇒ 污染无害；v116 在**训练开始前**跑 ⇒
        污染会**改变训练起点** ⇒ 违反『训练数学逐字未改』的可证性。
      ⇒ 故：副本上跑 + 跑完**校验原模型 buffer 逐位未变**（双保险）。
    ★ 绝不 exit（纪律：不许在主路径里抛 SystemExit）⇒ 只返回读数。
    ★ 返回 dict：{"ok":bool, "error":str|None, ...}。
    """
    _res = {"ok": False, "error": None, "used_copy": False,
             "model_state_unchanged": None}
    # ---- 原模型 buffer 快照（用于跑完后的**逐位校验**）----
    _snap = {}
    try:
        for _bn_name, _bn_buf in model.named_buffers():
            if torch.is_tensor(_bn_buf):
                _snap[_bn_name] = _bn_buf.detach().clone()
    except Exception:
        _snap = {}
    # ---- 副本 ----
    _probe_model = model
    _copy_mod = None
    try:
        # ★ 用 __import__ 而不是 `import copy as _copy_mod`：
        #   后者会在函数作用域留一个别名名 ⇒ symtable 报成『未绑定名』⇒ 闸门误报。
        _copy_mod = __import__("copy")
        _probe_model = _copy_mod.deepcopy(model)
        _res["used_copy"] = True
    except Exception as _e:
        _res["copy_error"] = "%s: %s" % (type(_e).__name__, _e)
        _probe_model = model
    try:
        _pr = _v116_pinface_probe(_probe_model, inputs, epoch)
        _res["probe"] = _pr
        _res["ok"] = bool(isinstance(_pr, dict) and _pr.get("ok"))
        if not _res["ok"]:
            _res["error"] = _pr.get("error") if isinstance(_pr, dict) else "no probe dict"
    except Exception as _e:
        _res["error"] = "%s: %s" % (type(_e).__name__, _e)
    finally:
        # ---- 双保险：若（万一）用了原模型，则**还原全部 buffer**并校验 ----
        if not _res.get("used_copy"):
            try:
                with torch.no_grad():
                    for _bn_name2, _bn_buf2 in model.named_buffers():
                        if _bn_name2 in _snap and torch.is_tensor(_bn_buf2):
                            _bn_buf2.copy_(_snap[_bn_name2])
            except Exception:
                pass
        # ---- 校验：原模型 buffer 必须**逐位未变** ----
        try:
            _same = True
            for _bn_name3, _bn_buf3 in model.named_buffers():
                if _bn_name3 in _snap and torch.is_tensor(_bn_buf3):
                    if not torch.equal(_snap[_bn_name3], _bn_buf3.detach()):
                        _same = False
                        break
            _res["model_state_unchanged"] = bool(_same)
        except Exception:
            _res["model_state_unchanged"] = None
    return _res

# ★★ [v116] v115 的旧探针体（`_v113_pinface_probe`）**已整体移除**：
#      它有 D1（负数开方）/ D2（levels 不写）/ D3（grad_leaks 不写）三处缺陷，
#      且**一个大 try 包全部** ⇒ 单点异常吃掉整轴读数（三次白跑的共同根因）。
#      新探针 = `_v116_pinface_probe`（分节降级 + 双写 + 真写 leaks + 安全 sqrt）。
#      ★ 训练数学与 `_v113_roughness` / `_v113_band_frac` / `_v113_grad_split_probe`
#        / `_v113_pinface_verdict` / `_v113_pinface_selftest` **全部逐字未改**。
AXIS_V113 = (
    "v113 = cnn10_v110.py's training math VERBATIM except ONE axis: the ACTION SURFACE of the "
    "physics constraint term `pinn`. In GradientStablePhysResNetV3.forward, pinn (the "
    "physics-constraint term that penalizes roughness of the final feature map) is computed on "
    "`deep` = the layer4 output, i.e. AFTER gate3 has INJECTED the physics field. Because "
    "Gate.forward is `feat + scale * a * bn(proj(phys))`, the gradient of pinn therefore flows "
    "deep -> gate3 -> ph3 -> phys3's s/lam/nu/aniso parameters. Consequence: the physics "
    "parameters are continuously shaped by 'the injected feature must be smooth', which makes it "
    "cheapest for the optimizer to RAISE l (make the field DC) rather than smooth the backbone "
    "-- a self-reinforcing loop. v111 measured exactly that: band_frac_mean = 0.0968 (90% of the "
    "field's energy still in DC), raw_lambda_c_px = 7.627, raw_ratio_mean = 1.4736; and v111 "
    "therefore ARCHIVED the additive injection channel ('the field output is a filtered copy of "
    "the backbone feature map => informationally redundant'). v113 moves pinn's action point to "
    "h2 = the layer2 output, i.e. BEFORE gate2 injects the FIRST physics field, so pinn's "
    "gradient can no longer reach ANY physics parameter (topological cut, not a scalar "
    "reweighting: changing PINN_WEIGHT would only shrink the field-parameter gradient "
    "proportionally, and the scalar-tweak family is already dead). This realizes v36's "
    "principle that physics may only act as a PASSIVE prior. "
    "MEASURED FACT (real model, non-zero input, per-parameter grads): on the OLD action point "
    "deep, pinn sends gradient to phys2/phys3 lam_param (0.31/0.90), nu_param (0.27/0.79) and "
    "aniso_param (0.39/0.94) => the physics parameters ARE continuously shaped by 'the injected "
    "feature must be smooth'. Moving the action point to the layer3 output is NOT enough: the "
    "gradient still leaks back to phys2 (2.3e-2) through layer3->layer2->gate2->ph2. Only the "
    "layer2 output (before gate2 injects the first field) disconnects BOTH fields. SEPARATE "
    "MEASURED FACT: s_param receives exactly ZERO gradient on every action point (torch.clamp"
    "(s_vals, 0.3, 1.3) saturates at the initial value) => the field smoothness exponent s has "
    "never been trainable through pinn. WHY THIS IS A FALSIFICATION TEST, NOT JUST A TWEAK: v111's "
    "archival rests on an implicit premise -- that the field is a 'copy' because the physics "
    "parameters are continuously shaped by the data/loss toward matching the backbone features. "
    "If the pinn -> physics-parameter path is cut, the field should stop being a copy, and the "
    "injection channel could revive => v111's archival conclusion would be falsified. If the "
    "field is STILL near-DC after the cut, the archival is REINFORCED (redundancy is "
    "structural, and the remaining culprit is the other constraint path, w_phys * smooth). "
    "FORWARD IS BITWISE UNCHANGED: pinn only enters the loss (PINN_WEIGHT * pinn_reg), never the "
    "forward values => same weights give identical logits => this version changes ONLY the "
    "gradient path. CRITERION (pre-registered): pass_line = 97.838 (= v110 delivered 97.50 + "
    "2*sigma_diff, sigma_diff = 0.169 from the four-pair zero-change RMS); redline = 97.04; "
    "mechanism delivery requires the physics-parameter gradient fraction under the new action "
    "point to be <= 1e-6 while the old action point's fraction is > 0 (the probe must not be "
    "blind); band_frac_mean >= 0.20 means the field recovered content (v111 measured 0.0968). "
    "EXPECTATION (pre-registered, from discipline 336 '末态吸引子律'): precision NULL is "
    "expected, because this is still a trajectory-reshaping constraint edit; the ONLY mechanism "
    "that could refute that expectation is the field regaining content, which is exactly what "
    "the band_frac readout measures."
)


# ---- [v114] 约束通道：pinn 作用面（轴沿用 v113；本版修模块级顺序）--------
#   ★ 本版的**唯一一根轴**与 v113 **完全相同**（pinn 的作用点从 layer4 输出
#     deep 移到 layer2 输出 h2 ⇒ 梯度到不了任何物理参数）。
#     为什么沿用：v113 的训练跑完了，但**收尾读数被模块级顺序 bug 吃掉**
#     ⇒ 那根轴**一次都没测到** ⇒ **未测 ≠ 判负** ⇒ 必须重测同一根轴。
#
#   ★ v114 与 v113 的**唯一差别在收尾顺序**（训练数学与全部计算逐字未动）：
#     v113: 探针/判词在 `final_tta` 之后（模块顶层）⇒ 读 `delivered`
#           而 `delivered` 在更后面 ⇒ **NameError** ⇒ 整跑的收尾全丢。
#     v114: 探针/判词在 `delivered = float(legacy_delivered)` **之后**
#           ⇒ 顺序正确 ⇒ 读数能活着出来。
V114_ORDER_FIX_NOTE = (
    "v114 = v113 + 模块级顺序修正（探针/判词移到 delivered 绑定之后）"
    "+ 判词先落盘 + 探针读数打 stdout。训练数学与全部计算逐字未动。")


# ---- [v115] 约束通道：pinn 作用面（轴沿用 v113/v114；本版修读数三处）--------
#   ★ 本版的**唯一一根轴**与 v113/v114 **完全相同**（pinn 的作用点从 layer4 输出
#     deep 移到 layer2 输出 h2 ⇒ 梯度到不了任何物理参数）。
#     为什么第三次沿用：v113 收尾读数被模块级顺序 bug 吃掉；v114 顺序修好了但
#     探针输入没搬 device ⇒ 机制读数**两次都没测到** ⇒ 未测 ≠ 判负 ⇒ 重测同一根轴。
#
#   ★ v115 相对 v114 的**唯一差别在读数层**（训练数学与全部计算逐字未动）：
#     P0-A 探针输入搬到模型 device（v114 在这崩 ⇒ 读数全丢）
#     P0-B 探针前快照 / 探针后还原 BN running stats（探针不许改写末态模型）
#     P0-C OUT_DIR 开机即建（v114 判词预落盘因目录不存在而失败）
V115_READOUT_FIX_NOTE = (
    "v115 = v114 + 读数三修（探针 device 对齐 / BN 统计还原 / OUT_DIR 早建）"
    "+ 判词先落盘 + 探针读数打 stdout。训练数学与全部计算逐字未动。"
    "⇒ 预期 delivered 仍 ≈ 97.4（NULL，同 v113/v114）；本版的价值是**拿到机制读数**。"
)

# ★★ [v115] v114 实跑的历史裁定（静态留档，供 v115 读者对照；不是本跑的读数）
V115_V114_HISTORY = {
    "version": "v114",
    "delivered": 97.42,
    "precision_vs_v110": "-0.08pp = -0.47σ ⇒ NULL（非断崖）",
    "verdict": "PINN_FACE_READOUT_ERROR",
    "probe_ok": False,
    "probe_error": ("RuntimeError: Input type (torch.FloatTensor) and weight type "
                     "(torch.cuda.FloatTensor) should be the same ..."),
    "prewrite_ok": False,
    "prewrite_error": "FileNotFoundError: ./out_v114_s42/v114_verdict_v114_seed42.json.tmp",
    "note": ("v114 的机制读数**两次都没测到**（v113 顺序 bug、v114 device bug）⇒ "
             "该轴**尚未被判定** ⇒ v115 重测同一根轴（未测 ≠ 判负）。"),
}


def _v115_align_probe_inputs(model, inputs):
    """[v115] 把探针输入搬到**模型参数所在的 device**（v114 实跑崩在这里）。

    v114 实跑：探针输入取自 `next(iter(testloader))` ⇒ 张量留在 **CPU**；
    而模型已 `.to(device)`（CUDA）⇒ `_net.backbone.conv1(inputs)` 抛
        RuntimeError: Input type (torch.FloatTensor) and weight type
        (torch.cuda.FloatTensor) should be the same ...
    ⇒ `probe_ok=false` ⇒ grad_frac_new/old、band_frac_mean **全部读不到**
    ⇒ 那根轴**第二次**没被测到（v113 是第一次：模块级顺序 bug）。

    ★ 本函数只搬输入、**不碰模型、不碰 RNG、不碰 .grad** ⇒ 训练数学零影响。
    ★ 返回 (inputs, device_str, ok)：ok=False 时调用方**必须**跳过探针并
      如实记 READOUT_ERROR（不许伪装成『机制没切断』—— 覆辙 267）。
    """
    if inputs is None:
        return None, "NO_INPUTS", False
    try:
        _dev = next(model.parameters()).device
    except Exception as _e:
        return None, "NO_DEVICE:%s" % type(_e).__name__, False
    try:
        _in_dev = inputs.device
    except Exception:
        return None, "NO_INPUT_DEVICE", False
    if str(_in_dev) != str(_dev):
        try:
            inputs = inputs.to(_dev)
        except Exception as _e:
            return None, "MOVE_FAIL:%s" % type(_e).__name__, False
    try:
        if str(inputs.device) != str(_dev):
            return None, "MOVE_INEFFECTIVE:%s->%s" % (inputs.device, _dev), False
    except Exception:
        pass
    return inputs, str(inputs.device), True


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
        # [v101] 各向异性轴（v90 重开）：a_g = 1.0 + ANISO_RANGE*tanh(aniso_param_g)。
        #   zeros 初始化是双保险：① tanh(0)=0 ⇒ a_g 精确 1.0 ⇒ 第 0 步核与 v100 逐位相同；
        #   ② torch.zeros 不耗 RNG ⇒ 初始化序列与 v100 逐位相同（覆铣 299：按"不改变
        #      消耗量"验，不按"看起来没调随机 API"验）。
        self.aniso_param = nn.Parameter(torch.zeros(self.groups))

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

        ky = _dft_axis(H, x.device, x.dtype)      # [v96] 真实 DFT 频率轴（覆辙 273/275）
        kx = _dft_axis(W, x.device, x.dtype)      # [v96] step 2*pi/N 半开区间，DC 精确在 N//2
        ky, kx = torch.meshgrid(ky, kx, indexing="ij")
        # [v101] 各向异性轴：k_sq_g = kx^2 + a_g*ky^2（v90 原式逐字重开）。
        #   a_vals 整张量算（tanh 可微）⇒ 梯度能回流到 aniso_param；
        #   zeros 初始化时 a_g 精确 1.0 ⇒ kx_sq + 1.0*ky_sq 与 v100 的 kx_sq + ky_sq
        #   **操作数相同** ⇒ 逐位相同（不是"接近"，是同一个 f32 加法，生成前已实测）。
        kx_sq = kx**2
        ky_sq = ky**2
        a_vals = 1.0 + ANISO_RANGE * torch.tanh(self.aniso_param)

        x_g = x.reshape(B, self.groups, self.cpg, H, W).mean(dim=2)

        phys_list = []
        for g in range(self.groups):
            # [v9] Matern kernel: (1 + lam*|k|^2/nu)^{-s}；[v101] k_sq 各向异性化
            k_sq_g = kx_sq + a_vals[g] * ky_sq
            kernel = (1.0 + lam_vals[g] * k_sq_g / nu_vals[g]) ** (-s_vals[g])
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
        ky = _dft_axis(H, x.device, x.dtype)      # [v96] 真实 DFT 频率轴（覆辙 273/275）
        kx = _dft_axis(W, x.device, x.dtype)      # [v96] step 2*pi/N 半开区间，DC 精确在 N//2
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

        self.phys1 = FixedDiffusion(256, groups=PHYS1_GROUPS)
        self.gate1 = Gate(256, learnable=False)
        self.phys2 = MaternDiffusion(512, groups=PHYS2_GROUPS)   # [v98] 唯一一根轴：通道粒度 8 -> 64
        self.gate2 = Gate(512, learnable=True)
        self.phys3 = MaternDiffusion(1024, groups=PHYS3_GROUPS)
        self.gate3 = Gate(1024, learnable=True)

    def forward(self, x, epoch):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = F.relu(x)

        x = self.backbone.layer1(x)
        x, _ = self.gate1(x, self.phys1(x))

        x = self.backbone.layer2(x)
        # [v113] 唯一一根轴：保存 layer2 输出（= gate2 **之前** = **物理注入之前**）
        #   ★ 为什么必须是 layer2 而不是 layer3：实测（真模型 + 非零输入）
        #     pinn 在 layer3 输出处算 ⇒ 梯度仍会经 layer3→layer2→gate2→ph2
        #     **漏回** phys2 的 lam/nu/aniso（占比 ~2.3e-2，只比 deep 小 13 倍，
        #     并非 0）⇒ 切断不彻底。layer2 输出处两个场都断开。
        h2 = x
        ph2, sm2, s2, l2, nu2 = self.phys2(x, x, epoch)
        x, a2 = self.gate2(x, ph2)

        x = self.backbone.layer3(x)
        ph3, sm3, s3, l3, nu3 = self.phys3(x, x, epoch)
        x, a3 = self.gate3(x, ph3)

        x = self.backbone.layer4(x)
        # [v113] `deep` 别名已删：pinn 不再在 gate3 之后算（见下）⇒ 无未使用变量
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        out = self.backbone.fc(x)

        # [v113] ★ pinn 的**新作用点** = h2（gate2 之前 = 物理注入之前）
        #   ⇒ pinn 的计算图里**不含 ph2 也不含 ph3** ⇒ 梯度**到不了任何物理参数**
        #   ⇒ 兑现 v36「物理只作被动先验」（物理参数不再被注入后特征的平滑度塑形）
        #   ★ 前向数值逐位不变（pinn 只进 loss，不进前向）
        dx = h2[..., 1:] - h2[..., :-1]
        dy = h2[..., 1:, :] - h2[..., :-1, :]
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


# ---- [v113] 作用面探针用的固定一批输入（惰性取；**不消耗训练 RNG**）----
#   ★ 用 testloader 的第一批（测试集、确定性顺序）⇒ 不碰训练数据流、不改 RNG 位点。
_v113_probe_inputs = None
# ★★ [v114] v114 的探针用**同一批**输入（同一 testloader 第一批 ⇒ 完全一致）
#   ⇒ v113 与 v114 的探针读数**可直接对比**（同输入同权重同测试集）。
_v114_probe_inputs = None
try:
    _v113_probe_inputs, _v113_probe_targets = next(iter(testloader))
    _v114_probe_inputs = _v113_probe_inputs
except Exception:
    _v113_probe_inputs = None
    _v114_probe_inputs = None
# ====================== Model Init ==================
best_acc = 0.0
best_epoch = 0
best_tta_acc = 0.0
best_tta_epoch = 0
no_improve_count = 0
no_test_improve = 0
# [v43] 剂量读数：每个 epoch 实际生效的 StochasticDepth drop_rate 均值
sd_trace = []
# [v101] 各向异性轴的 per-epoch |param| 运行最大值（**主本侧**收集）。
#   ★ 为什么不在 MaternDiffusion.forward 里用 buffer 累加：DataParallel 下 forward
#     跑在 replica 上，buffer 更新**不会回流** master ⇒ master 侧永远读到初值
#     （v98 的 running_* 读数其实就是这个形态，只是当时没人点破）。
#     参数（aniso_param）由优化器在 master 侧更新 ⇒ 读参数才是可靠的。
#   ★ 每 epoch 两次 .item() 同步，与 sd_trace 同阶；不碰每 batch 的那条线。
aniso_trace = []   # [(epoch, phys2_abs_max, phys3_abs_max), ...]
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


# ====================== [v116] 训练开始前的可运行性 + 探针自检 ======================
#   ★★ 教训（v113/v114/v115 三次白跑）：崩点都在**训练跑完之后**才暴露 ⇒
#      本版把『能不能跑起来』提前到**开机 1 秒内**回答。
#   ★ 只读：探针在**深拷贝**上跑 ⇒ 原模型零污染（runnable 自检会校验）。
#   ★ **绝不 exit**（纪律：不许在主路径里抛 SystemExit）⇒ 只打横幅 + 落盘。
_v116_runnable = _v117_verify_runnable()
flush_print("  [v116] 可运行性自检：ok=%s | missing=%r"
            % (_v116_runnable.get("ok"), _v116_runnable.get("missing")))
# ====================== [v117] 训练开始前的探针自检（小批 + 清扫 + CPU 兜底）======================
#   ★ v116 实跑：自检 grad=False ⇒ 紧接着 run_determinism_probe 抛
#     CUDNN_STATUS_NOT_INITIALIZED ⇒ **整跑当场死在训练开始前**。
#   ⇒ v117 三修：① 自检**固定小批**（显存峰值降 ~30×）；② 跑完立刻清扫；
#     ③ 不 ok 就在 **CPU 深拷贝**上再跑一次 ⇒ 把「逻辑」与「显存」解耦。
#   ★★ 自检**绝不 exit**（纪律）：只打横幅 + 落盘。
_v117_precheck = {"ok": False, "error": None, "ran": False}
try:
    if _v114_probe_inputs is None:
        _v117_precheck = {"ok": False, "error": "NO_PROBE_INPUTS", "ran": False}
    else:
        _v117_pc_in, _v117_pc_dev, _v117_pc_ok = _v115_align_probe_inputs(
            model, _v114_probe_inputs)
        _v117_precheck = _v117_precheck2(
            model, _v117_pc_in, TRAIN_EPOCHS)
        _v117_precheck["ran"] = True
        _v117_precheck["device"] = _v117_pc_dev
        _v117_precheck["align_ok"] = bool(_v117_pc_ok)
except Exception as _v117_pc_e:
    _v117_precheck = {"ok": False, "ran": True,
                      "error": "%s: %s" % (type(_v117_pc_e).__name__, _v117_pc_e)}
_v117_cuda_sweep()
flush_print("  [v117] 训练前探针自检：ok=%s | device=%s | probe_n=%s | error=%s"
            % (_v117_precheck.get("ok"), _v117_precheck.get("device"),
               _v117_precheck.get("probe_n"), _v117_precheck.get("error")))
flush_print("        副本运行=%s | 原模型状态未变=%s | OOM疑似=%s | CPU兜底=%s"
            % (_v117_precheck.get("used_copy"),
               _v117_precheck.get("model_state_unchanged"),
               _v117_precheck.get("oom_suspected"),
               _v117_precheck.get("cpu_fallback")))
# ★★ [v117] F4：**逐段打印 section_errors 原文**。
#   v116 只打 "incomplete: sections_ok={...}" ⇒ 真错被吞 ⇒ 下一轮仍然瞎。
_v117_pc_se = ((_v117_precheck.get("probe", {}) or {}).get("section_errors") or {})
if _v117_pc_se:
    for _k117 in sorted(_v117_pc_se.keys()):
        flush_print("        §%s 错误原文 = %s"
                    % (_k117, str(_v117_pc_se[_k117])[:300]))
if _v117_precheck.get("ok"):
    _v117_pc_pr = _v117_precheck.get("probe", {}) or {}
    flush_print("        grad_frac_new = %.6e | grad_frac_old = %.6e | band_frac_mean = %.6f"
                % (float(_v117_pc_pr.get("grad_frac_new", -1.0)),
                   float(_v117_pc_pr.get("grad_frac_old", -1.0)),
                   float((_v117_pc_pr.get("levels", {}) or {}).get(
                       "band_frac_mean", -1.0))))
else:
    flush_print("        ⚠ 训练前自检未通过（**不 exit**，继续训练；读数可能不可信）")

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
# ====================== [v110] Sobolev 度量 SAM（本版唯一一根轴） ======================
#   一句话：**把 SAM 的欧氏信任域换成场自己的度量**。
#   SAM：eps = rho*g/||g||，取 w+eps 处的梯度更新 w（找平坦极小）。本轴保留 SAM 的
#   全部动作面，只把*度量*从 ||.||_2 换成 Bessel 势诱导的 Sobolev 度量：
#       K(k) = (1 + (lam/nu)*|k|^2)^{-s}      eps = rho * K*g / ||K*g||_2
#   物理读法：(1 + ell^2 k^2)^{-s} 正是 Bessel 势 (I - ell^2 \Delta)^{-s} 的谱表示。
#   在这个度量下"最陡上升"只取梯度的**场许可的平滑分量**：描述纹理级细节的、脆的
#   高频参数方向被排除在信任域之外 ⇒ 我们要求的"平坦"是**沿扩散场所关心的方向**
#   的平坦，而不是全参数空间的平坦。
#   ★ 为什么不用各向异性 a_g：那需要复写 v101 的 `1 + ANISO_RANGE*tanh(param)`
#     变换 ⇒ 同一份逻辑两个实现（覆铣 306）⇒ 只取各向同性部分（lam/nu）。
SAM_RHO = 0.05            # 信任域 Euclidean 半径（SAM 文献 CIFAR 常用 0.05；本线 LR_BASE 同为 0.1）
SAM_SOBOLEV_S = 1.0       # Bessel 势阶 s（越大 ⇒ 越强调低频方向）
SAM_RATIO_FALLBACK = 1.5649   # 无场模块时的 lam/nu（取场实跑末态读数；不动 pills）
SAM_RATIO_MIN = 0.02
SAM_RATIO_MAX = 20.0
SAM_DIR_RATIO_LO = 0.05   # ||K*g||/||g|| 下界：低于此 ⇒ 度量把自己压没了（等于没在探险区域内走）
SAM_DIR_RATIO_HI = 0.95   # 高于此 ⇒ 低通失效（等于没换度量，退化成裸 SAM）
SAM_DELIVERED_MIN_FRAC = 0.9
_V110_SAM = {"n_steps": 0, "n_applied": 0, "n_degenerate": 0, "n_second_fail": 0,
             "ratio_sum": 0.0, "ratio_n": 0, "dir_sum": 0.0,
             "dir_min": None, "dir_max": None, "error": None}


def _v110_freq_axis(N, device=None, dtype=None):
    """fftshift 之后第 i 个 bin 的真实角频率 —— **奇偶通用**。

    ★★ 教训 331：不能直接用生产里的 `_dft_axis`。那句是 -pi + 2*pi*i/N，只对
       **偶数** N 等于真实 bin（已被 v96 自测在 N=8/16/32 上验证）；生产路径只用在
       16/8/4 这类特征图尺寸上，所以从未暴露。卷积核绝大多数是 **3x3（奇数）**：
         N=3 时 `_dft_axis` 给 [-pi, -pi/3, pi/3]，而 fftshift 后的真值是
         [-2pi/3, 0, +2pi/3] ⇒ **错半格**，把最大增益配到了真正的 DC 之外，
         后果是常数核被衰减到 0.313（而非 1.0），低通事实上变成了高通。
       这里直接用 torch.fft.fftfreq + fftshift 表达同一对象，奇偶都对。
    """
    _f = torch.fft.fftfreq(int(N), device=device, dtype=dtype)
    return 2.0 * math.pi * torch.fft.fftshift(_f)


def _v110_matern_kernel(h, w, ratio, s_exp, device=None, dtype=None):
    """Bessel 势核 K(k) = (1 + ratio*|k|^2)^{-s}（在 fftshift 后的 bin 顺序上）。"""
    ky = _v110_freq_axis(int(h), device=device, dtype=dtype)
    kx = _v110_freq_axis(int(w), device=device, dtype=dtype)
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")
    k_sq = KX * KX + KY * KY
    return (1.0 + float(ratio) * k_sq) ** (-float(s_exp))


def _v110_lowpass_grad(g, ratio, s_exp=SAM_SOBOLEV_S):
    """把梯度沿卷积核的两个**空间**维做一次 Bessel 势低通。

    1x1 卷积 / BN / bias / Linear 等非二维空间张量 ⇒ 原样返回（同一对象，不是拷贝）。
    device/dtype 全部跟随梯度本身 ⇒ 不需要 cpu generator（覆铣 312/325）。
    """
    if g.dim() != 4:
        return g
    kh = int(g.shape[-2])
    kw = int(g.shape[-1])
    if kh < 2 or kw < 2:
        return g
    k = _v110_matern_kernel(kh, kw, ratio, s_exp, device=g.device, dtype=g.dtype)
    return torch.real(ifft2(ifftshift(fftshift(fft2(g)) * k)))


def _v110_field_ratio(m):
    """从场的 running 缓冲读当前的 lam/nu（**pooled**：总 lam 均值 / 总 nu 均值）。

    零新参数、零 RNG、只读 ⇒ 任何异常都不影响训练。无场模块 ⇒ 回退常量。
    """
    num = torch.zeros((), dtype=torch.float32)
    den = torch.zeros((), dtype=torch.float32)
    n = 0
    for sub in m.modules():
        if hasattr(sub, "running_lam") and hasattr(sub, "running_nu"):
            num = num + sub.running_lam.detach().to(torch.float32).reshape(-1).to(num.device).mean()
            den = den + sub.running_nu.detach().to(torch.float32).reshape(-1).to(den.device).mean()
            n += 1
    if n == 0 or float(den) <= 0.0:
        return float(SAM_RATIO_FALLBACK), n
    r = float(num) / float(den)
    if r < float(SAM_RATIO_MIN):
        r = float(SAM_RATIO_MIN)
    if r > float(SAM_RATIO_MAX):
        r = float(SAM_RATIO_MAX)
    return r, n


def _v110_build_directions(grads, ratio):
    """构造上升方向与两个范数平方（贝塞尔势低通后 / 原始）。

    设备侧的标量累加 ⇒ 全程只在这一行与 float() 处同步一次，避免逐参数 sync。
    """
    dev = grads[0].device
    s2 = torch.zeros((), dtype=torch.float32, device=dev)
    r2 = torch.zeros((), dtype=torch.float32, device=dev)
    dirs = []
    for g in grads:
        d = _v110_lowpass_grad(g, ratio)
        dirs.append(d)
        s2 = s2 + d.detach().to(torch.float32).pow(2).sum()
        r2 = r2 + g.detach().to(torch.float32).pow(2).sum()
    return dirs, float(s2), float(r2)


def _v110_perturb(entries, dirs, s2):
    """把参数沿方向推 SAM_RHO（Euclidean 长度）。返回实际 scale。"""
    scale = float(SAM_RHO) / (float(s2) ** 0.5)
    with torch.no_grad():
        for p, d in zip(entries, dirs):
            p.data.add_(d.detach(), alpha=scale)
    return scale


def _v110_restore(backup):
    """从**快照**精确还原（不是加回去再减回来 —— 浮点加法不闭环）。"""
    with torch.no_grad():
        for p, saved in backup:
            p.data.copy_(saved)


def _v110_forward_and_loss(inputs, targets, targets_a, targets_b, lam, use_mixup, epoch, w_phys):
    """[v110] 前向 + 完整损失，**生产同一式**：两遍 fwd/bwd 共用这一个实现
    （覆铣 306：同一份逻辑只许一个实现）。返回顺序同 v103 的局部变量名。
    """
    outputs, s_vals, l_vals, nu_vals, a_vals, smooth, pinn_reg = net(inputs, epoch)
    if use_mixup:
        cls_loss = mixup_criterion(ce, outputs, targets_a, targets_b, lam)
    else:
        cls_loss = ce(outputs, targets)
    smooth = _reduce(smooth)
    pinn_reg = _reduce(pinn_reg)
    s_vals = [_reduce(s) for s in s_vals]
    l_vals = [_reduce(l) for l in l_vals]
    nu_vals = [_reduce(n) for n in nu_vals]
    a_vals = [_reduce(a) for a in a_vals]
    reg_loss = physics_reg(s_vals, epoch)
    loss = cls_loss + w_phys * smooth + PINN_WEIGHT * pinn_reg + reg_loss
    return loss, outputs, s_vals, l_vals, nu_vals, a_vals, smooth, pinn_reg


def _v110_sam_second_pass(inputs, targets, targets_a, targets_b, lam, use_mixup, epoch, w_phys,
                          model_ref=None, opt_ref=None, loss_fn=None):
    """[v110] 唯轴的第二个 pass。

    ① 用第一遍梯度构造 Sobolev 度量下的最陡上升方向（Bessel 势低通 + 全局归一）；
    ② 把参数推到 w + eps（eps 的 Euclidean 长度 = SAM_RHO）；
    ③ 同一 batch 再跑一遍 forward + backward ⇒ .grad 现在是 **w+eps 处**的梯度；
    ④ 从快照精确还原 w。调用方随后照常 clip + optimizer.step()。
    任何异常 ⇒ 就地记录 + 还原 + 把第一遍（未扰动）梯度补回来 ⇒ 退化成普通 SGD 步，
    **绝不把异常抛给训练主循环**（覆铣 279：读数装配不许有能力毁掉主结果）。
    """
    _m = model if model_ref is None else model_ref
    _opt = optimizer if opt_ref is None else opt_ref
    _fn = _v110_forward_and_loss if loss_fn is None else loss_fn
    _V110_SAM["n_steps"] += 1
    backup = None
    ratio = float(SAM_RATIO_FALLBACK)
    try:
        entries = [p for p in _m.parameters() if p.grad is not None]
        if not entries:
            return
        grads = [p.grad for p in entries]
        ratio, _nmod = _v110_field_ratio(_m)
        dirs, s2, r2 = _v110_build_directions(grads, ratio)
        if not (s2 > 0.0):
            _V110_SAM["n_degenerate"] += 1
            return
        dir_ratio = ((s2 / r2) ** 0.5) if r2 > 0.0 else 0.0
        backup = [(p, p.detach().clone()) for p in entries]
        _v110_perturb(entries, dirs, s2)
        _opt.zero_grad()
        _loss2 = _fn(inputs, targets, targets_a, targets_b, lam, use_mixup, epoch, w_phys)[0]
        _loss2.backward()
    except Exception as _e:
        _V110_SAM["n_second_fail"] += 1
        _V110_SAM["error"] = repr(_e)
        if backup is not None:
            _v110_restore(backup)
        try:
            _opt.zero_grad()
            _rec = _fn(inputs, targets, targets_a, targets_b, lam, use_mixup, epoch, w_phys)[0]
            _rec.backward()
        except Exception:
            pass
        return
    _v110_restore(backup)
    _V110_SAM["n_applied"] += 1
    _V110_SAM["ratio_sum"] += float(ratio)
    _V110_SAM["ratio_n"] += 1
    _V110_SAM["dir_sum"] += float(dir_ratio)
    _dmin = _V110_SAM["dir_min"]
    _dmax = _V110_SAM["dir_max"]
    _V110_SAM["dir_min"] = float(dir_ratio) if _dmin is None else min(_dmin, float(dir_ratio))
    _V110_SAM["dir_max"] = float(dir_ratio) if _dmax is None else max(_dmax, float(dir_ratio))


def _v110_sam_stats():
    """训练收尾的机制读数（只读 _V110_SAM；异常就地捕获，绝不毁主结果）。"""
    try:
        n_steps = int(_V110_SAM["n_steps"])
        n_applied = int(_V110_SAM["n_applied"])
        rn = int(_V110_SAM["ratio_n"])
        return {
            "n_steps": n_steps,
            "n_applied": n_applied,
            "n_degenerate": int(_V110_SAM["n_degenerate"]),
            "n_second_fail": int(_V110_SAM["n_second_fail"]),
            "applied_frac": (n_applied / n_steps) if n_steps > 0 else 0.0,
            "ratio_mean": (_V110_SAM["ratio_sum"] / rn) if rn > 0 else None,
            "dir_ratio_mean": (_V110_SAM["dir_sum"] / rn) if rn > 0 else None,
            "dir_ratio_min": _V110_SAM["dir_min"],
            "dir_ratio_max": _V110_SAM["dir_max"],
            "error": _V110_SAM["error"],
        }
    except Exception as _e:
        return {"error": repr(_e)}


def _v110_sam_verdict(stats, st_rep):
    """五档判词（判**机制**，不判精度）。任何入口异常 ⇒ READOUT_ERROR，不许崩。"""
    try:
        if st_rep is None or (not st_rep.get("all_pass", False)) or st_rep.get("error") is not None:
            return "SAM_SELFTEST_FAILED"
        if (not isinstance(stats, dict)) or stats.get("error") is not None:
            return "SAM_READOUT_ERROR"
        n_applied = int(stats.get("n_applied") or 0)
        frac = float(stats.get("applied_frac") or 0.0)
        if n_applied == 0 or frac < float(SAM_DELIVERED_MIN_FRAC):
            return "SAM_NOT_APPLIED"
        dm = stats.get("dir_ratio_mean")
        if dm is None:
            return "SAM_READOUT_ERROR"
        if float(dm) < float(SAM_DIR_RATIO_LO) or float(dm) > float(SAM_DIR_RATIO_HI):
            return "SAM_METRIC_DEGENERATE"
        return "SAM_SOBOLEV_DELIVERED"
    except Exception as _e:
        return "SAM_READOUT_ERROR:%s" % (_e,)


def _v110_sam_selftest(device=None):
    """[v110] 开机自测：构造性零随机（全 CPU 造张量再 .to(device)；**不用 RNG**
    —— 覆铣 325：随机张量一律 CPU 造 + 搬过去，不许 cpu generator 配 cuda device）。

    分母 = len(cases)（不是手数）⇒ 结构上免疫 #326 分母型假绿。
    """
    cases = {}

    def _add(name, ok, why):
        cases[name] = (bool(ok), why)

    dev = device

    # ---- case1: 奇数 N 下 fftshift 后的 bin 里必须有精确的 DC=0（老网格没有）----
    ax3 = _v110_freq_axis(3, device=dev, dtype=torch.float32)
    _add("case1_odd_dc_bin_is_zero",
         abs(float(ax3[3 // 2])) < 1e-12,
         "N=3 ⇒ 第 N//2=1 个 bin 频率恰 0（_dft_axis 给的是 -pi/3 ⇒ 错半格）")

    # ---- case2: 偶数 N 与现役 _dft_axis 一致（不许自造一套与生产不同的语义）----
    _ok2 = True
    _msg2 = []
    for _N in (4, 8, 16):
        _a = torch.sort(_v110_freq_axis(_N, device=dev, dtype=torch.float32)).values
        _b = torch.sort(_dft_axis(_N, dev, torch.float32)).values
        _d = float((_a - _b).abs().max())
        _msg2.append("N%d:%.1e" % (_N, _d))
        _ok2 = _ok2 and (_d < 1e-5)
    _add("case2_even_matches_production_grid", _ok2, " ".join(_msg2))

    # ---- case3: 核在 DC 处精确为 1 ----
    _k3 = _v110_matern_kernel(3, 3, 1.5, 1.0, device=dev, dtype=torch.float32)
    _add("case3_kernel_dc_exact_one", abs(float(_k3[1, 1]) - 1.0) < 1e-12,
         "K(0)=1 ⇒ 场不改动均匀分量")

    # ---- case4: 核随 |k| 单调衰减 ----
    _k7 = _v110_matern_kernel(7, 7, 1.0, 1.0, device=dev, dtype=torch.float32)
    _c = 7 // 2
    _row = [float(_k7[_c, j]) for j in range(_c, 7)]
    _add("case4_kernel_monotone", all(_row[i] > _row[i + 1] for i in range(len(_row) - 1)),
         "沿半径单调递减：%s" % ", ".join("%.4f" % v for v in _row))

    # ---- case5: ★ 常数卷积核必须**原样通过**（这条精确咬住教训 331 的半格错位）----
    _g1 = torch.ones(2, 2, 3, 3, device=dev, dtype=torch.float32)
    _g2 = torch.ones(2, 2, 4, 4, device=dev, dtype=torch.float32)
    _e1 = float((_v110_lowpass_grad(_g1, 1.5) - 1.0).abs().max())
    _e2 = float((_v110_lowpass_grad(_g2, 1.5) - 1.0).abs().max())
    _add("case5_constant_kernel_preserved", (_e1 < 1e-5) and (_e2 < 1e-5),
         "3x3 与 4x4 常数核保留率误差 %.2e / %.2e（老网格下 3x3 会掉到 0.313）" % (_e1, _e2))

    # ---- case6: 单频模式的增益等于解析值（不调用自己的核函数 —— 独立真值）----
    _cb = torch.zeros(1, 1, 4, 4, device=dev, dtype=torch.float32)
    for _i in range(4):
        for _j in range(4):
            _cb[0, 0, _i, _j] = 1.0 if ((_i + _j) % 2 == 0) else -1.0
    _r6 = 1.0
    _want6 = (1.0 + _r6 * 2.0 * (math.pi ** 2)) ** (-1.0)
    _f6 = _v110_lowpass_grad(_cb, _r6)
    _got6 = float(_f6[0, 0, 0, 0])
    _add("case6_nyquist_gain_matches_analytic",
         abs(_got6 - _want6) < 1e-5 and float((_f6 - _want6 * _cb).abs().max()) < 1e-5,
         "棋盘(Nyquist) 模式 ⇒ 增益 %.6f == 解析 %.6f" % (_got6, _want6))

    # ---- case7: s=0 ⇒ 核恒 1 ⇒ 恒等算子（退化为裸 SAM 的开关在手，不是死后 append）----
    _g7 = torch.arange(9, device=dev, dtype=torch.float32).reshape(1, 1, 3, 3)
    _f7 = _v110_lowpass_grad(_g7, 1.5, s_exp=0.0)
    _add("case7_s_zero_is_identity", float((_f7 - _g7).abs().max()) < 1e-5,
         "s=0 ⇒ K≡1 ⇒ 与原梯度逐点同（这是下一步裸 SAM 对照跑的开关位）")

    # ---- case8/9: 非二维空间参数原样返回（同一对象，不拷贝）----
    _g1x1 = torch.ones(4, 4, 1, 1, device=dev, dtype=torch.float32)
    _gv = torch.ones(10, device=dev, dtype=torch.float32)
    _add("case8_one_by_one_untouched", _v110_lowpass_grad(_g1x1, 1.5) is _g1x1,
         "1x1 卷积核原样返回（只有一个空间 bin，没得低通）")
    _add("case9_non4d_untouched", _v110_lowpass_grad(_gv, 1.5) is _gv,
         "bias/BN/Linear 原样返回")

    # ---- case10: lam/nu 读数（pooled：总 lam / 总 nu）----
    class _FakeField(nn.Module):
        def __init__(self, lam_val, nu_val):
            super(_FakeField, self).__init__()
            self.register_buffer("running_lam",
                                 torch.tensor([lam_val, lam_val], dtype=torch.float32))
            self.register_buffer("running_nu",
                                 torch.tensor([nu_val, nu_val], dtype=torch.float32))

    _fake = nn.Sequential(_FakeField(1.0, 2.0), _FakeField(3.0, 4.0))
    _fake = _fake.to(dev)
    _r10, _n10 = _v110_field_ratio(_fake)
    _add("case10_field_ratio_pooled",
         abs(_r10 - ((1.0 + 3.0) / (2.0 + 4.0))) < 1e-6 and _n10 == 2,
         "pooled lam/nu = (1+3)/(2+4) = %.6f（读到 %d 个场模块）" % (_r10, _n10))

    # ---- case11: 无场模块 ⇒ 回退常量（不许除零、不许崩）----
    _r11, _n11 = _v110_field_ratio(nn.Sequential().to(dev))
    _add("case11_field_ratio_fallback",
         (_n11 == 0) and abs(_r11 - float(SAM_RATIO_FALLBACK)) < 1e-12,
         "无 running_lam/nu ⇒ 回退 %.4f" % SAM_RATIO_FALLBACK)

    # ---- case12: 极端比值被钳住（场参数本身有 band clamp，这里是双保险）----
    _r12a, _ = _v110_field_ratio(nn.Sequential(_FakeField(1e9, 1.0)).to(dev))
    _r12b, _ = _v110_field_ratio(nn.Sequential(_FakeField(1.0, 1e9)).to(dev))
    _add("case12_field_ratio_clamped",
         abs(_r12a - float(SAM_RATIO_MAX)) < 1e-6 and abs(_r12b - float(SAM_RATIO_MIN)) < 1e-12,
         "上钳 %.2f / 下钳 %.4f" % (_r12a, _r12b))

    # ---- case13: 梯度整体缩放不改变方向（只缩放へ。）
    _ga = torch.arange(9, device=dev, dtype=torch.float32).reshape(1, 1, 3, 3) + 1.0
    _gb = _ga * 7.0
    _da, _sa, _ra = _v110_build_directions([_ga], 1.5)
    _db, _sb, _rb = _v110_build_directions([_gb], 1.5)
    _add("case13_direction_scale_invariant",
         abs(_sb - 49.0 * _sa) < 1e-3 and float((_db[0] - 7.0 * _da[0]).abs().max()) < 1e-4,
         "g→7g ⇒ 方向 7 倍、范数平方 49 倍（归一后同方向）")

    # ---- case14: 推出去的 Euclidean 长度恰等于 rho ----
    class _Tiny(nn.Module):
        def __init__(self):
            super(_Tiny, self).__init__()
            self.w3 = nn.Parameter(torch.zeros(2, 2, 3, 3))
            self.v = nn.Parameter(torch.zeros(5))

    _t = _Tiny().to(dev)
    _t.w3.grad = torch.ones(2, 2, 3, 3, device=dev) * 0.5
    _t.v.grad = torch.ones(5, device=dev) * 0.25
    _ent = [p for p in _t.parameters() if p.grad is not None]
    _dr, _s2, _r2 = _v110_build_directions([p.grad for p in _ent], 1.5)
    _before = [p.detach().clone() for p in _ent]
    _v110_perturb(_ent, _dr, _s2)
    _delta2 = float(sum(float((p.detach() - b).pow(2).sum()) for p, b in zip(_ent, _before)))
    _add("case14_perturb_radius_is_rho",
         abs((_delta2 ** 0.5) - float(SAM_RHO)) < 1e-5,
         "||Δw|| = %.8f == rho = %.4f" % (_delta2 ** 0.5, SAM_RHO))

    # ---- case15: 快照还原是**逐位**精确（不是加回去再减回来）----
    _v110_restore([(p, b) for p, b in zip(_ent, _before)])
    _maxdiff = max(float((p.detach() - b).abs().max()) for p, b in zip(_ent, _before))
    _add("case15_restore_bit_exact", _maxdiff == 0.0,
         "还原后逐位相同（maxdiff=%r，浮点减法做不到这个）" % (_maxdiff,))

    # ---- case16: 第二遍抛异常 ⇒ 还原 + 计数器 + 不把异常抛出去 ----
    #   ★ 隔离条款（本轮自引入缺陷的修法，见文件尾纪律）：case16/17 会真的去改
    #     **生产**那个 _V110_SAM 全局计数（n_steps / n_second_fail / n_degenerate /
    #     error）。不隔离 ⇒ 真训练一开场就带着假的 n_second_fail=1，读数终生不可信，
    #     更会在 case18 的 error is None 断言上爆出来（本轮就是这样抓到的）。
    _saved_gl = dict(_V110_SAM)
    try:
        _saved_opt = globals()["optimizer"]
        _t2 = _Tiny().to(dev)
        _t2.w3.grad = torch.ones(2, 2, 3, 3, device=dev)
        _t2.v.grad = torch.ones(5, device=dev)
        _opt2 = torch.optim.SGD([{"params": list(_t2.parameters())}], lr=0.1)
        _before2 = [p.detach().clone() for p in _t2.parameters()]
        globals()["optimizer"] = _opt2

        def _boom(*_a, **_k):
            raise RuntimeError("second pass boom")

        _n_fail_before = int(_V110_SAM["n_second_fail"])
        try:
            _v110_sam_second_pass(None, None, None, None, None, False, 1, 0.1,
                                  model_ref=_t2, opt_ref=_opt2, loss_fn=_boom)
            _ok16 = True
        except Exception as _e:
            _ok16 = False
            flush_print("  [v110 自测] case16 异常外泄: %r" % (_e,))
        finally:
            globals()["optimizer"] = _saved_opt
        _maxdiff2 = max(float((p.detach() - b).abs().max())
                        for p, b in zip(_t2.parameters(), _before2))
        _add("case16_second_pass_failure_is_contained",
             _ok16 and (_maxdiff2 == 0.0)
             and int(_V110_SAM["n_second_fail"]) == _n_fail_before + 1,
             "第二遍崩 ⇒ 权重逐位还原、n_second_fail+1、异常不外泄（调用方退化成普通步）")

        # ---- case17: 梯度全零 ⇒ degenerate，且**不调用**第二遍 ----
        _t3 = _Tiny().to(dev)
        for p in _t3.parameters():
            p.grad = torch.zeros_like(p.detach())

        class _CountingBoom(object):
            n = 0

        def _boom2(*_a, **_k):
            _CountingBoom.n += 1
            raise RuntimeError("boom2")

        _n_deg_before = int(_V110_SAM["n_degenerate"])
        _v110_sam_second_pass(None, None, None, None, None, False, 1, 0.1,
                              model_ref=_t3, opt_ref=_opt2, loss_fn=_boom2)
        _add("case17_zero_grad_degenerate",
             int(_V110_SAM["n_degenerate"]) == _n_deg_before + 1
             and _CountingBoom.n == 0,
             "范数为 0 ⇒ n_degenerate+1 且第二遍一次都没被调用")
    finally:
        _V110_SAM.clear()
        _V110_SAM.update(_saved_gl)

    # ---- case18: 记账（临时改全局态，finally 还原）----
    _saved = dict(_V110_SAM)
    try:
        _V110_SAM["n_steps"] = 100
        _V110_SAM["n_applied"] = 95
        _V110_SAM["ratio_sum"] = 10.0
        _V110_SAM["ratio_n"] = 5
        _V110_SAM["dir_sum"] = 2.5
        _V110_SAM["dir_min"] = 0.3
        _V110_SAM["dir_max"] = 0.7
        _s18 = _v110_sam_stats()
        _ok18 = (_s18["n_steps"] == 100 and _s18["n_applied"] == 95
                 and abs(_s18["applied_frac"] - 0.95) < 1e-15
                 and abs(_s18["ratio_mean"] - 2.0) < 1e-12
                 and abs(_s18["dir_ratio_mean"] - 0.5) < 1e-12
                 and _s18["error"] is None)
        _add("case18_stats_bookkeeping", _ok18,
             "n/applied_frac/ratio_mean/dir_ratio_mean 与手算对上，error=None")
    finally:
        _V110_SAM.clear()
        _V110_SAM.update(_saved)

    # ---- case19:健康读数 ⇒ DELIVERED ----
    _st_ok = {"n_pass": len(cases) + 2, "n_expected": len(cases) + 2,
              "all_pass": True, "error": None, "completed": True}
    _healthy = {"n_steps": 100, "n_applied": 95, "applied_frac": 0.95,
                "ratio_mean": 1.2, "dir_ratio_mean": 0.45,
                "dir_ratio_min": 0.3, "dir_ratio_max": 0.6, "error": None}
    _add("case19_verdict_delivered",
         _v110_sam_verdict(_healthy, _st_ok) == "SAM_SOBOLEV_DELIVERED",
         "健康 stats + 自测绿 ⇒ DELIVERED")

    # ---- case20: 其余四档全都要能落到（右下墓）----
    _na = dict(_healthy)
    _na["n_applied"] = 0
    _nf = dict(_healthy)
    _nf["applied_frac"] = 0.5
    _dg1 = dict(_healthy)
    _dg1["dir_ratio_mean"] = 0.01
    _dg2 = dict(_healthy)
    _dg2["dir_ratio_mean"] = 0.99
    _st_bad = dict(_st_ok)
    _st_bad["all_pass"] = False

    class _Boom(dict):
        def get(self, *a, **k):
            raise RuntimeError("boom")

    _add("case20_verdict_other_branches",
         _v110_sam_verdict(_na, _st_ok) == "SAM_NOT_APPLIED"
         and _v110_sam_verdict(_nf, _st_ok) == "SAM_NOT_APPLIED"
         and _v110_sam_verdict(_dg1, _st_ok) == "SAM_METRIC_DEGENERATE"
         and _v110_sam_verdict(_dg2, _st_ok) == "SAM_METRIC_DEGENERATE"
         and _v110_sam_verdict(_healthy, _st_bad) == "SAM_SELFTEST_FAILED"
         and str(_v110_sam_verdict(_Boom(), _st_ok)).startswith("SAM_READOUT_ERROR"),
         "NOT_APPLIED x2 / METRIC_DEGENERATE x2 / SELFTEST_FAILED / READOUT_ERROR 全可达")

    rep = {"n_pass": 0, "n_expected": len(cases), "all_pass": False, "failed": [],
           "error": None, "completed": False}
    try:
        for _k in sorted(cases):
            _ok, _why = cases[_k]
            if _ok:
                rep["n_pass"] += 1
            else:
                rep["failed"].append(_k)
                flush_print("  [v110 自测 FAIL] %s: %s" % (_k, _why))
        rep["completed"] = True
        # 覆铣 312：all_pass 同防「case 红」与「提前崩」——
        # 分母 = len(cases)（不是手写的数字），completed + error 双守卫。
        rep["all_pass"] = (rep["completed"]
                           and rep["error"] is None
                           and rep["n_pass"] == rep["n_expected"])
    except Exception as _e:
        rep["error"] = repr(_e)
        rep["all_pass"] = False
    return rep


def _v110_sam_selftest_cases_count():
    """[v110] 自测条数的**独立真值**： literally 数源码里的 _add( 调用点。

    覆铣 326（手数分母必错）：部署检查用这条去对 n_expected，红 ⇒ 分母又漂了。
    """
    try:
        with open(os.path.abspath(__file__), "r", encoding="utf-8") as _f:
            _txt = _f.read()
    except Exception:
        return -1
    _i = _txt.find("def _v110_sam_selftest(")
    _j = _txt.find("def _v110_sam_selftest_cases_count(")
    if _i < 0 or _j < 0:
        return -1
    _seg = _txt[_i:_j]
    _n = 0
    for _ln in _seg.split("\n"):
        if "_add(" in _ln and "def _add" not in _ln:
            _n += 1
    return _n


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
            targets_a_ctx, targets_b_ctx, lam_ctx = targets_a, targets_b, lam
        else:
            targets_a_ctx, targets_b_ctx, lam_ctx = None, None, None

        # [v110] 前向 + 完整损失抽成**单点实现**，两遍 fwd/bwd 共用同一份
        #   （覆铣 306：同一份逻辑只许一个实现）。算式与 v103 逐字同。
        loss, outputs, s_vals, l_vals, nu_vals, a_vals, smooth, pinn_reg = _v110_forward_and_loss(
            inputs, targets, targets_a_ctx, targets_b_ctx, lam_ctx, use_mixup, epoch, w_phys)
        loss.backward()
        # [v110] 唯一一根轴：在 w+eps 处再取一遍梯度（完成后 .grad 换成该处梯度，权重已精确还原）
        _v110_sam_second_pass(inputs, targets, targets_a_ctx, targets_b_ctx, lam_ctx,
                              use_mixup, epoch, w_phys)

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
        _safe_save(model.state_dict(), 'best_physics_resnet_v51.pth', 'best_plain')
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
    # [v97] 走守卫；没写成就**不进 pool** —— pool 里的每一份都必须真的在盘上，
    #   否则 ensemble 阶段会去 load 一个不存在的文件（那才是真的事故）。
    if not _safe_save(model.state_dict(), filepath, 'ens_ep%d' % int(epoch)):
        return False
    ckpt_pool.append((epoch, test_acc, filepath))
    # Sort by test_acc descending, keep top ENS_MAX_KEEP
    ckpt_pool.sort(key=lambda x: x[1], reverse=True)
    if len(ckpt_pool) > ENS_MAX_KEEP:
        # Remove worst checkpoint from disk
        removed = ckpt_pool.pop()
        try:
            os.remove(removed[2])
        except OSError as _rm_e:
            # [v97] 原来是裸 `except OSError: pass` —— 删失败被静默吞掉，
            #   被淘汰的 ckpt 变成幽灵文件继续占盘，而且完全无声。现在计数并打印。
            DISK_GUARD["remove_failed"] = int(DISK_GUARD.get("remove_failed") or 0) + 1
            _disk_push_event("RM_FAIL %s %s: %s" % (removed[2], type(_rm_e).__name__, _rm_e))
            flush_print("  [disk] 淘汰旧 ckpt 失败（幽灵文件，占盘）: %s" % removed[2])
    return True

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
# [v49] 探针必须在训练循环之前跑（秒级）；它自己负责还原 RNG / 权重 / 梯度 / 模式。
# ★★ [v117] **只读仪器绝不许杀主流程**：v116 就是死在这个模块级裸调用上
#     （CUDNN_STATUS_NOT_INITIALIZED ⇒ 整跑当场死，180ep 一次没跑）。
#     ⇒ 换成守卫壳：崩了只记 error + 无条件还原确定性开关 + 保证
#       `det_probe_report` 一定是 dict（否则下游 .get() 二次崩）。
_v117_det_guard = _v117_guarded_determinism_probe()
flush_print("  [v117] 确定性探针守卫：ok=%s | verdict=%s | error=%s"
            % (_v117_det_guard.get("ok"),
               (det_probe_report or {}).get("verdict"),
               (det_probe_report or {}).get("error")))
# ★★ [v117] CUDA 健康探针：v116 的崩溃是**次生**的（探针 OOM ⇒ 之后任何 conv
#   都报 CUDNN_STATUS_NOT_INITIALIZED）⇒ 训练前最后一刻直接问一次 cuDNN。
_v117_canary = _v117_cuda_canary()
flush_print("  [v117] CUDA 健康探针：ran=%s | ok=%s | error=%s"
            % (_v117_canary.get("ran"), _v117_canary.get("ok"),
               _v117_canary.get("error")))
if _v117_canary.get("ran") and not _v117_canary.get("ok"):
    flush_print("        ⚠⚠ cuDNN 现场不健康：训练首步可能崩；本条是**独立证据**（**不 exit**）")

# [v96] 频率标尺自测（秒级；**不消耗全局 RNG** ⇒ RNG 序列与 v51 共享 ⇒ 跨跑可比）
_v96_dft_st = _dft_grid_selftest(device)
flush_print(f"  [v96 频率标尺自测] {_v96_dft_st['n_pass']}/{_v96_dft_st['n_expected']} "
            f"{'ALL PASS' if _v96_dft_st['all_pass'] else 'FAIL'}"
            + (f"  error={_v96_dft_st['error']}" if _v96_dft_st.get('error') else ""))

# [v101] 各向异性轴自测（秒级；构造 nn.Module 会耗 RNG ⇒ 快照/还原，case5 验证还原）
_aniso_st = _aniso_selftest(device)
_v110_st = _v110_sam_selftest(device)
_v113_pfst = _v113_pinface_selftest(device)
flush_print("  [v116] ★★ 唯一一根轴 = 约束通道：物理约束项 pinn 的作用面（**同轴第四次重测** · 探针内部五修）")
flush_print("        ★ v113/v114/v115 三次都没测到这根轴（三次都是探针自己崩）⇒ 未测 ≠ 判负 ⇒ 第四次重测同一根轴")
flush_print("        ★ 本版**只改探针**（仪器层）：分节降级 | levels 双写 | grad_leaks 真写 | 安全 sqrt | 训练前自检")
flush_print("        ★ 训练数学与全部计算**逐字零改动** ⇒ 预期 delivered 仍 ≈ 97.4（NULL），价值在机制读数")
flush_print("        轴沿用 v113/v114（那根轴两次都没测到 ⇒ 未测 ≠ 判负 ⇒ 第三次重测）")
flush_print("        P0-A 探针输入搬到模型 device | P0-B BN 统计快照/还原 | P0-C OUT_DIR 开机即建")
flush_print("        [v117] ★ 发车路径三修：探针小批(%d) | CUDA 清扫 | 确定性探针不杀主流程"
            % int(V117_PROBE_N_SMALL))
flush_print("               v116 实跑死在 run_determinism_probe() 的 CUDNN_STATUS_NOT_INITIALIZED"
            " ⇒ 整跑当场死；本版把它关进守卫壳 + 探针改小批 + 跑完清显存")
flush_print("        训练数学与全部计算逐字未改 ⇒ 预期 delivered 仍 ≈ 97.4（NULL）；本版要的是机制读数")

flush_print(f"  [v113 作用面自测] {_v113_pfst['n_pass']}/{_v113_pfst['n_expected']} "
             f"{'ALL PASS' if _v113_pfst['all_pass'] else 'FAIL'}"
             + (f"  error={_v113_pfst.get('error')}" if _v113_pfst.get("error") else ""))
flush_print(f"  [v110 SobolevSAM自测] {_v110_st['n_pass']}/{_v110_st['n_expected']} "
            f"{'ALL PASS' if _v110_st['all_pass'] else 'FAIL'}")
flush_print(f"  [v101 各向异性自测] {_aniso_st['n_pass']}/{_aniso_st['n_expected']} "
            f"{'ALL PASS' if _aniso_st['all_pass'] else 'FAIL'}"
            + (f"  error={_aniso_st['error']}" if _aniso_st.get('error') else ""))
if not _v96_dft_st["all_pass"]:
    flush_print("  [v96] 未通过项: " + ", ".join(
        k for k, v in _v96_dft_st["checks"].items() if not v["pass"]))
flush_print(f"  [v96] flip 等变差: 新网格 {_v96_dft_st.get('flip_diff_new')} "
            f"vs 旧网格 {_v96_dft_st.get('flip_diff_old')}  |  "
            f"DC 偏离: 新 {_v96_dft_st.get('dc_dev_new')} vs 旧 {_v96_dft_st.get('dc_dev_old')}")
flush_print(f"  [v96] 核差 max|K_new - K_old| = {_v96_dft_st.get('kernel_max_abs_diff')} "
            f"(recipe 域改动：第 0 步就与 v51 不同 ⇒ 只有跨跑判据)")
_v96_print_fingerprint()
flush_print(f"  [v100] ★★ 本版**唯一一根轴** = 判据/仪器修正（**零训练数学改动**）")
flush_print(f"        训练数学与 v99 逐位相同：无参数/调度/RNG/损失项/物理场常量变化。")
flush_print(f"        改的是尺子，且尺子被证明错在五处：")
flush_print(f"          (1) σ_diff 0.108→{PLATEAU_SIGMA_MEASURED}（一对→四对 RMS，低估 1.565×）")
flush_print(f"          (2) 四条判据线全部由常量派生（v99 的 97.27/97.61 还是硬编码）")
flush_print(f"          (3) lr_tail 剂量门拆两门（平台通道=硬守卫 / 交付通道=事实+四档判词）")
flush_print(f"          (4) argmax 并列成机器字段（v99 的并列是读曲线才发现的）")
flush_print(f"          (5) 修掉被自己数据证伪的 expected_delivered_vs_v98（97.42 vs 实跑 97.44）")
flush_print(f"        派生线：平台 PASS>={PLATEAU_PASS} / 红线<={PLATEAU_REDLINE} | "
            f"v51 通道 PASS>={PASS_LINE_96} / 红线<={REDLINE_96}")
flush_print(f"        （熔断红线由 BACKTRACK_BASELINE_V46 − 2σ 派生，打印在收尾判据块；"
            f"它在本横幅之后才绑定 ⇒ 这里不引用它的名字，覆辙 279）")
flush_print(f"        ⚠ 诚实：平台 PASS 线 {PLATEAU_PASS} > 历史最好平台 {PLATEAU_BEST_EVER_PP}"
            f" ⇒ 单跑平台判词算术上预先决定 NULL（可达? {PLATEAU_PASS_REACHABLE}）。")
flush_print(f"        验收三眼：① 本横幅；② stdout 的 `[v100] 剂量门拆分` 块；"
            f"③ JSON 的 v100_instrument_fix + lr_tail_delivered_verdict。")
flush_print(f"  [v101] ★★ 唯一一根轴 = 各向异性 Matérn 核（v90 轴在修正标尺上重开）")
flush_print(f"        核从径向 |k|^2 = kx^2+ky^2 改为 k_sq_g = kx^2 + a_g*ky^2，"
            f"a_g = 1 + {ANISO_RANGE}*tanh(param_g) ∈ (0,2]；只开 Matern 两支，Fixed 不动")
flush_print(f"        归因纯度：aniso_param zeros 初始化 ⇒ 第 0 步核与 v100 逐位相同"
            f"（1.0*t 精确）且 zeros 不耗 RNG ⇒ 初始化序列与 v100 逐位相同")
flush_print(f"        v90 两条假红已校准：① f32 tanh({ANISO_TANH_SAT_PARAM:.0f})=1.0 精确饱和 "
            f"⇒ a=2.0 精确是**预期**；② flip 等变在修正 DFT 网格上成立（旧网格上各向同性核也不等变）")
flush_print(f"        验收三眼：① 本横幅；② [v101 各向异性自测] N/N ALL PASS；"
            f"③ 收尾 aniso verdict + JSON v101_anisotropy")
flush_print(f"  [v110] ★★ 唯一一根轴 = Sobolev 度量 SAM（用场自己的 Bessel 势度量换掉欧氏信任域）")
flush_print(f"        eps = rho * K*g / ||K*g||, K(k) = (1 + (lam/nu)|k|^2)^(-s)，lam/nu 取场 running 统计")
flush_print(f"        rho={SAM_RHO} s={SAM_SOBOLEV_S}；**lr / momentum / mixup 强度一个都不动**（绕开 v106-v109 根因）")
flush_print(f"        杂交：每步 2 次 fwd+bwd（墙钟 ~1.8-2x），第二轮多抽一次 SD 掩码 ⇒ RNG 流不再逐位可比")
flush_print(f"        验收三眼：① 本横幅；② [v110 SobolevSAM自测] N/N ALL PASS；③ 收尾 JSON v110_sam.verdict")
flush_print(f"  [v113] 约束通道：物理约束项 **pinn 的作用面**（v113 原版；本版 v115 已做读数三修）")
flush_print(f"        pinn 的计算点：layer4 输出 deep（gate3 **之后** = 含物理注入）"
             f"  ->  layer2 输出 h2（gate2 **之前** = 注入之前）")
flush_print(f"        ⇒ pinn 的梯度**结构性到不了任何物理参数**（拓扑切断，不是标量调权）"
             f" ⇒ 兑现 v36「物理只作被动先验」")
flush_print(f"        ★ 前向**逐位不变**（pinn 只进 loss）⇒ 本版只改**梯度路径**")
flush_print(f"        ★ 为什么是 layer2 而不是 layer3：实测 layer3 处仍会经 layer2→gate2→ph2 漏回 phys2")
flush_print(f"        ★ 这是对 v111 归档结论的**证伪性检验**：切掉塑形路径后场若恢复内容"
             f"（band_frac ≥ {V113_BAND_FRAC_MIN}，v111 实测 {V113_BASELINE_BAND_FRAC}）⇒ 注入通道可能复活")
flush_print(f"        判据（预注册）：基线 v110 = {V113_BASE_DELIVERED} / pass_line = {V113_PASS_LINE} (+2σ) / 红线 {V113_REDLINE}")
flush_print(f"        预期（纪律 336）：精度多半 NULL；唯一能推翻预期的机制 = 场恢复内容")
flush_print(f"        验收三眼：① 本横幅 + [v113 作用面自测] N/N ALL PASS；② 收尾 JSON v113_pinface.verdict；"
             f"③ grad_frac_new ≈ 0 且 grad_frac_old > 0")
flush_print(f"  [v103] ★★ 唯一一根轴 = 换 seed：SEED 1234 -> {SEED}（协议 mandated 的第 2 个 seed）")
flush_print(f"        训练数学与 v102 **逐字相同**：只改 SEED 一个整数；参数/调度/损失/物理常量/RNG 位点全不动。")
flush_print(f"        seed 的全部消费者（4 个 manual_seed + worker_init_fn + 路径/JSON 名）都从 SEED 派生 ⇒ 换得干净；")
flush_print(f"        遗留 1234 字面仅剩历史记录与三个独立探针 seed（12345/2026096/2026092，跨版可比必须固定）。")
flush_print(f"        为什么是 seed 42：归档只有两个 seed（v43=42、v44 起全 1234），v90/v101/v102 三次")
flush_print(f"        各向异性观测其实是**同一个** seed ⇒ 协议要求补第 2 个 seed 才能谈轴有效性。")
flush_print(f"        顺带修 provenance：v102 实跑文件名 loss.mmm1.py、self_sha1 与交付片不符（B 线第一次")
flush_print(f"        内容级破损）⇒ 本版必须 cnn10_v103.py 之名跑，开机横幅 sha1 须等于交付值。")
flush_print(f"        判据预注册：与 v102 同代码不同 seed ⇒ 预期落在 σ_diff={PLATEAU_SIGMA} 内；本跑目的是")
flush_print(f"        **量当前代码的 seed 方差**，不主张精度结论（单跑 NULL 不得升级为'轴无效'，需 3 seed）。")
flush_print(f"        验收三眼：① 本横幅 + sha1 与交付片一致；② [v101 各向异性自测] 15/15 且 completed；"
            f"③ 收尾 aniso verdict + JSON v103_seed_pair")
flush_print(f"  [v99] ★★ 本版**无新轴** —— 纯 P0 崩溃修复，训练数学与 v98 逐位相同")
flush_print(f"        修的是：`_v98_gran` 在 results **字面量内部**被引用，却在 251 行之后才赋值")
flush_print(f"        ⇒ 模块级 NameError ⇒ v98 跑满 180 轮后整份 JSON 丢失（覆辙 279 同款，v93 亦同）。")
flush_print(f"        修法：results 字面量**永不**引用后置变量；读数改为构造后回填 + _safe_readout。")
flush_print(f"        另加：主结果先落 *_core.json；判词 stdout 双通道；namescope 已并入部署自证。")
flush_print(f"        验收：delivered 应**精确等于** 97.42（v98 同 seed）；不等 ⇒ 非确定性，不是轴效应。")
flush_print(f"  [v98] ★★ 本版**唯一一根轴** = gate2 通道粒度：phys2 groups 8 -> {PHYS2_GROUPS}"
            f"（每 {512 // PHYS2_GROUPS} 通道一套 Matern 核；v97 是每 64 通道一套）")
flush_print(f"        对照组：phys1 groups={PHYS1_GROUPS} / phys3 groups={PHYS3_GROUPS}"
            f"（必须仍是 8 ⇒ 结构上保证只动了一处）")
flush_print(f"        跨跑只差这一处：s/lam/nu 是 torch.zeros、running_* 是 torch.full"
            f" ⇒ 改 groups 不耗 RNG，初始化与 v97 逐位相同。")
flush_print(f"        代价（诚实）：phys2 的 FFT 调用 8 -> {PHYS2_GROUPS} 次/前向"
            f"（@16x16，估算整跑 +~80s；**不**改批量 FFT，那不保证逐位相同）。")

flush_print("\n" + "="*60)
flush_print(f"{VERSION} Training Configuration (= v51 逐字回退 + DFT 频率网格修正):")
flush_print(f"  [v96] ★★ 本版**唯一一根轴** = 频率网格修正（继承 bug，非新机制）：")
flush_print(f"        ky/kx: torch.linspace(-pi,pi,N) [step 2pi/(N-1)] -> _dft_axis(N) [step 2pi/N]")
flush_print(f"        ⇒ flip 等变恢复（旧 6.4e-02 -> 新 ~2e-07）、DC 增益回到 1；")
flush_print(f"        ⇒ 交付口径 delivered = max-of-N TTA 唯一依赖 flip 等变，故直接命中。")
flush_print(f"  [v96] 下面 [v51] 各行 = 逐字继承的配方说明，**不是**本版的轴。")
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
flush_print(f"  [v51] 判据用**实测** σ_diff={PLATEAU_SIGMA:.3f}pp（四对同口径跑 |Δdelivered| 的 RMS）⇒ 平台 PASS>={PLATEAU_PASS} / 红线<={PLATEAU_REDLINE}（可达? {PLATEAU_PASS_REACHABLE}）。")
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
    # [v101] 各向异性 per-epoch 运行最大值（主本侧读；理由见 aniso_trace 初始化处）
    aniso_trace.append((epoch,
                        round(float(model.phys2.aniso_param.detach().abs().max()), 6),
                        round(float(model.phys3.aniso_param.detach().abs().max()), 6)))

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
        _ens_ok = save_ckpt(epoch, val_acc)
        flush_print(f"  [ENS] {('Saved' if _ens_ok else 'SKIPPED')} ckpt ep{epoch} (acc={val_acc:.2f}%) | pool={len(ckpt_pool)}")

    run_tta = epoch >= TTA_START_EPOCH and epoch % TTA_INTERVAL == 0
    if run_tta:
        tta_acc = test_tta(epoch)
        history_tta.append({"epoch": int(epoch), "acc": round(float(tta_acc), 4)})
        if tta_acc > best_tta_acc:
            best_tta_acc = tta_acc
            best_tta_epoch = epoch
            no_improve_count = 0
            _safe_save(model.state_dict(), 'best_tta_resnet_v51.pth', 'best_tta')
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
# ---- [v113] 约束通道：pinn 作用面探针（只读；不改任何训练态）----
# ★★ [v115] 本块沿用 v114 的**位置**（`delivered` 绑定之后 ⇒ 不再 NameError），
#   并在其上做**三处读数修复**（训练数学与全部计算逐字未改）：
#     【P0-A】探针输入搬到模型 device（v114 实跑在这崩 ⇒ 机制读数全丢）
#     【P0-B】BN running stats 快照 / 还原（探针不许改写末态模型的 BN 统计）
#     【P0-C】OUT_DIR 开机即建 ⇒ 判词预落盘不再因目录不存在而失败
#
#   ★ P0-B 的必要性（v113/v114 都藏着这个坑）：探针为求梯度会 `_net.train()`
#     ⇒ BatchNorm 按 batch 统计更新 running_mean/running_var ⇒
#     **一次读数就把末态模型的 BN 统计改写了** ⇒ 快照 + 还原。
_V115_BN_SNAP = {}
try:
    for _bn_name, _bn_buf in model.named_buffers():
        if (("running_" in _bn_name) or ("num_batches_tracked" in _bn_name)):
            _V115_BN_SNAP[_bn_name] = _bn_buf.detach().clone()
except Exception:
    _V115_BN_SNAP = {}
_v115_rng_state_before = torch.random.get_rng_state()
_v115_probe_inputs, _v115_probe_dev, _v115_probe_align_ok = _v115_align_probe_inputs(model, _v114_probe_inputs)
flush_print("  [v115] 探针输入 device 对齐：%s（ok=%s）"
            % (_v115_probe_dev, bool(_v115_probe_align_ok)))
# ★★ [v116] 探针调用点改用**加固版** `_v116_pinface_probe`（训练数学无关）。
# ★★ [v117] 主探针改为**整批优先 + 小批自动降级**。
#   训练后显存碎片化，整批 256 张的带图前反向 + 两次 autograd.grad 极易 OOM
#   （v116 的 precheck 就是这么死的）⇒ 整批失败就清显存改小批，并**自曝**批量。
#   ★ 返回值**仍是 probe dict 本体**（附加 probe_n / small_batch_fallback）
#     ⇒ 下游 `_v113_probe.get(...)` 的口径**完全不变**。
_v113_probe = _safe_readout(
    lambda: _v117_main_probe(model, _v115_probe_inputs, TRAIN_EPOCHS),
    {"ok": False, "error": "PROBE_NOT_RUN"})
_v117_cuda_sweep()
flush_print("  [v117] 主探针：ok=%s | probe_n=%s | 小批降级=%s | OOM疑似=%s"
            % (bool(isinstance(_v113_probe, dict) and _v113_probe.get("ok")),
               (_v113_probe or {}).get("probe_n"),
               (_v113_probe or {}).get("small_batch_fallback"),
               (_v113_probe or {}).get("oom_suspected")))
# ★★ [v115] P0-B：探针跑完**还原** BN running stats（探针是只读读数）
_v115_bn_restore_ok = True
try:
    with torch.no_grad():
        for _bn_name2, _bn_buf2 in model.named_buffers():
            if _bn_name2 in _V115_BN_SNAP:
                _bn_buf2.copy_(_V115_BN_SNAP[_bn_name2])
except Exception:
    _v115_bn_restore_ok = False
_v115_rng_changed = bool(not torch.equal(_v115_rng_state_before,
                                      torch.random.get_rng_state()))
_v113_probe_ok = bool(isinstance(_v113_probe, dict) and _v113_probe.get("ok"))
_v113_pfst_ok = bool(isinstance(_v113_pfst, dict) and _v113_pfst.get("all_pass"))
_v113_verdict = _v113_pinface_verdict(
    _v113_probe.get("grad_frac_new", 1.0) if _v113_probe_ok else 1.0,
    _v113_probe.get("grad_frac_old", 0.0) if _v113_probe_ok else 0.0,
    (_v113_probe.get("levels", {}) or {}).get(
        "band_frac_mean",
        _v113_probe.get("band_frac_mean", 0.0)) if _v113_probe_ok else 0.0,
    float(delivered),
    selftest_ok=_v113_pfst_ok, readout_ok=_v113_probe_ok)
# ★★ [v115] P0-C：探针读数**同时打 stdout**（崩了也能从控制台捞回）。
flush_print("  [v115] ★ 作用面探针读数：grad_frac_new = %.6e | grad_frac_old = %.6e"
            " | band_frac_mean = %.6f"
            % (float(_v113_probe.get("grad_frac_new", -1.0)) if _v113_probe_ok else -1.0,
               float(_v113_probe.get("grad_frac_old", -1.0)) if _v113_probe_ok else -1.0,
               float((_v113_probe.get("levels", {}) or {}).get(
                   "band_frac_mean", _v113_probe.get("band_frac_mean", -1.0)))
               if _v113_probe_ok else -1.0))
if _v113_probe_ok:
    flush_print("  [v115]   新作用点漏到的物理参数：%r（应为空）"
                % (list(_v113_probe.get("grad_leaks", [])),))
else:
    flush_print("  [v115] ⚠ 探针未成功：%r（判词走 READOUT_ERROR，不许当机制读数）"
                % (_v113_probe.get("error"),))
flush_print("  [v115] 作用面判词 = %s" % (_v113_verdict,))
# ★★ [v115] 判词块**先落盘**（results 落盘之前）⇒ 后面再崩也不丢。
_v115_prewrite_done = False
try:
    os.makedirs(OUT_DIR, exist_ok=True)
    _v115_pw = os.path.join(OUT_DIR, "v115_verdict_%s_seed%s.json" % (VERSION, SEED))
    _v115_payload = {"kind": "v115_verdict_prewrite",
                     "version": VERSION,
                     "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "verdict": _v113_verdict,
                     "probe": _v113_probe,
                     "probe_device": _v115_probe_dev,
                     "probe_align_ok": bool(_v115_probe_align_ok),
                     "bn_restore_ok": bool(_v115_bn_restore_ok),
                     "delivered": float(delivered)}
    _v115_r = _safe_write_json(_v115_payload, _v115_pw)
    _v115_prewrite_done = bool(_v115_r)
    flush_print("  [v115] 判词预落盘 %s（ok=%s）" % (_v115_pw, _v115_prewrite_done))
except Exception as _exc:
    flush_print("  [v115] ⚠ 判词预落盘失败（已隔离）: %s" % type(_exc).__name__)
delivered_source_v47 = "final_tta_legacy_max_of_n(v49 交付链零后处理，口径与 v45/v47/v48 同尺)"
# 判据（预注册，不放宽）。参照系的选取见下"为什么红线由 v46 派生"。
#   <1> 交付口径（跨跑可比，与 v43/v44/v45/v46 同一把尺):
#       PASS      delivered >= PASS_LINE_96   (= v51 97.56 + 2σ, σ=0.169pp)
#       NULL      BACKTRACK_REDLINE <= delivered < PASS_LINE_96
#       BACKTRACK delivered <= BACKTRACK_REDLINE (= v46 97.38 − 2σ)
#   <2> 为什么本版没有同跑内配对:（v46 有，v47 没有 —— 这点必须写清）
#       本版无法在单跑内构造配对对照（lr 一改，整条轨迹就变了，不存在"同一跑里的 control"）
#       ⇒ 本版**只有跨跑判据**，因此必须靠 **3 个 seed** 才敢说"有/无效应"。
#       单跑读到 NULL **不得**升级为"轴无效"（那条结论要 3 seed）。
#   ★★ [v100] 回退红线**由公式派生**，不再写死 97.27：
#      97.27 是"v46 97.38 − 2σ(0.06)"再手工对齐到的数，而 0.06 这个 σ 与
#      PLATEAU_SIGMA（0.169）不是同一把尺 ⇒ 两处 σ 打架（覆辙 145 的同源变种）。
#      现在统一用 SIGMA_DIFF_PP：97.38 − 2×0.169 = 97.042 ⇒ **97.04**（归档定案同值）。
#      它是"防止断崖"的保险丝，不是统计检验（同 seed ≠ 可复现，覆辙 141）。
#   ★★★ 档位与红线必须用**同一个比较算子**，否则边界会自相矛盾
#     （自测发现：`verdict` 用 `<=` 而 `backtrack_advice` 用 `<`，在 delivered 恰为红线时
#      会给出 "BACKTRACK" 的档位却配 "DO_NOT_BACKTRACK" 的动作 —— 同一份 JSON 自相矛盾。
#     这正是覆辙 145 要防的病：红线必须落在机器字段上，且字段之间不许打架。）
#     ⇒ 统一定义：`backtrack_needed = (delivered <= BACKTRACK_REDLINE)`，两处共用。
BACKTRACK_BASELINE_V46 = 97.38   # [v100] v46 实跑 delivered（同 seed=1234 的唯一可直接对齐参照）
BACKTRACK_REDLINE = round(BACKTRACK_BASELINE_V46 - 2.0 * float(SIGMA_DIFF_PP), 2)   # -> 97.04
backtrack_needed = bool(float(delivered) <= float(BACKTRACK_REDLINE))
if delivered >= float(PASS_LINE_96):
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

# ---- [v100] 剂量门**拆成两门**（v99 用一把门判两个通道 ⇒ 过度严格）----
#   v99 实测（results_cnn_v99_seed1234.json）证明一把门不够：
#     · 剂量 ep121–156 **真跑了 36 轮**（n_changed_full_horizon=80、n_changed_in_this_run=36）
#     · 平台窗 ep135–156 **完整落在剂量窗内** ⇒ 平台读数**是带剂量的**
#     · 但交付选择点 = ep120（TTA 在 ep120 与 ep126 并列 97.44，first-wins 选中前者）
#       ⇒ "剂量窗 ∩ 交付窗 = ∅" 是**选点规则的产物**，不是剂量没送达
#   ⇒ 一把 `dose_delivered` 把"平台带剂量"这个事实一起作废 ⇒ INSTRUMENT_FAIL 是假红。
#   ⇒ 拆成：平台通道（主门，仍是硬守卫）+ 交付通道（事实字段 + 独立四档判词）。
_tail_changed_set = set(int(e) for e in _lr_tail_changed)
_plateau_epochs = ([int(x["epoch"]) for x in history_tta][-int(PLATEAU_WINDOW):]
                   if len(history_tta) >= 1 else [])
_tail_n_in_plateau = len([e for e in _plateau_epochs if e in _tail_changed_set])
_tail_plateau_fully_dosed = bool(len(_plateau_epochs) >= 1
                                 and _tail_n_in_plateau == len(_plateau_epochs))
_tail_dose_reaches_plateau = bool(_tail_n_in_plateau >= 1)
# argmax 并列：v99 的并列是**读曲线才发现的** ⇒ v100 起它是机器字段
_tta_accs_all = [float(x["acc"]) for x in history_tta]
_tta_eps_all = [int(x["epoch"]) for x in history_tta]
_tta_max_val = max(_tta_accs_all) if _tta_accs_all else None
_argmax_eps = ([e for e, a in zip(_tta_eps_all, _tta_accs_all)
                if _tta_max_val is not None and abs(a - _tta_max_val) <= 1e-12]
               if _tta_accs_all else [])
_any_tie_in_dose = bool(len([e for e in _argmax_eps if e in _tail_changed_set]) >= 1)
_selection_in_dose = bool(int(best_tta_epoch) in _tail_changed_set)
# ---- [v100] 交付通道的**独立**判词（四档；v99 之前只有"送达/没送达"一档）----
#   DOSE_NOT_RUN                                剂量一轮都没跑（v47 真空集）
#   DOSE_DELIVERED                              剂量起点 <= 交付选择点
#   DOSE_AFTER_SELECTION_BUT_TIE_INSIDE_DOSE    v99 形态：剂量晚于选择点，但并列点在剂量窗内
#                                               ⇒ 换 tie-break 规则就会改变读数
#   DOSE_AFTER_SELECTION_DELIVERED_BLIND        v47 形态：剂量晚于选择点且无并列点落在剂量窗内
#                                               ⇒ 交付读数在数学上不含本轴信息
if int(_lr_tail_n_run) < 1:
    lr_tail_delivered_verdict = "DOSE_NOT_RUN(n_changed_in_this_run=0 => 本跑不可归因)"
elif bool(_tail_dose_reaches_delivery):
    lr_tail_delivered_verdict = "DOSE_DELIVERED"
elif bool(_any_tie_in_dose):
    lr_tail_delivered_verdict = ("DOSE_AFTER_SELECTION_BUT_TIE_INSIDE_DOSE"
                                 "(first_wins 选点把剂量窗排除在交付窗外；argmax 并列点 %s "
                                 "落在剂量窗内 => 换 tie-break 规则就会改变读数)"
                                 % str(_argmax_eps))
else:
    lr_tail_delivered_verdict = "DOSE_AFTER_SELECTION_DELIVERED_BLIND"

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
    # [v100] 一门拆两门（v99 的一把门把"平台带剂量"这个事实一起作废了）
    "dose_reaches_delivered_selection": bool(_tail_dose_reaches_delivery),
    "dose_reaches_plateau_window": bool(_tail_dose_reaches_plateau),
    "n_plateau_epochs_under_dose": int(_tail_n_in_plateau),
    "n_plateau_epochs_total": int(len(_plateau_epochs)),
    "plateau_window_fully_inside_dose": bool(_tail_plateau_fully_dosed),
    "lr_mult_at_best_tta": (round(float(_tail_mult_at_best_tta), 4)
                            if _tail_mult_at_best_tta is not None else None),
    # ---- [v100] argmax 并列的机器字段（v99 的并列是读曲线才发现的）----
    "best_tta_argmax_epochs": _argmax_eps,
    "best_tta_argmax_n": int(len(_argmax_eps)),
    "best_tta_tie_break_rule": "first_wins",
    "best_tta_last_tie_epoch": (int(_argmax_eps[-1]) if _argmax_eps else None),
    "selection_epoch_inside_dose": bool(_selection_in_dose),
    "any_argmax_epoch_inside_dose": bool(_any_tie_in_dose),
    # ---- [v100] 判据可达性（事实字段，不是守卫）----
    "plateau_best_ever_pp": float(PLATEAU_BEST_EVER_PP),
    "plateau_pass_line_reachable": bool(PLATEAU_PASS_REACHABLE),
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
    # ---- [v100] 拆门后的两个"送达"字段（语义不同，不许再合并）----
    #   主门（平台通道，**仍是硬守卫**）：平台读数有没有带剂量。v99 实测 = True。
    "dose_delivered": bool(_tail_dose_reaches_plateau),
    #   交付通道（选择点，**事实字段**）：v99 实测 = False，且成因是 first-wins 并列，不是没跑。
    "dose_delivered_delivered_channel": bool(int(_lr_tail_n_run) >= 1 and _tail_dose_reaches_delivery),
    #   交付通道的独立判词（四档）
    "lr_tail_delivered_verdict": lr_tail_delivered_verdict,
}
# [v100] FACT_KEYS 扩容的依据（不是"把红的塞进去"）：判词/事实与守卫是两类东西。
#   · **守卫**（不在集合里 ⇒ False 即 INSTRUMENT_FAIL）：剂量真的送到了没有。
#   · **事实/档位**（在集合里）：选点规则的产物、并列点的位置、判据可达性 ——
#     这些是**读数**，False 只说明"这一跑的结构长这样"，不说明仪器坏了。
#   ⇒ `dose_delivered`（平台主门）**故意留在集合外**：v99 的严格性在这一门上不放松。
_LR_TAIL_FACT_KEYS = {"plateau_verdict", "lr_tail_delivered_verdict",
                      "dose_reaches_delivered_selection", "dose_reaches_plateau_window",
                      "selection_epoch_inside_dose", "any_argmax_epoch_inside_dose",
                      "plateau_window_fully_inside_dose",
                      "dose_delivered_delivered_channel",
                      "plateau_pass_line_reachable"}   # 纯事实/档位字段，不参与"坏键"判定
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

# [v103] v102 已实跑的常量（**同代码不同 seed** 的配对参照 —— 本版唯一的测量目的）。
# ⚠ provenance 注记：v102 实跑文件是 loss.mmm1.py、self_sha1=ad097731...，
#   与交付片 cnn10_v102.py（c12256ab...）不符（B 线第一次内容级破损）⇒ 下面三个数
#   取自 results_cnn_v102_seed1234.json，待训练机核对实跑文件与交付片的 diff 后才算钉死。
V102_SEED = 1234
V102_DELIVERED = 97.42
V102_FINAL_TTA = 97.42
V102_PLATEAU_MEAN = 97.3088
V102_PROVENANCE_NOTE = ("v102 实跑 provenance 破损：文件名 loss.mmm1.py、"
                        "self_sha1=ad09773175920ecde37cb82d88777d13eec3f1b4 "
                        "!= 交付片 c12256abdedbd37c13e677b2ae995deaea697e25；"
                        "读数待训练机核对实跑文件与交付片的 diff 之后采信")

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
# [v103] v44 = seed 1234、本版 = seed 42 ⇒ 与 v44 是**真不同 seed**（但代码栈也不同，
#   所以既不是同代码配对（那是 v103 vs v102），也不是同 seed 复现 —— 两个字段各读各的）。
_indep_of_v44 = bool(int(SEED) != int(V44_SEED))
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
    "purpose": "[v103] seed swap is the only axis (SEED 1234->42); this instrument block stays read-only",
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
    "purpose": "[v103] z-combination with v43 (kept for continuity); v103 shares seed 42 with v43 => NOT combinable as independent",
    "note": "[v103] v43 was seed 42 and this run is seed 42 => same seed, different code "
            "(v43 predates the whole v45-v102 stack) => the two z-scores must NOT be merged "
            "(pitfall 106). Against v44 (seed 1234) this run IS a different seed, but v44 "
            "also differs by the entire v45-v102 code stack, so it is neither a same-code "
            "pair (that is v103 vs v102, see v103_seed_pair) nor a same-seed replicate.",
    "code_differs_from_v44_by": ["the entire v45-v102 stack", "SEED 1234 -> 42"],
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
    "independent_of_v44": bool(_indep_of_v44),
    "seed_differs_from_v43": bool(_seed_differs),
    "curves_differ_from_v43": bool(_curves_differ),
    "fp_test": _fp_test,
    "fp_tta": _fp_tta,
    "v43_fp_test": V43_TEST_FP,
    "v43_fp_tta": V43_TTA_FP,
    "backtrack_advice": "DO_NOT_BACKTRACK_CODE",
    "backtrack_reason": "[v103] code differs from cnn10_v102.py ONLY by the SEED line (1234->42); "
                       "a low reading is seed variance, the action is another seed, not a code rollback",
    "scope_note": ("本字段属于【复跑判据】块（当前口径：代码只差 SEED ⇒ 低读数是 seed 方差，"
                   "动作是换 seed 而不是回退代码）。"
                   "★ 与本版【交付判据】的红线（delivered <= %.2f ⇒ 回退 cnn10_v46.py）"
                   "是两个不同作用域，不要混读：前者管'代码要不要退'，后者管'交付权重链要不要退'。"
                   % float(BACKTRACK_REDLINE)),
}

# ====================== [v103] 同代码跨 seed 配对（本版唯一的测量目的，只读） ======================
# v103 = v102 逐字相同的训练数学 + SEED 1234->42 ⇒ 这是一对**同代码不同 seed**的配对，
# 是当前代码栈 seed 方差的第一个直接测量（归档里此前的配对都跨着代码栈）。
_v103_delta_delivered = float(delivered) - float(V102_DELIVERED)
_v103_seed_pair = {
    "purpose": ("first same-code cross-seed pair on the CURRENT code stack "
                "(v103 seed 42 vs v102 seed 1234); measures the seed spread the "
                "3-seed protocol needs; claims no accuracy conclusion"),
    "this_version": "v110",
    "this_seed": int(SEED),
    "this_delivered": round(float(delivered), 4),
    "this_final_tta": round(float(final_tta), 4),
    "this_plateau_mean": (lr_tail_instrument.get("plateau_mean")
                          if isinstance(lr_tail_instrument, dict) else None),
    "pair_version": "v102",
    "pair_seed": int(V102_SEED),
    "pair_delivered": float(V102_DELIVERED),
    "pair_final_tta": float(V102_FINAL_TTA),
    "pair_plateau_mean": float(V102_PLATEAU_MEAN),
    "delta_delivered_pp": round(float(_v103_delta_delivered), 4),
    "abs_delta_within_sigma_diff": bool(abs(float(_v103_delta_delivered)) <= float(PLATEAU_SIGMA)),
    "sigma_diff_pp": float(PLATEAU_SIGMA),
    "pair_provenance_note": V102_PROVENANCE_NOTE,
    "verdict": ("SEED_SPREAD_WITHIN_SIGMA" if abs(float(_v103_delta_delivered)) <= float(PLATEAU_SIGMA)
                else "SEED_SPREAD_EXCEEDS_SIGMA_READ_IT"),
    "note": ("|delta| <= sigma_diff=%.3fpp ⇒ 两个 seed 的差落在噪声底内，seed 方差量到了；"
             "超出 ⇒ 要么 seed 方差比四对 RMS 估的宽，要么 v102 读数有 provenance 问题，"
             "两种情况都要读，都不是自动的回退信号。" % float(PLATEAU_SIGMA)),
}


# ---- [v110] 本版轴常量 + 收尾判词 ----
AXIS_V110 = ("v110 = cnn10_v103.py training math + ONE axis: SOBOLEV-METRIC SAM. The SAM trust "
             "region ball (Euclidean) is replaced by the metric induced by the network's own Bessel "
             "potential operator, K(k) = (1 + (lam/nu)|k|^2)^{-s}, taken from the field's running "
             "lam/nu statistics (pooled) - zero new parameters, zero new RNG in the metric. The "
             "ascent direction is eps = rho * K*g / ||K*g||_2 (K acts on the two spatial dims of "
             "conv kernels only), and the update uses the gradient at w+eps. Nothing else moves: "
             "no lr / momentum / mixup-strength scalar is touched. WHY (v109 readout = NULL, "
             "2026-10-23): six consecutive versions across three action surfaces (feature "
             "injection v103-v105, optimizer scalars v106-v108, data side v109) are all NULL or "
             "negative, and diagnosed defects are shared - (a) the control signal smooth is ITSELF "
             "inside the loss (w_phys*smooth, peak 0.15) so it is actively pushed down and any "
             "z-score degenerates into one-way drift (v107 scale_mean=1.0998, v108 mu_std=0.011 "
             "with mu_mean=0.9186, v109 NULL): the signal has no usable high-frequency content; "
             "(b) the ones that do act act on the fragile early high-lr phase (v107 -0.45pp, "
             "v108 -1.60pp early) while the tail floor-lr regime is dose-insensitive, i.e. any "
             "extra time-varying control is just a perturbation of an already tuned schedule. SAM "
             "changes WHERE we step instead of HOW FAST, and is one of the few mechanisms with a "
             "documented effect (+0.3 to 1.0pp clean accuracy) far above this line's "
             "sigma_diff=0.169 measurement floor. TECHNICAL PREREQUISITE (new pitfall 331): the "
             "line's existing _dft_axis equals the true DFT bins only for EVEN N; conv kernels are "
             "mostly 3x3 (ODD), where the old grid is off by half a bin and a constant kernel is "
             "attenuated to 0.313 instead of 1.0 - so this version carries its own parity-safe "
             "_v110_freq_axis (fftfreq + fftshift) and self-tests 'a constant kernel passes "
             "unchanged'. DISCLOSED PURITY LOSS: two forward-backward per step (~1.8-2x wall "
             "clock) and the second forward draws another StochasticDepth mask, so the RNG stream "
             "is no longer bit-comparable with v103; cross-version pairing still uses "
             "sigma_diff=0.169. ATTRIBUTION PRE-REGISTRATION: if this version gains beyond "
             "sigma_diff, the next version MUST be plain SAM (K identical 1, same rho) as the "
             "control run before claiming the Sobolev metric itself is what helped.")

results_axis_used = AXIS_V113          # [v113] 供 text_sync 做"文案同源"自证

# ---- [v96] 本版轴的收尾判词（必须是函数 —— 覆辙 263）----
_v96_dft_vd = _dft_grid_verdict_now(_v96_dft_st)
_v96_dft = {
    "axis": AXIS_V96,
    "selftest": _v96_dft_st,
    "verdict": _v96_dft_vd,
    "self_sha1": SELF_SHA1,
    "criterion": {
        "baseline_v51_acc": float(BASELINE_V51_ACC),
        "sigma_diff_pp": float(SIGMA_DIFF_PP),
        "pass_line": float(PASS_LINE_96),
        "redline": float(REDLINE_96),
        "margin_in_sigma": 2.0,
        "margin_2sigma_ok": bool(DFT_GRID_2SIGMA_MARGIN_OK),
        "note": ("PASS 边距 = 2σ_diff。依据 v48 noise_floor/note：v47/v48 的边距"
                 "0.10pp 只有 0.93σ ⇒ 那两个 NULL 是算术上被预先决定的。"
                 "recipe 域改动 ⇒ 只有跨跑判据，单跑 NULL 不得升级为『轴无效』。"),
    },
    "rolled_back_to": "cnn10_v51.py (逐字; v95 delivered 97.33 = -0.23pp = -2.13σ, "
                      "NULL 偏负 ⇒ 不触红线但不带进下一版)",
    "prev_version": "v109",
    "prev_sha1": "b22097bb6773f0b2da30889e590fe9ba893893a2",
}

# ---- [v110] 本版轴的收尾判词（必须是函数 —— 覆辙 263）----
_v110_sam_stats = _v110_sam_stats()
_v110_sam = {
    "axis": AXIS_V110,
    "selftest": _v110_st,
    "selftest_cases_source": int(_v110_sam_selftest_cases_count()),
    "stats": _v110_sam_stats,
    "thresholds": {
        "rho": float(SAM_RHO),
        "sobolev_s": float(SAM_SOBOLEV_S),
        "ratio_fallback": float(SAM_RATIO_FALLBACK),
        "dir_ratio_lo": float(SAM_DIR_RATIO_LO),
        "dir_ratio_hi": float(SAM_DIR_RATIO_HI),
        "delivered_min_frac": float(SAM_DELIVERED_MIN_FRAC),
        "selftest_cases": int(_v110_st["n_expected"]),
    },
    "criterion": {
        "baseline_v103_acc": 97.37,
        "baseline_note": ("v103 的 97.42 是 selection_bias_pp=0.0475 修正前的 max-of-N 孤点；"
                          "配对基线取 97.37。sigma_diff=0.169（跨版本配对 RMS）。"),
        "sigma_diff_pp": 0.169,
        "note": ("本版改的是更新几何 ⇒ 训练数学已变，不与 v103 同ան trajectory；"
                 "第二轮 forward 多抽一次 StochasticDepth 掩码 ⇒ RNG 流不再逐位可比。"
                 "归因预注册：**若涨幅越过 sigma_diff，下一版必须是裸 SAM（K≡1，同 rho）**"
                 "作对照，才能拆开『SAM 的功劳』与『Sobolev 度量的功劳』。"),
    },
    "verdict": _v110_sam_verdict(_v110_sam_stats, _v110_st),
    "self_sha1": SELF_SHA1,
    "note": ("判的是**机制**：applied_frac>=0.9 且 ||K*g||/||g|| 均值 ∈ (0.05, 0.95)。"
             "高于 0.95 ⇒ 低通没起作用（退化成裸 SAM）；低于 0.05 ⇒ 度量把自己压没了。"),
}

# ---- [v113] 约束通道块（results 之前构造；字段全在这里，不散落）----
_v113_pinface_block = {
    "axis": AXIS_V113,
    "selftest": _v113_pfst if isinstance(_v113_pfst, dict) else {"error": "no selftest"},
    "selftest_cases_source": int(_v113_pinface_selftest_cases_count()),
    "probe": _v113_probe,
    "verdict": _v113_verdict,
    "criterion": {
        "baseline_v110_delivered": float(V113_BASE_DELIVERED),
        "pass_line": float(V113_PASS_LINE),
        "redline": float(V113_REDLINE),
        "sigma_diff_pp": float(V113_SIGMA_DIFF),
        "pass_margin_sigma": 2.0,
        "sigma_diff_applies": True,
        "sigma_diff_note": ("本轴改的是**训练数学**（pinn 的作用点）⇒ 轨迹与 v110 不同，"
                            "⇒ 跨跑口径（σ_diff=0.169）适用；与 v112 的**同权重配对**读数不同。"),
        "mechanism_gate_grad_frac_max": float(V113_PINN_FACE_GRAD_FRAC_MAX),
        "mechanism_gate_band_frac_min": float(V113_BAND_FRAC_MIN),
        "baseline_band_frac_v111": float(V113_BASELINE_BAND_FRAC),
        "baseline_lambda_c_px_v111": float(V113_BASELINE_LAMBDA_C_PX),
        "note": ("delivered 口径不变（2 视图 TTA，与 v45/v47/v48/v110/v112 同尺 ⇒ 跨版可比）。"
                 "★ 预期（纪律 336 末态吸引子律）：精度多半 NULL；能推翻预期的只有"
                 "『场恢复内容』（band_frac_mean ≥ 0.20）这一条机制链。"),
    },
    "followup": {
        "if_DELIVERED": ("作用面族成立 ⇒ 下一版把 w_phys * smooth 也沿同一思路处理"
                         "（它同样在罚场输出粗糙度 ⇒ 同样会推 l 上去），并**重锚** PINN_WEIGHT。"),
        "if_MECHANISM_ONLY": ("作用面切成功、场也恢复内容，但精度没过 2σ 线 ⇒ "
                              "机制成立而收益不足 ⇒ 下一版**只做**『把 2σ 线换成配对口径』"
                              "（同权重两种作用面读数），不再加新机制。"),
        "if_CUT_NO_CONTENT": ("★ 关键否证：切掉 pinn 这条路后场**仍是近常数** ⇒ "
                              "说明压 DC 的主因是**另一条约束路径**（w_phys * smooth）"
                              "⇒ v111 的归档结论被**强化**（冗余是结构性的）"
                              "⇒ 下一版若再动约束，必须动 w_phys 的作用面，且这是最后一张牌。"),
        "if_NOT_BINDING": ("轴空转（切不掉或旧路径本来就没梯度）⇒ 探针读数先查，"
                           "再决定是改实现还是换轴。"),
        "if_BACKTRACK": ("精度触红线 ⇒ 逐字回退 cnn10_v110.py（本版是唯一改动点）。"),
        "if_NULL_no_mechanism": ("精度 NULL 且机制也没切成功 ⇒ 约束通道（作用面族）判死 ⇒ "
                                 "下一版只能动**目标函数主项**（分类损失形式 / 标签结构 / 损失权重结构）。"),
    },
    "self_sha1": SELF_SHA1,
}




_v117_pinface_block = {
    "launch_fix": V117_LAUNCH_FIX_NOTE,
    "v116_crash": V117_V116_HISTORY,
    "verdict": _v113_verdict,
    "probe_ok": bool(_v113_probe_ok),
    "probe_error": (_v113_probe.get("error") if not _v113_probe_ok else None),
    "probe_n": (_v113_probe or {}).get("probe_n"),
    "small_batch_fallback": (_v113_probe or {}).get("small_batch_fallback"),
    "oom_suspected": (_v113_probe or {}).get("oom_suspected"),
    "probe_attempts": (_v113_probe or {}).get("attempts"),
    "grad_frac_new": (float(_v113_probe.get("grad_frac_new"))
                        if _v113_probe_ok else None),
    "grad_frac_old": (float(_v113_probe.get("grad_frac_old"))
                        if _v113_probe_ok else None),
    "grad_leaks": (list(_v113_probe.get("grad_leaks", []))
                     if _v113_probe_ok else None),
    "band_frac_mean": (float((_v113_probe.get("levels", {}) or {}).get(
        "band_frac_mean", _v113_probe.get("band_frac_mean", 0.0)))
        if _v113_probe_ok else None),
    "levels": ((_v113_probe.get("levels", {}) or {}) if _v113_probe_ok else None),
    "probe_sections_ok": (_v113_probe.get("sections_ok")
                            if _v113_probe_ok else None),
    "probe_section_errors": (_v113_probe.get("section_errors")
                               if _v113_probe_ok else None),
    "precheck_ok": bool(_v117_precheck.get("ok")),
    "precheck_probe_n": _v117_precheck.get("probe_n"),
    "precheck_cpu_fallback": bool(_v117_precheck.get("cpu_fallback")),
    "precheck_oom_suspected": bool(_v117_precheck.get("oom_suspected")),
    "precheck_model_state_unchanged": _v117_precheck.get("model_state_unchanged"),
    "det_guard_ok": bool(_v117_det_guard.get("ok")),
    "det_guard_error": _v117_det_guard.get("error"),
    "det_verdict": (det_probe_report or {}).get("verdict"),
    "cuda_canary": _v117_canary,
    "self_sha1": SELF_SHA1,
}

_v116_pinface_block = {
    "verdict": _v113_verdict,
    "delivered_acc": float(delivered),
    "grad_frac_new": (float(_v113_probe.get("grad_frac_new"))
                      if _v113_probe_ok else None),
    "grad_frac_old": (float(_v113_probe.get("grad_frac_old"))
                     if _v113_probe_ok else None),
    "grad_leaks": (list(_v113_probe.get("grad_leaks", []))
                   if _v113_probe_ok else None),
    "band_frac_mean": (float((_v113_probe.get("levels", {}) or {}).get(
        "band_frac_mean", _v113_probe.get("band_frac_mean", 0.0)))
        if _v113_probe_ok else None),
    "levels": ((_v113_probe.get("levels", {}) or {}) if _v113_probe_ok else None),
    "probe_ok": bool(_v113_probe_ok),
    "probe_error": (_v113_probe.get("error") if not _v113_probe_ok else None),
    "probe_sections_ok": (_v113_probe.get("sections_ok")
                          if _v113_probe_ok else None),
    "probe_section_errors": (_v113_probe.get("section_errors")
                             if _v113_probe_ok else None),
    "probe_device": _v115_probe_dev,
    "probe_align_ok": bool(_v115_probe_align_ok),
    "bn_snap_n": int(len(_V115_BN_SNAP)),
    "bn_restore_ok": bool(_v115_bn_restore_ok),
    "rng_changed_by_probe": bool(_v115_rng_changed),
    "selftest_ok": bool(_v113_pfst_ok),
    "prewrite_ok": bool(_v115_prewrite_done),
    "precheck": _v117_precheck,
    "precheck_ok": bool(_v117_precheck.get("ok")),
    "runnable": _v116_runnable,
    "runnable_ok": bool(_v116_runnable.get("ok")),
    "det_guard": _v117_det_guard,
    "det_guard_ok": bool(_v117_det_guard.get("ok")),
    "readout_fix_note": V116_READOUT_FIX_NOTE,
    "v115_history": V116_V115_HISTORY,
    "axis_note": (
        "本版被测轴 = 约束通道：pinn 的作用面（**沿用 v113/v114/v115**）。"
        "v113 顺序 bug、v114 device 不匹配、v115 探针内部 math domain error "
        "⇒ 同一根轴**三次都没测到** ⇒ 未测 ≠ 判负 ⇒ 本版**第四次**重测同一根轴。"
        "判据：grad_frac_new <= V113_PINN_FACE_GRAD_FRAC_MAX "
        "且 grad_frac_old > 0 ⇒ 切断成立；band_frac_mean >= V113_BAND_FRAC_MIN "
        "⇒ 场恢复内容 ⇒ 加性注入通道可能复活；切断成立但场仍近常数 ⇒ 冗余是结构性的 "
        "⇒ v111 的归档被强化。"),
    "fingerprint_note": ("训练数学与全部计算逐字未改 ⇒ 精度读数与 v113/v114/v115 同尺"
                         "（σ_diff=0.169）⇒ 预期 delivered 仍 ≈ 97.4（NULL）；"
                         "本版价值在**机制读数**而非精度。"),
    "self_sha1": SELF_SHA1,
}
_v115_pinface_block = {
    "verdict": _v113_verdict,
    "delivered_acc": float(delivered),
    "grad_frac_new": (float(_v113_probe.get("grad_frac_new"))
                      if _v113_probe_ok else None),
    "grad_frac_old": (float(_v113_probe.get("grad_frac_old"))
                      if _v113_probe_ok else None),
    "grad_leaks": (list(_v113_probe.get("grad_leaks", []))
                   if _v113_probe_ok else None),
    "band_frac_mean": (float((_v113_probe.get("levels", {}) or {}).get(
        "band_frac_mean", _v113_probe.get("band_frac_mean", 0.0)))
        if _v113_probe_ok else None),
    "levels": ((_v113_probe.get("levels", {}) or {}) if _v113_probe_ok else None),
    "probe_ok": bool(_v113_probe_ok),
    "probe_error": (_v113_probe.get("error") if not _v113_probe_ok else None),
    "probe_device": _v115_probe_dev,
    "probe_align_ok": bool(_v115_probe_align_ok),
    "bn_snap_n": int(len(_V115_BN_SNAP)),
    "bn_restore_ok": bool(_v115_bn_restore_ok),
    "rng_changed_by_probe": bool(_v115_rng_changed),
    "selftest_ok": bool(_v113_pfst_ok),
    "prewrite_ok": bool(_v115_prewrite_done),
    "readout_fix_note": V115_READOUT_FIX_NOTE,
    "v114_history": V115_V114_HISTORY,
    "axis_note": (
        "本版被测轴 = 约束通道：pinn 的作用面（**沿用 v113/v114**）。"
        "v113 的收尾读数被模块级顺序 bug 吃掉、v114 的探针输入没搬 device ⇒ "
        "同一根轴**两次都没测到** ⇒ 未测 ≠ 判负 ⇒ 本版第三次重测同一根轴。"
        "判据：grad_frac_new <= V113_PINN_FACE_GRAD_FRAC_MAX "
        "且 grad_frac_old > 0 ⇒ 切断成立；band_frac_mean >= V113_BAND_FRAC_MIN "
        "⇒ 场恢复内容 ⇒ 加性注入通道可能复活；切断成立但场仍近常数 ⇒ "
        "冗余是结构性的 ⇒ v111 的归档被强化。"),
    "fingerprint_note": (
        "训练数学与全部计算逐字未改 ⇒ 精度读数与 v113/v114 同尺（σ_diff=0.169）"
        "⇒ 预期 delivered 仍 ≈ 97.4（NULL）；本版价值在**机制读数**而非精度。"),
}



results = {
    "version": VERSION,
    "base": "cnn10_v51.py",
    "base_sha1": "3516a363737c1930019ba97f7c2853e67fc0eee2",
    "prev_version": "v115",
    "prev_sha1": "4e4b5fd77d0f778fb3f1895df8ac2eed716119d6",
    "rolled_back_to": "cnn10_v110.py (逐字父本；v112 的视图群轴实测判负（10 视图 -0.18pp、角位比中心位 -0.4575pp）⇒ 回退优先：先回 v110 训练态，再换约束通道 pinn 的作用面)",
    "axis": AXIS_V113,
    "seed": int(SEED),
    "v110_sam": _v110_sam,
    "v113_pinface": _v113_pinface_block,
    "v115_pinface_history": V116_V115_HISTORY,
    "v116_pinface": _v116_pinface_block,
    "v117_pinface": _v117_pinface_block,
    "v117_parent": "cnn10_v116.py",
    "v117_parent_sha1": "d454925073b2acb4e4aea94690c0a00e32f10ce0",
    "v114_pinface_history": V115_V114_HISTORY,
    "v115_pinface": _v115_pinface_block,
    "v115_parent": "cnn10_v114.py",
    "v115_parent_sha1": "bbd7da58d09eff40ff59cdd173b44c70f4158f94",
    "v116_parent": "cnn10_v115.py",
    "v116_parent_sha1": "4e4b5fd77d0f778fb3f1895df8ac2eed716119d6",
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
    "delivery_backtrack_scope": ("本字段属于【交付判据】块：delivered <= %.2f 时回退到 cnn10_v46.py（逐字）。"
                                 "与复跑判据块的 backtrack_advice(DO_NOT_BACKTRACK_CODE) 作用域不同。"
                                 % float(BACKTRACK_REDLINE)),
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
        "note": ("★ v47/v48 的 PASS 边距 0.10pp 只有 %.3fσ（按 v100 修正后的 σ=%.3f）⇒ "
                 "那两个 NULL 是算术上被预先决定的，不构成『轴无效』的证据。"
                 "以后 B 线 PASS 边距不得小于 2σ_diff。"
                 "★ [v100] σ 自身也修正过：0.108（一对）→ %.3f（四对 RMS），"
                 "旧常数只有新值的 %.3f，即低估 %.3f×。"
                 % (0.10 / float(PLATEAU_SIGMA), float(PLATEAU_SIGMA),
                    float(PLATEAU_SIGMA), 0.108 / float(PLATEAU_SIGMA),
                    float(PLATEAU_SIGMA) / 0.108)),
    },
    # ★ [v51] 文本失同步机检（v48 的三处：axis 文案 / banner 版本行 / banner 里的 phase2 终点字面量）
    "text_sync": {
        "axis_field_mentions_version": bool(VERSION in str(AXIS_V113)),
        "axis_field_mentions_real_axis": bool("action surface" in str(AXIS_V113).lower()),
        "axis_field_is_shared_constant": bool(str(AXIS_V113) == str(results_axis_used)),
        # ★ 下面两条在 dict 建好之后由 `results["text_sync"][...] = ...` 补写（字面量里看不到自己）
        "axis_field_actually_equals_constant": None,
        "version_is_v113": None,
        "banner_phase2_end_derived": _V96_BANNER_PHASE2,
        "banner_phase2_end_hardcoded_lines": _V96_BANNER_PHASE2.get("hardcoded_lines"),
        "note": ("[v110] axis 文案由常量 AXIS_V110 提供、banner 里的 phase2 终点与版本号从常量派生 ⇒ "
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
        # [v100] 一门拆两门（详见 lr_tail_instrument 内注释与 lr_tail_delivered_verdict）
        "dose_reaches_delivered_selection": bool(_tail_dose_reaches_delivery),
        "dose_reaches_plateau_window": bool(_tail_dose_reaches_plateau),
        "n_plateau_epochs_under_dose": int(_tail_n_in_plateau),
        "plateau_window_fully_inside_dose": bool(_tail_plateau_fully_dosed),
        "lr_mult_at_best_tta": lr_tail_instrument["lr_mult_at_best_tta"],
        "dose_delivered": bool(lr_tail_instrument["dose_delivered"]),
        "dose_delivered_delivered_channel": bool(lr_tail_instrument["dose_delivered_delivered_channel"]),
        "lr_tail_delivered_verdict": lr_tail_delivered_verdict,
        "best_tta_argmax_epochs": lr_tail_instrument["best_tta_argmax_epochs"],
        "best_tta_argmax_n": int(lr_tail_instrument["best_tta_argmax_n"]),
        "best_tta_tie_break_rule": str(lr_tail_instrument["best_tta_tie_break_rule"]),
        "selection_epoch_inside_dose": bool(lr_tail_instrument["selection_epoch_inside_dose"]),
        "any_argmax_epoch_inside_dose": bool(lr_tail_instrument["any_argmax_epoch_inside_dose"]),
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
    "v103_seed_pair": _v103_seed_pair,
    "v96_dft_grid": _v96_dft,
    "v96_file_name_selfcheck": _V96_FILE_NAME_CHECK,
    "self_sha1": SELF_SHA1,
}

# ★ [v49] text_sync 的事后自证：dict 字面量里引用不到自身，所以在这里补写两条**真检查**。
results["text_sync"]["axis_field_actually_equals_constant"] = bool(results.get("axis") == AXIS_V113)
results["text_sync"]["version_is_v113"] = bool(results.get("version") == VERSION)

# [v97] 磁盘守卫读数。故意写在 text_sync **之后**：判词要读的是最终计数，
#      提前读会拿到全零，那才是一个永远绿的假检查（覆辙 267）。
_dg_min = DISK_GUARD.get("min_free_gb")
results["disk_guard"] = {
    "writes_ok": int(DISK_GUARD.get("writes_ok") or 0),
    "writes_skipped": int(DISK_GUARD.get("writes_skipped") or 0),
    "writes_failed": int(DISK_GUARD.get("writes_failed") or 0),
    "remove_failed": int(DISK_GUARD.get("remove_failed") or 0),
    "min_free_gb_seen": (None if _dg_min == float("inf") else float(_dg_min)),
    "verdict": _disk_verdict_now(),
    "events": list(DISK_GUARD.get("events") or [])[-DISK_MAX_EVENTS:],
    "note": ("[v97 继承，v98 原样保留] v97 axis = DISK WRITE GUARD (3 lines), training recipe verbatim v96. "
             "Line 1 (the real one) = pre-write free-space check via os.statvfs; "
             "line 2 = atomic write via .tmp + os.replace; line 3 = python-level "
             "exception fallback + counters into this JSON. NOTE HONESTLY: try/except "
             "CANNOT catch the C++ abort that follows a partially-written torch.save "
             "(c10::Error from PyTorchStreamWriter destructor) - only line 1 prevents "
             "that by never calling torch.save when space is short. Because this run "
             "has NO resume (user decision), a core dump from any other cause still "
             "loses everything; these guards only remove the disk from the list of "
             "things that can cause one."),
}

results["v100_instrument_fix"] = {
    # ★ AXIS_V100 必须是**活引用**（覆辙 251：数引用，不读函数名）。
    "axis_this_version": AXIS_V100,
    "parent_version": "v99",
    "parent_sha1": "d11403c2e7c94b796db202e016be4e9a1ff36d56",
    "training_math_changed": False,
    "training_math_evidence": [
        "no parameter / schedule / RNG draw / loss term / physics constant differs from v99",
        "SEED / BATCH_SIZE / LR_BASE / PHASE2_LR_END / PINN_WEIGHT / GATE1_INJECT all verbatim",
        "function-level md5 vs v99: DIFF confined to criterion + instrument functions",
    ],
    "corrections": {
        "1_sigma_diff_0.108_to_0.169": {
            "old_pp": 0.108,
            "new_pp": float(PLATEAU_SIGMA),
            "old_source": "v45/v46 ONE pair (|delta delivered| = 0.120pp)",
            "new_source": "FOUR same-recipe pairs, |delta delivered| RMS",
            "pairs_pp": {"v45_v46": 0.120, "v48_v49": 0.040,
                         "v44_v54": 0.140, "v51_v55": 0.280},
            "rms_pp": float(PLATEAU_SIGMA),
            "old_over_new_ratio": round(0.108 / float(PLATEAU_SIGMA), 3),
            "understatement_factor": round(float(PLATEAU_SIGMA) / 0.108, 3),
            "lesson": "a single pair measures that pair's LUCK, not the variance "
                      "(pitfall 113 in its measurement form)",
        },
        "2_all_lines_derived_from_constants": {
            "plateau_pass": float(PLATEAU_PASS),
            "plateau_redline": float(PLATEAU_REDLINE),
            "v51_pass_line": float(PASS_LINE_96),
            "v51_redline": float(REDLINE_96),
            "backtrack_redline": float(BACKTRACK_REDLINE),
            "was_hardcoded_in_v99": ["BACKTRACK_REDLINE = 97.27", "if delivered >= 97.61:"],
            "now": "round(baseline -/+ 2*sigma, 2) everywhere; guards are EXACT equalities",
        },
        "3_dose_gate_split_in_two": {
            "old": "one field `dose_delivered` gated both the plateau channel and the "
                   "delivered-selection channel",
            "v99_measurement": {
                "n_changed_full_horizon": 80,
                "n_changed_in_this_run": 36,
                "dose_window_epochs": [121, 156],
                "plateau_window_epochs": [135, 156],
                "plateau_fully_inside_dose": True,
                "best_tta_epoch": 120,
                "argmax_tie_epochs": [120, 126],
                "tie_break_rule": "first_wins",
            },
            "new_fields": ["dose_delivered (plateau channel, HARD guard)",
                           "dose_delivered_delivered_channel (fact)",
                           "dose_reaches_delivered_selection (fact)",
                           "dose_reaches_plateau_window (fact)",
                           "lr_tail_delivered_verdict (4-way)"],
            "why": "the delivery-channel blindness is an artifact of the first-wins "
                   "tie-break, NOT a missing dose; one gate falsified a fact that was true",
        },
        "4_argmax_tie_is_now_a_machine_field": {
            "fields": ["best_tta_argmax_epochs", "best_tta_argmax_n",
                       "best_tta_tie_break_rule", "best_tta_last_tie_epoch",
                       "selection_epoch_inside_dose", "any_argmax_epoch_inside_dose"],
            "why": "in v99 the tie (ep120 == ep126 == 97.44) was only visible by reading "
                   "the curve by eye; it is the entire explanation of the dose verdict",
        },
        "5_falsified_expectation_corrected": {
            "where": "results['v99_crash_fix']['expected_delivered_vs_v98']",
            "as_written": "identical (97.42)",
            "actually_ran": 97.44,
            "fix": "original text kept verbatim under a *_AS_WRITTEN_AT_DELIVERY key plus a "
                   "*_CORRECTED_BY_V100 key (history is annotated, never erased - pitfall 292)",
        },
    },
    "sigma_block": {
        "sigma_diff_pp": float(PLATEAU_SIGMA),
        "margin_2sigma_pp": float(PLATEAU_PASS_MARGIN),
        "pass_margin_ok": bool(PLATEAU_MARGIN_OK),
        "lines_derived_ok": bool(PLATEAU_DERIVED_OK),
        "source": PLATEAU_BASELINE_SOURCE,
    },
    "dose_split_block": {
        "plateau_gate_dose_delivered": bool(_tail_dose_reaches_plateau),
        "n_plateau_epochs_under_dose": int(_tail_n_in_plateau),
        "n_plateau_epochs_total": int(len(_plateau_epochs)),
        "delivered_channel_dose_delivered": bool(_tail_dose_reaches_delivery),
        "lr_tail_delivered_verdict": lr_tail_delivered_verdict,
        "argmax_epochs": _argmax_eps,
    },
    "reachability_note": (
        "★ σ 0.108 → 0.169 后，平台 PASS 线 = %.2f，而本线历史最好平台 = %.4f "
        "⇒ plateau_pass_line_reachable = %s。单跑平台判词在算术上预先决定为 NULL；"
        "这不是轴无效的证据，是尺子变准后的必然结果（v47/v48 之错的镜像：那次线太低）。"
        "本版的交付物是仪器修正本身，精度主张留到尺子稳定之后（且仍需 3 seed）。"
        % (float(PLATEAU_PASS), float(PLATEAU_BEST_EVER_PP),
           str(bool(PLATEAU_PASS_REACHABLE)))),
    "expected_delivered_vs_v99": (
        "STATISTICALLY INDISTINGUISHABLE, NOT 'IDENTICAL'. v100's training math is "
        "bit-identical to v99 (no parameter/schedule/RNG/loss/constant change), so any "
        "difference is run-to-run noise on the same code path. But this platform has "
        "same-seed non-determinism measured three times (pitfall 141) plus cudnn/cuBLAS "
        "non-associativity, and v99 itself proved the point: it claimed 'identical (97.42)' "
        "and ran 97.44. => this field does NOT promise a number. Expected magnitude: "
        "|delta| within sigma_diff = %.3fpp of v99's 97.44; a larger deviation is a "
        "measurement worth reading, not automatically a regression."
        % float(PLATEAU_SIGMA)),
    "self_sha1": SELF_SHA1,
}

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f'results_cnn_{VERSION}_seed{SEED}.json')
_core_path = os.path.join(OUT_DIR, f'results_cnn_{VERSION}_seed{SEED}_core.json')

# ★★ [v99] 主结果**先落盘**（覆辙 279 三件套之三）。
#   顺序是刻意的：results 字面量 → 回填(text_sync/disk_guard) → **core 落盘**
#   → 后置机制读数 → 回填 → 完整落盘。
#   v98 的教训：把机制读数塞进 results **字面量内部**，而它在字面量之后才绑定
#   ⇒ 收尾 NameError ⇒ core 和完整版一起没了。
#   现在即使后置读数全崩，盘上已经有一份不含读数的主结果。
_core_ok = _safe_write_json(results, _core_path)
if not _core_ok:
    flush_print("  [disk] ★★ 主结果 core 未落盘： %s" % _core_path)
else:
    flush_print("  [v99] 主结果 core 已先落盘： %s" % _core_path)

# ---- [v99] gate2 通道粒度轴的后置读数：results 构造**完成之后**才算 ----
#   ★ 为什么用回填而不是写进字面量：字面量内部不能引用"之后才绑定"的名字。
#     挪位置只能救这一处，**回填**才能让这一类 bug 结构性消失。
#   ★ 为什么仍在 disk_guard / text_sync **之后**：判词读的必须是跑完之后的
#     最终参数态；提前读会拿到零初始化（跨组 std 全 0）⇒ 必然红的假检查（覆辙 267）。
_v98_gran = _safe_readout(_v98_granularity_readout, {})
_v98_gran_verdict = _safe_readout(
    lambda: _v98_granularity_verdict(_v98_gran), "GRAN_READOUT_ERROR")
results["v98_gate2_granularity"] = {
    "axis": AXIS_V98,
    "readout": _v98_gran,
    "verdict": _v98_gran_verdict,
    "thresholds": {
        "gran_separation_min": float(GRAN_SEPARATION_MIN),
        "alpha_bound_eps": float(GRAN_ALPHA_BOUND_EPS),
        "phys2_groups_target": int(PHYS2_GROUPS),
        "phys1_groups_control": int(PHYS1_GROUPS),
        "phys3_groups_control": int(PHYS3_GROUPS),
    },
    "negctrl_note": (
        "负向对照（两个方向，均由执行级自证实测）："
        "(a) 能红的方向 —— s_param / lam_param / nu_param 全为 "
        "nn.Parameter(torch.zeros(groups))，phase_bias / dynamic_bias 是标量广播 "
        "⇒ 第 0 步这三个参数张量的跨组 std **精确为 0.0**（实测 C5）。"
        "所以 'max(std_s, std_lam, std_nu) >= 1e-3' **能红**，不是同义反复（覆辙 267）。"
        "★ 精确起见：running_s/running_lam/running_nu 是 f32 的 torch.full（64 个同一值），"
        "其 std 实测为 **5.96e-08 而非 0**（f32 两遍归约的舍入，覆辙 272 的新形态）——"
        "故判据**只用 s/lam/nu 三个 param 的 std**，不读 running_*；阈值 1e-3 比该噪声大 4 个数量级。"
        "(b) 另一个方向 —— 把 PHYS2_GROUPS 改回 8，phys2.groups 就不等于 "
        "phys2_groups_target ⇒ 判词立刻 GRAN_GROUPS_NOT_APPLIED。"),
    "note": (
        "★ 这是**机制级**判词，与精度无关，单跑即可判。"
        "精度另走跨跑口径（PASS>=%.2f / 红线<=%.2f，σ_diff=%.3fpp，全部由常量派生）；"
        "档案已定案 13 版 delivered 极差 0.280pp < 2σ=%.3fpp ⇒ "
        "跨跑比精度已不可判，单跑 NULL 不得升级为'轴无效'（需 3 seed）。"
        % (float(PLATEAU_PASS), float(PLATEAU_REDLINE),
           float(PLATEAU_SIGMA), float(PLATEAU_PASS_MARGIN))),
    "self_sha1": SELF_SHA1,
}
# ---- [v101] 各向异性轴的后置读数（与 v98 块同一模式：results 构造**完成之后**回填；
#      字面量内部永不引用后置变量 —— v98/v93 的 NameError 就是这样消灭的，覆铣 279）----
_v101_aniso = _safe_readout(_v101_aniso_readout, {})
_aniso_verdict = _safe_readout(
    lambda: _v101_aniso_verdict(_v101_aniso, _aniso_st), "ANISO_READOUT_ERROR")
results["v101_anisotropy"] = {
    "axis": AXIS_V101,
    "selftest": _aniso_st,
    "readout": _v101_aniso,
    "verdict": _aniso_verdict,
    "thresholds": {
        "aniso_range": float(ANISO_RANGE),
        "aniso_deliver_min": float(ANISO_DELIVER_MIN),
        "aniso_frozen_max": float(ANISO_FROZEN_MAX),
        "aniso_modules_expected": int(ANISO_MODULES_EXPECTED),
        "aniso_equiv_tol": float(ANISO_EQUIV_TOL),
        "aniso_tanh_sat_param": float(ANISO_TANH_SAT_PARAM),
    },
    "instrument_fix_note": (
        "[v103] 唯一一根轴 = SEED 1234->42（seed swap）；各向异性轴从 v101 **原样继承**，"
        "本版不重开、不改一个字节。v102 的仪器修复（覆铣 #312：_on_dev 搬家、"
        "completed 分母、selftest-error 判据档）全部保留 —— v102 实跑已用 15/15 "
        "completed=True 兑现。本版实跑的两件事：① 协议 mandated 的第 2 个 seed"
        "（v90/v101/v102 三次观测同为 seed 1234，本质是一个）；② provenance 修复"
        " —— v102 实跑文件名 loss.mmm1.py、self_sha1 ad097731... != 交付片"
        " c12256ab...（B 线第一次内容级破损），本版必须 cnn10_v103.py 之名跑，"
        "开机横幅 sha1 须等于交付值。精度判据：与 v102 同代码不同 seed ⇒ 预期落在"
        " σ_diff=0.169 内；本跑目的是量当前代码的 seed 方差，单跑 NULL 不得升级"
        "为'轴无效'（需 3 seed）。"),
    "negctrl_note": (
        "负向对照（三个方向，均由执行级自证实测）："
        "(a) 第 0 步逐位相同**能红** —— case0d 把 aniso_param 置 0.5 后 a 必须 != 1；"
        "(b) 等变判据**能红** —— case2c 拿错轴的翻转做对照，差必须 > 容差"
        "（否则输出全零之类也能'通过'等变检查，覆铣 267）；"
        "(c) f32 饱和是**预期不是失败** —— case1b 断言 tanh(10) 在 f32 下精确等于 1.0 "
        "⇒ a 精确等于 2.0；v90 的旧自测期望'a 严格小于 2'，必然被饱和咬成假红。"),
    "note": (
        "★ 这是**机制级**判词，与精度无关，单跑即可判。"
        "精度另走跨跑口径（PASS>=%.2f / 红线<=%.2f，σ_diff=%.3fpp，全部由常量派生）；"
        "档案已定案跨跑比精度不可判，单跑 NULL 不得升级为'轴无效'（需 3 seed）。"
        % (float(PLATEAU_PASS), float(PLATEAU_REDLINE),
           float(PLATEAU_SIGMA))),
    "self_sha1": SELF_SHA1,
}
results["v99_crash_fix"] = {
    # ★ AXIS_V99 必须是**活引用**而不是死常量（覆辙 251：数引用，不读函数名）。
    #   它声明"本版无新轴"，只有被真正读出来才算声明过了。
    "axis_this_version": AXIS_V99,
    "bug": "USED_BEFORE_BOUND at module level (pitfall 279, same shape as v93)",
    "symptom": ("NameError: name '_v98_gran' is not defined (loss.mmm1.py L3644), "
                "raised AFTER a full 180-epoch run => entire JSON lost"),
    "root_cause": ("`results = {` is at L3453 and references _v98_gran at L3644, "
                   "but the name is only bound at L3704 (251 lines later). Structural "
                   "cause: two rules held at once - 'readout must come after the "
                   "delivery chain' and 'readout must go into the JSON' - are mutually "
                   "exclusive inside one dict literal. Note the same file already used "
                   "the CORRECT pattern (`results['disk_guard'] = {...}`) two blocks up."),
    "fix": ("results literal no longer references any post-hoc name; the readout is "
            "computed AFTER construction and attached by assignment; every post-hoc "
            "readout is wrapped in _safe_readout."),
    "why_not_just_move_the_assignment": ("Moving it earlier fixes this one instance but "
                                         "leaves the pattern intact - the next version "
                                         "that adds a readout to the literal hits the same "
                                         "wall. Assignment-after-construction makes the "
                                         "whole class of bug structurally impossible."),
    "extra_guards": [
        "main results written to *_core.json BEFORE any post-hoc readout runs",
        "all post-hoc readouts wrapped in _safe_readout (assembly cannot kill the run)",
        "verdict echoed to stdout (dual channel: JSON may die, stdout cannot)",
        "namescope (module-level binding) check integrated into the deploy checker",
    ],
    "training_math_changed": False,
    # ★ [v100] 就地改正：v99 交付时写的是"expected identical (97.42)"，
    #   而 v99 实跑 **97.44** ⇒ 这句话被自己的数据证伪。
    #   不删历史（覆辙 292：改判断可以，抹掉记录不行）—— 原句与实测并排留下。
    "expected_delivered_vs_v98_AS_WRITTEN_AT_DELIVERY": (
        "identical (97.42); a deviation is evidence of "
        "non-determinism, not of an axis effect"),
    "expected_delivered_vs_v98_CORRECTED_BY_V100": (
        "WRONG AS ORIGINALLY WRITTEN. v99 actually ran delivered = 97.44 "
        "(best_tta 97.44 @ep120, plateau 97.3763), NOT 97.42. Two honest "
        "readings and they are NOT the same claim: (a) v99's training math IS "
        "bit-identical to v98 (no parameter/schedule/RNG/loss change), so the "
        "+0.02pp is run-to-run noise on the SAME code path; (b) but "
        "'bit-identical code' does NOT buy 'bit-identical result' on this "
        "platform - pitfall 141 (same-seed non-determinism, measured three "
        "times) plus cudnn/cuBLAS non-associativity. => NEVER write "
        "'expected identical to <number>' as a delivery claim again; write "
        "'statistically indistinguishable at sigma_diff' and let the "
        "measurement speak. v100 carries the same discipline for v99: see "
        "v100_instrument_fix.expected_delivered_vs_v99."),
}
# ★ 双通道（覆辙 279）：JSON 可能死，stdout 死不了。
#   v98 那次 stdout 上同样一个字都没有 —— 因为崩在 L3644，而打印在 L3843。
flush_print("")
flush_print("=" * 60)
flush_print("[v99] gate2 通道粒度轴 —— 机制判词（stdout 通道，先于 JSON 落盘）:")
flush_print("  verdict = %s" % _v98_gran_verdict)
for _k in ("phys2_groups", "phys1_groups", "phys3_groups",
           "s_param_std", "lam_param_std", "nu_param_std"):
    _c = (_v98_gran.get(_k) or {}) if isinstance(_v98_gran, dict) else {}
    flush_print("    %-20s = %s%s" % (_k, _c.get("value"),
                                      "" if _c.get("error") is None
                                      else "   [ERR %s]" % _c.get("error")))
flush_print("=" * 60)

# [v97] 主结果走守卫：写不出来就把整份 JSON 打到 stdout（双通道，覆辙 279）
_json_ok = _safe_write_json(results, out_path)
if not _json_ok:
    flush_print("  [disk] ★★ 主结果 JSON 未落盘： %s" % out_path)

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
flush_print("  --- [v100] 剂量门拆成两门（v99 一把门判两个通道 ⇒ 过度严格）---")
flush_print("  平台通道（主门/硬守卫）: dose_delivered = %s   "
            "(%d/%d 个平台 epoch 带剂量, 窗 ep%s..%s)"
            % (str(bool(lr_tail_instrument["dose_delivered"])),
               int(lr_tail_instrument["n_plateau_epochs_under_dose"]),
               int(lr_tail_instrument["n_plateau_epochs_total"]),
               str(_plateau_first_epoch), str(_plateau_last_epoch)))
flush_print("  交付通道（选择点/事实）: dose_delivered = %s   "
            "(选择点 ep%d, 剂量首改 ep%d)"
            % (str(bool(lr_tail_instrument["dose_delivered_delivered_channel"])),
               int(best_tta_epoch), int(_lr_tail_first)))
flush_print("  交付通道判词 = %s" % lr_tail_delivered_verdict)
flush_print("  argmax 并列点 = %s (n=%d, tie-break=%s) ⇒ "
            "选择点 ep%d 在剂量窗内? %s | 任一并列点在剂量窗内? %s"
            % (str(lr_tail_instrument["best_tta_argmax_epochs"]),
               int(lr_tail_instrument["best_tta_argmax_n"]),
               str(lr_tail_instrument["best_tta_tie_break_rule"]),
               int(best_tta_epoch),
               str(bool(lr_tail_instrument["selection_epoch_inside_dose"])),
               str(bool(lr_tail_instrument["any_argmax_epoch_inside_dose"]))))
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
flush_print("[v98] gate2 通道粒度轴 —— 机制读数（JSON 之外再打一遍，双通道）:")
flush_print("  verdict = %s   (阶梯见 _v98_granularity_verdict 的 docstring)" % _v98_gran_verdict)
for _k in ("phys2_groups", "phys2_cpg", "phys2_groups_times_cpg",
           "phys1_groups", "phys3_groups",
           "s_param_std", "lam_param_std", "nu_param_std",
           "running_s_std", "running_lam_std", "running_nu_std",
           "gate2_alpha", "gate2_scale", "gate2_inject_effective",
           "gate3_alpha", "gate3_scale", "gate3_inject_effective",
           "gate1_inject_fixed"):
    _c = _v98_gran.get(_k) or {}
    _err = _c.get("error")
    flush_print("    %-26s = %s%s" % (_k, _c.get("value"),
                                      "" if _err is None else "   [ERR %s]" % _err))
flush_print("=" * 60)

flush_print("")
flush_print("=" * 60)
flush_print("[v101] 各向异性 Matérn 核轴 —— 机制读数（JSON 之外再打一遍，双通道）:")
flush_print("  selftest = %s/%s %s" % (_aniso_st['n_pass'], _aniso_st['n_expected'],
                                       'ALL PASS' if _aniso_st['all_pass'] else 'FAIL'))
flush_print("  verdict = %s   (阶梯见 _v101_aniso_verdict 的 docstring)" % _aniso_verdict)
for _k in ("aniso_range", "phys1_has_aniso_param",
           "phys2_aniso_numel", "phys2_aniso_abs_max_final",
           "phys2_a_min", "phys2_a_max",
           "phys2_max_abs_a_minus_1", "phys2_mean_abs_a_minus_1",
           "phys2_aniso_abs_max_running", "phys2_aniso_saturated_groups",
           "phys3_aniso_numel", "phys3_aniso_abs_max_final",
           "phys3_a_min", "phys3_a_max",
           "phys3_max_abs_a_minus_1", "phys3_mean_abs_a_minus_1",
           "phys3_aniso_abs_max_running", "phys3_aniso_saturated_groups",
           "aniso_trace_len", "aniso_epochs_with_movement"):
    _c = _v101_aniso.get(_k) or {}
    _err = _c.get("error")
    flush_print("    %-30s = %s%s" % (_k, _c.get("value"),
                                      "" if _err is None else "   [ERR %s]" % _err))
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
flush_print("[v46 血脉] 同 seed 复现检验（seed 与 v44 相同 ⇒ 仪器改动零副作用；本版 seed=42 ⇒ 预期 SEED_CHANGED_NOT_A_REPLICATE）:")
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
flush_print("LR_FLOOR  = %s   (v47 遗留轴，本版非轴；剂量 %d 轮被咬住 ⇒ %s)"
            % (lr_floor_verdict, int(_lr_floor_bites_in_run),
               "干预已送到" if int(_lr_floor_bites_in_run) >= 1 else "★干预没送到，NULL 不可归因"))
flush_print("INSTRUMENT = %s   (仪器自曝；FAIL ⇒ 上面读数不可信)" % instrument_verdict)
flush_print("SAME_SEED_REPLICATE = %s   (seed 42 != v44 的 1234 ⇒ 预期 SEED_CHANGED_NOT_A_REPLICATE；同 seed 时 EXACT/NEAR 才证仪器零副作用)" % _repro)
flush_print("REPLICATE = %s" % replicate)
flush_print("DFT_GRID = %s   (v96 遗留轴，本版非轴：flip 等变 + DC 增益，开工前即可判)" % _v96_dft_vd)
if _v96_dft_vd != "DFT_DELIVERED":
    flush_print("  ★★ 频率标尺没修好 ⇒ 本次读数**不可**用于跨跑归因，看 v96_dft_grid.selftest.checks")
flush_print("判据(2σ) = PASS >= %.2f / NULL (%.2f, %.2f) / BACKTRACK <= %.2f  "
            "「基线 v51 %.2f, σ_diff %.3f」"
            % (PASS_LINE_96, REDLINE_96, PASS_LINE_96, REDLINE_96,
               BASELINE_V51_ACC, SIGMA_DIFF_PP))
flush_print("★ 本版只有跨跑判据：单跑 NULL 不得升级为‘轴无效’（需 3 seed）")
flush_print("=" * 60)
flush_print("-" * 60)
flush_print("v113 约束通道（pinn 作用面）: delivered = %.2f%%  (基线 v110 = %.2f, pass_line = %.2f)"
             % (float(delivered), float(V113_BASE_DELIVERED), float(V113_PASS_LINE)))
flush_print("  梯度分流: grad_frac_new(新作用点) = %s | grad_frac_old(旧作用点) = %s"
             % (("%.3e" % float(_v113_probe.get("grad_frac_new", -1.0))) if _v113_probe_ok else "NA",
                ("%.6f" % float(_v113_probe.get("grad_frac_old", -1.0))) if _v113_probe_ok else "NA"))
flush_print("  场内容: band_frac_mean = %s (v111 实测 %.4f) | raw_lambda_c_px = %s (v111 %.3f) | 判词 = %s"
             % (("%.4f" % float(_v113_probe.get("band_frac_mean", 0.0))) if _v113_probe_ok else "NA",
                float(V113_BASELINE_BAND_FRAC),
                ("%.3f" % float(_v113_probe.get("raw_lambda_c_px", 0.0))) if _v113_probe_ok else "NA",
                float(V113_BASELINE_LAMBDA_C_PX),
                _v113_verdict))
sys.stdout.flush()
