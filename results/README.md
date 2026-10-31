# results · 真跑产出的读数

本目录放**训练机真跑产出的原始 JSON / verify 输出**，不放任何合成 fixture。
仓库里所有表格中的数字都可以追溯到这里的文件。

## nn/ —— 神经网络线

| 文件 | 版本 | 关键读数 |
|---|---|---|
| `results_cnn_v51_seed1234.json` | B 线基线 | delivered **97.56** |
| `results_cnn_v110_seed42.json` / `_core.json` | v110 参数侧度量族 | **97.50**；`dir_ratio_mean = 0.9129` |
| `results_cnn_v111_seed42.json` / `_core.json` | v111 频带夹带 | 97.46；`band_frac_mean = 0.0968` |
| `results_cnn_v112_seed42.json` / `_core.json` | v112 视图族 | 97.52；配对 Δ **−0.18pp**、McNemar p=0.05045 |
| `results_cnn_v117_seed42.json` / `_core.json` | v117 约束通道·pinn | 97.59；`grad_frac_new = 0.0`、`band_frac_mean = 0.05526` |
| `results_cnn_v118_seed42.json` / `_core.json` | v118 约束通道·smooth | **97.53**；`grad_frac_smooth_new = 0.0`、`band_frac_mean = 0.038904`、`smooth_level` 43.9× |
| `results_cnn_info_v01_seed1234.json` | InfoDiff v01 | 97.06；`i_std ≈ 0.012` |
| `results_cnn_info_v02_seed1234.json` | InfoDiff v02 | **97.12**；`INFOMAP_NO_LEVERAGE` |

`*_core.json` 是从完整 JSON 抽出的核心块（顶层标量 + 判据块），便于快速比对；
完整 JSON 含逐 epoch 数组与全部仪器读数。

## rl/ —— 强化学习线

| 文件 | 版本 | 内容 |
|---|---|---|
| `verify_ppo_v9_out.txt` | v9 | α 释放调度（5M）验证输出 |
| `verify_ppo_v10_out.txt` | v10 | α 释放调度（10M）验证输出 |
| `verify_ppo_v21_out.txt` | v21 | 熵淬火 + 回火验证输出 |
| `verify_ppo_v22_out.txt` | v22 | DAP TAIL 形状验证输出 |
| `env_check.txt` | — | 训练环境自检 |

⚠ RL 线的回报曲线**没有落盘 JSON**（当时只留了 verify 输出），
`docs/03-rl-line.md` 中的回报数字来自实验日志的逐次记录。
这是本仓库已知的记录缺口，也是"方法论硬伤"的一部分。

## 怎么用

读一个版本的结论，按这个顺序：

1. `delivered` / `verdict` —— 精度结论（注意 `delivered_source` 是否同口径）；
2. 该版本的**机制块**（如 `v118_smoothface`）—— `probe_ok` 先看，再看读数；
3. `self_sha1` —— 与 `MANIFEST.md` 比对，确认跑的就是这个字节文件；
4. `determinism` / `det_guard` —— 确认是否在确定性算子下跑（影响是否同尺）。
