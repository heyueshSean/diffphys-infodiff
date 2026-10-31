# 发布到 GitHub

仓库目录已就绪（`git init` 完成，70 个文件已暂存）。下面是发布所需的全部命令。

## 0. 前置检查（已做）

| 项 | 状态 |
|---|---|
| `git init` | ✅ |
| `git add -A`（70 个文件，135,794 行） | ✅ |
| `core.autocrlf = false`（保护字节级 provenance） | ✅ |
| 全部 `.py` 通过 `py_compile`（34 个） | ✅ |
| `MANIFEST.md` 已生成（56 个入库文件的 sha1） | ✅ |
| `gh` CLI | ❌ 本机未安装 |

## 1. 设置身份（只需一次）

```bash
git config user.name  "<你的名字>"
git config user.email "<你的 GitHub 邮箱>"
```

## 2. 首次提交

```bash
git commit -F - <<'EOF'
Add differentiable-physics-field archive: two lines, all branches, best versions

Two independent experimental lines on differentiable physics fields
(information / energy diffusion as an inductive bias):

- RL line (PPO + Atari Breakout, v1-v34 + Route D): the field is a graph
  diffusion operator over rollouts, not a network layer. Only the credit
  assignment surface works (SOTA v26: best block 215.2, TAIL CV 4.34%,
  0.06% overhead). Archived 2026-08-26 with a directional negative verdict.

- NN line (CIFAR-10 / CIFAR-100 / InfoDiff): the field is a learnable
  Matérn/heat-kernel spectral multiplier. Additive injection, constraint
  channels, parameter-space metrics and readout aggregation are all judged
  dead on evidence; the alignment-target surface (v119) is still untested.

Ships 32 negative results with version-pinned measurements, the measurement
protocol (paired criteria, sigma floors, max-of-N caveats), and the
byte-level provenance discipline (sha1-pinned MANIFEST).
EOF
```

## 3. 建远端仓库并推送

### 方式 A：用 GitHub CLI（推荐，需先装 gh 并 `gh auth login`）

```bash
gh repo create diffphys-infodiff \
  --public \
  --source=. \
  --remote=origin \
  --description "Differentiable physics fields for information/energy diffusion: a two-line experimental archive (RL + NN) with all branches, best versions and 32 negative results" \
  --push
```

### 方式 B：手动

1. 在 https://github.com/new 建一个名为 `diffphys-infodiff` 的空仓库（**不要**勾选 README / .gitignore / LICENSE）；
2. 然后：

```bash
git remote add origin https://github.com/<你的用户名>/diffphys-infodiff.git
git branch -M main
git push -u origin main
```

## 4. 推完之后

```bash
python tools/make_manifest.py     # 若之后增删文件，重新生成清单
git diff --exit-code -- MANIFEST.md   # CI 会检查这条
```

- 建议开启 GitHub Actions（`.github/workflows/ci.yml` 已就绪）：
  它会编译全部脚本、并检查 `MANIFEST.md` 是否最新。
- 建议在仓库设置里加 topics：`physics-informed`, `differentiable-physics`,
  `reinforcement-learning`, `negative-results`, `cifar-10`, `cifar-100`, `ppo`。
- 建议在 About 里写上：**Most results here are NULL. The value is in the
  negative results and the measurement protocol.**

## 5. 未入库的内容（有意排除）

- 全部权重 / checkpoint（`ckpts_*/`、`out_*/`、`*.pth`）—— 见 `.gitignore`；
- 约 650 个中间版本脚本（本仓库只收**每个分支的最佳版本** + 关键反例）；
- 逐 epoch 训练日志与内部决策索引（`.workbuddy/memory/`）—— 含未脱敏的工作笔记。
