# 发布到 GitHub

本地仓库**已提交完毕**。剩下只有一步：建远端并推送。

## 0. 本地状态（已完成）

| 项 | 状态 |
|---|---|
| `git init` | ✅ |
| `git add -A`（**71** 个文件，135,900+ 行） | ✅ |
| 首次提交 `11ea123` | ✅ |
| 字节修正提交 `077ecb0` | ✅ |
| `core.autocrlf = false` + 目录级 `-text`（保护字节级 provenance） | ✅ |
| 全部 `.py` 通过 `py_compile`（34 个） | ✅ |
| `MANIFEST.md` 已生成（56 个入库文件的 sha1 / 字节 / 行数 / 换行） | ✅ |
| 提交者身份 | ✅ `heyueshSean <heyueshsta@163.com>` |
| **HEAD blob sha1 == 磁盘文件 sha1** | ✅ **71/71** |
| 工作树 | ✅ clean |
| `gh` CLI | ❌ 本机未安装 |

当前提交树：

```
077ecb0  Fix gitattributes precedence; add PUBLISH.md
11ea123  Add differentiable-physics-field archive: two lines, all branches, best versions
```

## 1. 建远端仓库并推送

### 方式 A：用 GitHub CLI（需先装 gh 并 `gh auth login`）

```bash
gh repo create diffphys-infodiff \
  --public \
  --source=. \
  --remote=origin \
  --description "Differentiable physics fields for information/energy diffusion: a two-line experimental archive (RL + NN) with all branches, best versions and 32 negative results" \
  --push
```

### 方式 B：手动（本机走这条）

1. 在 https://github.com/new 建一个名为 **`diffphys-infodiff`** 的空仓库
   （**不要**勾选 README / .gitignore / LICENSE —— 本地都已备好，勾了会冲突）；
2. 然后：

```bash
cd "C:/Users/Yuesheng He/WorkBuddy/2026-08-07-15-31-42/diffphys-infodiff"

git remote add origin https://github.com/heyueshSean/diffphys-infodiff.git
git branch -M main
git push -u origin main
```

若 GitHub 要求认证：用户名填 `heyueshSean`，密码栏用 **Personal Access Token**
（GitHub 已停用账号密码推送）。Token 在 Settings → Developer settings →
Personal access tokens → Tokens (classic)，勾选 `repo` 即可。

## 2. 推完之后

```bash
python tools/make_manifest.py            # 之后若增删文件，重新生成清单
git diff --exit-code -- MANIFEST.md      # CI 会检查这条
```

- 建议开启 GitHub Actions（`.github/workflows/ci.yml` 已就绪）：
  它会编译全部脚本、并检查 `MANIFEST.md` 是否最新。
- 建议 topics：`physics-informed`、`differentiable-physics`、
  `reinforcement-learning`、`negative-results`、`cifar-10`、`cifar-100`、`ppo`。
- 建议 About 写：**Most results here are NULL. The value is in the negative
  results and the measurement protocol.**

## 3. ★ 一个真实的坑：`.gitattributes` 的规则顺序

本仓第一次提交**就踩了**，靠提交后的字节校验才抓出来——肉眼看不出来。

`.gitattributes` 按「**同一文件匹配多行时，最后出现的那行赢**」解析，
**与路径写得有多具体无关**。第一版把扩展名规则写在目录规则之后：

```gitattributes
rl/**       -text      # 先写
results/**  -text
*.txt  text eol=lf     # 后写 → 反而赢了
```

结果 `results/rl/env_check.txt`（100 B，1 个 CR）和
`results/rl/verify_ppo_v21_out.txt`（11082 B，230 个 CR）被 Git 偷偷 LF 归一化，
blob sha1 与训练机上的文件对不上——**正是 `.gitattributes` 本来要防的那件事**。

修法：**目录级 `-text` 必须写在扩展名规则之后**。

```gitattributes
*.md    text eol=lf    # 文档类先写
*.txt   text eol=lf
rl/**       -text      # 代码与证据后写 → 覆盖
results/**  -text
```

然后 `git add --renormalize -A` 让 Git 按新规则重算已入库文件的 blob。

**校验手段**（每次提交后都应跑一遍，比什么都可靠）：

```bash
# 逐个文件比对 HEAD blob 与磁盘文件的 sha1，必须全等
git ls-files | while read f; do
  a=$(git show HEAD:"$f" | sha1sum | cut -d' ' -f1)
  b=$(sha1sum "$f"        | cut -d' ' -f1)
  [ "$a" = "$b" ] || echo "MISMATCH: $f"
done
```

## 4. 未入库的内容（有意排除）

- 全部权重 / checkpoint（`ckpts_*/`、`out_*/`、`*.pth`）—— 见 `.gitignore`；
- 约 650 个中间版本脚本（本仓库只收**每个分支的最佳版本** + 关键反例）；
- 逐 epoch 训练日志与内部决策索引（`.workbuddy/memory/`）—— 含未脱敏的工作笔记。
