# 发布到 GitHub

## 🎉 已上线

**https://github.com/heyueshSean/diffphys-infodiff**

73 个文件 / 6.8 MB，推送时间 2026-10-31T04:28:10Z，默认分支 `main`。
走的是 **SSH**（`github.com:443` 在这台机器被网络策略挡死，443 走不通、22 通）。

推送后又从 GitHub **克隆回来逐字节比对过**：**73/73 文件 sha1 完全一致**，
包括 CRLF 的 CRLF（`CR=230`、`CR=12240` 一个 `\r` 没丢）。
⇒ 仓库里的 `MANIFEST.md` 指纹在远端同样成立。

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

> ### ⚠ 网络实况（2026-10-31 实测，先读这段）
>
> 远端仓库 `heyueshSean/diffphys-infodiff` 已建好（public、空仓、默认分支 `main`），
> 但**这台机器到 GitHub 的网络是受限的**：
>
> | 通道 | 结果 |
> |---|---|
> | `https://github.com` 经代理 | ❌ `CONNECT tunnel failed, response 502` |
> | `https://github.com` 直连（去代理） | ❌ 20 秒超时 |
> | **`git@github.com:22`（SSH）** | ✅ **TCP 开放，SSH 握手成功** |
> | **`https://api.github.com`** | ✅ **200** |
> | `codeload` / `raw` / `objects.githubusercontent.com` | ✅ 可达（只读） |
>
> 代理只对 `api.github.com` 等少数 host 放行。⇒ **`git push` 默认走的
> `https://github.com` 是死路**，必须改用下面两种方式之一。

### ✅ 方式 0（实际采用）：SSH

443 被拦、22 通。把公钥加到 GitHub 后，`git push -u origin main` 一次成功。

```bash
git remote add origin git@github.com:<owner>/diffphys-infodiff.git
git branch -M main
git push -u origin main
```

### 方式 0b：SSH（如尚未加公钥）—— 已配好，只差把公钥加到 GitHub

本地已就绪：`origin` 指向 `git@github.com:heyueshSean/diffphys-infodiff.git`，
分支已改名 `main`，3 个 commit 待推。

**你需要做一次**：把下面这把 ed25519 公钥加到
https://github.com/settings/ssh/new （Title 随便填，Key type 保持 Authentication Key）：

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMGH7jQ182tPItOa0q+sXTR4wRinQU+QOs28FJk4sF34 heyueshSean@diffphys-infodiff
```

（私钥在本机 `~/.ssh/id_ed25519`，无 passphrase。）

加完之后，我这边一条命令就推完：

```bash
cd "C:/Users/Yuesheng He/WorkBuddy/2026-08-07-15-31-42/diffphys-infodiff"
git push -u origin main
```

### 方式 1：用 API 推送（需要有效的 Personal Access Token）

`tools/publish_via_api.py` 走 `api.github.com`，用 Git Data API 手工构造
blob → tree → commit → ref。字节**完全保真**（已用 `git hash-object` 验证：
71/71 文件的 blob sha1 与 git 算的一致，CRLF 文件的 `\r` 一个不少）。

```bash
export GH_TOKEN=<你的 PAT>
python tools/publish_via_api.py          # 先 --dry-run 看清单
```

⚠ 注意：8 位那种短口令不是 PAT。GitHub 的 classic PAT 是 **40 位十六进制**
（或 `github_pat_` 开头的一长串）。在
https://github.com/settings/tokens 生成，勾选 `repo`。

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

## 3b. ★★ 第二个坑：本地一致 ≠ 克隆一致

第一个坑靠提交后的字节校验抓到。但**只校验本地是不够的**——推上去之后
我用**默认配置**（`-c core.autocrlf=true`，即 Git for Windows 的出厂设置）
重新克隆了一份，逐字节比对，结果：

```
缺失 = 0   多余 = 0   字节不同 = 3
   !! .gitattributes
   !! .gitignore
   !! LICENSE
```

**根因**：`core.autocrlf` 是**本地**配置，**不会随仓库传播**。那三个根文件
既不在 `-text` 目录里，也不匹配任何扩展名规则 ⇒ 落进了克隆者 `autocrlf=true`
的手里，被转成 CRLF。讽刺的是，其中一个是 `.gitattributes` 自己。

**修法**：加一条兜底，且必须是**最后一行**（后出现的规则赢）：

```gitattributes
# ---- 1. 文档类先写 ----
*.md    text eol=lf
*.txt   text eol=lf

# ---- 2. 目录级后写，覆盖上面 ----
rl/**       -text
results/**  -text

# ---- 3. 兜底放最后 ----
*           -text
```

修完再克隆：**73/73 字节全部一致**。

**工具**：`tools/verify_clone_bytes.py` —— 从外部验证这件事：

```bash
git clone --depth 1 -c core.autocrlf=true \
    git@github.com:heyueshSean/diffphys-infodiff.git /tmp/clone_check
python tools/verify_clone_bytes.py --clone /tmp/clone_check
```

★ 教训：**仓库级的 `core.autocrlf=false` 只保护你自己**。
要让别人拿到同样的字节，必须靠 `.gitattributes`，而且必须**覆盖到每一个文件**
——漏掉的那几个，正是你以为"不重要"的根文件。

## 4. 未入库的内容（有意排除）

- 全部权重 / checkpoint（`ckpts_*/`、`out_*/`、`*.pth`）—— 见 `.gitignore`；
- 约 650 个中间版本脚本（本仓库只收**每个分支的最佳版本** + 关键反例）；
- 逐 epoch 训练日志与内部决策索引（`.workbuddy/memory/`）—— 含未脱敏的工作笔记。
