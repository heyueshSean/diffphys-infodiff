#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
publish_via_api.py — 当 HTTPS/SSH 都被网络策略挡住时，用 GitHub REST API 推送。

为什么需要这个脚本
--------------------
git push 必须连 `github.com:443`。如果所在网络的代理只对 `api.github.com`
放行（CONNECT github.com:443 返回 502），`git push` 就彻底走不通。
本脚本改走 `https://api.github.com`，用 Git Data API 手工构造
blob → tree → commit → ref，效果等价于一次 git push，且**字节完全保真**。

字节保真的依据
----------------
`POST /git/blobs` 接受 `{"content": <base64>, "encoding": "base64"}`，
服务端按原始字节存 blob，不做任何换行转换（`.gitattributes` 是**客户端**
机制，API 不会应用它）。因此 blob 的 Git sha1 与本地文件的 sha1 一致，
`MANIFEST.md` 里的指纹仍然成立。

用法
----------------
    export GH_TOKEN=<你的 GitHub Personal Access Token>
    python tools/publish_via_api.py

或：
    python tools/publish_via_api.py --token <token>

可选参数：
    --owner   默认 heyueshSean
    --repo    默认 diffphys-infodiff
    --branch  默认 main
    --root    默认脚本所在目录的上级（仓库根）
    --dry-run 只列出将要上传的文件，不实际请求
    --verify  上传后逐个下载 blob 校验 sha1（默认开启）

Token 需要 `repo` 权限（对 public 仓库也要有 public_repo 写入权）。
脚本**不会**把 token 写进任何文件；只从命令行或环境变量读取。

单文件上限 100 MB（GitHub 限制）；本仓库最大约 0.5 MB，无压力。
"""

import argparse
import base64
import hashlib
import json
import os
import sys
import time

try:
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
except ImportError:  # pragma: no cover
    print("需要 Python 3")
    sys.exit(1)

API = "https://api.github.com"
# 本脚本刻意只访问 api.github.com —— 这是受限网络里通常唯一放行的 host。


def _req(method, path, token, body=None, expect=(200, 201, 204)):
    url = API + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    r = Request(url, data=data, method=method)
    r.add_header("Authorization", "Bearer " + token)
    r.add_header("Accept", "application/vnd.github+json")
    r.add_header("X-GitHub-Api-Version", "2022-11-28")
    r.add_header("User-Agent", "diffphys-infodiff-publisher")
    if data is not None:
        r.add_header("Content-Type", "application/json")
    try:
        with urlopen(r, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
            return (resp.status, json.loads(raw) if raw else {})
    except HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            detail = json.loads(raw)
        except Exception:
            detail = {"message": raw[:400]}
        if method == "GET" and e.code == 404:
            return (404, detail)
        raise RuntimeError("%s %s -> %d %s" % (
            method, path, e.code, detail.get("message", "")[:400]))


def collect_files(root):
    """收集仓库根下所有受版本控制的文件（尊重 .gitignore 的粗筛）。"""
    skip_dirs = {".git", "__pycache__", ".github"}
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace("\\", "/")
            # 绝对路径自身不入库
            if rel.startswith("../"):
                continue
            out.append(rel)
    return sorted(out)


def blob_sha1(data):
    """Git blob 的 sha1：'blob <len>\\0' + content"""
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("GH_TOKEN", ""))
    ap.add_argument("--owner", default="heyueshSean")
    ap.add_argument("--repo", default="diffphys-infodiff")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--root", default=os.path.dirname(here))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    files = collect_files(args.root)
    print("仓库根：%s" % args.root)
    print("待上传文件：%d" % len(files))
    total = sum(os.path.getsize(os.path.join(args.root, f)) for f in files)
    print("总大小：%.2f MB" % (total / 1048576.0))

    # dry-run 不需要凭据 —— 先处理，方便无条件自检
    if args.dry_run:
        for f in files:
            print("  " + f)
        return 0

    if not args.token:
        print("错误：未提供 token。用 --token 或环境变量 GH_TOKEN。")
        print("token 需要 repo 权限：https://github.com/settings/tokens")
        return 2

    base = "/repos/%s/%s" % (args.owner, args.repo)

    # 0) 确认 token 有效
    st, me = _req("GET", "/user", args.token)
    print("认证为：%s" % me.get("login"))

    # 1) 逐个创建 blob
    tree = []
    for i, rel in enumerate(files, 1):
        with open(os.path.join(args.root, rel), "rb") as fh:
            data = fh.read()
        b64 = base64.b64encode(data).decode("ascii")
        st, b = _req("POST", base + "/git/blobs", args.token,
                     {"content": b64, "encoding": "base64"})
        got = b.get("sha")
        want = blob_sha1(data)
        if got != want:
            raise SystemExit("字节不一致！%s API=%s 本地=%s" % (rel, got, want))
        # 可执行位保留
        mode = "100755" if os.access(os.path.join(args.root, rel), os.X_OK) else "100644"
        tree.append({"path": rel, "mode": mode, "type": "blob", "sha": got})
        if i % 10 == 0 or i == len(files):
            print("  blob %d/%d  %s" % (i, len(files), rel))

    # 2) 建 tree（GitHub 支持带斜杠的完整路径，会自动构建子树）
    st, t = _req("POST", base + "/git/trees", args.token, {"tree": tree})
    print("tree  sha = %s" % t.get("sha"))

    # 3) 建 commit
    st, c = _req("POST", base + "/git/commits", args.token, {
        "message": "Add differentiable-physics-field archive: two lines, all branches, best versions\n\n"
                   "Pushed via Git Data API because the network blocks github.com:443.\n"
                   "Blobs are byte-identical to the local files (verified by blob sha1).",
        "tree": t.get("sha"),
        "parents": [],
    })
    print("commit sha = %s" % c.get("sha"))

    # 4) 指向分支（空仓没有 ref，用 POST；已有则用 PATCH）
    st, ref = _req("GET", base + "/git/refs/heads/" + args.branch, args.token)
    if st == 404:
        _req("POST", base + "/git/refs", args.token,
             {"ref": "refs/heads/" + args.branch, "sha": c.get("sha")})
        print("已创建 refs/heads/%s" % args.branch)
    else:
        _req("PATCH", base + "/git/refs/heads/" + args.branch, args.token,
             {"sha": c.get("sha")})
        print("已更新 refs/heads/%s" % args.branch)

    print("\n推送完成：https://github.com/%s/%s" % (args.owner, args.repo))

    # 5) 回读校验
    if not args.no_verify:
        print("\n回读校验（逐个比对 blob sha1）：")
        st, tr = _req("GET", base + "/git/trees/" + args.branch + "?recursive=1", args.token)
        remote = {x["path"]: x["sha"] for x in tr.get("tree", []) if x["type"] == "blob"}
        bad = []
        for rel in files:
            with open(os.path.join(args.root, rel), "rb") as fh:
                want = blob_sha1(fh.read())
            if remote.get(rel) != want:
                bad.append(rel)
        print("  远端 blob 数=%d  本地文件数=%d  不一致=%d"
              % (len(remote), len(files), len(bad)))
        for b in bad[:20]:
            print("   !! " + b)
        if not bad:
            print("  ✓ 全部字节一致，MANIFEST.md 的 sha1 在远端同样成立。")


if __name__ == "__main__":
    main()
