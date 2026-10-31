#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_clone_bytes.py — 验证「别人 clone 到的字节 == 我这台机器上的字节」。

为什么需要
------------
本仓库的全部价值建立在字节级 provenance 上：MANIFEST.md 里的 sha1
必须等于训练机上的文件 sha1。但 `core.autocrlf` 是**本地**配置，不会随
仓库传播，克隆者默认是 `true`。只要有文件没被 `.gitattributes` 的
`-text` 规则命中，Git 就会在克隆时把它转成 CRLF，字节就断了。

本仓库正是这么踩到的：修好目录级规则后重新 clone，`.gitattributes` /
`.gitignore` / LICENSE 三个根文件仍然变了 —— 它们既不在 `-text` 目录里，
也不匹配任何扩展名规则，于是落进了克隆者 autocrlf 的手里。
修法是加 `* -text` 兜底（作为最后一行），之后任何人 clone 都字节一致。

用法
------------
    # 1) 在别处用**默认**配置克隆（模拟真实克隆者，关键！）
    git clone --depth 1 -c core.autocrlf=true \
        git@github.com:heyueshSean/diffphys-infodiff.git /tmp/clone_check

    # 2) 比对
    python tools/verify_clone_bytes.py --clone /tmp/clone_check

不传 --clone 时，自动用脚本旁的同名目录 `_clone_verify/diffphys-infodiff`。

退出码：0 = 全部一致；1 = 有差异或缺失。
"""

import argparse
import hashlib
import os
import sys

SKIP_DIRS = {".git", "__pycache__"}


def walk(root):
    """返回 {相对 posix 路径: 绝对路径}，跳过 .git 与 __pycache__。"""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            # 统一成 posix 形式，便于跨系统比对
            out.append((rel.replace(os.sep, "/"), full))
    return dict(out)


def sha1_of(path):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_clone = os.path.join(
        os.path.dirname(here), "_clone_verify", "diffphys-infodiff")

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.dirname(here))
    ap.add_argument("--clone", default=default_clone)
    ap.add_argument("--show", action="store_true", help="打印每个文件的 sha1")
    args = ap.parse_args()

    if not os.path.isdir(args.clone):
        print("找不到克隆目录：%s" % args.clone)
        print("先用默认配置克隆一份：")
        print("  git clone --depth 1 -c core.autocrlf=true \\")
        print("      git@github.com:heyueshSean/diffphys-infodiff.git " + args.clone)
        return 2

    src = walk(args.src)
    cln = walk(args.clone)

    missing = sorted(k for k in src if k not in cln)
    extra = sorted(k for k in cln if k not in src)
    diff = sorted(
        k for k in src
        if k in cln and sha1_of(src[k]) != sha1_of(cln[k]))

    print("源仓库   %s  (%d 个文件)" % (args.src, len(src)))
    print("克隆副本 %s  (%d 个文件)" % (args.clone, len(cln)))
    print("缺失 = %d   多余 = %d   字节不同 = %d"
          % (len(missing), len(extra), len(diff)))

    for k in (missing + extra + diff)[:20]:
        print("   !! " + k)

    if missing or extra or diff:
        print("\n结论：克隆副本与源仓库字节不一致。")
        print("最常见原因：某些文件未被 .gitattributes 的 -text 规则命中，")
        print("被克隆端的 core.autocrlf 转了换行。检查规则的**顺序**：")
        print("gitattributes 是「后出现的那行赢」，所以目录级/兜底 -text 必须写在最后。")
        return 1

    print("\n*** 全部一致：克隆者拿到的字节与本机完全相同 ***")
    if args.show:
        for k in sorted(src):
            print("  %s  %s" % (sha1_of(src[k])[:12], k))
    else:
        # 挑几个代表性文件：LF 代码、CRLF 证据（两种换行都要覆盖）、无扩展名根文件
        probes = [".gitattributes", ".gitignore", "LICENSE",
                  "nn/cifar10/cnn10_v119.py",
                  "results/rl/verify_ppo_v21_out.txt",
                  "nn/cifar100/vit100_v111.py"]
        for k in probes:
            if k in cln:
                with open(cln[k], "rb") as fh:
                    data = fh.read()
                print("  %-40s %s %8dB CR=%d"
                      % (k, hashlib.sha1(data).hexdigest()[:12],
                         len(data), data.count(b"\r")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
