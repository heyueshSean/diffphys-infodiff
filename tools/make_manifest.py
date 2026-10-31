#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate MANIFEST.md: sha1 / bytes / lines / line-ending for every tracked file.

Run from the repository root:

    python tools/make_manifest.py

The manifest is the byte-level provenance record of this repository.  Every
version file shipped here must be reproducible from its sha1, because the
whole research log is keyed on sha1 (see docs/06-methodology.md).
"""
from __future__ import annotations

import hashlib
import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories that hold shipped source / evidence.  Everything else (docs,
# LICENSE, ...) is text and is listed without hashes.
SCAN_DIRS = ["rl", "nn", "results", "tools"]

# Human-readable role tags, keyed by basename.  Used to build the
# "what is this file" column in MANIFEST.md.
ROLES = {}


def _role(path: str) -> str:
    return ROLES.get(os.path.basename(path), "")


def _file_info(path: str) -> dict:
    with io.open(path, "rb") as fh:
        data = fh.read()
    sha1 = hashlib.sha1(data).hexdigest()
    cr = data.count(b"\r\n")
    lf = data.count(b"\n")
    if cr and cr == lf:
        eol = "CRLF"
    elif cr:
        eol = "MIXED"
    else:
        eol = "LF"
    return {
        "path": os.path.relpath(path, ROOT).replace("\\", "/"),
        "sha1": sha1,
        "bytes": len(data),
        "lines": lf + (1 if data and not data.endswith(b"\n") else 0),
        "eol": eol,
    }


def main() -> None:
    rows = []
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames.sort()
            for name in sorted(filenames):
                if name.startswith("."):
                    continue
                rows.append(_file_info(os.path.join(dirpath, name)))

    out = []
    out.append("# MANIFEST — 字节级 provenance")
    out.append("")
    out.append("由 `python tools/make_manifest.py` 生成。本仓库的每一份代码都按 **sha1** 归档：")
    out.append("训练机上发车前必须核对 sha1 与行数，否则会「跑了旧版却以为改过了」。")
    out.append("（换行约定：B 线/C 线 = LF，A 线 = CRLF，这是各线的历史约定，不要改动。）")
    out.append("")
    out.append("| 文件 | sha1 | 字节 | 行数 | 换行 |")
    out.append("|---|---|---:|---:|---|")
    for r in rows:
        out.append(
            "| `{p}` | `{s}` | {b} | {l} | {e} |".format(
                p=r["path"], s=r["sha1"], b=r["bytes"], l=r["lines"], e=r["eol"]
            )
        )
    out.append("")
    out.append("总计 **{n}** 个文件。".format(n=len(rows)))
    out.append("")

    with io.open(os.path.join(ROOT, "MANIFEST.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out))

    print("MANIFEST.md written: {n} files".format(n=len(rows)))
    for r in rows:
        print("  {0:58s} {1}  {2:>8d}B  {3:>6d}L  {4}".format(
            r["path"], r["sha1"], r["bytes"], r["lines"], r["eol"]))


if __name__ == "__main__":
    main()
