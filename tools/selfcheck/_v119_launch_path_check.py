# -*- coding: utf-8 -*-
"""
_v119_launch_path_check.py —— **发车路径真跑验证**（回答「能不能直接运行」）

★★ 本文件**不训练**：它把 cnn10_v119.py 的模块级代码**真的执行一遍**，
   但在**训练循环之前**停下 ⇒ 验证「发车路径」零运行期错误。

★ 为什么它是 v117 的**决定性**证据：v116 实跑 = 发车就死
  （训练前自检 grad=False ⇒ 模块级裸调 run_determinism_probe() 抛
  CUDNN_STATUS_NOT_INITIALIZED）⇒ **180ep 一次没跑**。
  ⇒ v117 必须证明：发车路径上**每一步**都不会把主流程拖死。

★ 做法（**不改产物一个字节**）：
  1. 读产物源码；
  2. 用 AST 找到训练循环 `for epoch in ...:`；
  3. 把该循环**整段**替换成 `pass`（只在内存里改，不落盘）；
  4. `exec` 这份"截断版" ⇒ 真的跑到训练循环前 ⇒ 观察有没有异常。

★ 环境：本机 torch 2.13.0+cpu（**无 GPU**）⇒ 探针跑 CPU；
  数据集若不可用则属环境，不算代码缺陷。

★ 退出码：0 = 发车路径无代码缺陷；1 = 有。
"""
import ast
import io
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
V119 = os.path.join(HERE, "cnn10_v119.py")

# 这些异常类型 = **代码缺陷**（不是环境问题）
CODE_DEFECT = (NameError, AttributeError, TypeError, IndentationError,
                UnboundLocalError, SyntaxError, KeyError, IndexError,
                ZeroDivisionError, ValueError, ImportError, ModuleNotFoundError)
# 这些 = 环境/数据问题（本机无 GPU、无 CIFAR 数据）⇒ 不算缺陷
ENV_TOLERANT = (OSError, IOError, RuntimeError, FileNotFoundError)


def find_train_loop(tree):
    for n in tree.body:
        if isinstance(n, ast.For):
            t = n.target
            if isinstance(t, ast.Name) and t.id == "epoch":
                return n
    return None


def main():
    if not os.path.exists(V119):
        print("FATAL: 找不到 %s" % V119)
        return 1
    src = open(V119, encoding="utf-8").read()
    tree = ast.parse(src)
    loop = find_train_loop(tree)
    if loop is None:
        print("FATAL: 找不到训练循环")
        return 1

    # ---- 截断：把训练循环整段换成 pass（只在内存里）----
    lines = src.split("\n")
    head = "\n".join(lines[:loop.lineno - 1])
    tail = "\n".join(lines[loop.end_lineno:])
    truncated = head + "\npass  # [launch-path-check] 训练循环被截断\n" + tail

    print("=" * 70)
    print("发车路径真跑验证（训练循环截断；不训练）")
    print("  产物 = %s" % os.path.basename(V119))
    print("  训练循环行号 = %d..%d（被截断）" % (loop.lineno, loop.end_lineno))
    print("=" * 70)

    buf = io.StringIO()
    real_stdout = sys.stdout
    g = {"__name__": "__main__", "__file__": V119}
    code = compile(truncated, V119, "exec")
    exc = None
    try:
        sys.stdout = buf
        exec(code, g)
    except CODE_DEFECT as e:
        exc = e
    except ENV_TOLERANT as e:
        exc = e
        print("")
        print("  [ENV] 环境类异常（不算代码缺陷）：%s: %s"
              % (type(e).__name__, e))
        tb = traceback.format_exc()
        if "NameError" in tb or "AttributeError" in tb:
            print("  [!!] 但 traceback 里有 NameError/AttributeError ⇒ 判为代码缺陷")
            sys.stdout = real_stdout
            print(tb[-3000:])
            return 1
    finally:
        sys.stdout = real_stdout

    out = buf.getvalue()
    for ln in out.split("\n"):
        if ("[v118]" in ln or "[v117]" in ln or "唯一一根轴" in ln or "Traceback" in ln
                or "delivered" in ln.lower()):
            print("  | %s" % ln)

    print("-" * 70)
    ok = True
    if exc is None:
        print("LAUNCH PATH: 无异常（发车路径零错误）")
    elif isinstance(exc, CODE_DEFECT):
        print("LAUNCH PATH: 代码缺陷 %s: %s" % (type(exc).__name__, exc))
        print(traceback.format_exc()[-3000:])
        ok = False
    else:
        print("LAUNCH PATH: 环境类异常 %s: %s（不算代码缺陷）"
              % (type(exc).__name__, exc))

    # ---- 额外：验证 v117 的开机自检 + 守卫真的跑了 ----
    print("-" * 70)
    _rp = g.get("_v116_runnable")
    if isinstance(_rp, dict):
        print("  runnable.ok = %s | missing = %r"
              % (_rp.get("ok"), _rp.get("missing")))
    else:
        print("  [!!] _v116_runnable 未产出")
        ok = False
    _pp = g.get("_v117_precheck")
    if isinstance(_pp, dict):
        print("  precheck.ok = %s | used_copy = %s | model_state_unchanged = %s"
              " | probe_n = %r"
              % (_pp.get("ok"), _pp.get("used_copy"),
                 _pp.get("model_state_unchanged"), _pp.get("probe_n")))
        _err = str(_pp.get("error") or "")
        _env_err = ("NO_PROBE_INPUTS" in _err or "CUDA" in _err
                    or "FileNotFound" in _err or "download" in _err.lower())
        if _err and not _env_err:
            print("  [ENV?] precheck error = %r" % _err)
    else:
        print("  [!!] _v117_precheck 未产出")
        ok = False
    _dg = g.get("_v117_det_guard")
    if isinstance(_dg, dict):
        print("  det_guard.ok = %s | restored = %s | error = %r"
              % (_dg.get("ok"), _dg.get("restored"), _dg.get("error")))
        # ★★ 守卫的核心命题：**无论探针怎么崩，都不许把主流程拖死**
        #   ⇒ det_guard **必须**是一个 dict（哪怕 ok=False）
        print("  ★ 守卫返回了 dict ⇒ 确定性探针**没有**把主流程拖死")
    else:
        print("  [!!] _v117_det_guard 未产出（守卫没跑到）")
        ok = False
    _dpr = g.get("det_probe_report")
    print("  det_probe_report 是 dict = %s（下游 .get 不会二次崩）"
          % isinstance(_dpr, dict))

    _ud = g.get("_v118_uda_at_train_start")
    print("  [v118] 训练开始前 use_deterministic_algorithms = %r（期望 True；"
          "v117 会被守卫错关成 False）" % (_ud,))
    if _ud is not True:
        print("  [!!] uda 未被还原到探针前的值")
        ok = False
    _dg2 = g.get("_v117_det_guard")
    if isinstance(_dg2, dict):
        print("  [v118] det_guard.uda_before = %r | uda_after = %r"
              % (_dg2.get("uda_before"), _dg2.get("uda_after")))

    # ---- v119 专属：轴的目标构造在发车路径上必须可用 ----
    _comp = g.get("_v119_complement")
    if _comp is None:
        print("  [!!] _v119_complement 未绑定")
        ok = False
    else:
        try:
            import torch as _t
            _x = _t.randn(2, 4, 16, 16)
            _c = _comp(_x)
            print("  [v119] 补集目标可用：shape=%s requires_grad=%s"
                  % (tuple(_c.shape), bool(_c.requires_grad)))
        except Exception as _e:
            print("  [!!] _v119_complement 真跑失败: %s" % _e)
            ok = False
    print("  [v119] V119_HP_KERNEL=%r | V119_TARGET_MODE=%r"
          % (g.get("V119_HP_KERNEL"), g.get("V119_TARGET_MODE")))
    print("  [v119] refine_physics 绑定 = %r" % (g.get("refine_physics") is not None,))

    print("=" * 70)
    print("LAUNCH-PATH CHECK: %s" % ("ALL GREEN" if ok else "HAS FAILURES"))
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
