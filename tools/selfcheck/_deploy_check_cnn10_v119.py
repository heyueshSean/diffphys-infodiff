# -*- coding: utf-8 -*-
"""_deploy_check_cnn10_v119.py —— cnn10_v119.py 的**部署级**自证（静态 + 负向对照）

面板：
  A 指纹 / 硬纪律（LF / 零 argparse / SystemExit 不增 / py_compile）
  B 父本 v118 逐字性（added 恰 5 / removed 0 / changed 恰 {refine_physics}
    + refine_physics 的 diff **恰好**那根轴 + 全文件删除行白名单）
  C 轴闸门（**每条都对父本 v118 原文做负向对照** ⇒ 闸门必须有区分力）
  D 变异体（文本层）咬红
  E preflight 通用闸门 G1~G8（与父本比差值）
  F 版本同步（五处）
  G 顺序（目标定义在 refine 之前 / 探针在收尾 / 结果块在 results 里）
"""
import ast
import difflib
import hashlib
import os
import py_compile
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V119 = os.path.join(HERE, "cnn10_v119.py")
V118 = os.path.join(HERE, "cnn10_v118.py")
LINT = os.path.join(
    os.path.expanduser("~"), ".workbuddy", "skills",
    "long-run-preflight-audit", "scripts", "lint_train_script.py")

_n = [0]
_fail = [0]


def chk(name, ok, detail=""):
    _n[0] += 1
    if ok:
        print("  [PASS] %s" % name)
    else:
        _fail[0] += 1
        print("  [FAIL] %s   << %s" % (name, detail))


def read(path):
    return open(path, encoding="utf-8", newline="").read()


def src_of(text, name):
    tree = ast.parse(text)
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n.name == name:
                return ast.get_source_segment(text, n) or ""
    return ""


def top_defs(text):
    tree = ast.parse(text)
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            s = ast.get_source_segment(text, n)
            if s:
                out[n.name] = hashlib.md5(s.encode()).hexdigest()
    return out


def main():
    if not os.path.exists(V119) or not os.path.exists(V118):
        print("FATAL: 需要 cnn10_v118.py 与 cnn10_v119.py")
        return 1
    txt = read(V119)
    par = read(V118)
    raw = open(V119, "rb").read()
    sha = hashlib.sha1(raw).hexdigest()

    # ============================================================ A
    print("== A 指纹与硬纪律 ==")
    print("  sha1  = %s" % sha)
    print("  bytes = %d | lines = %d" % (len(raw), txt.count("\n")))
    try:
        py_compile.compile(V119, doraise=True)
        chk("A1 py_compile 通过", True)
    except Exception as e:
        chk("A1 py_compile 通过", False, "%s: %s" % (type(e).__name__, e))
    chk("A2 纯 LF（B 线约定）", raw.count(b"\r") == 0, "CR=%d" % raw.count(b"\r"))
    chk("A3 零 argparse", txt.count("add_argument") == 0)
    chk("A4 raise SystemExit 不新增（父本 %d）" % par.count("raise SystemExit"),
        txt.count("raise SystemExit") == par.count("raise SystemExit"))
    chk("A5 无 smoke / fake-data 死开关",
        ("smoke" not in txt.lower().replace("smoo_", ""))
        and "fake-data" not in txt)

    # ============================================================ B
    print("== B 父本 v118 逐字性 ==")
    a, b = top_defs(par), top_defs(txt)
    changed = sorted([k for k in a if k in b and a[k] != b[k]])
    deleted = sorted([k for k in a if k not in b])
    added = sorted([k for k in b if k not in a])
    chk("B1 父本只有 `refine_physics` 被改（**恰好那根轴**）",
        set(changed) == {"refine_physics"}, "changed=%r" % changed)
    chk("B2 父本顶层**零删除**", len(deleted) == 0, "deleted=%r" % deleted)
    chk("B3 新增**恰好 5 个** `_v119_*`",
        set(added) == {"_v119_align_probe", "_v119_complement",
                       "_v119_refine_verdict", "_v119_std_pair",
                       "_v119_verify_runnable"}, "added=%r" % added)

    _p = src_of(par, "refine_physics").replace("\r\n", "\n")
    _c = src_of(txt, "refine_physics").replace("\r\n", "\n")
    _rem, _add = [], []
    for ln in difflib.unified_diff(_p.split("\n"), _c.split("\n"), n=0,
                                   lineterm=""):
        if ln[:3] in ("---", "+++", "@@ "):
            continue
        if ln.startswith("-"):
            _rem.append(ln[1:])
        elif ln.startswith("+"):
            _add.append(ln[1:])
    _exp_rem = ["            loss_align = structure_alignment_loss(ph2, feat2) + \\",
                "                         structure_alignment_loss(ph3, feat3)"]
    chk("B4 refine_physics 被删的行**恰好**是那两行（不多不少）",
        _rem == _exp_rem, "removed=%r" % _rem)
    chk("B5 refine_physics 新增 8 行（4 注释 + 2 目标构造 + 2 调用）",
        len([x for x in _add if x.strip()]) == 8, "added=%r" % _add)
    chk("B6 新增的两行目标构造**就是** `_v119_complement`",
        any("_v119_complement(feat2)" in x for x in _add)
        and any("_v119_complement(feat3)" in x for x in _add))

    _sm = difflib.SequenceMatcher(None, par.replace("\r\n", "\n").split("\n"),
                                  txt.split("\n"), autojunk=False)
    _rem_all = []
    for tag, i1, i2, j1, j2 in _sm.get_opcodes():
        if tag in ("replace", "delete"):
            for x in par.replace("\r\n", "\n").split("\n")[i1:i2]:
                if x.strip():
                    _rem_all.append(x)
    _allowed = set(_exp_rem + [
        'VERSION         = "v118"   # [v110] 五处同步的唯一真相源：CKPT_DIR / OUT_DIR / JSON 名 / results / banner'])
    _bad = [x for x in _rem_all if x not in _allowed]
    chk("B7 全文件范围内父本被删/被改的行**只有** VERSION + 那两行",
        not _bad, "%r" % _bad[:6])

    _frozen = ["_v110_forward_and_loss", "_v110_sam_second_pass", "train",
               "test_tta", "physics_reg", "get_w_phys", "get_mixup_alpha",
               "structure_alignment_loss", "MaternDiffusion", "Gate",
               "_v113_roughness", "_v113_band_frac", "_v113_pinface_verdict",
               "_v95_train_loop"]
    _frozen = [k for k in _frozen if k in a and k in b]
    chk("B8 训练数学/判据关键函数 md5 **逐字未改**（%d 个）" % len(_frozen),
        all(a[k] == b[k] for k in _frozen),
        "%r" % [k for k in _frozen if a[k] != b[k]])

    # ============================================================ C 轴闸门
    print("== C 轴闸门（每条都对父本原文做**负向对照**）==")
    GATES = [
        ("C1 refine 的对齐目标来自 `_v119_complement`",
         lambda t: ("_v119_tg2 = _v119_complement(feat2)" in t
                    and "_v119_tg3 = _v119_complement(feat3)" in t)),
        ("C2 `structure_alignment_loss` 本体**未改**（父本逐字）",
         lambda t: src_of(t, "structure_alignment_loss")
         == src_of(par, "structure_alignment_loss")),
        ("C3 目标全程 no_grad + detach（不接受梯度）",
         lambda t: ("with torch.no_grad():" in src_of(t, "_v119_complement")
                    and "return (_f - _low).detach()" in src_of(t, "_v119_complement"))),
        ("C4 低通核是**参数为零**的算子（avg_pool2d，不新增可学参数）",
         lambda t: "F.avg_pool2d(_f, _k, stride=1, padding=_k // 2" in t),
        ("C5 V119_* 常量齐全（HP_KERNEL / TARGET_MODE / MIN_HP_FRAC / MAX_CORR）",
         lambda t: all(("V119_" + x) in t for x in
                       ("HP_KERNEL", "TARGET_MODE", "MIN_HP_FRAC", "MAX_CORR"))),
        ("C6 探针经 `_safe_readout`（不裸调，崩了不杀主流程）",
         lambda t: "_v119_align = _safe_readout(" in t),
        ("C7 结果键 `v119_refineface` 落盘（含 verdict + 双口径 + band）",
         lambda t: ('"v119_refineface": {' in t
                    and '"verdict": _v119_verdict,' in t
                    and '"band_frac_mean": (float(_v118_band)' in t)),
        ("C8 判词**内容优先**（band 闸门**单独**先行，不许并上 regime）",
         lambda t: ('        if float(band_frac) >= float(V113_BAND_FRAC_MIN):\n'
                    '            return "REFINE_HP_DELIVERED"\n'
                    in src_of(t, "_v119_refine_verdict"))),
        ("C9 可运行性自检覆盖 v119 的新函数与新常量",
         lambda t: all(x in src_of(t, "_v119_verify_runnable") for x in
                       ("_v119_complement", "_v119_align_probe",
                        "V119_HP_KERNEL", "V119_MIN_HP_FRAC"))),
        ("C10 判据数值沿用既有常量（一个数不改）",
         lambda t: all(x in src_of(t, "_v119_refine_verdict") for x in
                       ("V113_BAND_FRAC_MIN",))),
    ]
    # ★ C2 是**恒等项**（判的是"父子同文"，父本必然通过）⇒ 不做负向对照
    _NO_NEG = ("C2 `structure_alignment_loss` 本体**未改**（父本逐字）",)
    for _nm, _fn in GATES:
        try:
            ok_v, msg = _fn(txt), ""
        except Exception as e:
            ok_v, msg = False, "%s: %s" % (type(e).__name__, e)
        chk("C " + _nm, ok_v, msg)
        if _nm in _NO_NEG:
            print("    | %s ⇒ 恒等项，免负向对照" % _nm)
            continue
        try:
            neg = not _fn(par)
        except Exception:
            neg = True
        chk("C " + _nm + " [负向对照：父本 v118 必须**不过**]", neg,
            "父本也通过 ⇒ 闸门无区分力")

    # ============================================================ D 变异体
    print("== D 变异体（文本层）咬红 ==")
    muts = [
        ("M1 目标回退成 feat（父本语义）",
         txt.replace("            _v119_tg2 = _v119_complement(feat2)\n"
                     "            _v119_tg3 = _v119_complement(feat3)\n",
                     "            _v119_tg2 = feat2\n"
                     "            _v119_tg3 = feat3\n")),
        ("M2 用高通用替代补集（目标变回 feat 的常数倍）",
         txt.replace("            return (_f - _low).detach()",
                     "            return (_f - 0.0 * _low).detach()")),
        ("M3 判词把内容闸门降级（regime 优先）",
         txt.replace('        if float(band_frac) >= float(V113_BAND_FRAC_MIN):\n'
                     '            return "REFINE_HP_DELIVERED"\n',
                     '        if (float(band_frac) >= float(V113_BAND_FRAC_MIN)\n'
                     '                and str(align.get("regime")) == "HP_ALIGNED"):\n'
                     '            return "REFINE_HP_DELIVERED"\n')),
        ("M4 探针不经 `_safe_readout`（裸调 ⇒ 崩了会杀主流程）",
         txt.replace("_v119_align = _safe_readout(\n"
                     "    lambda: _v119_align_probe(model, _v115_probe_inputs, TRAIN_EPOCHS),\n"
                     '    {"ok": False, "error": "PROBE_RAISED"})',
                     "_v119_align = _v119_align_probe(model, _v115_probe_inputs, TRAIN_EPOCHS)")),
    ]
    for _tag, _msrc in muts:
        chk("D %s ⇒ 源确实变了" % _tag, _msrc != txt)
        try:
            ast.parse(_msrc)
            chk("D %s 仍是合法 Python（**语义**变异，不是语法破坏）" % _tag, True)
        except SyntaxError as e:
            chk("D %s 可解析" % _tag, False, str(e))
        # 必须至少有一条 C 闸门因此翻红 ⇒ 变异真的被测到
        _bites = 0
        for _nm, _fn in GATES:
            try:
                if _fn(txt) and not _fn(_msrc):
                    _bites += 1
            except Exception:
                _bites += 1
        chk("D %s 至少咬红 1 条 C 闸门（实际 %d 条）" % (_tag, _bites), _bites >= 1)

    # ============================================================ E preflight
    print("== E preflight 通用闸门 G1~G8 ==")
    if not os.path.exists(LINT):
        chk("E0 lint_train_script.py 存在", False, LINT)
        g4_child = g4_parent = 0
    else:
        def _run(path):
            r = subprocess.run([sys.executable, LINT, path],
                               capture_output=True, text=True, timeout=900)
            o = (r.stdout or "") + (r.stderr or "")
            fl = [ln for ln in o.split("\n")
                  if any(("[G%d FAIL]" % i) in ln for i in range(1, 9))]
            return fl
        try:
            fc, fp = _run(V119), _run(V118)
            nc = len([x for x in fc if "[G4 FAIL]" not in x])
            np_ = len([x for x in fp if "[G4 FAIL]" not in x])
            g4_child = len([x for x in fc if "[G4 FAIL]" in x])
            g4_parent = len([x for x in fp if "[G4 FAIL]" in x])
            chk("E1 G1/G2/G3/G5/G6/G7/G8 全 PASS（非 G4）", nc == 0,
                "%r" % [x for x in fc if "[G4 FAIL]" not in x][:3])
            chk("E2 G4 继承债**不新增**（子 %d <= 父 %d）" % (g4_child, g4_parent),
                g4_child <= g4_parent,
                "child=%d parent=%d" % (g4_child, g4_parent))
        except Exception as e:
            chk("E0 lint 运行", False, "%s: %s" % (type(e).__name__, e))
            g4_child = g4_parent = 0

    # ============================================================ F 版本同步
    print("== F 版本同步（五处）==")
    chk("F1 VERSION = v119", 'VERSION         = "v119"' in txt)
    chk("F2 OUT_DIR 由 VERSION 派生",
        "OUT_DIR = f'./out_{VERSION}_s{SEED}'" in txt)
    chk("F3 CKPT_DIR 由 VERSION 派生",
        "CKPT_DIR = f'./ckpts_{VERSION}_s{SEED}'" in txt)
    chk("F4 结果键含 v119", '"v119_refineface"' in txt)
    chk("F5 banner 含 v119 轴说明（恰好 1 条）",
        txt.count("[v119] ★ 唯一一根轴") == 1)
    chk("F6 无残留旧版本号派生标识（ckpts_v118 / out_v118 / results_cnn_v118）",
        all(x not in txt for x in ("ckpts_v118", "out_v118", "results_cnn_v118")))

    # ============================================================ G 顺序
    print("== G 顺序（定义在前 / 调用在后 / 读数在收尾）==")
    i_comp = txt.find("def _v119_complement(")
    i_refine = txt.find("def refine_physics(")
    i_probe = txt.find("def _v119_align_probe(")
    i_verdict = txt.find("def _v119_refine_verdict(")
    i_call = txt.find("_v119_align = _safe_readout(")
    i_res = txt.find('"v119_refineface": {')
    chk("G1 目标构造定义 < refine_physics", i_comp < i_refine)
    chk("G2 探针/判词的**定义**都早于**调用点**（定义之间的先后不影响可用性）",
        i_probe < i_call and i_verdict < i_call,
        "probe=%d verdict=%d call=%d" % (i_probe, i_verdict, i_call))
    chk("G3 探针调用点在**收尾**（结果块之前）", i_call < i_res)
    chk("G4 探针在 BN 还原之前（与父本 v115 同序）",
        i_call < txt.find("_v115_bn_restore_ok"))

    print("=" * 66)
    print("DEPLOY CHECK: %d/%d PASS%s"
          % (_n[0] - _fail[0], _n[0],
             "" if _fail[0] == 0 else "  <<< %d FAIL" % _fail[0]))
    return 0 if _fail[0] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
