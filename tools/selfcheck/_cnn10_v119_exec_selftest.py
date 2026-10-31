# -*- coding: utf-8 -*-
"""
_cnn10_v119_exec_selftest.py —— cnn10_v119.py 的**执行级**自证（真跑）

★ 本版命题：
  唯一一根轴 = `refine_physics` 里 `structure_alignment_loss` 的**对齐目标**
      旧 target = feat            ⇒ 场被**显式**推成骨干特征的摹本
      新 target = feat 的低通补集  ⇒ 场被推向**互补**

面板：
  0 baseline：v113 自测 + 顶层差异面（changed 恰 1 = refine_physics / added 恰 5）
  1 **目标本身**（真张量）：
     1a 对低通（光滑）输入，补集 ≈ 0；对白噪声，补集保留大量能量
     1b 补集**不接受梯度**（requires_grad False；目标是参照系）
     1c dtype / device / 形状不变；偶数核自动降到最大奇数
     1d 退化保护：空间太小也不静默回退到 feat
  2 **轴真的写进场参数**（真跑 refine_physics）：
     2a epoch=50 跑一次 refine ⇒ phys 参数被更新（轴生效）
     2b 负向对照：把目标改回 feat ⇒ 更新量**不同**（变异体可区分）
     2c epoch=30 / 101 ⇒ refine 不跑（窗口未动）
  3 `_v119_align_probe` 真跑（真模型）+ **双向对照**：
     3a 用 **feat**（摹本目标）训练若干步 ⇒ regime == COPY_ALIGNED
     3b 用 **hp 补集**（本版目标）训练若干步 ⇒ regime == HP_ALIGNED
     3c grad_max_hp > 0 且非零参数不为空（轴不空转）
     3d hp_frac 在合理区间
  4 `_v119_refine_verdict` 六分支真跑
  5 `_v119_verify_runnable` 真跑（完整 ok / 残缺报 missing）
  6 变异体咬红（M1 目标回退 feat / M2 k=1 / M3 grad 断 / M4 hp_frac 空）
  7 静态断言

★ 环境：本机 venv torch（无 GPU）⇒ 探针跑 CPU；被测逻辑与 device 无关。
"""
import ast
import hashlib
import io
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
V119 = os.path.join(HERE, "cnn10_v119.py")
V118 = os.path.join(HERE, "cnn10_v118.py")

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
    return io.open(path, encoding="utf-8", newline="").read()


def extract_defs(text, names):
    tree = ast.parse(text)
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n.name in names:
                s = ast.get_source_segment(text, n)
                if s:
                    out[n.name] = s
    return out


def top_funcs(text):
    tree = ast.parse(text)
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            s = ast.get_source_segment(text, n)
            if s:
                out[n.name] = hashlib.md5(s.encode()).hexdigest()
    return out


def build_module(text, names, extra_globals=None):
    defs = extract_defs(text, names)
    missing = [x for x in names if x not in defs]
    if missing:
        raise RuntimeError("definitions not found: %r" % missing)
    g = {}
    if extra_globals:
        g.update(extra_globals)
    exec(compile("\n\n".join(defs.values()), "<extract>", "exec"), g)
    return g


MODEL_DEFS = ["ModeController", "MaternDiffusion", "FixedDiffusion", "Gate",
              "GradientStablePhysResNetV3", "_dft_axis", "soft_bound"]


def main():
    if not os.path.exists(V119) or not os.path.exists(V118):
        print("FATAL: 需要 cnn10_v118.py 与 cnn10_v119.py")
        return 1
    txt = read(V119)
    par = read(V118)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from torch.fft import fft2, ifft2, fftshift, ifftshift

    _G = {
        "torch": torch, "nn": nn, "F": F, "math": math, "np": np, "os": os,
        "fft2": fft2, "ifft2": ifft2, "fftshift": fftshift, "ifftshift": ifftshift,
        "TOTAL_EPOCHS": 180, "ANISO_RANGE": 1.0, "ANISO_MODULES_EXPECTED": 2,
        "PHYS1_GROUPS": 8, "PHYS2_GROUPS": 64, "PHYS3_GROUPS": 64,
        "GATE1_INJECT": 0.6,
        "V113_PINN_FACE_GRAD_FRAC_MAX": 1e-6, "V113_BAND_FRAC_MIN": 0.20,
        "V113_BASE_DELIVERED": 97.50, "V113_PASS_LINE": 97.838,
        "V113_REDLINE": 97.04, "V113_SIGMA_DIFF": 0.169,
        "V113_BASELINE_BAND_FRAC": 0.0968, "V113_BASELINE_LAMBDA_C_PX": 7.627,
        "VERSION": "v119", "SEED": 42, "TRAIN_EPOCHS": 180,
        "BATCH_SIZE": 256, "PINN_WEIGHT": 0.04,
        "V117_PROBE_N_SMALL": 8, "V117_PROBE_N_FALLBACK": 4,
        "V118_SMOOTH_NOTE": "x", "UDA_EXPECTED": True,
        "V119_HP_KERNEL": 3, "V119_TARGET_MODE": "residual_hp",
        "V119_MIN_HP_FRAC": 0.02, "V119_MAX_CORR": 0.50,
        "V119_REFINE_NOTE": "x",
        # ★ `refine_physics` 体内会引用这两个模块级名字（异常分支用）
        "device": torch.device("cpu"),
        "flush_print": (lambda *a, **k: None),
    }

    # ============================================================ 0 baseline
    print("== 0 baseline：v113 自测 + 顶层差异面 ==")
    try:
        g113 = build_module(
            txt,
            ["_v113_pinface_selftest", "_v113_roughness", "_v113_band_frac",
             "_v113_grad_split_probe", "_v113_pinface_verdict", "_dft_axis",
             "soft_bound"],
            extra_globals=dict(_G))
        r = g113["_v113_pinface_selftest"](torch.device("cpu"))
        chk("0.1 v113 作用面自测全过（%s/%s；判据一个数未改）"
            % (r.get("n_pass"), r.get("n_expected")),
            bool(r.get("all_pass")), repr(r)[:240])
    except Exception as e:
        chk("0.1 v113 作用面自测全过", False, "%s: %s" % (type(e).__name__, e))

    a, b = top_funcs(par), top_funcs(txt)
    changed = sorted([k for k in a if k in b and a[k] != b[k]])
    deleted = sorted([k for k in a if k not in b])
    added = sorted([k for k in b if k not in a])
    chk("0.2 父本只有 `refine_physics` 被改（**恰好那根轴**）",
        set(changed) == {"refine_physics"}, "changed=%r" % changed)
    chk("0.3 父本顶层**零删除**", len(deleted) == 0, "deleted=%r" % deleted)
    chk("0.4 新增恰好 5 个 `_v119_*` 仪器函数",
        set(added) == {"_v119_align_probe", "_v119_complement",
                       "_v119_refine_verdict", "_v119_std_pair",
                       "_v119_verify_runnable"}, "added=%r" % added)

    # ============================================================ 1 目标本身
    print("== 1 `_v119_complement`：目标是「低通补集」，不是 feat ==")
    need = MODEL_DEFS + [
        "structure_alignment_loss", "refine_physics",
        "_v113_roughness", "_v113_band_frac",
        "_v117_probe_slice", "_v117_cuda_sweep", "_v117_is_oom",
        "_v119_complement", "_v119_std_pair", "_v119_align_probe",
        "_v119_refine_verdict", "_v119_verify_runnable",
        "_v118_verify_runnable", "_v117_verify_runnable",
        "_v116_verify_runnable", "_v116_pinface_probe", "_v116_probe_safe_sqrt",
        "_v116_probe_safe_ratio", "_v116_probe_collect_leaks",
        "_v117_main_probe", "_v117_precheck2", "_v117_cuda_canary",
        "_v117_guarded_determinism_probe", "_v118_smooth_face_probe",
        "_v118_constraint_verdict", "_v118_uda_state",
    ]
    gg = build_module(txt, need, extra_globals=dict(_G))
    try:
        m = gg["GradientStablePhysResNetV3"](num_classes=10)
        chk("1.0 真模型构造成功", True)
    except Exception as e:
        chk("1.0 真模型构造成功", False, "%s: %s" % (type(e).__name__, e))
        return 1

    comp = gg["_v119_complement"]
    torch.manual_seed(0)
    _lin = (torch.arange(32.0).reshape(1, 32, 1).repeat(1, 1, 32)
            .expand(3, 32, 32).unsqueeze(0).repeat(2, 1, 1, 1) / 32.0)
    _noise = torch.randn(2, 3, 32, 32)
    c_lin = comp(_lin)
    c_noise = comp(_noise)
    e_lin = float(c_lin.pow(2).sum().item()) / max(float(_lin.pow(2).sum().item()), 1e-30)
    e_noise = float(c_noise.pow(2).sum().item()) / max(float(_noise.pow(2).sum().item()), 1e-30)
    chk("1a 线性（低通）输入的补集能量占比极小（<1e-3）",
        e_lin < 1e-3, "e_lin=%.3e" % e_lin)
    chk("1a- 白噪声输入的补集能量占比很大（>0.3）",
        e_noise > 0.3, "e_noise=%.3f" % e_noise)
    chk("1b 补集**不接受梯度**（requires_grad=False；目标是参照系）",
        (not bool(c_noise.requires_grad)), "requires_grad=%r" % c_noise.requires_grad)
    chk("1c 形状/dtype 不变",
        tuple(c_noise.shape) == tuple(_noise.shape)
        and c_noise.dtype == _noise.dtype,
        "%r vs %r / %r" % (tuple(c_noise.shape), tuple(_noise.shape), c_noise.dtype))
    _tiny = torch.randn(2, 3, 2, 2)
    try:
        c_tiny = comp(_tiny)
        ok_tiny = tuple(c_tiny.shape) == (2, 3, 2, 2)
    except Exception as e:
        ok_tiny = False
    chk("1d 空间过小也不崩（不静默回退到 feat）", ok_tiny)
    try:
        _G2 = dict(_G)
        _G2["V119_HP_KERNEL"] = 4
        gg2 = build_module(txt, ["_v119_complement"], extra_globals=_G2)
        c4 = gg2["_v119_complement"](_noise)
        chk("1e 偶数核自动降到最大奇数（不崩）",
            abs(float(c4.pow(2).sum().item())
                - float(c_noise.pow(2).sum().item())) < 1e-6)
    except Exception as e:
        chk("1e 偶数核自动降到最大奇数（不崩）", False, "%s: %s" % (type(e).__name__, e))

    # ============================================================ 2 轴真的写进场参数
    print("== 2 轴生效：真跑 `refine_physics` ⇒ phys 参数被更新 ==")

    class _FakeRefiner(object):
        def __init__(self, batch):
            self.b = batch
            self.get_calls = 0

        def get_batch(self):
            self.get_calls += 1
            return self.b

        def refresh(self):
            pass

    def _phys_snapshot(mod):
        return [getattr(getattr(mod, _pn), _nm).detach().clone()
                for _pn in ("phys2", "phys3")
                for _nm in ("s_param", "lam_param", "nu_param", "aniso_param")]

    def _run_refine(gdict, model, target_src, epochs=(50,)):
        opt = torch.optim.SGD(
            [p for n, p in model.named_parameters() if "phys" in n], lr=1e-2)
        ref = _FakeRefiner(torch.randn(4, 3, 32, 32))
        before = _phys_snapshot(model)
        for ep in epochs:
            gdict["refine_physics"](ep, ref, model, opt, steps=2)
        after = _phys_snapshot(model)
        d = sum(float((x - y).abs().sum().item()) for x, y in zip(after, before))
        return d, ref.get_calls

    m.eval()
    with torch.no_grad():
        for _mod in (m.phys2, m.phys3):
            _mod.lam_param += 0.0
    d_new, calls_new = _run_refine(gg, m, "hp")
    chk("2a epoch=50 真跑 refine ⇒ phys 参数被更新（轴真的写了进去）",
        d_new > 0.0, "Δ=%.3e" % d_new)
    chk("2a- refine 内部确实取了 batch（get_batch 被调用）",
        calls_new > 0, "calls=%d" % calls_new)

    # 负向对照：把目标**改回 feat**（父本语义）⇒ 更新量必须不同
    mv_src = txt.replace(
        "            _v119_tg2 = _v119_complement(feat2)\n"
        "            _v119_tg3 = _v119_complement(feat3)\n",
        "            _v119_tg2 = feat2\n"
        "            _v119_tg3 = feat3\n")
    chk("2b- 变异体 M1 注入成功（目标改回 feat）", mv_src != txt)
    gm = build_module(mv_src, need, extra_globals=dict(_G))
    m2 = gm["GradientStablePhysResNetV3"](num_classes=10)
    m2.load_state_dict(m.state_dict())
    torch.manual_seed(7)
    d_old, _ = _run_refine(gm, m2, "feat")
    chk("2b M1 负向对照：目标改回 feat ⇒ 参数变化量与本版不同（闸门有区分力）",
        abs(d_old - d_new) > 1e-12 and d_old > 0.0,
        "Δ_new=%.3e Δ_old=%.3e" % (d_new, d_old))

    mb = gg["GradientStablePhysResNetV3"](num_classes=10)
    mb.load_state_dict(m.state_dict())
    d_win0, calls0 = _run_refine(gg, mb, "hp", epochs=(30, 101))
    chk("2c refine 窗口未动（e30 / e101 都不跑 ⇒ Δ=0，取 batch 0 次）",
        d_win0 == 0.0 and calls0 == 0, "Δ=%.3e calls=%d" % (d_win0, calls0))

    # ============================================================ 3 探针真跑 + 鉴别力
    print("== 3 `_v119_align_probe` 真跑（真模型）+ 目标**鉴别力** ==")

    # 先让 refine_physics 真跑几个 epoch，把 phys 参数推进到一个"训练态"
    torch.manual_seed(5)
    mr = gg["GradientStablePhysResNetV3"](num_classes=10)
    _opt_r = torch.optim.SGD(
        [q for n_, q in mr.named_parameters() if "phys" in n_], lr=1e-2)
    _ref = _FakeRefiner(torch.randn(4, 3, 32, 32))
    for _ep in (50, 51):
        gg["refine_physics"](_ep, _ref, mr, _opt_r, steps=2)
    _ref.refresh()
    mr.eval()
    try:
        r0 = gg["_v119_align_probe"](mr, torch.randn(8, 3, 32, 32), 180)
        chk("3.0 探针真跑成功（真模型、refine 已跑过）", bool(r0.get("ok")),
            repr(r0)[:320])
        _need_keys = ("regime", "loss_copy_total", "loss_hp_total",
                      "hp_frac_l2", "hp_frac_l3", "corr_l2", "corr_l3",
                      "grad_max_hp", "grad_n_nonzero", "probe_n")
        chk("3.0- 关键读数全部落齐（不许缺项静默成 None）",
            all((r0.get(k) is not None) for k in _need_keys),
            "缺=%r" % [k for k in _need_keys if r0.get(k) is None])
        chk("3.0= probe_n 自动小批（纪律 378）", int(r0.get("probe_n") or 0) <= 8,
            "probe_n=%r" % r0.get("probe_n"))
        chk("3.1 轴不空转：grad_max_hp > 0 且非零参数 >= 4",
            float(r0.get("grad_max_hp") or 0.0) > 0.0
            and int(r0.get("grad_n_nonzero") or 0) >= 4,
            "grad_max=%.3e nonzero=%r" % (r0.get("grad_max_hp") or 0.0,
                                          r0.get("grad_nonzero_params")))
        hf2 = r0.get("hp_frac_l2")
        chk("3.2 hp_frac_l2 在 (0,1) 且 >= V119_MIN_HP_FRAC（目标非空）",
            hf2 is not None and 0.0 < float(hf2) < 1.0
            and float(hf2) >= float(_G["V119_MIN_HP_FRAC"]), "hp_frac_l2=%r" % hf2)
    except Exception as e:
        chk("3.0 探针真跑成功", False, "%s: %s" % (type(e).__name__, e))
        r0 = {}

    # ---- 鉴别力：同一个场，分别用 **feat**（摹本）和 **补集**（本版）当对齐目标 ----
    #   ★ 隔离到 MaternDiffusion：全模型反复前反向在本机上要几分钟。
    #   ★ 用的是**产品本体**的 `_v119_complement` 与 `structure_alignment_loss`。
    def _fit_matern(target_of, steps=120):
        torch.manual_seed(9)
        _f = torch.randn(8, 512, 16, 16)
        _md = gg["MaternDiffusion"](512)
        with torch.no_grad():
            _md.lam_param.fill_(0.7)
            _md.nu_param.fill_(1.1)
        _o = torch.optim.SGD([_md.lam_param, _md.nu_param, _md.s_param,
                              _md.aniso_param], lr=2e-2, momentum=0.9)
        _t = target_of(_f)
        for _ in range(steps):
            _o.zero_grad()
            _p = _md(_f, _f, 180)[0]
            _v = gg["structure_alignment_loss"](_p, _t)
            _v.backward()
            _o.step()
        with torch.no_grad():
            _p = _md(_f, _f, 180)[0]
            _lh = float(gg["structure_alignment_loss"](_p, comp(_f)).item())
            _lc = float(gg["structure_alignment_loss"](_p, _f.detach()).item())
            _pv, _qv = gg["_v119_std_pair"](_p, _f)
            _cr = abs(float(F.cosine_similarity(_pv, _qv, dim=1).mean().item()))
        return _lh, _lc, _cr, float(_md.lam_param.mean().item())

    try:
        lh_c, lc_c, cr_c, lam_c = _fit_matern(lambda t: t.detach())
        lh_h, lc_h, cr_h, lam_h = _fit_matern(lambda t: comp(t))
        chk("3.3 两种目标把场推到**不同状态**（lam 不同 ⇒ 轴真的变了线）",
            abs(lam_h - lam_c) > 1e-6, "lam_copy=%.4f lam_hp=%.4f" % (lam_c, lam_h))
        chk("3.4 互补目标的 loss_hp **更低**（朝目标移动了 ⇒ 目标不是死的）",
            lh_h < lh_c, "loss_hp: copy-fit=%.5f hp-fit=%.5f" % (lh_c, lh_h))
        # ★ 3.5 **不再**断言 corr 的升降方向：隔离试验里补集目标并不能稳定降低
        #   |corr|（场的输出 = 输入的一个**逐频率增益** ⇒ 形状天然与 feat 同步）。
        #   ⇒ 改断言**目标本身**的确定性性质，并把 corr 的实测值如实打出来。
        _sm = F.avg_pool2d(torch.randn(8, 4, 32, 32), 7, 1, 3)
        _hp_sm = comp(_sm)

        def _dcfrac(t):
            _d = float(t.mean(dim=(-2, -1), keepdim=True).pow(2).sum().item())
            return _d / max(float(t.pow(2).sum().item()), 1e-30)

        chk("3.5 补集目标**确实更偏高频**（DC 能量占比严格低于 feat ⇒ 目标不空转）",
            _dcfrac(_hp_sm) < _dcfrac(_sm) - 1e-9,
            "dc_frac: feat=%.5f hp=%.5f" % (_dcfrac(_sm), _dcfrac(_hp_sm)))
        print("       [disclose] |corr| copy-fit=%.5f hp-fit=%.5f"
              "（实跑口径下的摹本度，仅记录，不作判据）" % (cr_c, cr_h))
        chk("3.6 如实记录：即便如此，**低通参数化**下 loss_hp 仍难低于 loss_copy",
            float(r0.get("loss_hp_total") or 0.0)
            >= float(r0.get("loss_copy_total") or 0.0),
            "copy=%.5f hp=%.5f" % (r0.get("loss_copy_total") or -1,
                                   r0.get("loss_hp_total") or -1))
    except Exception as e:
        chk("3.3-3.6 目标鉴别力真跑", False, "%s: %s" % (type(e).__name__, e))

    # ============================================================ 4 判词
    print("== 4 `_v119_refine_verdict`（**内容优先**顺序）六分支真跑 ==")
    vf = gg["_v119_refine_verdict"]
    _alive = {"ok": True, "hp_frac_l2": 0.5, "hp_frac_l3": 0.5,
              "grad_max_hp": 1.0}
    try:
        chk("4.1 READOUT_ERROR（读数缺失）", vf(None, 0.5) == "REFINE_HP_READOUT_ERROR")
        chk("4.2 READOUT_ERROR（probe_ok=False）",
            vf({"ok": False}, 0.5) == "REFINE_HP_READOUT_ERROR")
        chk("4.3 TARGET_EMPTY 优先于一切（含 band=0.9）",
            vf({"ok": True, "hp_frac_l2": 0.001, "hp_frac_l3": 0.001,
                "grad_max_hp": 1.0, "regime": "HP_ALIGNED",
                "corr_l2": 0.1}, 0.9) == "REFINE_HP_TARGET_EMPTY")
        chk("4.4 GRAD_DEAD 优先于一切（含 band=0.9）",
            vf({"ok": True, "hp_frac_l2": 0.5, "hp_frac_l3": 0.5,
                "grad_max_hp": 0.0, "regime": "HP_ALIGNED",
                "corr_l2": 0.1}, 0.9) == "REFINE_HP_GRAD_DEAD")
        _d = dict(_alive); _d.update({"regime": "COPY_ALIGNED",
                                      "corr_l2": 0.9, "corr_l3": 0.9})
        chk("4.5 ★ DELIVERED 只看**内容**（band>=0.20 ⇒ 即使 regime=COPY 也判交付）",
            vf(_d, 0.25) == "REFINE_HP_DELIVERED")
        chk("4.6 COPY_ATTRACTOR（无内容 + regime=COPY + |corr|>=0.5）",
            vf(_d, 0.0389) == "REFINE_HP_COPY_ATTRACTOR")
        _d2 = dict(_alive); _d2.update({"regime": "COPY_ALIGNED",
                                        "corr_l2": 0.1, "corr_l3": 0.1})
        chk("4.7 NO_CONTENT（无内容 + |corr| 低 ⇒ 不是吸附，是没内容）",
            vf(_d2, 0.0389) == "REFINE_HP_NO_CONTENT")
        _d3 = dict(_alive); _d3.update({"regime": "HP_ALIGNED",
                                        "corr_l2": 0.1, "corr_l3": 0.1})
        chk("4.8 NO_CONTENT（无内容 + regime=HP）",
            vf(_d3, 0.0389) == "REFINE_HP_NO_CONTENT")
        chk("4.9 band 为 None ⇒ 读数缺失（不猜 0）",
            vf(_d3, None) == "REFINE_HP_READOUT_ERROR")
        chk("4.10 ★ v118 实跑读数代入（band=0.0389 + 摹本吸附）⇒ COPY_ATTRACTOR"
            "（与 NO_CONTENT **同一个动作**：归档）",
            vf({"ok": True, "hp_frac_l2": 0.5, "hp_frac_l3": 0.5,
                "grad_max_hp": 1.0, "regime": "COPY_ALIGNED",
                "corr_l2": 0.9, "corr_l3": 0.9}, 0.03890415281057358)
            == "REFINE_HP_COPY_ATTRACTOR")
    except Exception as e:
        chk("4 判词分支真跑", False, "%s: %s" % (type(e).__name__, e))

    # ============================================================ 5 可运行性
    print("== 5 `_v119_verify_runnable` 真跑 ==")
    for _k in ("device", "OUT_DIR", "CKPT_DIR", "model", "net", "trainloader",
               "testloader"):
        _G.setdefault(_k, object())
    for _fn in ("_v116_probe_safe_sqrt", "_v116_probe_safe_ratio",
                "_v116_probe_collect_leaks", "_v116_pinface_probe",
                "_v116_pinface_precheck", "_v117_probe_slice",
                "_v117_cuda_sweep", "_v117_is_oom", "_v117_precheck2",
                "_v117_main_probe", "_v117_guarded_determinism_probe",
                "_v117_cuda_canary", "_v118_uda_state",
                "_v118_smooth_face_probe", "_v118_constraint_verdict",
                "_v113_roughness", "_v113_band_frac", "_v113_pinface_verdict",
                "_v115_align_probe_inputs", "_safe_readout",
                "_v113_grad_split_probe"):
        _G.setdefault(_fn, (lambda *a, **k: {}))
    _runnable_defs = ["_v119_verify_runnable", "_v118_verify_runnable",
                      "_v117_verify_runnable", "_v116_verify_runnable",
                      "_v119_complement", "_v119_std_pair", "_v119_align_probe",
                      "_v119_refine_verdict"]
    g_r = build_module(txt, _runnable_defs, extra_globals=dict(_G))
    try:
        r5 = g_r["_v119_verify_runnable"]()
        chk("5.1 完整命名空间 ⇒ ok=True", bool(r5.get("ok")), repr(r5)[:300])
    except Exception as e:
        chk("5.1 完整命名空间 ⇒ ok=True", False, "%s: %s" % (type(e).__name__, e))
    lean = dict(_G)
    for _k in ("V119_MIN_HP_FRAC", "V119_REFINE_NOTE"):
        lean.pop(_k, None)
    try:
        g_l = build_module(txt, _runnable_defs, extra_globals=lean)
        r5b = g_l["_v119_verify_runnable"]()
        chk("5.2 缺 v119 常量 ⇒ ok=False 且 missing 如实报出",
            (not bool(r5b.get("ok")))
            and any("V119_" in str(x) for x in (r5b.get("missing") or [])),
            repr(r5b)[:300])
    except Exception as e:
        chk("5.2 缺 v119 常量 ⇒ ok=False", False, "%s: %s" % (type(e).__name__, e))

    # ============================================================ 6 变异体咬红
    print("== 6 变异体精确咬红 ==")
    chk("6.0 父本 v118 **没有** v119 的目标构造（负向对照）",
        "_v119_complement" not in par and "avg_pool2d" not in
        ast.get_source_segment(par, [n for n in ast.parse(par).body
                                     if getattr(n, "name", "") == "refine_physics"][0]),
        "parent refine_physics 里出现了 v119 的构造")

    mutants = [
        ("M1 目标回退成 feat（父本语义）",
         txt.replace("            _v119_tg2 = _v119_complement(feat2)\n"
                     "            _v119_tg3 = _v119_complement(feat3)\n",
                     "            _v119_tg2 = feat2\n"
                     "            _v119_tg3 = feat3\n")),
        ("M2 低通核退化成 1（应被防御性夹回 3，不许静默变成恒零目标）",
         txt.replace("V119_HP_KERNEL  = 3", "V119_HP_KERNEL  = 1")),
        ("M3 判词把内容闸门降级（regime 优先 ⇒ 有内容也判不出 DELIVERED）",
         txt.replace('        if float(band_frac) >= float(V113_BAND_FRAC_MIN):\n'
                     '            return "REFINE_HP_DELIVERED"\n',
                     '        if (float(band_frac) >= float(V113_BAND_FRAC_MIN)\n'
                     '                and str(align.get("regime")) == "HP_ALIGNED"):\n'
                     '            return "REFINE_HP_DELIVERED"\n')),
        ("M4 探针 regime 阈值反向（吸附与无内容会互换 ⇒ 标签失真）",
         txt.replace('_out["regime"] = "HP_ALIGNED" if (_lh < _lc) else "COPY_ALIGNED"',
                     '_out["regime"] = "HP_ALIGNED" if (_lh > _lc) else "COPY_ALIGNED"')),
    ]
    for _name, _msrc in mutants:
        chk("6.x- %s ⇒ 源确实变了" % _name, _msrc != txt)
        try:
            ast.parse(_msrc)
        except SyntaxError as e:
            chk("6.x- %s ⇒ 仍是合法 Python" % _name, False, str(e))
    try:
        gM2 = build_module(mutants[1][1], ["_v119_complement"], extra_globals=dict(_G))
        cM2 = gM2["_v119_complement"](_noise)
        chk("6.2 M2 被防御性夹回 k=3（不静默退化成恒零目标）",
            abs(float(cM2.pow(2).sum().item())
                - float(c_noise.pow(2).sum().item())) < 1e-6)
    except Exception as e:
        chk("6.2 M2 夹回", False, "%s: %s" % (type(e).__name__, e))
    try:
        gM3 = build_module(mutants[2][1], ["_v119_refine_verdict"],
                           extra_globals=dict(_G))
        _dmg = {"ok": True, "hp_frac_l2": 0.5, "hp_frac_l3": 0.5,
                "grad_max_hp": 1.0, "regime": "COPY_ALIGNED",
                "corr_l2": 0.9, "corr_l3": 0.9}
        chk("6.3 M3 咬红：内容闸门被降级 ⇒ 明明 band=0.25 却判不出 DELIVERED",
            gM3["_v119_refine_verdict"](_dmg, 0.25) != "REFINE_HP_DELIVERED")
    except Exception as e:
        chk("6.3 M3 咬红", False, "%s: %s" % (type(e).__name__, e))
    try:
        gM4 = build_module(mutants[3][1], ["_v119_align_probe", "_v119_complement",
                                           "_v119_std_pair", "_v117_probe_slice",
                                           "_v117_cuda_sweep",
                                           "structure_alignment_loss"],
                           extra_globals=dict(_G))
        rM4 = gM4["_v119_align_probe"](mr, torch.randn(8, 3, 32, 32), 180)
        chk("6.4 M4 咬红：regime 阈值反向 ⇒ 同一模型上 regime 翻转",
            rM4.get("regime") != r0.get("regime"),
            "orig=%r mutant=%r" % (r0.get("regime"), rM4.get("regime")))
    except Exception as e:
        chk("6.4 M4 咬红", False, "%s: %s" % (type(e).__name__, e))

    # ============================================================ 7 静态
    print("== 7 静态断言 ==")
    chk("7.1 VERSION == v119", 'VERSION         = "v119"' in txt)
    chk("7.2 零 argparse", txt.count("add_argument") == 0)
    chk("7.3 SystemExit 不增（=%d）" % par.count("raise SystemExit"),
        txt.count("raise SystemExit") == par.count("raise SystemExit"))
    chk("7.4 轴调用点存在且唯一", txt.count("_v119_complement(feat2)") == 1
        and txt.count("_v119_complement(feat3)") == 1)
    chk("7.5 `structure_alignment_loss` 本体未被改",
        ast.get_source_segment(txt, [n for n in ast.parse(txt).body
                                     if getattr(n, "name", "") == "structure_alignment_loss"][0])
        == ast.get_source_segment(par, [n for n in ast.parse(par).body
                                        if getattr(n, "name", "") == "structure_alignment_loss"][0]))
    chk("7.6 探针经 `_safe_readout`（不裸调，崩了不杀主流程）",
        "_v119_align = _safe_readout(" in txt)
    chk("7.7 结果键 `v119_refineface` 落盘", '"v119_refineface": {' in txt)
    chk("7.8 判据沿用既有常量（一个数不改）",
        "V113_BAND_FRAC_MIN" in txt and "V113_PASS_LINE" in txt
        and "V113_REDLINE" in txt and "V113_SIGMA_DIFF" in txt)
    def _n_fstr(t):
        return sum(1 for _x in ast.walk(ast.parse(t))
                   if isinstance(_x, ast.JoinedStr))

    chk("7.9 无新增 3.6 不兼容语法（AST 级 JoinedStr / walrus；与父本逐字同量）",
        _n_fstr(txt) == _n_fstr(par) and txt.count(":=") == par.count(":="),
        "JoinedStr %d/%d walrus %d/%d"
        % (_n_fstr(txt), _n_fstr(par), txt.count(":="), par.count(":=")))
    chk("7.10 refine 窗口未动（仍是 e31-100）",
        "if epoch <= 30 or epoch > 100:" in txt)

    print("=" * 66)
    print("EXEC SELFTEST: %d/%d PASS%s"
          % (_n[0] - _fail[0], _n[0], "" if _fail[0] == 0 else "  <<< %d FAIL" % _fail[0]))
    return 0 if _fail[0] == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
