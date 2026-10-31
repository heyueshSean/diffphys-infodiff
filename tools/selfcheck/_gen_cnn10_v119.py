# -*- coding: utf-8 -*-
"""[v119] 幂等生成器：cnn10_v118.py  ->  cnn10_v119.py

父本 cnn10_v118.py（sha1 399f5dad764f9097c78205b9bdb90f264d960613）。

**唯一一根轴**：`refine_physics` 里 `structure_alignment_loss` 的**对齐目标**
    旧 = feat             （⇒ 场被显式推成骨干特征的**摹本**）
    新 = feat 的低通补集   （`feat - avgpool_k(feat)` ⇒ 场被推向**互补**）
这是自 v9 起唯一从未读过数、从未被改过的一条**写场通路**。

除此之外全部改动都在**仪器/读数层**（新增 `_v119_*`，父本函数零改动零删除）。
"""
import ast
import difflib
import hashlib
import io
import os
import py_compile
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "cnn10_v118.py")
DST = os.path.join(HERE, "cnn10_v119.py")

if not os.path.exists(SRC):
    print("FATAL: 找不到父本 %s" % SRC)
    sys.exit(1)

src = io.open(SRC, encoding="utf-8").read()
_ops = []


def rep(tag, old, new, expect=1):
    """精确替换（exactly `expect` 次）。"""
    n = src.count(old)
    assert n == expect, "锚点 %s 出现 %d 次（期望 %d）:\n%r" % (tag, n, expect, old[:120])
    out = src.replace(old, new) if expect else src
    _ops.append(("rep", tag, old, new))
    return out


def work(dev_src):
    s = dev_src

    # ---------------------------------------------------------------- P1 版本同步
    old = 'VERSION         = "v118"   # [v110] 五处同步的唯一真相源：CKPT_DIR / OUT_DIR / JSON 名 / results / banner'
    new = 'VERSION         = "v119"   # [v110] 五处同步的唯一真相源：CKPT_DIR / OUT_DIR / JSON 名 / results / banner'
    assert s.count(old) == 1, "P1"
    s = s.replace(old, new)
    _ops.append(("rep", "P1-VERSION", old, new))

    # ---------------------------------------------------------------- P2 v119 常量
    OLD_CONST_ANCHOR = (
        "# ====================== [v118] 约束通道「最后一张牌」的自证常量 ======================\n")
    NEW_CONST = (
        "# ====================== [v119] 对齐目标族（refine_physics）的自证常量 ======================\n"
        "#   ★ 为什么是这根轴：v113/v117 切断 pinn、v118 切断 w_phys*smooth 之后，\n"
        "#     场**仍是**近 DC（band_frac 0.055 -> 0.039 << 0.20）⇒ 约束通道判死。\n"
        "#     但 arbol还有一条**从未读过数**的写场通路：`refine_physics`（e31-100，\n"
        "#     独立优化器、冻结主干）里的 `structure_alignment_loss`，它把场**显式**\n"
        "#     对齐到骨干特征 ⇒ 数学上就是在**制造摹本**。v111 实测的『冗余/DC』\n"
        "#     正是它的预测表型。⇒ 未测 ≠ 判负（纪律 365）⇒ 先测这最后一条。\n"
        "V119_HP_KERNEL  = 3                # 低通核边长（奇数）；补集 = feat - avgpool_k(feat)\n"
        "V119_TARGET_MODE = \"residual_hp\"   # 对齐目标模式；写死常开，**没有开关**\n"
        "V119_MIN_HP_FRAC = 0.02            # 目标的最低高频能量占比（低于此 ⇒ 补集是空的 ⇒ 轴空转）\n"
        "V119_MAX_CORR    = 0.50            # 『场不再是摹本』的相关上限（|corr| 越低越互补）\n"
        "V119_REFINE_NOTE = (\n"
        "    \"v119 = v118 逐字 + 唯一一根轴：refine_physics 的**对齐目标**。\"\n"
        "    \"旧：target = feat ⇒ structure_alignment_loss 在标准化后最大化 \"\n"
        "    \"cos(phys, feat) ⇒ 场被**显式**推成骨干特征的摹本 ⇒ v111 实测 \"\n"
        "    \"band_frac=0.0968（96% 能量在 DC）、加性注入通道以『信息冗余』归档。\"\n"
        "    \"新：target = feat - avgpool_k(feat)（零参数盒子低通）⇒ 场被推向 \"\n"
        "    \"**骨干没有的那一份**（高频补集）⇒ 直接逆练『摹本』这个已测表型。\"\n"
        "    \"★ 目标是**参照系**：全程 no_grad + detach，不接受梯度。\"\n"
        "    \"★ 不加开关、不加参数、不改 `structure_alignment_loss` 本体、不改 refine 窗口（e31-100）。\"\n"
        "    \"★ 判据 band_frac_mean 沿用 V113_BAND_FRAC_MIN=0.20，一个数不改。\"\n"
        ")\n"
        "\n"
        + OLD_CONST_ANCHOR)
    assert s.count(OLD_CONST_ANCHOR) == 1, "P2"
    s = s.replace(OLD_CONST_ANCHOR, NEW_CONST)
    _ops.append(("rep", "P2-CONST", OLD_CONST_ANCHOR, NEW_CONST))

    # ---------------------------------------------------------------- P3~P6 仪器函数
    OLD_FN_ANCHOR = "def refine_physics(epoch, refiner, model, phys_optimizer, steps=3):"

    NEW_FNS = '''def _v119_complement(feat):
    """[v119] **唯一一根轴的作用面**：对齐目标 = 骨干特征的低通补集。

    旧（v9 ~ v118）：target = feat
        ⇒ `structure_alignment_loss` 在 per-map 标准化后最大化 cos(phys_norm, feat_norm)
        ⇒ 场被**显式**推成骨干特征的摹本 ⇒ v111 实测 band_frac = 0.0968（96% DC）
          ⇒ 加性注入通道以「信息冗余」归档。
    新（v119）    ：target = feat - avgpool_k(feat)
        ⇒ 场被推向**骨干没有的那一份**（高频补集）⇒ 直接检验「场能不能承载互补信息」。

    ★ 目标是**参照系**：全程 no_grad + detach ⇒ 不接受梯度、不带进计算图。
    ★ 零新增参数（盒子低通是固定算子），保持 dtype / device / 形状不变。
    ★ 退化保护：核边长取 <= 空间边长的最大奇数；取不到就用 feat 自身相减
      （**绝不**静默回退到 `feat` —— 那等于把轴悄悄关掉，纪律 358）。
    """
    try:
        _k = int(V119_HP_KERNEL)
    except Exception:
        _k = 3
    if _k < 3 or _k % 2 == 0:
        _k = 3
    _f = feat.detach()
    with torch.no_grad():
        try:
            _h = int(_f.shape[-2])
            _w = int(_f.shape[-1])
            _lim = min(_h, _w)
            while _k > _lim and _k > 3:
                _k -= 2
            if _k > _lim:
                return _f - _f
            try:
                _low = F.avg_pool2d(_f, _k, stride=1, padding=_k // 2,
                                    count_include_pad=False)
            except TypeError:
                _low = F.avg_pool2d(_f, _k, stride=1, padding=_k // 2)
            return (_f - _low).detach()
        except Exception:
            return (_f - _f).detach()


def _v119_std_pair(phys, feat):
    """[v119] 读数助手：取 gray + per-map 标准化后的一对扁平向量。

    ★ 与 `structure_alignment_loss` **同一套口径**（标准化 ⇒ 平移/缩放不变）
      ⇒ 两处的数字可直接互读。只读，不建图。
    """
    def _g(t):
        return t.mean(dim=1, keepdim=True)
    with torch.no_grad():
        _a = _g(phys.detach())
        _b = _g(feat.detach())
        _a = (_a - _a.mean(dim=(-2, -1), keepdim=True))
        _b = (_b - _b.mean(dim=(-2, -1), keepdim=True))
        _sa = _a.std() + 1e-8
        _sb = _b.std() + 1e-8
        return (_a / _sa).flatten(1), (_b / _sb).flatten(1)


def _v119_align_probe(model, inputs, epoch):
    """[v119] refine 对齐目标的**作用面探针**（只读，不杀主流程）。

    读数（全部**自我对照**，不需要另跑一趟父本）：
      · `hp_frac_l2/l3`  ：目标的高频能量占比 ‖feat-blur(feat)‖ / ‖feat‖
                           （低于 V119_MIN_HP_FRAC ⇒ 补集是空的 ⇒ 轴空转）
      · `loss_copy_l2/l3`：structure_alignment_loss(ph, feat)          （摹本口径）
      · `loss_hp_l2/l3`  ：structure_alignment_loss(ph, hp-complement) （互补口径）
      · `regime`         ：loss_hp < loss_copy ⇒ **HP_ALIGNED**（场已离开摹本）
                           否则 COPY_ALIGNED（场仍是摹本）
      · `corr_l2/l3`     ：|cos(phys_norm, feat_norm)| —— 摹本度的一致性读数
      · `grad_*`         ：新目标对**场参数**的梯度（必须 > 0 ⇒ 轴不是死的）

    ★ 全程小批（纪律 378）+ 跑完清扫（纪律 379）+ 不抛（纪律 375）+ 恒返回 dict（纪律 381）。
    """
    _out = {"ok": False, "error": None, "sections_ok": {}, "section_errors": {}}
    _pairs = []
    try:
        _net = model
        if hasattr(_net, "module"):
            _net = _net.module
        for _pn in ("phys2", "phys3"):
            _p = getattr(_net, _pn, None)
            if _p is None:
                continue
            for _nm in ("s_param", "lam_param", "nu_param", "aniso_param"):
                _q = getattr(_p, _nm, None)
                if _q is not None:
                    _pairs.append((_pn + "." + _nm, _q))
    except Exception as _e:
        _out["error"] = "phys_params: %s: %s" % (type(_e).__name__, _e)
        return _out
    _tgt = [_q for (_n, _q) in _pairs]
    if not _tgt:
        _out["error"] = "no physics params found"
        return _out
    try:
        _in_s, _n_s = _v117_probe_slice(inputs, V117_PROBE_N_SMALL)
    except Exception as _e:
        _out["error"] = "slice: %s: %s" % (type(_e).__name__, _e)
        return _out
    _out["probe_n"] = _n_s
    _out["target_mode"] = str(V119_TARGET_MODE)
    _out["hp_kernel"] = int(V119_HP_KERNEL)
    _feats = {}
    try:
        # ---------- §1 特征 + 场输出（no_grad，不建图）----------
        try:
            _net.eval()
            with torch.no_grad():
                _h = _net.backbone.conv1(_in_s)
                _h = _net.backbone.bn1(_h)
                _h = torch.relu(_h)
                _h = _net.backbone.layer1(_h)
                _h, _ = _net.gate1(_h, _net.phys1(_h))
                _h = _net.backbone.layer2(_h)
                _feats["f2"] = _h.detach().clone()
                _feats["p2"], _, _, _, _ = _net.phys2(_h, _h, epoch)
                _feats["p2"] = _feats["p2"].detach().clone()
                _x, _ = _net.gate2(_h, _feats["p2"])
                _x = _net.backbone.layer3(_x)
                _feats["f3"] = _x.detach().clone()
                _p3, _, _, _, _ = _net.phys3(_x, _x, epoch)
                _feats["p3"] = _p3.detach().clone()
            _out["sections_ok"]["feats"] = True
        except Exception as _e:
            _out["sections_ok"]["feats"] = False
            _out["section_errors"]["feats"] = "%s: %s" % (type(_e).__name__, _e)
            return _out
        # ---------- §2 量级 / 双口径损失 / 相关系数（仍然 no_grad）----------
        try:
            _lc = 0.0
            _lh = 0.0
            with torch.no_grad():
                for _lv in ("2", "3"):
                    _f = _feats["f" + _lv]
                    _p = _feats["p" + _lv]
                    _t = _v119_complement(_f)
                    _fn = float(_f.detach().pow(2).sum().item())
                    _tn = float(_t.detach().pow(2).sum().item())
                    _out["hp_frac_l" + _lv] = ((_tn ** 0.5) / ((_fn ** 0.5) + 1e-12)
                                               if _fn > 0 else None)
                    _a, _b = _v119_std_pair(_p, _f)
                    # ★ 用**内建** abs：`.item()` 已经是 float，
                    #   `torch.abs(float)` 在新版 torch 直接 TypeError（仪器自己崩）。
                    _out["corr_l" + _lv] = float(
                        abs(float(F.cosine_similarity(_a, _b, dim=1).mean().item())))
                    _c = float(structure_alignment_loss(_p, _f).item())
                    _hpv = float(structure_alignment_loss(_p, _t).item())
                    _out["loss_copy_l" + _lv] = _c
                    _out["loss_hp_l" + _lv] = _hpv
                    _lc += _c
                    _lh += _hpv
            _out["loss_copy_total"] = _lc
            _out["loss_hp_total"] = _lh
            _out["regime"] = "HP_ALIGNED" if (_lh < _lc) else "COPY_ALIGNED"
            _out["sections_ok"]["levels"] = True
        except Exception as _e:
            _out["sections_ok"]["levels"] = False
            _out["section_errors"]["levels"] = "%s: %s" % (type(_e).__name__, _e)
        # ---------- §3 梯度：新目标**真的**写进场参数 ----------
        try:
            _net.train()
            # ★★ 只有**骨干**在 no_grad 下取特征；被优化的**场**必须**带图**跑，
            #    否则 `_lhp` 没有 grad_fn ⇒ autograd.grad 必抛
            #    "element 0 of tensors does not require grad" ⇒ grad 段恒失败。
            with torch.no_grad():
                _h = _net.backbone.conv1(_in_s)
                _h = _net.backbone.bn1(_h)
                _h = torch.relu(_h)
                _h = _net.backbone.layer1(_h)
                _h, _ = _net.gate1(_h, _net.phys1(_h))
                _h = _net.backbone.layer2(_h)
                _f2g = _h.detach()
                _t2 = _v119_complement(_f2g)
                _xd, _ = _net.gate2(_h, _net.phys2(_h, _h, epoch)[0].detach())
                _xg = _net.backbone.layer3(_xd)
                _f3g = _xg.detach()
                _t3 = _v119_complement(_f3g)
            _p2g, _, _, _, _ = _net.phys2(_f2g, _f2g, epoch)
            _p3g, _, _, _, _ = _net.phys3(_f3g, _f3g, epoch)
            _lhp = structure_alignment_loss(_p2g, _t2) + structure_alignment_loss(_p3g, _t3)
            _gs = torch.autograd.grad(_lhp, _tgt, retain_graph=False,
                                      allow_unused=True)
            _mx = 0.0
            _nz = []
            for (_nm, _q), _g in zip(_pairs, _gs):
                if _g is None:
                    continue
                _v = float(_g.abs().max().item())
                if _v > 0.0:
                    _nz.append(_nm)
                if _v > _mx:
                    _mx = _v
            _out["grad_max_hp"] = _mx
            _out["grad_nonzero_params"] = _nz
            _out["grad_n_nonzero"] = len(_nz)
            _out["sections_ok"]["grad"] = True
        except Exception as _e:
            _out["sections_ok"]["grad"] = False
            _out["section_errors"]["grad"] = "%s: %s" % (type(_e).__name__, _e)
        finally:
            try:
                _net.eval()
            except Exception:
                pass
        _req = ("feats", "levels", "grad")
        _out["ok"] = bool(all(_out["sections_ok"].get(_k2, False) for _k2 in _req)
                          and (_out.get("regime") is not None)
                          and (_out.get("grad_max_hp") is not None))
        if not _out["ok"] and _out.get("error") is None:
            _miss = [k for k in _req if not _out["sections_ok"].get(k, False)]
            _out["error"] = ("incomplete: missing=%s | section_errors=%s"
                             % (_miss, _out.get("section_errors")))
    except Exception as _e:
        _out["error"] = "%s: %s" % (type(_e).__name__, _e)
    try:
        _v117_cuda_sweep()
    except Exception:
        pass
    return _out


def _v119_refine_verdict(align, band_frac):
    """[v119] 对齐目标族的**终局判词**（判据数值沿用既有常量，一个数不改）。

      · 目标非空：hp_frac >= V119_MIN_HP_FRAC（否则补集是空的 ⇒ 轴空转）
      · 轴不空转：grad_max_hp > 0（新目标**真的**把梯度送进了场参数）
      · 场离开摹本：regime == HP_ALIGNED
      · 场有了内容：band_frac_mean >= V113_BAND_FRAC_MIN (0.20)

      ★★ 判定的**先后顺序**是有意的，而且是**最少次方final**（见 `_v119_refineface.criterion.note`）：

       先用两根「轴空转」闸门（目标是否非空 / 目标是否真的送进了梯度），
       再用 **内容闸门**（band_frac）做**唯一**的成败判据，
       `regime` 只用于把失败细分成 COPY_ATTRACTOR / NO_CONTENT（两者**同一个动作**：归档）。

      REFINE_HP_TARGET_EMPTY     ：补集空（目标 auto  degenerate）⇒ 先修目标，不看场
      REFINE_HP_GRAD_DEAD        ：目标没送出梯度 ⇒ 轴空转，先修实现
      REFINE_HP_READOUT_ERROR    ：读数缺失 ⇒ 未测（**不等于判负**）
      REFINE_HP_DELIVERED        ：band_frac >= 0.20 ⇒ 场被推出了内容
                                  ⇒ 场的信息价值被救回 ⇒ 下一版复活注入通道
      REFINE_HP_COPY_ATTRACTOR   ：无内容 **且** regime=COPY 且 |corr| 仍 >= V119_MAX_CORR
                                  ⇒ 场是「吸附」在摹本解上的 ⇒ **物理场整条线归档**
      REFINE_HP_NO_CONTENT       ：无内容且不是摹本吸附 ⇒ 场本身没有可塑信息
                                  ⇒ **物理场整条线归档**
    """
    try:
        if not isinstance(align, dict) or not align.get("ok"):
            return "REFINE_HP_READOUT_ERROR"
        _hf = [align.get("hp_frac_l2"), align.get("hp_frac_l3")]
        _hf = [float(_x) for _x in _hf if _x is not None]
        if _hf and (min(_hf) < float(V119_MIN_HP_FRAC)):
            return "REFINE_HP_TARGET_EMPTY"
        _gm = align.get("grad_max_hp")
        if _gm is None or float(_gm) <= 0.0:
            return "REFINE_HP_GRAD_DEAD"
        if band_frac is None:
            return "REFINE_HP_READOUT_ERROR"
        if float(band_frac) >= float(V113_BAND_FRAC_MIN):
            return "REFINE_HP_DELIVERED"
        _cs = [align.get("corr_l2"), align.get("corr_l3")]
        _cs = [abs(float(_x)) for _x in _cs if _x is not None]
        if (str(align.get("regime")) == "COPY_ALIGNED"
                and _cs and max(_cs) >= float(V119_MAX_CORR)):
            return "REFINE_HP_COPY_ATTRACTOR"
        return "REFINE_HP_NO_CONTENT"
    except Exception:
        return "REFINE_HP_READOUT_ERROR"


def _v119_verify_runnable():
    """[v119] 可运行性自检 = v118 的检查项 **+ v119 新增的函数/常量**。

    ★ 不修改父本的 `_v118_verify_runnable`（逐字保留）⇒ 这里做**包装**。
    """
    try:
        _r = _v118_verify_runnable()
    except Exception as _e:
        _r = {"ok": False, "missing": ["_v118_verify_runnable raised: %s" % _e],
              "notes": []}
    if not isinstance(_r, dict):
        _r = {"ok": False, "missing": ["parent returned non-dict"], "notes": []}
    _r.setdefault("missing", [])
    _r.setdefault("notes", [])
    for _fn in ("_v119_complement", "_v119_std_pair", "_v119_align_probe",
                "_v119_refine_verdict", "_v119_verify_runnable"):
        if _fn not in globals():
            _r["missing"].append("func:" + _fn)
    for _cn in ("V119_HP_KERNEL", "V119_TARGET_MODE", "V119_MIN_HP_FRAC",
                "V119_MAX_CORR", "V119_REFINE_NOTE"):
        if _cn not in globals():
            _r["missing"].append("const:" + _cn)
    _r["ok"] = bool(len(_r["missing"]) == 0)
    return _r


'''
    assert s.count(OLD_FN_ANCHOR) == 1, "P3"
    s = s.replace(OLD_FN_ANCHOR, NEW_FNS + OLD_FN_ANCHOR)
    _ops.append(("rep", "P3-FUNCS", OLD_FN_ANCHOR, NEW_FNS + OLD_FN_ANCHOR))

    # ---------------------------------------------------------------- P7 唯一一根轴
    OLD_AXIS = (
        "            loss_align = structure_alignment_loss(ph2, feat2) + \\\n"
        "                         structure_alignment_loss(ph3, feat3)\n")
    NEW_AXIS = (
        "            # ★★ [v119] **唯一一根轴**：对齐目标 = 骨干特征的低通补集\n"
        "            #   旧 target = feat ⇒ 场被显式推成摹本（见 V119_REFINE_NOTE）；\n"
        "            #   新 target = feat - avgpool_k(feat) ⇒ 场被推向**互补**。\n"
        "            #   ★ 目标由 `_v119_complement` 产出（no_grad + detach ⇒ 不接受梯度）。\n"
        "            _v119_tg2 = _v119_complement(feat2)\n"
        "            _v119_tg3 = _v119_complement(feat3)\n"
        "            loss_align = structure_alignment_loss(ph2, _v119_tg2) + \\\n"
        "                         structure_alignment_loss(ph3, _v119_tg3)\n")
    assert s.count(OLD_AXIS) == 1, "P7"
    s = s.replace(OLD_AXIS, NEW_AXIS)
    _ops.append(("rep", "P7-AXIS", OLD_AXIS, NEW_AXIS))

    # ---------------------------------------------------------------- P8 收尾读数
    OLD_TAIL = (
        'flush_print("  [v118] ★ 约束通道终局判词 = %s | band_frac_mean = %s"\n'
        '            % (_v118_verdict,\n'
        '               ("%.6f" % float(_v118_band)) if _v118_band is not None else "None"))\n')
    NEW_TAIL = OLD_TAIL + (
        '# ★★ [v119] refine 对齐目标的作用面探针（小批；只读；不杀主流程）\n'
        '_v119_align = _safe_readout(\n'
        '    lambda: _v119_align_probe(model, _v115_probe_inputs, TRAIN_EPOCHS),\n'
        '    {"ok": False, "error": "PROBE_RAISED"})\n'
        '_v119_ok = bool(isinstance(_v119_align, dict) and _v119_align.get("ok"))\n'
        '_v119_verdict = _v119_refine_verdict(\n'
        '    _v119_align if _v119_ok else None, _v118_band)\n'
        'flush_print("  [v119] 对齐目标探针：ok=%s | mode=%s | k=%s | probe_n=%s"\n'
        '            % (_v119_ok, V119_TARGET_MODE, V119_HP_KERNEL,\n'
        '               (_v119_align or {}).get("probe_n")))\n'
        'flush_print("  [v119]             regime=%s | loss_copy=%.5f | loss_hp=%.5f"\n'
        '            % ((_v119_align or {}).get("regime"),\n'
        '               float((_v119_align or {}).get("loss_copy_total") or 0.0),\n'
        '               float((_v119_align or {}).get("loss_hp_total") or 0.0)))\n'
        'flush_print("  [v119]             hp_frac l2=%s l3=%s | |corr| l2=%s l3=%s"\n'
        '            % ((_v119_align or {}).get("hp_frac_l2"),\n'
        '               (_v119_align or {}).get("hp_frac_l3"),\n'
        '               (_v119_align or {}).get("corr_l2"),\n'
        '               (_v119_align or {}).get("corr_l3")))\n'
        'flush_print("  [v119]             grad_max_hp=%.3e | 非零参数=%d/%d | err=%s"\n'
        '            % (float((_v119_align or {}).get("grad_max_hp") or 0.0),\n'
        '               int((_v119_align or {}).get("grad_n_nonzero") or 0), 8,\n'
        '               (_v119_align or {}).get("error")))\n'
        'flush_print("  [v119] ★ 对齐目标族终局判词 = %s | band_frac_mean = %s"\n'
        '            % (_v119_verdict,\n'
        '               ("%.6f" % float(_v118_band)) if _v118_band is not None else "None"))\n')
    assert s.count(OLD_TAIL) == 1, "P8"
    s = s.replace(OLD_TAIL, NEW_TAIL)
    _ops.append(("rep", "P8-TAIL", OLD_TAIL, NEW_TAIL))

    # ---------------------------------------------------------------- P9 results 块
    OLD_RES = '    "v117_parent": "cnn10_v116.py",\n'
    NEW_RES = OLD_RES + (
        '    "v119_refineface": {\n'
        '        "axis": V119_REFINE_NOTE,\n'
        '        "target_mode": str(V119_TARGET_MODE),\n'
        '        "hp_kernel": int(V119_HP_KERNEL),\n'
        '        "probe_ok": _v119_ok,\n'
        '        "probe_error": ((_v119_align or {}).get("error")\n'
        '                         if not _v119_ok else None),\n'
        '        "probe_n": (_v119_align or {}).get("probe_n"),\n'
        '        "regime": ((_v119_align or {}).get("regime") if _v119_ok else None),\n'
        '        "loss_copy_total": ((_v119_align or {}).get("loss_copy_total")\n'
        '                             if _v119_ok else None),\n'
        '        "loss_hp_total": ((_v119_align or {}).get("loss_hp_total")\n'
        '                           if _v119_ok else None),\n'
        '        "hp_frac_l2": ((_v119_align or {}).get("hp_frac_l2") if _v119_ok else None),\n'
        '        "hp_frac_l3": ((_v119_align or {}).get("hp_frac_l3") if _v119_ok else None),\n'
        '        "corr_l2": ((_v119_align or {}).get("corr_l2") if _v119_ok else None),\n'
        '        "corr_l3": ((_v119_align or {}).get("corr_l3") if _v119_ok else None),\n'
        '        "grad_max_hp": ((_v119_align or {}).get("grad_max_hp") if _v119_ok else None),\n'
        '        "grad_n_nonzero": ((_v119_align or {}).get("grad_n_nonzero")\n'
        '                            if _v119_ok else None),\n'
        '        "grad_nonzero_params": ((_v119_align or {}).get("grad_nonzero_params")\n'
        '                                 if _v119_ok else None),\n'
        '        "band_frac_mean": (float(_v118_band)\n'
        '                            if _v118_band is not None else None),\n'
        '        "delivered_acc": float(delivered),\n'
        '        "verdict": _v119_verdict,\n'
        '        "criterion": {\n'
        '            "band_frac_min": float(V113_BAND_FRAC_MIN),\n'
        '            "min_hp_frac": float(V119_MIN_HP_FRAC),\n'
        '            "max_corr": float(V119_MAX_CORR),\n'
        '            "baseline_v110_delivered": float(V113_BASE_DELIVERED),\n'
        '            "pass_line": float(V113_PASS_LINE),\n'
        '            "redline": float(V113_REDLINE),\n'
        '            "sigma_diff_pp": float(V113_SIGMA_DIFF),\n'
        '            "note": (\n'
        '                "判据数值沿用既有常量（band>=0.20 / PASS 97.838 / 红线 97.04 / "\n'
        '                "sigma 0.169），一个数不改。\\n"\n'
        '                "本版是最后一条从未读过数的写场通路：refine_physics 的对齐目标。\\n"\n'
        '                "★ 预承诺：若本版判出 NO_CONTENT / COPY_ATTRACTOR / GRAD_DEAD，\\n"\n'
        '                "则物理场整条线归档，下下版起只许动目标函数主项\\n"\n'
        '                "（分类损失形式 / 标签结构 / 损失权重结构），不再回来动场。"),\n'
        '        },\n'
        '        "self_sha1": SELF_SHA1,\n'
        '    },\n')
    assert s.count(OLD_RES) == 1, "P9"
    s = s.replace(OLD_RES, NEW_RES)
    _ops.append(("rep", "P9-RESULTS", OLD_RES, NEW_RES))

    # ---------------------------------------------------------------- P10 banner
    OLD_BANNER = (
        'flush_print("        [v118] ★ 三修：uda 还原到探针前 | 整批 section_errors 落盘 | 缺项如实报")\n')
    NEW_BANNER = OLD_BANNER + (
        'flush_print("        [v119] ★ 唯一一根轴 = refine_physics 的**对齐目标**："\n'
        '            "feat（摹本） -> feat 的低通补集（互补）| 对齐目标族最后一条未读通路")\n'
        'flush_print("        [v119] ★ 这是自 v9 起唯一从未读过数、也从未被切过的一条**写场通路**")\n'
        'flush_print("        [v119] ★ 目标 k=%d | 判据沿用 band_frac>=%.2f / PASS %.3f / 红线 %.2f"\n'
        '            % (int(V119_HP_KERNEL), float(V113_BAND_FRAC_MIN),\n'
        '               float(V113_PASS_LINE), float(V113_REDLINE)))\n')
    assert s.count(OLD_BANNER) == 1, "P10"
    s = s.replace(OLD_BANNER, NEW_BANNER)
    _ops.append(("rep", "P10-BANNER", OLD_BANNER, NEW_BANNER))
    return s


def _apply(dev_src):
    return work(dev_src)


# ================================================================ 幂等：先查 MARKERS
# ★ MARKERS 必须是**产物里真实存在的字面**（用 _ops 的 tag 会永远缺失 ⇒ 每次都重生成）
MARKERS = [
    'VERSION         = "v119"',
    "V119_HP_KERNEL  = 3",
    "_v119_complement(feat2)",
    "def _v119_align_probe(model, inputs, epoch):",
    "def _v119_refine_verdict(align, band_frac):",
    "def _v119_verify_runnable():",
    '"v119_refineface": {',
    "_v119_align = _safe_readout(",
    '[v119] ★ 对齐目标族终局判词',
    "[v119] ★ 唯一一根轴 = refine_physics",
    # ★ 仪器自证抓出的两处真 bug 也必须是 marker
    "abs(float(F.cosine_similarity(_a, _b, dim=1).mean().item()))",
    "_p2g, _, _, _, _ = _net.phys2(_f2g, _f2g, epoch)",
]
if os.path.exists(DST):
    cur = io.open(DST, encoding="utf-8").read()
    missing = [m for m in MARKERS if m not in cur]
else:
    cur, missing = None, MARKERS

if cur is not None and not missing:
    print("[gen v119] 产物已含全部 %d 个 MARKERS ⇒ NO-OP（不重写）" % len(MARKERS))
    out = cur
else:
    print("[gen v119] 缺 %d/%d marker ⇒ 从父本重跑" % (len(missing), len(MARKERS)))
    out = _apply(src)

# ================================================================ 写出 + 自检
_b = out.encode("utf-8")
_t = out
assert _b.count(b"\r") == 0, "产物不是纯 LF"
with io.open(DST, "wb") as f:
    f.write(_b)

_t = io.open(DST, encoding="utf-8").read()
_b = _t.encode("utf-8")
print("[gen v119] sha1=%s bytes=%d lines=%d CR=%d"
      % (hashlib.sha1(_b).hexdigest()[:12], len(_b), _t.count("\n"),
         _b.count(b"\r")))

try:
    import tempfile as _tf
    _cc = os.path.join(_tf.gettempdir(), "_v119_compile_probe.pyc")
    py_compile.compile(DST, doraise=True, cfile=_cc)
    print("[gen v119] py_compile OK")
except Exception as e:
    print("[gen v119] py_compile FAIL: %s" % e)
    raise


def _tops(t):
    m = ast.parse(t)
    d = {}
    for n in m.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            d[n.name] = n
    return d


A = _tops(src)
C = _tops(_t)
added = sorted([k for k in C if k not in A])
removed = sorted([k for k in A if k not in C])
dupbody = []
for k in A:
    if k in C:
        if ast.dump(A[k]) != ast.dump(C[k]):
            dupbody.append(k)
print("[gen v119] 顶层 def/class: 父本 %d -> 本版 %d" % (len(A), len(C)))
print("[gen v119] ADDED   = %r" % added)
print("[gen v119] REMOVED = %r" % removed)
print("[gen v119] CHANGED = %r" % dupbody)
assert not removed, "有父本顶层定义被删除"
for k in added:
    assert k.startswith("_v119_"), "非预期的新增顶层定义 %r" % k
assert dupbody == ["refine_physics"], (
    "只允许 refine_physics 被改（那正是本版唯一一根轴），实际 = %r" % dupbody)

# ---- refine_physics 的改动必须**恰好**是那根轴（不多不少）
_p = ast.get_source_segment(src, A["refine_physics"]).replace("\r\n", "\n")
_c = ast.get_source_segment(_t, C["refine_physics"]).replace("\r\n", "\n")
_rem, _add = [], []
for line in difflib.unified_diff(_p.split("\n"), _c.split("\n"), n=0, lineterm=""):
    if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
        continue
    if line.startswith("-"):
        _rem.append(line[1:])
    elif line.startswith("+"):
        _add.append(line[1:])
_expect_rem = [
    "            loss_align = structure_alignment_loss(ph2, feat2) + \\",
    "                         structure_alignment_loss(ph3, feat3)",
]
assert _rem == _expect_rem, "refine_physics 被删的行不符预期: %r" % _rem
assert len([x for x in _add if x.strip()]) == 8, (
    "refine_physics 新增行应为 4 注释 + 2 目标 + 2 调用 = 8 行，实际 %r" % _add)

# ---- 全文件级：父本被删/被改的行只允许 VERSION + 上面那两行
_rem_all = []
_sm = difflib.SequenceMatcher(None, src.replace("\r\n", "\n").split("\n"),
                              _t.split("\n"), autojunk=False)
for tag, i1, i2, j1, j2 in _sm.get_opcodes():
    if tag in ("replace", "delete"):
        for x in src.replace("\r\n", "\n").split("\n")[i1:i2]:
            if x.strip():
                _rem_all.append(x)
_allowed_rem = set(_expect_rem + [
    'VERSION         = "v118"   # [v110] 五处同步的唯一真相源：CKPT_DIR / OUT_DIR / JSON 名 / results / banner',
])
_bad = [x for x in _rem_all if x not in _allowed_rem]
assert not _bad, "父本有非预期被删/被改行:\n%s" % "\n".join(repr(x) for x in _bad[:10])

# ---- 硬指标
_n_arg = _t.count("add_argument")
_n_sysexit = _t.count("raise SystemExit")
PARENT_SYSEXIT = src.count("raise SystemExit")
print("[gen v119] argparse=%d SystemExit=%d（父本 %d）" % (_n_arg, _n_sysexit, PARENT_SYSEXIT))
assert _n_arg == 0, "出现了 argparse 开关"
assert _n_sysexit == PARENT_SYSEXIT, "SystemExit 不得增加"
# ★ 计数断言换成**派生路径断言**：版本派生量必须真的跟着 VERSION 走
#   （"v118" 的字面计数会因为新增 note 提到父本而上升，那不是缺陷）
for _bad_v in ("ckpts_v118", "out_v118", "results_cnn_v118"):
    assert _bad_v not in _t, "残留旧版本的派生标识 %r" % _bad_v
assert '"v119"' in _t and 'VERSION         = "v119"' in _t

# ---- Python 3.6 兼容（训练机 3.6）：新增行不许有 f-string / walrus
_new_only = [l for l in _add if l.strip()]
for l in _new_only:
    assert "f\"" not in l and "f'" not in l, "3.6 不兼容的 f-string: %r" % l
    assert ":=" not in l, "3.6 不兼容的 walrus: %r" % l

print("[gen v119] ALL STRUCTURAL CHECKS PASSED")
print("[gen v119] wrote %s" % os.path.basename(DST))
