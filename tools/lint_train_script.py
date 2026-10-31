"""lint_train_script.py — 训练脚本发车前静态体检 (torch-free, 秒级)。

用途: 长训练最贵的失败模式不是"写错公式", 而是"错误藏在只有跑到第 N 个 epoch
才会执行的分支里"。v29 就是这样: get_sigma_max(start=..., end=...) 与签名
(s_start/s_end) 不符, py_compile 通过、AST 文本检查通过, 结果跑满 150 个干净
epoch 后在 e151 首次进入 PFC 分支时崩掉, 白烧 1.7 小时。同一脚本第二次崩溃是
`sigma_max`: 只在 `if cons_teacher is not None:` 里绑定, 却在 `if w_eff > 0:`
的日志串里读 —— PFC 模式下永远绑不上, 仍在 e151 炸。

六道闸门:
  G1  调用点 vs 签名: 关键字名拼错、位置参数超量、必填参数缺失。
  G2  MAYBE-UNBOUND 局部读取: 守卫敏感 (path-sensitive) 的 definite-assignment
      分析。判据是**守卫蕴含**: 在守卫 R 处读 x 安全 <=> 存在一条 x 的绑定
      路径, 其守卫被 R 蕴含。于是
          if use_mixup: t_a, t_b, lam = ...        # 同守卫读 -> 放过
          if use_mixup: ... t_a, t_b, lam ...
          if cons_teacher is not None: sigma_max = ...   # 异守卫读 -> 必报
          if w_eff > 0: f"{sigma_max}"
      G2 与 verify_vit_v29.py 共用 static_defassign.py 的实现。
  G3  每 epoch 记录项: `history["k"]` 用了但 dict 字面量里没初始化 -> 首个 batch KeyError。
  G4  裸 torch.save: 没有写前查盘守卫（磁盘满会在 C++ 析构里 abort，except 抓不到）。
  G5  读了从没被绑定的名字: 忘 import / 忘定义（语法合法 ⇒ py_compile 不报）。
  G6  **模块级 use-before-def**: 顶层语句是顺序执行的，把常量写在使用它的
      横幅/辅助调用之后 -> 发车即 NameError。**G5 按设计抓不到这一类**
      （它把"定义在 use 之后"算作已绑定，因为函数体里确实如此），
      py_compile 也一定过。2026-09-24 v56 实测：横幅 L2203 引用 `BACKTRACK_REDLINE`，
      而它定义在 L2618 ⇒ `NameError`，白跑一次发车。
      实现见 scan_top_order.py（含"函数被提前调用、函数体读后面的常量"这一层）。
  G7  硬编码字面量的自检字段（**警告级，不影响退出码**）:
      `bool("<lit>" in str(<常量>).lower())` 这类"文案同步自检"，在常量内容变化或
      "轴搬家"后会**静默翻转** —— 2026-09-24 v57 实测：`bool("gate1" in str(AXIS_V57).lower())`
      在 v51~v56 恒为 True（轴恰好在 gate1），本版把轴搬到 gate2 后**恒为 False**，
      在一个真的同步正确的版本里伪造出"文案失同步"的假警（"配置常量伪装成观测"，
      覆辙 138/140）。判据：字面量必须真的出现在该常量的字面量里；
      常量能被 ast.literal_eval 静态求值时判死，否则列为待核。
      ⇒ 自检的字面量应由**单一真源派生**（如 `f"gate{N}"`），不要写死。

闸门本身也要被验证 —— "全绿"可能只是因为检测器太宽松。用
    python test_defassign_gate.py [script.py]     # 变异测试 G1~G5
    python test_top_order_gate.py                 # 变异测试 G6（3 正例 / 6 负例）
    python test_literal_gate.py                   # 变异测试 G7（3 正例 / 4 负例）
做变异测试: 把已修好的 bug 逐条塞回去, 要求闸门必须报出来 (v29 因此揪出
sigma_max / s_feat / loss 三处真缺陷)。

用法:
    python lint_train_script.py aa1.py
    python lint_train_script.py train_vit_cifar100_aug_v29.py
退出码 0 = 干净, 1 = 有问题 (不要发车)。
"""
import ast
import sys

# G2 与 verify_vit_v29.py 共用同一份判定逻辑（两份实现漂移本身就是故障源）
from static_defassign import maybe_unbound
from scan_unbound import scan_src
from scan_top_order import scan_src as scan_top_order_src


def collect_sigs(tree):
    """只登记**能被裸名字调用**的函数: 模块级 + 嵌套。类方法不登记。

    ★ 2026-09-17 修 (真实假阳性): 原来用 `ast.walk` 把**所有** FunctionDef 都按
    裸名字登记，于是 `class EnsembleTeacher: def head(self, tokens)` 会被注册成
    `head(pos=['self','tokens'])`；而别处有一句 `head = model.head` 再 `head(tokens)`
    (绑定后的别名，self 已经绑好了) ⇒ 被误报成 "missing required arg(s): tokens"。
    这个方法只能通过属性访问到达 (`obj.m(...)`) 或经别名，**结构上不可能**由裸名字
    匹配，所以登记它只会制造假阳性。
    ⇒ 闸门恒红与闸门恒绿同样致命: 一个永远报错的闸门等于没有闸门
      (与「④ 变异 pattern 失效」同源, 见 SKILL 的「闸门/仪器自身的失败模式」)。
    """
    sigs = {}

    def visit(node):
        for ch in node.body:
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = ch.args
                pos = [p.arg for p in a.posonlyargs] + [p.arg for p in a.args]
                sigs.setdefault(ch.name, {
                    "pos": pos,
                    "kwonly": [p.arg for p in a.kwonlyargs],
                    "n_req": len(pos) - len(a.defaults),
                    "vararg": a.vararg is not None,
                    "kwarg": a.kwarg is not None,
                    "lineno": ch.lineno,
                })
                visit(ch)                      # 嵌套函数仍按裸名字登记
            elif isinstance(ch, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                visit(ch)                      # 穿透语句块找嵌套 def
            # ClassDef 刻意不访问: 方法走属性访问, 不是裸名字调用

    visit(tree)
    return sigs


def assigned_names(tree):
    """作用域 -> 被赋值过的名字集合 (用于排除 `head = model.head` 这类别名)。

    裸名字调用只有在"该名字就是那个函数"时才有意义; 一旦在作用域里被重新赋值,
    它指向的是别的对象 (绑定方法 / 模块 / 属性), 拿函数签名去比就是误报。
    """
    out = []

    def scope(node):
        names = set()
        stack = [node]
        while stack:
            cur = stack.pop()
            for ch in ast.iter_child_nodes(cur):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef, ast.Lambda)):
                    continue                   # 内层作用域自己管
                if isinstance(ch, ast.Name) and isinstance(ch.ctx, ast.Store):
                    names.add(ch.id)
                if isinstance(ch, (ast.Global, ast.Nonlocal)):
                    names.update(ch.names)
                stack.append(ch)
        out.append((node, names))
        for ch in ast.walk(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scope(ch)

    scope(tree)
    return out


def walk_scopes(tree):
    """一次遍历: (作用域 id -> 该作用域内被赋值过的名字, 调用 id -> 所属最内层作用域).

    裸名字调用只有在"该名字真的就是那个函数"时才有意义; 一旦在作用域里被重新赋值
    (`head = model.head`), 它指向的是别的对象 (绑定方法 / 模块 / 属性), 再拿函数
    签名去比就是误报。
    ★ 注意别把 `ClassDef` 当函数作用域递归 —— 用 `ast.walk` 递归会在函数自身那一层
      无限自调用 (RecursionError, 2026-09-17 实测踩过)。
    """
    names = {}
    calls = {}

    def enter(node, key):
        names.setdefault(key, set())
        stack = list(ast.iter_child_nodes(node))
        while stack:
            cur = stack.pop()
            if isinstance(cur, ast.Lambda):
                continue                             # lambda 内不产生裸名字调用
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                enter(cur, id(cur))                  # 新作用域, 不与外层共享赋值集
                continue
            if isinstance(cur, ast.Name) and isinstance(cur.ctx, ast.Store):
                names[key].add(cur.id)
            if isinstance(cur, (ast.Global, ast.Nonlocal)):
                names[key].update(cur.names)
            if isinstance(cur, ast.Call):
                calls.setdefault(id(cur), []).append(key)
            stack.extend(ast.iter_child_nodes(cur))

    enter(tree, id(tree))
    return names, calls


def gate1_callsites(tree):
    """Keyword-name / arity mismatches against local signatures."""
    sigs = collect_sigs(tree)
    scope_names, call_scope = walk_scopes(tree)

    def alias_rebound(call):
        for key in call_scope.get(id(call), ()):
            if call.func.id in scope_names.get(key, ()):
                return True
        return False

    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Name):
            continue
        s = sigs.get(n.func.id)
        if s is None:
            continue
        if alias_rebound(n):
            continue                           # 别名 (head = model.head) ⇒ 不按签名比

        allowed = set(s["pos"]) | set(s["kwonly"])
        for kw in n.keywords:
            if kw.arg is not None and kw.arg not in allowed and not s["kwarg"]:
                out.append((n.lineno, n.func.id,
                            f"unexpected keyword '{kw.arg}' "
                            f"(signature @L{s['lineno']}: "
                            f"{', '.join(s['pos'] + s['kwonly'])})"))
        npos = len([x for x in n.args if not isinstance(x, ast.Starred)])
        if any(isinstance(x, ast.Starred) for x in n.args):
            continue
        if npos > len(s["pos"]) and not s["vararg"]:
            out.append((n.lineno, n.func.id,
                        f"{npos} positional args > {len(s['pos'])} params"))
        supplied = set(s["pos"][:npos]) | {k.arg for k in n.keywords}
        missing = [p for p in s["pos"][:s["n_req"]] if p not in supplied]
        if missing and not any(k.arg is None for k in n.keywords):
            out.append((n.lineno, n.func.id,
                        f"missing required arg(s): {', '.join(missing)}"))
    return out, len(sigs)


def gate2_partial_binding(tree):
    """MAYBE-UNBOUND locals — 守卫敏感的 definite-assignment 分析。

    实现在 static_defassign.py，与 verify_vit_v29.py 的 G2 闸门共用同一份
    判定逻辑。核心判据是**守卫蕴含**：在守卫 R 处读取 x 安全 <=> 存在一条
    x 的绑定路径，其守卫被 R 蕴含。这样既能放掉
        if use_mixup: ... t_a, t_b, lam = ...
        if use_mixup: ... t_a, t_b, lam ...        (同守卫，运行时安全)
    又能抓到
        if cons_teacher is not None: sigma_max = ...
        if w_eff > 0: f"{sigma_max}"                (异守卫，PFC 模式下必崩)
    """
    return maybe_unbound(tree)


def gate3_history_keys(tree, dict_name="history"):
    """G3 — every `history["k"]` subscript must be initialised in the dict literal.

    v30 shipped with exactly this bug: `history["intra_disp"].append(...)` was
    added but `"intra_disp": []` never made it into the `history = {...}` literal
    -> KeyError on the FIRST batch, i.e. the run dies in seconds — but only after
    you have copied the file over and launched it.

    Cheap, exact, and it catches the whole family: any per-epoch metric that is
    recorded but never declared.
    """
    init = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id == dict_name:
                    for k in n.value.keys:
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            init.add(k.value)
    if not init:
        return []                      # no such dict -> not applicable
    used = {}
    for n in ast.walk(tree):
        if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                and n.value.id == dict_name):
            s = n.slice
            if isinstance(s, ast.Constant) and isinstance(s.value, str):
                used.setdefault(s.value, n.lineno)
    return [(ln, k) for k, ln in sorted(used.items(), key=lambda kv: kv[1])
            if k not in init]


def gate5_unbound(src_text):
    """G5: 读了从没被绑定的名字 —— py_compile 抓不到, 只有跑到那一行才 NameError。

    由来 (2026-09-17, cnn10_v40 e0 崩溃): 给 v40 新增 dump_results() 用 io.open(),
    但 import 块里只补了 json/shutil, 漏了 io ⇒ 第 0 轮刷新 best 时立刻 NameError。
    当时 py_compile + G1/G2/G3 全绿, 一次都没报警 —— 因为语法完全合法,
    名字解析是运行期的事。

    难点在**别喊狼来了**: 收集绑定必须全量先行(定义在 use 之后、定义在 if 分支里
    都算绑定), 类作用域不参与方法内部解析, 推导式自成一域。实现见 scan_unbound.py。

    返回 [(lineno, name)]; 文件里有 from x import * 时返回 [] (解析不可靠, 不拦)。
    """
    hits, star = scan_src(src_text)
    return [] if star else hits


def gate4_disk_guard(tree):
    """G4: 长跑里的写盘必须有"写前查剩余空间"的守卫（**含模块级**）。

    由头 (覆辙 117)：磁盘写满时 torch.save 先在 C++ 侧抛 OSError，随后在**异常展开期**
    的析构函数里二次抛 c10::Error ⇒ std::terminate ⇒ abort(core dumped)，
    **Python 的 try/except 一次都跑不到**。所以唯一可靠的守卫在"进入写盘之前"。

    判定：模块内**每个** torch.save 调用点，其所在函数必须(传递地)引用到磁盘余量查询；
    模块级的写则要求模块级代码（排除 def/class 子树）里出现 disk_usage、或调用已守卫的函数。
    ★ 旧版把"模块级"一律放行，理由是"一次性初始化写盘"——对**本线这种没有 __main__ 守卫、
      训练循环写在模块级的脚本**，这条豁免把最热的一次写盘漏掉了（2026-09-28 实测）。
    """
    funcs = {n.name: n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    def mentions_disk(node):
        for ch in ast.walk(node):
            if (isinstance(ch, ast.Call)
                    and isinstance(ch.func, ast.Attribute)
                    and ch.func.attr == "disk_usage"):
                return True
        return False

    # 传递闭包: 引用过 disk_usage 的函数, 以及调用了这些函数的函数
    guarded = {name for name, f in funcs.items() if mentions_disk(f)}
    changed = True
    while changed:
        changed = False
        for name, f in funcs.items():
            if name in guarded:
                continue
            for ch in ast.walk(f):
                if (isinstance(ch, ast.Call) and isinstance(ch.func, ast.Name)
                        and ch.func.id in guarded):
                    guarded.add(name)
                    changed = True
                    break

    # ★ [2026-09-28 修] 模块级代码也参与守卫判定。
    #   原来这里是 `if info is None: continue  # 模块级一次性写盘, 不拦`。这条豁免的动机是
    #   "初始化期写一次配置文件"，但本线的脚本**没有 __main__ 守卫**、训练循环就写在**模块级**
    #   ⇒ 被豁免掉的恰恰是整跑里最热的那次写盘。实测（CNN v66，2026-09-28）：
    #   三处 torch.save 里有 1 处（模块级循环中的 best_tta 落盘）G4 从来看不见；
    #   而"跑到第 111 轮才崩"的形态就意味着**这类写盘只能在循环里**。
    #   新判定：模块级语句（**排除 def/class 子树**）里出现 disk_usage、或调用已守卫的函数
    #   ⇒ 模块级算守卫；否则模块级 torch.save 一律报出来（函数名记为 `<module-level>`）。
    def _mentions_guarded(node):
        for ch in ast.walk(node):
            if (isinstance(ch, ast.Call) and isinstance(ch.func, ast.Name)
                    and ch.func.id in guarded):
                return True
        return False

    mod_top = [n for n in tree.body
               if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    module_guarded = (any(mentions_disk(n) for n in mod_top)
                      or any(_mentions_guarded(n) for n in mod_top))

    # torch.save 调用点 -> 所属最内层函数
    owner = {}
    for name, f in funcs.items():
        for ch in ast.walk(f):
            if isinstance(ch, ast.Call):
                owner.setdefault(id(ch), (name, f))
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        if not (isinstance(n.func, ast.Attribute) and n.func.attr == "save"):
            continue
        base = n.func.value
        if not (isinstance(base, ast.Name) and base.id == "torch"):
            continue
        info = owner.get(id(n))
        fname = info[0] if info is not None else "<module-level>"
        if info is not None:
            if fname in guarded:
                continue
        elif module_guarded:
            continue
        out.append((n.lineno, fname))
    return sorted(out)


def gate7_literal_selfcheck(tree):
    """G7: 硬编码字面量的自检字段 —— `"<lit>" in str(<CONST>)`。

    这类字段的返回值**只由字面量与常量内容共同决定**。一旦常量被改写、或"轴搬家"
    （自检里写死的那一项换到了别处），它就会静默翻转，且没人会去复查一个"自检"。
    2026-09-24 B 线 v57 实测：`bool("gate1" in str(AXIS_V57).lower())` 在 v51~v56
    恒 True（那些版本的轴恰好在 gate1），本版把轴搬到 gate2 后恒 False
    ⇒ 在同步完全正确的版本里伪造"文案失同步"。

    返回 (definite, suspect)：
      definite —— 常量能被 ast.literal_eval 静态求值且字面量确实不在里面 ⇒ 布尔值已定死。
      suspect  —— 常量求不出值（运行时构造）或不是模块级顶层常量 ⇒ 需要人工核。
    """
    consts = {}
    module_names = set()
    for node in tree.body:
        tgt = None
        val = None
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    module_names.add(t.id)
            if len(node.targets) == 1:
                tgt, val = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                module_names.add(node.target.id)
            tgt, val = node.target, node.value
        elif isinstance(node, (ast.For, ast.With, ast.AugAssign)):
            for t in ast.walk(node):
                if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                    module_names.add(t.id)
        if isinstance(tgt, ast.Name) and val is not None:
            try:
                consts[tgt.id] = ast.literal_eval(val)
            except Exception:                                    # noqa: BLE001
                pass

    definite, suspect = [], []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1
                and isinstance(node.ops[0], ast.In)):
            continue
        lit = node.left
        if not (isinstance(lit, ast.Constant) and isinstance(lit.value, str) and lit.value):
            continue
        name = _unwrap_str_name(node.comparators[0])
        if name is None:
            continue
        if name in consts:
            val = consts[name]
            if isinstance(val, str):
                if lit.value.lower() not in val.lower():
                    definite.append((node.lineno, lit.value, name, val[:70]))
            elif name in module_names:
                suspect.append((node.lineno, lit.value, name))
        elif name in module_names:
            suspect.append((node.lineno, lit.value, name))
        # 其余（函数/循环里的局部名，如 `for n, p in named_parameters()`）与本闸门无关
        # ⇒ 直接跳过，否则每份脚本都会刷出十几条无害 NOTE，闸门会被噪声淹没。
    return definite, suspect


def _unwrap_str_name(node):
    """从 `str(N)`, `str(N).lower()`, `str(N.lower())`, `N.lower()`, `N` 里取出 N。"""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "lower"):
        node = node.func.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "str":
        if len(node.args) != 1:
            return None
        node = node.args[0]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "lower"):
        node = node.func.value
    return node.id if isinstance(node, ast.Name) else None


def gate8_helper_defined_before_module_loop(tree, src):
    """G8: 模块级 for 循环会**先于其后的 def 执行** —— 若循环体（或它调用的方法）
    用到某个模块级私有助手，而该助手 `def` 在循环**之后**，则首步必 NameError。

    2026-10-02 B 线 v71 实测：训练循环 `for epoch in range(0, TRAIN_EPOCHS):` 在 L2219，
    而轴用到的五个谱测量助手 `def` 在 L2447 ⇒ 第一步 `NameError: name '_freq_spectral_mse'
    is not defined`，被 `except Exception` 吞掉 ⇒ `wired=False`，整根轴空转。
    py_compile / 逐函数 md5 / G5 / G6 / scan_top_order **全部抓不到**（它是跨作用域 + 调用时解析）。

    判据（保守，只报确定性形态）：
      1. 找出所有**模块级** `ast.For` 的 lineno，取最小的 = loop_ln；
      2. 收集所有**模块级** `def` 且名字以 `_` 开头的 (name, lineno)；
      3. 若 def_ln > loop_ln，且该 name 在循环之前的区段里被**真切调用过**
         （AST `Call` 的 func 是该 Name —— **不是注释/字符串里的提及**）⇒ 判 FAIL。

    ★ 标定（覆辙 164）：必须对"注释里提到名字"免疫。v72 的 L875/L1641 只在注释里写
      `` `_lr_tail_dose()` ``，其真正调用在 def 之后 ⇒ **不得误报**。
    """
    loop_ln = None
    for n in tree.body:
        if isinstance(n, ast.For):
            loop_ln = n.lineno if loop_ln is None else min(loop_ln, n.lineno)
    if loop_ln is None:
        return []

    # every name that is actually CALLED before the loop (AST-based, comment-immune)
    called_before = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if node.lineno >= loop_ln:
            continue
        fn = node.func
        if isinstance(fn, ast.Name):
            called_before.add(fn.id)
        elif isinstance(fn, ast.Attribute):
            called_before.add(fn.attr)

    out = []
    for n in tree.body:
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not n.name.startswith("_"):
            continue
        if n.lineno <= loop_ln:
            continue
        if n.name in called_before:
            out.append((n.lineno, n.name, loop_ln))
    return sorted(out)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)

    bad_calls, n_fn = gate1_callsites(tree)
    partial = gate2_partial_binding(tree)
    hist_bad = gate3_history_keys(tree)
    disk_bad = gate4_disk_guard(tree)
    unbound = gate5_unbound(src)
    order_bad = scan_top_order_src(src)
    lit_bad, lit_suspect = gate7_literal_selfcheck(tree)
    hlp_bad = gate8_helper_defined_before_module_loop(tree, src)
    n_calls = sum(isinstance(n, ast.Call) for n in ast.walk(tree))

    print(f"lint {path}: {n_fn} local functions, {n_calls} call nodes")
    print()
    if bad_calls:
        print(f"[G1 FAIL] {len(bad_calls)} call-site mismatch(es):")
        for ln, name, msg in sorted(bad_calls):
            print(f"  L{ln}: {name}() -> {msg}")
    else:
        print("[G1 PASS] every local call site matches its signature")
    if partial:
        print(f"[G2 FAIL] {len(partial)} MAYBE-UNBOUND local read(s):")
        for ln, fname, risky in sorted(partial):
            print(f"  L{ln} in {fname}(): {', '.join(risky)} may be unbound on some path")
    else:
        print("[G2 PASS] no MAYBE-UNBOUND local read (guard-sensitive definite-assignment)")
    if hist_bad:
        print(f"[G3 FAIL] {len(hist_bad)} history key(s) used but never initialised:")
        for ln, k in hist_bad:
            print(f"  L{ln}: history['{k}'] — add \"{k}\": [] to the "
                  f"history dict literal")
    else:
        print("[G3 PASS] every history[...] key is initialised in the dict literal")

    if disk_bad:
        print(f"[G4 FAIL] {len(disk_bad)} 个 torch.save 没有写前查盘守卫:")
        for ln, fname in disk_bad:
            print(f"  L{ln} in {fname}(): 磁盘写满会在 C++ 析构里 abort, except 抓不到 "
                  f"(覆辙 117) -> 写之前先 disk_usage 估算余量")
    else:
        print("[G4 PASS] every torch.save inside a function is disk-guarded")
    if unbound:
        print(f"[G5 FAIL] {len(unbound)} 处读了没绑定的名字 (运行时 NameError):")
        for ln, nm in unbound:
            print(f"  L{ln}: {nm} —— 补 import / 补定义; 语法合法所以 py_compile 不报")
    else:
        print("[G5 PASS] no name is read without being bound (no runtime NameError)")

    if order_bad:
        print(f"[G6 FAIL] {len(order_bad)} 处**模块级** use-before-def "
              f"(顺序执行 ⇒ 发车即 NameError, py_compile/G1~G5 都抓不到):")
        for ln, nm, def_ln, via in order_bad:
            _how = ("顶层语句直接读" if via is None
                    else f"经 {via}() 的调用读到")
            print(f"  L{ln}: 读到 `{nm}`，但首次绑定在 L{def_ln}（更靠后）—— {_how} "
                  f"-> 把该常量上移到被引用之前（覆辙 161）")
    else:
        print("[G6 PASS] no module-level use-before-def (top-level order is safe)")

    if lit_bad:
        print(f"[G7 WARN] {len(lit_bad)} 处自检字段的字面量**不可能**出现在它检查的常量里"
              f"（布尔值已定死，与本次改动无关就能翻转）:")
        for ln, lit, name, prev in sorted(lit_bad):
            print(f"  L{ln}: \"{lit}\" in str({name}).lower() -> 恒 False; "
                  f"{name} = \"{prev}...\"")
            print(f"         修：字面量改成由单一真源派生（如 f\"gate{{N}}\"），不要写死"
                  f"（覆辙 138/140）")
    if lit_suspect:
        print(f"[G7 NOTE] {len(lit_suspect)} 处自检字段的字面量无法静态核对"
              f"（常量是运行时构造的）—— 请人工确认它在本版仍然成立:")
        for ln, lit, name in sorted(lit_suspect):
            print(f"  L{ln}: \"{lit}\" in str({name})")
    if not lit_bad and not lit_suspect:
        print("[G7 PASS] no hardcoded-literal self-check over a module constant")

    if hlp_bad:
        print(f"[G8 FAIL] {len(hlp_bad)} 个模块级私有助手定义在**模块级 for 循环之后**，"
              f"但循环之前已被引用 ⇒ 首步必 NameError（覆辙 213）:")
        for def_ln, name, loop_ln in hlp_bad:
            print(f"  L{def_ln}: def {name}(...) 在训练循环 L{loop_ln} 之后，"
                  f"但循环前已调用 `{name}(` -> 把它的 def 上移到循环之前")
    else:
        print("[G8 PASS] no axis helper is defined after a module-level training loop "
              "(no first-step NameError)")

    bad = bool(bad_calls or partial or hist_bad or disk_bad or unbound or order_bad or hlp_bad)
    print()
    print("=== RESULT: " + ("DO NOT LAUNCH — fix the above first"
                           if bad else "clean, safe to launch") + " ===")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
