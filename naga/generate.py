"""自回归生成循环（P1：带 KV-Cache 的两段式 + 采样）。

两段式：
  1) 预填充 prefill —— 第一次调用喂入整段提示，一次算完、填满缓存；
  2) 解码 decode    —— 之后每次只喂 1 个新 token，复用缓存，O(1) 推进。

缓存的 offset 会随 update() 自动增长，模型据此给新 token 排正确的位置。
"""

from __future__ import annotations

from typing import Iterator

import mlx.core as mx


def _sample(logits: mx.array, temp: float, top_p: float, top_k: int) -> int:
    """从 logits 采样一个 token。

    temp=0  -> 贪心（取最大）
    top_k   -> 只在分数最高的 k 个里采
    top_p   -> 只在累计概率达到 p 的"核"里采（nucleus sampling）
    """
    if temp <= 0.0:
        return int(mx.argmax(logits).item())

    logits = logits.astype(mx.float32) * (1.0 / temp)

    order = mx.argsort(-logits)          # 按 logit 从大到小的下标排列
    s_logits = logits[order]
    n = s_logits.shape[0]

    if top_k and top_k > 0:
        keep = mx.arange(n) < top_k
        s_logits = mx.where(keep, s_logits, -1e9)

    if top_p < 1.0:
        probs = mx.softmax(s_logits)
        cum = mx.cumsum(probs)
        keep = (cum - probs) < top_p     # 第一个永远保留，逐个纳入直到累计达 p
        s_logits = mx.where(keep, s_logits, -1e9)

    choice = mx.random.categorical(s_logits)   # 在排序空间里抽
    return int(order[choice].item())           # 映射回原始 token id


def generate(
    model,
    prompt_ids: list[int],
    max_tokens: int = 256,
    eos_ids: tuple[int, ...] = (),
    temp: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> Iterator[int]:
    caches = model.make_caches()
    tokens = mx.array([prompt_ids])     # 第一次：整段提示（预填充）

    for _ in range(max_tokens):
        logits = model(tokens, caches)[0, -1]      # [vocab]，最后一个位置
        next_id = _sample(logits, temp, top_p, top_k)
        if next_id in eos_ids:
            break
        yield next_id
        tokens = mx.array([[next_id]])             # 之后每次：只喂新 token（解码）


def generate_cached(
    model,
    prompt_ids: list[int],
    radix,
    max_tokens: int = 256,
    eos_ids: tuple[int, ...] = (),
    temp: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> Iterator[int]:
    """带前缀缓存的生成（P13）：命中前缀只复用、不重算，prefill 只跑后缀。

    流程：match 最长前缀 -> 把命中的 KV 塞进缓存 -> 只喂未命中后缀做 prefill ->
    照常解码 -> 把"整段提示+生成"的 KV 回填进树，供后续请求复用。
    """
    matched, kv = radix.match(prompt_ids)

    # 至少要留 1 个 token 现场 prefill：模型得靠它产出下一个 logits
    if matched >= len(prompt_ids):
        matched = len(prompt_ids) - 1
        kv = [(K[:, :, :matched, :], V[:, :, :matched, :]) for (K, V) in kv] if matched > 0 else None

    # 监视：本次前缀缓存命中多少（P13 复用效果的直接体现）
    from .monitor import monitor
    _total = len(prompt_ids)
    monitor.emit("prefix_cache", matched=matched, total=_total,
                 reuse=round(matched / _total, 3) if _total else 0.0,
                 prefilled=_total - matched)

    caches = model.make_caches()
    if matched > 0:
        for L, c in enumerate(caches):
            c.keys, c.values = kv[L]                 # 直接把命中前缀的 KV 装进缓存

    tokens = mx.array([prompt_ids[matched:]])        # 只喂未命中后缀
    gen_ids: list[int] = []
    for _ in range(max_tokens):
        logits = model(tokens, caches)[0, -1]
        next_id = _sample(logits, temp, top_p, top_k)
        if next_id in eos_ids:
            break
        gen_ids.append(next_id)
        yield next_id
        tokens = mx.array([[next_id]])

    # 回填：以缓存里"真实存在的位置数"为准截断。
    # 关键：若因 max_tokens 截断，最后一个 yield 的 token 尚未被 forward，
    # 它的 KV 不在缓存里——按 offset 截断才不会把缺失/越界的 KV 存进树。
    cached_len = caches[0].offset
    full = (list(prompt_ids) + gen_ids)[:cached_len]
    radix.insert(full, caches)


def _ban(logits: mx.array, idx: int) -> mx.array:
    """把某个 token 的 logit 压到极小，等效从候选里剔除（不可再被 argmax 选中）。"""
    n = logits.shape[0]
    return mx.where(mx.arange(n) == idx, mx.array(-1e9, logits.dtype), logits)


def _constrained_pick(logits, gen_ids, committed, decode, constraint, eos_ids):
    """在约束下选一个 token：从 logit 最高开始试，第一个"文本增量全程合法"的即选中。

    非法 token 当场屏蔽再取下一个——这就是约束解码的本质：把违反语法的 token
    从采样分布里抹掉，模型只能在合法 token 中挑。选中即让约束机沿其字符前进。
    """
    work = logits
    for _ in range(64):                       # 最多试 64 个候选，足够避开非法 token
        idx = int(mx.argmax(work).item())
        if idx in eos_ids:
            return idx if constraint.complete() else None  # 只有已合法收尾才允许停
        delta = decode(gen_ids + [idx])[len(committed):]
        if not delta:                         # 零宽 token，跳过以防死循环
            work = _ban(work, idx); continue
        snap = constraint.snapshot()
        ok = True
        for ch in delta:
            if constraint.complete():         # 已收尾后多出的字符：只容忍尾随空白
                if ch in " \t\n\r":
                    continue
                ok = False; break
            if not constraint.step(ch):
                ok = False; break
        if ok:
            return idx                        # 约束机已随 delta 前进，直接采纳
        constraint.restore(snap)
        work = _ban(work, idx)
    return None


def generate_constrained(
    model,
    prompt_ids: list[int],
    decode,
    constraint,
    max_tokens: int = 128,
    eos_ids: tuple[int, ...] = (),
    ) -> Iterator[int]:
    """约束解码生成（P14）：每步只在"保持输出合法"的 token 里采，约束收尾即停。

    decode: 把 token id 列表还原成文本的函数（用来取每个候选 token 的文本增量）。
    constraint: 状态机（如 ToolCallConstraint），提供 step/snapshot/restore/complete。
    """
    caches = model.make_caches()
    tokens = mx.array([prompt_ids])
    gen_ids: list[int] = []
    committed = ""
    for _ in range(max_tokens):
        logits = model(tokens, caches)[0, -1]
        idx = _constrained_pick(logits, gen_ids, committed, decode, constraint, eos_ids)
        if idx is None or idx in eos_ids:
            break
        gen_ids.append(idx)
        committed = decode(gen_ids)
        yield idx
        tokens = mx.array([[idx]])
        if constraint.complete():             # 合法 JSON 已闭合，到此为止
            break


def generate_vlm(
    model,
    input_ids: list[int],
    pixel_values,
    max_tokens: int = 256,
    eos_ids: tuple[int, ...] = (),
    temp: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> Iterator[int]:
    """多模态生成：预填充时把图片一起喂进去，之后照常增量解码。"""
    caches = model.make_caches()
    tokens = mx.array([input_ids])
    px = pixel_values                              # 只在第一次（预填充）用到

    for _ in range(max_tokens):
        logits = model(tokens, pixel_values=px, caches=caches)[0, -1]
        next_id = _sample(logits, temp, top_p, top_k)
        if next_id in eos_ids:
            break
        yield next_id
        tokens = mx.array([[next_id]])
        px = None
