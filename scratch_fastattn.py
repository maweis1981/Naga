"""A/B：手写注意力 vs 融合快路径。先验数值等价（正确性），再比 prefill 速度。"""
from __future__ import annotations
import time
import mlx.core as mx
from naga.loader import load_model
from naga.tokenizer import ChatTokenizer
from naga.generate import generate

MODEL = "Qwen/Qwen2.5-3B-Instruct"
LENS = [256, 1024, 2048, 4096]


def prefill_ms(model, ids, n=3):
    x = mx.array([ids])
    c = model.make_caches(); mx.eval(model(x, c))   # 预热
    best = 1e9
    for _ in range(n):
        c = model.make_caches()
        t = time.perf_counter()
        mx.eval(model(x, c))
        best = min(best, (time.perf_counter() - t) * 1000)
    return best


def first_logits(model, ids):
    x = mx.array([ids])
    c = model.make_caches()
    out = model(x, c)[0, -1]
    mx.eval(out)
    return out


def main():
    print("加载 Q4 + 手写注意力 ...")
    m_slow, margs, path = load_model(MODEL, quantize=True, bits=4, fast_attn=False)
    print("加载 Q4 + 融合快路径 ...")
    m_fast, _, _ = load_model(MODEL, quantize=True, bits=4, fast_attn=True)
    tk = ChatTokenizer(path)

    # —— 正确性：同输入下两条路径 logits 应几乎一致 ——
    ids = tk.encode(tk.apply_chat_template(
        [{"role": "user", "content": "用一句话解释相对论。"}]))
    a = first_logits(m_slow, ids)
    b = first_logits(m_fast, ids)
    diff = float(mx.max(mx.abs(a - b)).item())
    same_argmax = int(mx.argmax(a).item()) == int(mx.argmax(b).item())
    print(f"\n正确性检查：logits 最大绝对差 = {diff:.4f}，argmax 一致 = {same_argmax}")

    # 实际生成对比（贪心，应逐字相同）
    g1 = list(generate(m_slow, ids, 40, tk.eos_ids, temp=0.0))
    g2 = list(generate(m_fast, ids, 40, tk.eos_ids, temp=0.0))
    print(f"手写: {tk.decode(g1)}")
    print(f"快路: {tk.decode(g2)}")
    print(f"逐字相同 = {g1 == g2}")

    # —— prefill 速度随长度 ——
    base = tk.encode("人工智能是研究如何让机器具备智能的科学与工程领域。")
    print(f"\n{'L':>6}{'手写(ms)':>12}{'快路(ms)':>12}{'加速':>8}")
    for L in LENS:
        seq = (base * (L // len(base) + 1))[:L]
        ts = prefill_ms(m_slow, seq)
        tf = prefill_ms(m_fast, seq)
        print(f"{L:>6}{ts:>12.1f}{tf:>12.1f}{ts/tf:>7.2f}x")


if __name__ == "__main__":
    main()
