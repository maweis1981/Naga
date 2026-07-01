"""A/B：串行逐条生成 vs 批量同批前向。测多条并发请求的总吞吐差异（P3）。

批量把 B 条序列打进一次前向，GPU 一次算 B 行——decode 是内存带宽受限的，权重只从显存
读一遍就服务了 B 条序列，故聚合吞吐随 B 提升（受显存与算力上限约束）。数值正确性见
tests/test_batched.py（fp32 下批量与串行逐位一致）。
"""
from __future__ import annotations

import time
from collections import defaultdict

from naga.engine import Engine
from naga.generate import batched_generate, generate

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = [
    "Name three primary colors.",
    "What is the capital of Japan?",
    "Write one sentence about the ocean.",
    "List two programming languages.",
    "Give me a synonym for happy.",
    "What is 12 times 12?",
]
N = 40


def main():
    eng = Engine(MODEL, prefix_cache=False)
    tok, model, eos = eng.tok, eng.model, eng.tok.eos_ids
    ids = [tok.encode(tok.apply_chat_template([{"role": "user", "content": p}])) for p in PROMPTS]

    # 串行：一条一条生成
    t0 = time.perf_counter()
    serial_tokens = 0
    for p in ids:
        serial_tokens += sum(1 for _ in generate(model, p, N, eos, temp=0.0))
    serial_s = time.perf_counter() - t0

    # 批量：一次前向服务全部
    t0 = time.perf_counter()
    bat = defaultdict(int)
    for b, _ in batched_generate(model, ids, N, eos, temp=0.0):
        bat[b] += 1
    batched_tokens = sum(bat.values())
    batched_s = time.perf_counter() - t0

    print(f"并发请求数 B = {len(ids)}，每条最多生成 {N} tokens\n")
    print(f"{'方式':<8}{'总生成 tok':>12}{'总耗时(s)':>12}{'聚合 tok/s':>14}")
    print(f"{'串行':<8}{serial_tokens:>12}{serial_s:>12.2f}{serial_tokens/serial_s:>14.1f}")
    print(f"{'批量':<8}{batched_tokens:>12}{batched_s:>12.2f}{batched_tokens/batched_s:>14.1f}")
    print(f"\n聚合吞吐提升：{(batched_tokens/batched_s)/(serial_tokens/serial_s):.2f}×")


if __name__ == "__main__":
    main()
