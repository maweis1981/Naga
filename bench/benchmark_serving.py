#!/usr/bin/env python3
"""Naga 服务基准测试（LLM serving benchmark）。

指标口径对齐业界标准（vLLM benchmark_serving / LLMPerf）：

  - TTFT  (Time To First Token)   首 token 延迟：请求发出到收到第 1 个内容 token
  - TPOT  (Time Per Output Token) 每输出 token 平均耗时 = (E2E - TTFT) / (输出tok - 1)
  - ITL   (Inter-Token Latency)   相邻 token 到达间隔（解码平顺度）
  - E2E   (End-to-End latency)    单请求端到端总耗时
  - Output token throughput       系统聚合输出吞吐 = 总生成 token / 墙钟时长
  - Request throughput            请求吞吐 = 成功请求数 / 墙钟时长

对每个指标报告 mean / median / p99。压测通过标准 OpenAI 流式接口
（/v1/chat/completions, stream=true, stream_options.include_usage=true），
用 include_usage 帧拿到精确 token 数，用内容帧到达时刻算 TTFT/ITL。

纯标准库（urllib + 线程），无额外依赖。

用法：
  python bench/benchmark_serving.py --url http://127.0.0.1:8131 \
      --model Qwen/Qwen2.5-0.5B-Instruct --num-prompts 32 --concurrency 8 --max-tokens 128
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

PROMPTS = [
    "Explain what a KV cache is in two sentences.",
    "Write a short poem about the ocean.",
    "List five uses for a paperclip.",
    "Summarize the theory of relativity for a child.",
    "What are the benefits of unit testing?",
    "Describe how photosynthesis works.",
    "Give me three tips for writing clean code.",
    "What is the difference between TCP and UDP?",
    "Explain recursion with a simple example.",
    "Why is the sky blue? Answer briefly.",
]


@dataclass
class Result:
    ok: bool = False
    ttft: float = 0.0                 # s
    e2e: float = 0.0                  # s
    prompt_tokens: int = 0
    output_tokens: int = 0
    itls: list[float] = field(default_factory=list)   # s，相邻 token 间隔
    error: str = ""


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    idx = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[idx]


def one_request(url: str, model: str, prompt: str, max_tokens: int, temperature: float) -> Result:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    r = Result()
    start = time.perf_counter()
    last = start
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                if obj.get("usage"):
                    r.prompt_tokens = obj["usage"].get("prompt_tokens", 0)
                    r.output_tokens = obj["usage"].get("completion_tokens", 0)
                for ch in obj.get("choices", []):
                    delta = ch.get("delta", {})
                    if delta.get("content"):
                        now = time.perf_counter()
                        if r.ttft == 0.0:
                            r.ttft = now - start
                        else:
                            r.itls.append(now - last)
                        last = now
        r.e2e = time.perf_counter() - start
        r.ok = r.output_tokens > 0 or r.ttft > 0
    except Exception as e:
        r.error = f"{type(e).__name__}: {e}"
    return r


def run_benchmark(url, model, num_prompts, concurrency, max_tokens, temperature):
    tasks = [PROMPTS[i % len(PROMPTS)] for i in range(num_prompts)]
    results: list[Result] = []
    bench_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(one_request, url, model, p, max_tokens, temperature) for p in tasks]
        for f in as_completed(futs):
            results.append(f.result())
    bench_dur = time.perf_counter() - bench_start
    return results, bench_dur


def _stat_row(name, xs_ms):
    return (f"{name:<28}{statistics.mean(xs_ms):>10.2f}{_percentile(xs_ms,50):>10.2f}"
            f"{_percentile(xs_ms,99):>10.2f}") if xs_ms else f"{name:<28}{'—':>10}{'—':>10}{'—':>10}"


def report(results, bench_dur, concurrency, max_tokens):
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    total_out = sum(r.output_tokens for r in ok)
    total_in = sum(r.prompt_tokens for r in ok)

    ttft_ms = [r.ttft * 1000 for r in ok]
    e2e_ms = [r.e2e * 1000 for r in ok]
    tpot_ms = [(r.e2e - r.ttft) / (r.output_tokens - 1) * 1000
               for r in ok if r.output_tokens > 1]
    itl_ms = [x * 1000 for r in ok for x in r.itls]

    print("\n" + "=" * 62)
    print("  Naga serving benchmark")
    print("=" * 62)
    print(f"  并发 (concurrency)          {concurrency}")
    print(f"  成功 / 总请求               {len(ok)} / {len(results)}"
          + (f"  (失败 {len(failed)})" if failed else ""))
    print(f"  墙钟时长 (s)                {bench_dur:.2f}")
    print(f"  每请求 max_tokens           {max_tokens}")
    print(f"  总输入 / 输出 token         {total_in} / {total_out}")
    print("-" * 62)
    print(f"  请求吞吐 (req/s)            {len(ok)/bench_dur:>10.2f}")
    print(f"  输出吞吐 (tok/s)           {total_out/bench_dur:>10.2f}")
    print(f"  总吞吐   (tok/s)           {(total_in+total_out)/bench_dur:>10.2f}")
    print("-" * 62)
    print(f"  {'指标 (ms)':<26}{'mean':>10}{'median':>10}{'p99':>10}")
    print("  " + _stat_row("TTFT  首token延迟", ttft_ms))
    print("  " + _stat_row("TPOT  每输出token", tpot_ms))
    print("  " + _stat_row("ITL   token间隔", itl_ms))
    print("  " + _stat_row("E2E   端到端", e2e_ms))
    print("=" * 62)
    if failed:
        print(f"  示例错误: {failed[0].error}")


def main():
    ap = argparse.ArgumentParser(description="Naga LLM serving benchmark")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="naga")
    ap.add_argument("--num-prompts", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    for _ in range(args.warmup):                       # 预热：排除首次加载/JIT
        one_request(args.url, args.model, "hello", 8, 0.0)

    results, dur = run_benchmark(args.url, args.model, args.num_prompts,
                                 args.concurrency, args.max_tokens, args.temperature)
    report(results, dur, args.concurrency, args.max_tokens)


if __name__ == "__main__":
    main()
