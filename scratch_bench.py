"""A/B benchmark: Naga (bf16) vs ollama (Q4 / fp16), same model & prompt.

只测引擎本身的两个关键速度：
  - prefill：吃完提示、吐出第一个 token 的耗时（TTFT）
  - decode ：之后每秒能产多少 token（tok/s）
外加 Naga 侧的 MLX 峰值显存。
全程 greedy(temp=0) 保证两边输出可比、结果稳定。
"""
from __future__ import annotations
import time, json, urllib.request
import mlx.core as mx
from naga.loader import load_model
from naga.generate import generate
from naga.tokenizer import ChatTokenizer

MODEL = "Qwen/Qwen2.5-3B-Instruct"
# 用一个稍长的提示，让 prefill 阶段有足够工作量、TTFT 可测
PROMPT = ("请详细解释什么是 KV-Cache，它在自回归生成里解决了什么性能问题，"
          "为什么预填充和解码要分两段处理。用条理清晰的中文回答。")
MAX_TOKENS = 200
RUNS = 3   # 取最好一轮（排除抖动/冷启动）


def bench_naga(quantize=False, bits=4, label="bf16"):
    print(f"⏳ Naga 加载 {MODEL} ({label}) ...", flush=True)
    t = time.perf_counter()
    model, margs, path = load_model(MODEL, quantize=quantize, bits=bits)
    tok = ChatTokenizer(path)
    load_s = time.perf_counter() - t
    prompt_ids = tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}]))

    best = None
    for r in range(RUNS):
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        first = None
        n = 0
        for tid in generate(model, prompt_ids, MAX_TOKENS, tok.eos_ids, temp=0.0):
            if first is None:
                first = time.perf_counter()
            n += 1
        dt = time.perf_counter() - t0
        ttft = (first - t0)
        decode_s = dt - ttft
        tok_s = (n - 1) / decode_s if decode_s > 0 and n > 1 else 0.0
        peak_gb = mx.get_peak_memory() / 1e9
        rec = dict(prompt_tok=len(prompt_ids), gen_tok=n, ttft_ms=ttft * 1000,
                   tok_s=tok_s, peak_gb=peak_gb)
        print(f"  run{r+1}: TTFT {ttft*1000:.0f}ms | decode {tok_s:.1f} tok/s | peak {peak_gb:.2f}GB")
        if best is None or rec["tok_s"] > best["tok_s"]:
            best = rec
    best["load_s"] = load_s
    return best


def bench_ollama(tag: str):
    url = "http://localhost:11434/api/generate"
    # 套 Qwen ChatML，让两边提示尽量等价；greedy
    body = {
        "model": tag,
        "prompt": PROMPT,
        "stream": False,
        "options": {"temperature": 0, "num_predict": MAX_TOKENS, "seed": 0},
    }
    best = None
    for r in range(RUNS):
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(req).read())
        # ollama 用纳秒计时
        ttft_ms = d["prompt_eval_duration"] / 1e6  # prefill 总耗时≈首字
        tok_s = d["eval_count"] / (d["eval_duration"] / 1e9)
        rec = dict(prompt_tok=d.get("prompt_eval_count", 0), gen_tok=d["eval_count"],
                   ttft_ms=ttft_ms, tok_s=tok_s)
        print(f"  run{r+1}: TTFT {ttft_ms:.0f}ms | decode {tok_s:.1f} tok/s")
        if best is None or rec["tok_s"] > best["tok_s"]:
            best = rec
    return best


if __name__ == "__main__":
    print("=" * 60)
    res = {}
    res["Naga (bf16)"] = bench_naga(quantize=False, label="bf16")
    print()
    res["Naga (Q8 自研)"] = bench_naga(quantize=True, bits=8, label="Q8 自研")
    print()
    res["Naga (Q4 自研)"] = bench_naga(quantize=True, bits=4, label="Q4 自研")
    print(f"\n⏳ ollama qwen2.5:3b (Q4_K_M) ...")
    res["ollama (Q4)"] = bench_ollama("qwen2.5:3b")
    print(f"\n⏳ ollama qwen2.5:3b-instruct-fp16 ...")
    res["ollama (fp16)"] = bench_ollama("qwen2.5:3b-instruct-fp16")

    print("\n" + "=" * 60)
    print(f"{'引擎':<18}{'提示tok':>8}{'生成tok':>8}{'TTFT(ms)':>10}{'tok/s':>9}{'峰值GB':>9}")
    for k, v in res.items():
        peak = f"{v.get('peak_gb', 0):.2f}" if v.get("peak_gb") else "—"
        print(f"{k:<18}{v['prompt_tok']:>8}{v['gen_tok']:>8}{v['ttft_ms']:>10.0f}{v['tok_s']:>9.1f}{peak:>9}")
    print(json.dumps(res, ensure_ascii=False, indent=2))
