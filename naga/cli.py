"""命令行入口：跑通"加载模型 -> 套对话模板 -> 流式生成 -> 报告速度"。

用法：
    python -m naga.cli "用三句话介绍一下你自己"
    python -m naga.cli --model Qwen/Qwen2.5-0.5B-Instruct --max-tokens 128 "你好"
"""

from __future__ import annotations

import argparse
import time


def main():
    ap = argparse.ArgumentParser(description="Naga —— 从零自研的 MLX 推理引擎")
    ap.add_argument("prompt", nargs="?", default="用三句话介绍一下你自己。", help="用户输入")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF 仓库名或本地路径")
    ap.add_argument("--system", default=None, help="可选的 system 提示")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.0, help="0=贪心，>0=温度采样")
    ap.add_argument("--top-p", type=float, default=1.0, help="nucleus 采样阈值")
    ap.add_argument("--top-k", type=int, default=0, help="只在前 k 个里采，0=不限")
    ap.add_argument("--quantize", action="store_true", help="加载时把线性层量化（默认 4-bit）")
    ap.add_argument("--bits", type=int, default=4, help="量化位宽（4 或 8），需配合 --quantize")
    ap.add_argument("--fast-attn", action="store_true", help="走融合注意力快路径（默认手写参考路径）")
    args = ap.parse_args()

    # 延迟导入重依赖，让 `--help`、打包检查、基础安装验证在无 Metal 环境下也可用。
    from .generate import generate
    from .loader import load_model
    from .tokenizer import ChatTokenizer

    print(f"⏳ 加载模型 {args.model} ...", flush=True)
    t = time.perf_counter()
    model, margs, path = load_model(args.model, quantize=args.quantize,
                                    bits=args.bits, fast_attn=args.fast_attn)
    tok = ChatTokenizer(path)
    print(f"✓ 加载完成（{time.perf_counter() - t:.1f}s），"
          f"{margs.num_hidden_layers} 层 / {margs.hidden_size} 维 / 词表 {margs.vocab_size}\n")

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    prompt_ids = tok.encode(tok.apply_chat_template(messages))
    print(f"👤 {args.prompt}")
    print(f"🤖 ", end="", flush=True)

    gen_ids: list[int] = []
    shown = ""
    t0 = time.perf_counter()
    first_token_at = None
    for tid in generate(model, prompt_ids, args.max_tokens, tok.eos_ids,
                        args.temp, args.top_p, args.top_k):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        gen_ids.append(tid)
        text = tok.decode(gen_ids)
        print(text[len(shown):], end="", flush=True)
        shown = text
    dt = time.perf_counter() - t0

    n = len(gen_ids)
    ttft = (first_token_at - t0) if first_token_at else 0.0
    speed = n / dt if dt > 0 else 0.0
    print(f"\n\n— 提示 {len(prompt_ids)} tok | 生成 {n} tok | "
          f"首字 {ttft*1000:.0f}ms | {speed:.1f} tok/s")


if __name__ == "__main__":
    main()
