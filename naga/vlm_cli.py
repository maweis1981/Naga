"""多模态命令行：给一张图 + 一个问题，让引擎"看图说话"。

用法：
    python -m naga.vlm_cli 图片.jpg "这张图里有什么？"
"""

from __future__ import annotations

import argparse
import time


def build_ids(tok: ChatTokenizer, question: str, image_token: int) -> list[int]:
    # 在 ChatML 的 user 段里，把图像占位符 token 夹在文本中间
    head = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    tail = "\n" + question + "<|im_end|>\n<|im_start|>assistant\n"
    return tok.encode(head) + [image_token] + tok.encode(tail)


def main():
    ap = argparse.ArgumentParser(description="Naga 多模态：看图说话")
    ap.add_argument("image", help="图片路径 / URL / data: base64")
    ap.add_argument("prompt", nargs="?", default="详细描述这张图片。")
    ap.add_argument("--model", default="llava-hf/llava-interleave-qwen-0.5b-hf")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.0)
    args = ap.parse_args()

    # 延迟导入 MLX 相关模块，让帮助命令和安装检查不依赖可用 GPU。
    from .generate import generate_vlm
    from .image import preprocess
    from .loader import load_vlm
    from .tokenizer import ChatTokenizer

    print(f"⏳ 加载多模态模型 {args.model} ...", flush=True)
    t = time.perf_counter()
    model, (va, ta, image_token), path = load_vlm(args.model)
    tok = ChatTokenizer(path)
    print(f"✓ 加载完成（{time.perf_counter()-t:.1f}s）。视觉塔 {va.num_hidden_layers} 层 / "
          f"每图 {(va.image_size//va.patch_size)**2} 个视觉 token\n")

    pixel_values = preprocess(args.image)
    input_ids = build_ids(tok, args.prompt, image_token)

    print(f"🖼️  {args.image}")
    print(f"👤 {args.prompt}")
    print("🤖 ", end="", flush=True)

    gen_ids: list[int] = []
    shown = ""
    t0 = time.perf_counter()
    for tid in generate_vlm(model, input_ids, pixel_values, args.max_tokens, tok.eos_ids, args.temp):
        gen_ids.append(tid)
        text = tok.decode(gen_ids)
        print(text[len(shown):], end="", flush=True)
        shown = text
    dt = time.perf_counter() - t0
    speed = len(gen_ids) / dt if dt > 0 else 0
    print(f"\n\n— 生成 {len(gen_ids)} tok | {speed:.1f} tok/s")


if __name__ == "__main__":
    main()
