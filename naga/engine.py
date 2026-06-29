"""Engine：把"模型 + 分词器 + 生成循环"封装成一个稳定的服务对象。

服务层（HTTP）只和 Engine 打交道，不关心 MLX 细节。Engine 负责：
  - 启动时加载一次模型，常驻内存；
  - 把 OpenAI 风格的 messages 套上对话模板、编码成 token；
  - 流式产出文本增量，并在结束时报告 token 用量（计费/统计要用）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

from .generate import generate, generate_cached, generate_vlm
from .image import preprocess
from .loader import load_model, load_vlm
from .monitor import monitor
from .radix import RadixCache
from .tokenizer import ChatTokenizer

# 私有区字符，仅用作"图片插入点"的内部占位标记
_IMG_MARK = ""


@dataclass
class Chunk:
    """流式输出的一块。done=True 的那块不带文本，只带最终用量统计。"""
    delta: str = ""
    done: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Engine:
    def __init__(self, model_id: str, prefix_cache: bool = True,
                 quantize: bool = False, bits: int = 4):
        t0 = time.perf_counter()
        self.model, self.args, path = load_model(model_id, quantize=quantize, bits=bits)
        self.tok = ChatTokenizer(path)
        self.model_id = model_id
        # P13：前缀 KV 缓存。多轮对话/RAG/Agent 的固定前缀只 prefill 一次，
        # 后续请求复用，TTFT 不再随历史变长。数值忠实（teacher-forcing 逐位一致）。
        self.radix = RadixCache(self.args.num_hidden_layers) if prefix_cache else None
        monitor.emit("model_load", model=model_id, kind_detail="text",
                     quantize=(f"Q{bits}" if quantize else "bf16"),
                     layers=self.args.num_hidden_layers,
                     load_s=round(time.perf_counter() - t0, 2),
                     prefix_cache=prefix_cache)

    def stream(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temp: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> Iterator[Chunk]:
        prompt_ids = self.tok.encode(self.tok.apply_chat_template(messages))
        monitor.emit("request", model=self.model_id, turns=len(messages),
                     prompt_tokens=len(prompt_ids))

        gen_ids: list[int] = []
        shown = ""
        stream = (
            generate_cached(self.model, prompt_ids, self.radix, max_tokens,
                            self.tok.eos_ids, temp, top_p, top_k)
            if self.radix is not None
            else generate(self.model, prompt_ids, max_tokens, self.tok.eos_ids, temp, top_p, top_k)
        )
        t0 = time.perf_counter()
        first_at = None
        for tid in stream:
            if first_at is None:
                first_at = time.perf_counter()      # 首字时刻 = prefill 结束
            gen_ids.append(tid)
            # 增量解码：解码全部已生成 id，取出比上次多出来的那段文本，
            # 这样能正确处理"一个汉字由多个 token 拼成"的情况。
            text = self.tok.decode(gen_ids)
            # 多字节字符还没拼完整时（解码出现 U+FFFD 替换符），先不吐出，等下一个 token
            if text.endswith("�"):
                continue
            delta = text[len(shown):]
            shown = text
            if delta:
                yield Chunk(delta=delta)

        # 生成统计：TTFT（prefill 速度）+ decode tok/s（你 P11/P13 优化的直接体现）
        n = len(gen_ids)
        ttft_ms = (first_at - t0) * 1000 if first_at else 0.0
        decode_s = (time.perf_counter() - first_at) if first_at else 0.0
        tok_s = (n - 1) / decode_s if decode_s > 0 and n > 1 else 0.0
        monitor.emit("generation", model=self.model_id,
                     prompt_tokens=len(prompt_ids), completion_tokens=n,
                     ttft_ms=round(ttft_ms), decode_tok_s=round(tok_s, 1))
        yield Chunk(done=True, prompt_tokens=len(prompt_ids), completion_tokens=len(gen_ids))


class VlmEngine:
    """多模态引擎：在 Engine 基础上支持 OpenAI 的图文混合 messages。

    OpenAI 多模态格式：content 是一个数组，元素可为
      {"type":"text","text":...} 或 {"type":"image_url","image_url":{"url":...}}
    我们解析出文本和（第一张）图片，按 ChatML 拼好、在图片处插入占位 token。
    """

    def __init__(self, model_id: str):
        self.model, (self.va, self.ta, self.image_token), path = load_vlm(model_id)
        self.tok = ChatTokenizer(path)
        self.model_id = model_id

    def _build(self, messages: list[dict]):
        """把 messages 拼成 ChatML，返回 (input_ids, pixel_values|None)。"""
        image = None
        parts: list[str] = []
        if not any(m["role"] == "system" for m in messages):
            parts.append("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")

        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n")
            content = m["content"]
            if isinstance(content, str):
                parts.append(content)
            else:  # 数组形式（图文混合）
                for c in content:
                    if c.get("type") == "text":
                        parts.append(c["text"])
                    elif c.get("type") == "image_url" and image is None:
                        image = c["image_url"]["url"]      # 只取第一张图
                        parts.append(_IMG_MARK)
            parts.append("<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        text = "".join(parts)

        if image is None:
            return self.tok.encode(text), None
        before, after = text.split(_IMG_MARK, 1)
        ids = self.tok.encode(before) + [self.image_token] + self.tok.encode(after)
        return ids, preprocess(image)

    def stream(self, messages, max_tokens=512, temp=0.0, top_p=1.0, top_k=0) -> Iterator[Chunk]:
        input_ids, pixel_values = self._build(messages)
        gen_ids: list[int] = []
        shown = ""
        for tid in generate_vlm(self.model, input_ids, pixel_values,
                                max_tokens, self.tok.eos_ids, temp, top_p, top_k):
            gen_ids.append(tid)
            text = self.tok.decode(gen_ids)
            # 多字节字符还没拼完整时（解码出现 U+FFFD 替换符），先不吐出，等下一个 token
            if text.endswith("�"):
                continue
            delta = text[len(shown):]
            shown = text
            if delta:
                yield Chunk(delta=delta)
        yield Chunk(done=True, prompt_tokens=len(input_ids), completion_tokens=len(gen_ids))
