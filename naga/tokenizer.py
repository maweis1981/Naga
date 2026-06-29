"""分词器 + 对话模板。

分词（把文字切成 BPE 子词）本身是个独立的大坑，我们用 HuggingFace 的
`tokenizers` 库直接加载模型自带的 tokenizer.json——这只是查表，不属于
"推理引擎"的范畴。但**对话模板**（怎么把多轮消息拼成模型认识的格式）
我们自己手写，因为这正是 OpenAI 风格 messages 和底层 token 之间的桥梁。
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer

# Qwen 使用 ChatML 格式：每条消息用 <|im_start|>{role}\n{content}<|im_end|> 包裹
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


class ChatTokenizer:
    def __init__(self, model_path: Path):
        self.tok = Tokenizer.from_file(str(Path(model_path) / "tokenizer.json"))
        # 生成停止符：助手说完会吐 <|im_end|>，整段文本结束是 <|endoftext|>
        self.eos_ids = tuple(
            i for i in (self.tok.token_to_id(IM_END), self.tok.token_to_id("<|endoftext|>"))
            if i is not None
        )

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def apply_chat_template(self, messages: list[dict]) -> str:
        """messages: [{"role": "system|user|assistant", "content": "..."}].

        末尾留一个 "<|im_start|>assistant\\n" 提示模型开始作答。
        """
        if not any(m["role"] == "system" for m in messages):
            messages = [{"role": "system", "content": DEFAULT_SYSTEM}] + messages
        parts = [f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in messages]
        parts.append(f"{IM_START}assistant\n")
        return "".join(parts)
