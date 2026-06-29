"""模型超参数。从 HuggingFace 的 config.json 读出来，避免任何硬编码。

每个推理引擎的第一件事都是把"模型长什么样"读进来：有多少层、
每层多少个注意力头、词表多大、用什么位置编码……这些数字决定了
后面所有矩阵的形状。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelArgs:
    hidden_size: int          # 每个 token 向量的维度 d_model
    intermediate_size: int    # 前馈网络中间层维度（通常是 hidden 的若干倍）
    num_hidden_layers: int    # Decoder 层数
    num_attention_heads: int  # 注意力头数（query 头）
    num_key_value_heads: int  # KV 头数；< query 头数即 GQA（分组查询注意力）
    vocab_size: int           # 词表大小
    rms_norm_eps: float       # RMSNorm 防止除零的小量
    rope_theta: float         # RoPE 的底数，决定位置编码的"波长"
    head_dim: int             # 每个头的维度 = hidden / num_heads（一般）
    tie_word_embeddings: bool # 输入词嵌入和输出 lm_head 是否共享权重
    fast_attn: bool = False   # 运行期开关：True=走融合注意力快路径（非模型超参）


def load_config(model_path: Path) -> ModelArgs:
    cfg = json.loads((Path(model_path) / "config.json").read_text())
    head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    return ModelArgs(
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg.get("rope_theta", 1_000_000.0),
        head_dim=head_dim,
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
    )
