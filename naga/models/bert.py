"""从零实现 BERT 编码器（用于文本嵌入，P8/P9 共用）。

和我们写过的 ViT/LLM 的关键区别：
  1. **后归一化**：BERT 是 `LayerNorm(x + sublayer(x))`，归一化在残差相加之后
     （LLM/ViT 是预归一化 `x + sublayer(LayerNorm(x))`）。
  2. 三种嵌入相加：词嵌入 + 位置嵌入 + 段落(token_type)嵌入。
  3. 双向注意力 + padding 掩码（句子不等长时忽略补位）。
  4. 句向量 = 取 [CLS]（第 0 个 token）的最后隐藏态，再 L2 归一化（在 embed.py 里做）。

模块属性名（attention.self.query / attention.output.dense ...）刻意对齐
HuggingFace BertModel 的权重命名。
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class BertArgs:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    type_vocab_size: int
    layer_norm_eps: float


def gelu(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.erf(x * 0.7071067811865476))


class BertEmbeddings(nn.Module):
    def __init__(self, a: BertArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(a.vocab_size, a.hidden_size)
        self.position_embeddings = nn.Embedding(a.max_position_embeddings, a.hidden_size)
        self.token_type_embeddings = nn.Embedding(a.type_vocab_size, a.hidden_size)
        self.LayerNorm = nn.LayerNorm(a.hidden_size, eps=a.layer_norm_eps)

    def __call__(self, ids: mx.array) -> mx.array:
        L = ids.shape[1]
        pos = mx.arange(L)
        tok_type = mx.zeros((L,), dtype=mx.int32)   # 单段输入，全 0
        e = (self.word_embeddings(ids)
             + self.position_embeddings(pos)
             + self.token_type_embeddings(tok_type))
        return self.LayerNorm(e)


class BertSelfAttention(nn.Module):
    def __init__(self, a: BertArgs):
        super().__init__()
        self.n_heads = a.num_attention_heads
        self.head_dim = a.hidden_size // a.num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(a.hidden_size, a.hidden_size)
        self.key = nn.Linear(a.hidden_size, a.hidden_size)
        self.value = nn.Linear(a.hidden_size, a.hidden_size)

    def __call__(self, x: mx.array, mask) -> mx.array:
        B, L, _ = x.shape
        q = self.query(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.key(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.value(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        scores = scores.astype(mx.float32)
        if mask is not None:
            scores = scores + mask
        w = mx.softmax(scores, axis=-1).astype(v.dtype)
        return (w @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)


class BertSelfOutput(nn.Module):
    def __init__(self, a: BertArgs):
        super().__init__()
        self.dense = nn.Linear(a.hidden_size, a.hidden_size)
        self.LayerNorm = nn.LayerNorm(a.hidden_size, eps=a.layer_norm_eps)

    def __call__(self, h, inp):
        return self.LayerNorm(self.dense(h) + inp)   # 后归一化 + 残差


class BertAttention(nn.Module):
    def __init__(self, a: BertArgs):
        super().__init__()
        self.self = BertSelfAttention(a)             # 对齐权重名 attention.self.*
        self.output = BertSelfOutput(a)

    def __call__(self, x, mask):
        return self.output(self.self(x, mask), x)


class BertIntermediate(nn.Module):
    def __init__(self, a: BertArgs):
        super().__init__()
        self.dense = nn.Linear(a.hidden_size, a.intermediate_size)

    def __call__(self, x):
        return gelu(self.dense(x))


class BertOutput(nn.Module):
    def __init__(self, a: BertArgs):
        super().__init__()
        self.dense = nn.Linear(a.intermediate_size, a.hidden_size)
        self.LayerNorm = nn.LayerNorm(a.hidden_size, eps=a.layer_norm_eps)

    def __call__(self, h, inp):
        return self.LayerNorm(self.dense(h) + inp)


class BertLayer(nn.Module):
    def __init__(self, a: BertArgs):
        super().__init__()
        self.attention = BertAttention(a)
        self.intermediate = BertIntermediate(a)
        self.output = BertOutput(a)

    def __call__(self, x, mask):
        a_ = self.attention(x, mask)
        return self.output(self.intermediate(a_), a_)


class BertEncoder(nn.Module):
    def __init__(self, a: BertArgs):
        super().__init__()
        self.layer = [BertLayer(a) for _ in range(a.num_hidden_layers)]


class BertModel(nn.Module):
    def __init__(self, a: BertArgs):
        super().__init__()
        self.embeddings = BertEmbeddings(a)
        self.encoder = BertEncoder(a)

    def __call__(self, ids: mx.array, attention_mask=None) -> mx.array:
        x = self.embeddings(ids)
        mask = None
        if attention_mask is not None:
            # [B,L] 的 1/0 -> 加性掩码 [B,1,1,L]，补位处置 -inf
            mask = (1.0 - attention_mask.astype(mx.float32))[:, None, None, :] * (-1e9)
        for layer in self.encoder.layer:
            x = layer(x, mask)
        return x   # [B, L, H]
