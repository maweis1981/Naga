"""从零实现 Qwen2 / Qwen2.5 的前向计算（基于 MLX 张量算子）。

我们只借用 MLX 提供的"积木"——矩阵乘法、softmax、Embedding 容器——
注意力、RoPE、GQA、SwiGLU 这些"引擎逻辑"全部手写。模块的属性名
（self_attn / q_proj / input_layernorm ...）刻意和 HuggingFace 权重
里的命名一一对应，这样加载权重时能直接对号入座。
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..config import ModelArgs


def silu(x: mx.array) -> mx.array:
    # SiLU(x) = x * sigmoid(x)，SwiGLU 的非线性门控
    return x * mx.sigmoid(x)


class RMSNorm(nn.Module):
    """RMSNorm：只按"均方根"缩放，不像 LayerNorm 那样减均值，更省算力。

    x_norm = x / sqrt(mean(x^2) + eps) * weight
    在 float32 下算以保证数值稳定，再转回原 dtype。
    """

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dt = x.dtype
        x = x.astype(mx.float32)
        norm = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (self.weight * norm).astype(dt)


def precompute_rope(seq_len: int, head_dim: int, theta: float, offset: int = 0):
    """预先算好每个位置的旋转角（cos/sin）。

    RoPE 的核心思想：不给 token 加一个"位置向量"，而是把 query/key
    向量按位置"旋转"一个角度。两个 token 的注意力分数只依赖它们的
    相对位置——这让模型能外推到比训练更长的序列。

    offset：增量解码时，新 token 的全局位置从已缓存长度开始，
    而不是从 0 开始——否则位置编码会错乱。
    """
    pos = mx.arange(offset, offset + seq_len, dtype=mx.float32)      # [L]
    inv_freq = 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    freqs = mx.outer(pos, inv_freq)                                  # [L, head_dim/2]
    emb = mx.concatenate([freqs, freqs], axis=-1)                   # [L, head_dim]
    return mx.cos(emb), mx.sin(emb)


def causal_mask(seq_len: int, offset: int) -> mx.array:
    """因果掩码，同时覆盖预填充和增量解码两种情形。

    query 的全局位置是 [offset, offset+L)，key 覆盖 [0, offset+L)。
    位置 i 只能看到 j <= i 的 key。解码时 L=1，新 token 能看到全部历史。
    """
    total = offset + seq_len
    q_idx = mx.arange(offset, total)[:, None]      # [L, 1]
    k_idx = mx.arange(total)[None, :]              # [1, total]
    return mx.where(k_idx <= q_idx, 0.0, -1e9).astype(mx.float32)


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    # x: [B, H, L, D]；cos/sin: [L, D]。NeoX 风格的 rotate_half。
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    """多头自注意力 + GQA + RoPE + 因果掩码。

    GQA（Grouped-Query Attention）：query 有很多头，但 key/value 头数更少，
    多个 query 头共享一组 KV。这样 KV-Cache 体积大幅缩小，是省显存的关键。
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5
        self.fast = args.fast_attn   # True=融合快路径，False=手写参考路径（默认）

        # Qwen2 的 q/k/v 投影带 bias，o_proj 不带 —— 这是该架构的特点
        self.q_proj = nn.Linear(args.hidden_size, self.n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, args.hidden_size, bias=False)

    def __call__(self, x: mx.array, cos, sin, mask, cache=None) -> mx.array:
        B, L, _ = x.shape

        # 投影并拆成多头：[B, L, H*D] -> [B, H, L, D]
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # 给 q、k 注入位置信息（新 token 的位置已经带上了 offset）
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # 把新 token 的 K/V 追加进缓存，取回包含全部历史的 K/V
        if cache is not None:
            k, v = cache.update(k, v)

        if self.fast:
            # —— 融合快路径 ——
            # mx.fast.scaled_dot_product_attention 是 MLX 提供的融合算子：
            # 内部分块计算 + 在线 softmax（Flash-Attention 思路），永不物化
            # 完整的 [H, L, L] 分数矩阵，prefill 的 O(L²) 显存爆炸由此消失；
            # 且原生支持 GQA（q 头多于 kv 头无需 repeat）。掩码沿用手写路径
            # 同一个加性 mask（转成计算 dtype，否则 SDPA 拒绝 fp32→bf16 降级）。
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=mask.astype(q.dtype))
        else:
            # —— 手写参考路径（默认，吃透原理用）——
            # GQA：把 KV 头复制到和 query 头一样多
            if self.n_kv_heads < self.n_heads:
                repeats = self.n_heads // self.n_kv_heads
                k = mx.repeat(k, repeats, axis=1)
                v = mx.repeat(v, repeats, axis=1)

            # 注意力分数 -> 加因果掩码（在 fp32 下做 softmax 更稳）
            scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
            scores = scores.astype(mx.float32) + mask
            weights = mx.softmax(scores, axis=-1).astype(v.dtype)
            out = weights @ v                                # [B, H, L, D]

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)    # 合并回 [B, L, H*D]
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU 前馈网络：down( silu(gate(x)) * up(x) )。

    比传统的 ReLU-MLP 多了一条"门控"通路（gate），表达力更强，
    是现代 LLM（Llama/Qwen 系）的标配。
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """一层 Decoder：注意力子层 + 前馈子层，都用"预归一化 + 残差连接"。

    残差连接 x + f(norm(x)) 让梯度能直通几十层而不消失。
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(self, x, cos, sin, mask, cache=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask, cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen2Model(nn.Module):
    """主干：词嵌入 -> N 层 Decoder -> 最终归一化。输出每个位置的隐藏状态。"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(self, input_ids=None, caches=None, inputs_embeds=None) -> mx.array:
        # 多模态时直接传入已拼好视觉向量的 embeds，跳过词嵌入查表
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        B, L = h.shape[0], h.shape[1]

        # 有缓存时，新 token 的全局起点 = 已缓存长度
        offset = caches[0].offset if caches is not None else 0

        cos, sin = precompute_rope(L, self.args.head_dim, self.args.rope_theta, offset)
        cos, sin = cos.astype(h.dtype), sin.astype(h.dtype)
        mask = causal_mask(L, offset)

        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin, mask, caches[i] if caches is not None else None)
        return self.norm(h)


class Model(nn.Module):
    """完整语言模型：主干 + lm_head（输出层）。

    tie_word_embeddings=True 时，输出层和输入词嵌入共享同一份权重
    （小模型常用，省一大块参数）。
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model = Qwen2Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, input_ids=None, caches=None, inputs_embeds=None) -> mx.array:
        h = self.model(input_ids, caches, inputs_embeds)
        if self.args.tie_word_embeddings:
            # 复用词嵌入矩阵做输出投影：h @ embed.weight.T
            return self.model.embed_tokens.as_linear(h)
        return self.lm_head(h)

    def make_caches(self):
        from ..cache import KVCache
        return [KVCache() for _ in range(self.args.num_hidden_layers)]
