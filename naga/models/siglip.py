"""SigLIP 视觉编码器 —— 把一张图变成一串"视觉 token"向量。

和文本 Transformer 的关键区别：
  1. 入口是"切 patch"：用一个步长=patch 的卷积把 384×384 图切成 27×27=729 块，
     每块投影成一个 1152 维向量 —— 这就是图片的"token"。
  2. 注意力是**双向**的（看图不需要因果掩码，每个 patch 能看到所有 patch）。
  3. 用 **LayerNorm**（带 bias）而不是 RMSNorm，激活是 gelu_tanh。
  4. 位置编码是**可学习的**位置嵌入（不是 RoPE）。

MLX 的 Conv2d 用 NHWC（通道在最后）布局，所以图片张量是 [B,H,W,3]，
卷积权重加载时也要从 PyTorch 的 [O,I,kh,kw] 转成 [O,kh,kw,I]（在 loader 里做）。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..backend import mx, nn


@dataclass
class VisionArgs:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    image_size: int
    patch_size: int
    layer_norm_eps: float


def gelu_tanh(x: mx.array) -> mx.array:
    # gelu_pytorch_tanh：GELU 的 tanh 近似，SigLIP 的 MLP 激活
    return 0.5 * x * (1.0 + mx.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


class VisionEmbeddings(nn.Module):
    def __init__(self, a: VisionArgs):
        super().__init__()
        self.num_patches = (a.image_size // a.patch_size) ** 2
        # 步长=核大小的卷积 = 不重叠地切 patch 并投影
        self.patch_embedding = nn.Conv2d(3, a.hidden_size, a.patch_size, stride=a.patch_size)
        self.position_embedding = nn.Embedding(self.num_patches, a.hidden_size)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        # pixel_values: [B, H, W, 3] (NHWC)
        patches = self.patch_embedding(pixel_values)          # [B, 27, 27, 1152]
        B = patches.shape[0]
        patches = patches.reshape(B, -1, patches.shape[-1])   # [B, 729, 1152]
        pos = mx.arange(self.num_patches)
        return patches + self.position_embedding(pos)


class VisionAttention(nn.Module):
    """标准双向多头自注意力（无掩码、无 RoPE、无 GQA）。"""

    def __init__(self, a: VisionArgs):
        super().__init__()
        self.n_heads = a.num_attention_heads
        self.head_dim = a.hidden_size // a.num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(a.hidden_size, a.hidden_size)
        self.k_proj = nn.Linear(a.hidden_size, a.hidden_size)
        self.v_proj = nn.Linear(a.hidden_size, a.hidden_size)
        self.out_proj = nn.Linear(a.hidden_size, a.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(out)


class VisionMLP(nn.Module):
    def __init__(self, a: VisionArgs):
        super().__init__()
        self.fc1 = nn.Linear(a.hidden_size, a.intermediate_size)
        self.fc2 = nn.Linear(a.intermediate_size, a.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(gelu_tanh(self.fc1(x)))


class VisionLayer(nn.Module):
    """预归一化 + 残差，结构和文本层一致，只是 norm 换成 LayerNorm。"""

    def __init__(self, a: VisionArgs):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(a.hidden_size, eps=a.layer_norm_eps)
        self.self_attn = VisionAttention(a)
        self.layer_norm2 = nn.LayerNorm(a.hidden_size, eps=a.layer_norm_eps)
        self.mlp = VisionMLP(a)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, a: VisionArgs):
        super().__init__()
        self.layers = [VisionLayer(a) for _ in range(a.num_hidden_layers)]


class SiglipVisionModel(nn.Module):
    def __init__(self, a: VisionArgs):
        super().__init__()
        self.embeddings = VisionEmbeddings(a)
        self.encoder = Encoder(a)
        self.post_layernorm = nn.LayerNorm(a.hidden_size, eps=a.layer_norm_eps)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        h = self.embeddings(pixel_values)
        for layer in self.encoder.layers:
            h = layer(h)
        # LLaVA 用 vision_feature_layer=-1：取最后一层输出，**不过** post_layernorm
        return h


class VisionTower(nn.Module):
    """外层包一下，让权重命名对上 vision_tower.vision_model.*"""

    def __init__(self, a: VisionArgs):
        super().__init__()
        self.vision_model = SiglipVisionModel(a)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        return self.vision_model(pixel_values)
