"""P11：权重量化（INT4/INT8）。

原理一句话：把每一组（group_size 个）相邻的 bf16 权重，用一个 scale 和一个
零点（biases）线性映射成低位整数；矩阵乘时由 `mx.quantized_matmul` 现场反
量化。瓶颈在内存带宽——Apple Silicon 解码每步都要把整个权重矩阵从内存搬到
计算单元，权重从 16-bit 压到 4-bit，搬运量降到 ~1/4，解码就快、显存也省。

边界（符合本项目硬约束）：我们只借用 MLX 的三个**张量算子**
  mx.quantize / mx.dequantize / mx.quantized_matmul
它们和 mx.matmul、mx.fast.rope 是同一层的积木。"哪些层量化、按什么分组打包、
前向怎么接线"这些引擎逻辑全部自己写——下面的 QuantizedLinear 就是手写的。
"""

from __future__ import annotations

from .backend import mx, nn


class QuantizedLinear(nn.Module):
    """量化版线性层：y = x @ dequant(W).T + bias。

    权重以三件套存储：
      weight  —— 打包成 uint32 的低位整数（每个 32-bit 槽塞多个权重）
      scales  —— 每组一个缩放系数
      biases  —— 每组一个零点偏移
    前向不显式还原整个 W，而是用 quantized_matmul 在乘法内部按组反量化，
    省去一次完整的反量化与回写。
    """

    def __init__(self, q_weight, scales, biases, bias, group_size: int, bits: int):
        super().__init__()
        self.weight = q_weight
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        if bias is not None:
            self.bias = bias

    @classmethod
    def from_linear(cls, lin: nn.Linear, group_size: int, bits: int) -> "QuantizedLinear":
        # mx.quantize 沿权重最后一维（输入维）每 group_size 个分一组
        q, scales, biases = mx.quantize(lin.weight, group_size=group_size, bits=bits)
        bias = lin.bias if ("bias" in lin) else None
        return cls(q, scales, biases, bias, group_size, bits)

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.quantized_matmul(
            x, self.weight, scales=self.scales, biases=self.biases,
            transpose=True, group_size=self.group_size, bits=self.bits,
        )
        if "bias" in self:
            y = y + self.bias
        return y


class QuantizedEmbedding(nn.Module):
    """量化版词嵌入。词表矩阵 [vocab, dim] 是 tied 模型里最大的单块权重，
    且每个 decode 步都被当 lm_head 复用一次（h @ W.T 过整个词表），所以
    量化它能直接砍 decode 的内存流量。

    两种用法都要支持：
      __call__(ids) —— 查表：gather 选中行的量化三件套，再反量化回向量
      as_linear(x)  —— 当输出投影：直接 quantized_matmul，不还原整张表
    """

    def __init__(self, q_weight, scales, biases, group_size: int, bits: int):
        super().__init__()
        self.weight = q_weight
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits

    @classmethod
    def from_embedding(cls, emb: nn.Embedding, group_size: int, bits: int) -> "QuantizedEmbedding":
        q, scales, biases = mx.quantize(emb.weight, group_size=group_size, bits=bits)
        return cls(q, scales, biases, group_size, bits)

    def __call__(self, ids: mx.array) -> mx.array:
        # 只反量化被查到的那些行（按 leading 维 gather，dequantize 自动广播）
        q = self.weight[ids]
        s = self.scales[ids]
        b = self.biases[ids]
        return mx.dequantize(q, s, b, group_size=self.group_size, bits=self.bits)

    def as_linear(self, x: mx.array) -> mx.array:
        return mx.quantized_matmul(
            x, self.weight, scales=self.scales, biases=self.biases,
            transpose=True, group_size=self.group_size, bits=self.bits)


def quantize_module(module: nn.Module, group_size: int = 64, bits: int = 4,
                    embed: bool = True) -> int:
    """递归遍历模块树，把每个 nn.Linear 换成 QuantizedLinear、
    （可选）每个 nn.Embedding 换成 QuantizedEmbedding。

    nn.Module 是 dict 子类：它的子模块/参数都是字典项，可直接遍历、就地改写。
    返回被量化的模块数，便于核验真生效。
    """
    count = 0
    for name, child in list(module.items()):
        if isinstance(child, nn.Linear):
            module[name] = QuantizedLinear.from_linear(child, group_size, bits)
            count += 1
        elif embed and isinstance(child, nn.Embedding):
            module[name] = QuantizedEmbedding.from_embedding(child, group_size, bits)
            count += 1
        elif isinstance(child, nn.Module):
            count += quantize_module(child, group_size, bits, embed)
        elif isinstance(child, list):
            for item in child:
                if isinstance(item, nn.Module):
                    count += quantize_module(item, group_size, bits, embed)
    return count
