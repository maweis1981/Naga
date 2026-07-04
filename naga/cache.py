"""KV-Cache：推理引擎性能的地基。

每一层注意力维护一份自己的 K/V 历史。预填充（prefill）阶段一次性把整段
提示的 K/V 算出来塞进缓存；解码（decode）阶段每步只算 1 个新 token 的
K/V 并追加进来。这样历史不再重算，生成从 O(n²) 降到 O(n)。

存的是"GQA 复制之前"的 KV（n_kv_heads 份），更省内存；复制到 query
头数的动作放在注意力里、读出缓存之后再做。
"""

from __future__ import annotations

from .backend import mx


class KVCache:
    def __init__(self):
        self.keys: mx.array | None = None
        self.values: mx.array | None = None

    @property
    def offset(self) -> int:
        # 已缓存的序列长度，也就是新 token 的起始位置
        return 0 if self.keys is None else self.keys.shape[2]

    def update(self, keys: mx.array, values: mx.array):
        # keys/values: [B, n_kv_heads, L_new, head_dim]
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        return self.keys, self.values
