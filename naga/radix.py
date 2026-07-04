"""P13：RadixAttention —— 跨请求的前缀 KV 缓存。

痛点：每个新请求都把整段提示从头 prefill。但多轮对话的 system prompt+历史、
RAG 的同一批检索块、Agent 每轮工具循环的固定前缀，都在**重复**——这些 token
的 KV 上一轮已经算过了。

思路：把历史请求的 KV 按 token 前缀存进一棵**基数树（radix tree）**。
  - 每条边携带一段连续 token，以及这段 token 对应的 KV 切片（每层一份）。
  - 新请求来时，从根沿树匹配最长公共前缀；命中部分的 KV 直接复用，
    只对未命中的后缀做 prefill。
  - 按 token 总量做 LRU 淘汰，控制显存。

这是纯"引擎逻辑"（一棵树 + KV 的增删查淘汰），不碰任何算子边界——正是本
项目该自研的层。底座是 cache.py 的 KVCache，KV 按 axis=2（序列维）切片/拼接。
"""

from __future__ import annotations

from .backend import mx


class _Node:
    """树节点。tokens 是父->本节点这条边上的 token；kv 是这段 token 的 KV 切片。"""
    __slots__ = ("children", "tokens", "kv", "parent", "last_access")

    def __init__(self, parent):
        self.children: dict[int, _Node] = {}   # 首 token -> 子节点
        self.tokens: list[int] = []            # 边上的 token 序列
        self.kv = None                         # list[(K,V)] 每层一份；根为 None
        self.parent = parent
        self.last_access = 0


def _slice_kv(kv_list, a: int, b: int):
    """把每层 (K,V) 沿序列维切出 [a,b)，并 eval 实体化（算一次、存起来）。"""
    out = []
    for K, V in kv_list:
        k, v = K[:, :, a:b, :], V[:, :, a:b, :]
        mx.eval(k, v)
        out.append((k, v))
    return out


class RadixCache:
    def __init__(self, n_layers: int, max_tokens: int = 16384):
        self.root = _Node(None)
        self.n_layers = n_layers
        self.max_tokens = max_tokens       # 缓存 token 上限，超了就 LRU 淘汰
        self.cached_tokens = 0
        self.clock = 0

    def _tick(self) -> int:
        self.clock += 1
        return self.clock

    # ---- 匹配：返回 (命中长度, 命中前缀的逐层 KV) ----
    def match(self, token_ids: list[int]):
        node = self.root
        i = 0
        path = []          # 沿途每条边的 kv（按层）
        while i < len(token_ids):
            child = node.children.get(token_ids[i])
            if child is None:
                break
            edge = child.tokens
            j = 0
            while j < len(edge) and i + j < len(token_ids) and edge[j] == token_ids[i + j]:
                j += 1
            if j == len(edge):
                path.append(child.kv)          # 整条边命中，下沉
                node = child
                node.last_access = self._tick()
                i += j
            else:
                if j > 0:                       # 部分命中：只取这条边前 j 个
                    path.append(_slice_kv(child.kv, 0, j))
                    i += j
                break
        if i == 0:
            return 0, None
        # 沿路径把每层 KV 拼成完整前缀
        kv = []
        for L in range(self.n_layers):
            ks = mx.concatenate([edge[L][0] for edge in path], axis=2)
            vs = mx.concatenate([edge[L][1] for edge in path], axis=2)
            kv.append((ks, vs))
        return i, kv

    # ---- 插入：把一段完整序列的 KV（在 caches 里）存进树 ----
    def insert(self, token_ids: list[int], caches):
        node = self.root
        i = 0
        while i < len(token_ids):
            child = node.children.get(token_ids[i])
            if child is None:
                break
            edge = child.tokens
            j = 0
            while j < len(edge) and i + j < len(token_ids) and edge[j] == token_ids[i + j]:
                j += 1
            if j == len(edge):
                node = child
                i += j
            else:
                node = self._split(node, child, j)   # 边中途分叉，先劈开
                i += j
                break
        if i < len(token_ids):                       # 把未命中后缀挂成新边
            new = _Node(node)
            new.tokens = list(token_ids[i:])
            kv_full = [(c.keys, c.values) for c in caches]
            new.kv = _slice_kv(kv_full, i, len(token_ids))
            new.last_access = self._tick()
            node.children[token_ids[i]] = new
            self.cached_tokens += len(new.tokens)
            self._evict()

    def _split(self, parent: _Node, child: _Node, j: int) -> _Node:
        """把 child 的边在第 j 个 token 处劈成两段，返回上半段（新中间节点）。"""
        mid = _Node(parent)
        mid.tokens = child.tokens[:j]
        mid.kv = _slice_kv(child.kv, 0, j)
        mid.last_access = child.last_access
        child.tokens = child.tokens[j:]
        child.kv = _slice_kv(child.kv, j, j + len(child.tokens))
        child.parent = mid
        mid.children[child.tokens[0]] = child
        parent.children[mid.tokens[0]] = mid
        return mid

    # ---- LRU 淘汰：超额时摘掉最久没碰过的叶子 ----
    def _evict(self):
        while self.cached_tokens > self.max_tokens:
            leaf = self._lru_leaf()
            if leaf is None or leaf.parent is None:
                break
            self.cached_tokens -= len(leaf.tokens)
            del leaf.parent.children[leaf.tokens[0]]

    def _lru_leaf(self) -> _Node | None:
        best, best_t = None, None
        stack = [self.root]
        while stack:
            n = stack.pop()
            if not n.children and n.parent is not None:      # 叶子
                if best_t is None or n.last_access < best_t:
                    best, best_t = n, n.last_access
            stack.extend(n.children.values())
        return best
