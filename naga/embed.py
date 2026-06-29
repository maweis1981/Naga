"""文本嵌入器：把文字编码成归一化向量（P8/P9 的语义检索地基）。

用我们手写的 BERT 前向，CLS 池化 + L2 归一化得到句向量。归一化后，
两个向量的"余弦相似度"就等于它们的点积，检索时算点积即可。
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

from .loader import ensure_local
from .models.bert import BertArgs, BertModel


class Embedder:
    def __init__(self, model_id: str = "BAAI/bge-small-zh-v1.5"):
        path = ensure_local(model_id)
        cfg = json.loads((path / "config.json").read_text())
        a = BertArgs(
            hidden_size=cfg["hidden_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            intermediate_size=cfg["intermediate_size"],
            vocab_size=cfg["vocab_size"],
            max_position_embeddings=cfg["max_position_embeddings"],
            type_vocab_size=cfg.get("type_vocab_size", 2),
            layer_norm_eps=cfg.get("layer_norm_eps", 1e-12),
        )
        self.model = BertModel(a)

        weights: dict[str, mx.array] = {}
        for f in sorted(path.glob("*.safetensors")):
            weights.update(mx.load(str(f)))
        # 只留我们用到的，丢掉 pooler / position_ids 等
        weights = {k: v for k, v in weights.items() if k.startswith(("embeddings.", "encoder."))}
        self.model.load_weights(list(weights.items()), strict=False)
        mx.eval(self.model.parameters())

        tj = path / "tokenizer.json"
        if tj.exists():
            self.tok = Tokenizer.from_file(str(tj))
        else:
            from tokenizers import BertWordPieceTokenizer
            self.tok = BertWordPieceTokenizer(str(path / "vocab.txt"), lowercase=False)

        self.dim = a.hidden_size
        self.model_id = model_id

    def encode(self, texts, batch_size: int = 32):
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        vecs: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encs = [self.tok.encode(t).ids[:512] for t in batch]
            maxlen = max(len(e) for e in encs)
            ids = [e + [0] * (maxlen - len(e)) for e in encs]
            mask = [[1] * len(e) + [0] * (maxlen - len(e)) for e in encs]

            h = self.model(mx.array(ids), mx.array(mask))    # [B, L, H]
            cls = h[:, 0]                                     # CLS 池化
            cls = cls * mx.rsqrt(mx.sum(cls * cls, axis=-1, keepdims=True) + 1e-12)  # L2 归一化
            mx.eval(cls)
            vecs.extend(cls.tolist())

        return vecs[0] if single else vecs


_SHARED: Embedder | None = None


def get_embedder() -> Embedder:
    """全局共享一个嵌入器（首次调用时才加载，Memory 和 RAG 共用）。"""
    global _SHARED
    if _SHARED is None:
        _SHARED = Embedder()
    return _SHARED
