"""本地 Memory：持久化记忆 + 语义检索（P8）。

每条记忆存一段文字和它的嵌入向量，落盘到 ~/.naga/memory.json。
对话时拿用户的话去做语义检索，把最相关的几条记忆注入到上下文，
模型就能"记住"之前告诉过它的事。
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from .embed import get_embedder

MEM_FILE = Path.home() / ".naga" / "memory.json"


class MemoryStore:
    def __init__(self):
        self.items: list[dict] = []
        if MEM_FILE.exists():
            try:
                self.items = json.loads(MEM_FILE.read_text())
            except Exception:
                self.items = []

    def _save(self):
        MEM_FILE.parent.mkdir(exist_ok=True)
        MEM_FILE.write_text(json.dumps(self.items, ensure_ascii=False, indent=2))

    def add(self, text: str) -> dict | None:
        text = (text or "").strip()
        if not text:
            return None
        vec = get_embedder().encode(text)
        item = {"id": uuid.uuid4().hex[:12], "text": text, "ts": int(time.time()), "vec": vec}
        self.items.append(item)
        self._save()
        return {"id": item["id"], "text": text, "ts": item["ts"]}

    def delete(self, mem_id: str) -> bool:
        n = len(self.items)
        self.items = [i for i in self.items if i["id"] != mem_id]
        if len(self.items) != n:
            self._save()
            return True
        return False

    def list(self) -> list[dict]:
        return [{"id": i["id"], "text": i["text"], "ts": i["ts"]}
                for i in sorted(self.items, key=lambda x: -x["ts"])]

    def search(self, query: str, top_k: int = 3, min_score: float = 0.35) -> list[dict]:
        """语义检索：返回与 query 最相关、且分数过阈值的记忆。"""
        if not self.items or not (query or "").strip():
            return []
        qv = get_embedder().encode(query)
        scored = [(sum(a * b for a, b in zip(qv, i["vec"])), i) for i in self.items]
        scored.sort(key=lambda x: -x[0])
        return [{"text": i["text"], "score": round(s, 3)}
                for s, i in scored[:top_k] if s >= min_score]
