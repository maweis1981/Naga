"""文档库 + RAG 检索（P9）。

把文档切成小块、各自嵌入成向量，提问时检索最相关的块注入上下文。
"源"可以是单个文件，也可以是一个目录（本地文件夹，或 macOS 已挂载的
网络盘 /Volumes/...）——注册目录后会递归索引里面所有受支持的文档。

支持格式：.txt / .md / .pdf / .csv / .log 等纯文本与 PDF。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from .embed import get_embedder

DOC_FILE = Path.home() / ".naga" / "documents.json"
SUPPORTED = {".txt", ".md", ".markdown", ".pdf", ".csv", ".log", ".text", ".json"}


def extract_text(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return p.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int = 400, overlap: int = 80) -> list[str]:
    text = text.strip()
    out, i = [], 0
    while i < len(text):
        piece = text[i:i + size].strip()
        if piece:
            out.append(piece)
        i += size - overlap
    return out


class DocumentStore:
    def __init__(self):
        self.sources: list[str] = []
        self.chunks: list[dict] = []
        if DOC_FILE.exists():
            try:
                d = json.loads(DOC_FILE.read_text())
                self.sources = d.get("sources", [])
                self.chunks = d.get("chunks", [])
            except Exception:
                pass

    def _save(self):
        DOC_FILE.parent.mkdir(exist_ok=True)
        DOC_FILE.write_text(json.dumps({"sources": self.sources, "chunks": self.chunks},
                                       ensure_ascii=False))

    def _ingest_file(self, path: str) -> dict:
        p = Path(path).resolve()
        full = str(p)
        try:
            text = extract_text(full)
        except Exception as e:
            return {"name": p.name, "ok": False, "error": str(e)}
        pieces = chunk_text(text)
        if not pieces:
            return {"name": p.name, "ok": False, "error": "无可提取文本"}
        # 同一文件重新导入时先清掉旧块
        self.chunks = [c for c in self.chunks if c["path"] != full]
        vecs = get_embedder().encode(pieces)
        for piece, vec in zip(pieces, vecs):
            self.chunks.append({"id": uuid.uuid4().hex[:12], "path": full,
                                "name": p.name, "text": piece, "vec": vec})
        return {"name": p.name, "ok": True, "chunks": len(pieces)}

    def add(self, path: str) -> dict:
        """自动判断：文件就导入它，目录就注册为源并递归索引。"""
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return {"ok": False, "error": f"路径不存在: {p}"}
        if p.is_file():
            r = self._ingest_file(str(p))
            self._save()
            return {"ok": r["ok"], "files": [r]}
        # 目录
        if str(p) not in self.sources:
            self.sources.append(str(p))
        results = [self._ingest_file(str(f)) for f in sorted(p.rglob("*"))
                   if f.is_file() and f.suffix.lower() in SUPPORTED]
        self._save()
        return {"ok": True, "dir": str(p), "files": results}

    def search(self, query: str, top_k: int = 4, min_score: float = 0.3) -> list[dict]:
        if not self.chunks or not (query or "").strip():
            return []
        qv = get_embedder().encode(query)
        scored = [(sum(a * b for a, b in zip(qv, c["vec"])), c) for c in self.chunks]
        scored.sort(key=lambda x: -x[0])
        return [{"name": c["name"], "text": c["text"], "score": round(s, 3)}
                for s, c in scored[:top_k] if s >= min_score]

    def list_docs(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for c in self.chunks:
            d = docs.setdefault(c["path"], {"name": c["name"], "path": c["path"], "chunks": 0})
            d["chunks"] += 1
        return list(docs.values())

    def remove(self, path: str) -> bool:
        n = len(self.chunks)
        self.chunks = [c for c in self.chunks if c["path"] != path]
        self.sources = [s for s in self.sources if s != path]
        self._save()
        return len(self.chunks) != n or path not in self.sources

    def state(self) -> dict:
        return {"sources": self.sources, "docs": self.list_docs(),
                "total_chunks": len(self.chunks)}
