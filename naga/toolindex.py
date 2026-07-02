"""MCP 工具索引 + 意图路由（工具语义检索）。

问题：工具一多（多个 MCP server 累计几十个），把**全部**工具描述塞进 system prompt
会淹没小模型的决策、还拖慢 prefill。做法：把每个工具的描述用自研 bge 嵌入器编码成向量、
在本地持久化；用户提问时对意图做语义检索，只把**最相关的 top-k** 工具交给模型。

复用 Naga 已有的嵌入器（naga.embed，memory/RAG 同一套），不引第三方。向量已 L2 归一化，
相似度=点积。工具集变化时按「名字+描述」的 hash 增量重嵌，存 ~/.naga/tool_index.json。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

INDEX_FILE = Path.home() / ".naga" / "tool_index.json"


def _sig(tool: dict) -> str:
    """工具签名：名字+描述变了才需要重嵌。"""
    raw = (tool.get("name", "") + "\n" + (tool.get("description", "") or "")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _doc(tool: dict) -> str:
    """把工具编码成一段可嵌入的文本（名字 + 描述）。"""
    return f"{tool.get('name', '')}: {tool.get('description', '') or ''}".strip()


class ToolIndex:
    def __init__(self, path: Path | None = None, embedder=None):
        self.path = Path(path) if path else INDEX_FILE
        self._embedder = embedder            # 可注入（测试用假嵌入器）；否则惰性取全局
        self.entries: dict[str, dict] = {}   # name -> {sig, vec, server, description}
        self._load()

    # ---- 嵌入器（惰性，复用全局 bge）----
    def _emb(self):
        if self._embedder is None:
            from .embed import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def _encode(self, text: str) -> list[float]:
        return self._emb().encode(text)

    # ---- 持久化 ----
    def _load(self):
        if self.path.exists():
            try:
                self.entries = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.entries = {}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, ensure_ascii=False), encoding="utf-8")

    # ---- 识别 + 记录：把当前工具集嵌入并持久化（增量）----
    def sync(self, tools: list[dict]) -> int:
        """按名字+描述的 hash 增量嵌入；移除已消失的工具。返回本次重嵌的数量。"""
        seen, changed = set(), 0
        for t in tools:
            name = t.get("name")
            if not name:
                continue
            seen.add(name)
            sig = _sig(t)
            e = self.entries.get(name)
            if e is None or e.get("sig") != sig:
                self.entries[name] = {"sig": sig, "vec": self._encode(_doc(t)),
                                      "server": t.get("server"), "description": t.get("description", "")}
                changed += 1
        for name in list(self.entries):          # 清掉不再存在的工具
            if name not in seen:
                del self.entries[name]; changed += 1
        if changed:
            self._save()
        return changed

    # ---- 意图匹配：按 query 返回最相关的 top-k 工具（在传入的当前可用工具里筛）----
    def search(self, query: str, tools: list[dict], top_k: int = 8,
               min_score: float = 0.25) -> list[dict]:
        query = (query or "").strip()
        if not query or len(tools) <= top_k:
            return list(tools)                    # 无意图或工具本就不多 → 全给
        qv = self._encode(query)
        scored = []
        for t in tools:
            e = self.entries.get(t.get("name"))
            s = sum(a * b for a, b in zip(qv, e["vec"])) if e else 0.0
            scored.append((s, t))
        scored.sort(key=lambda x: -x[0])
        top = [t for s, t in scored[:top_k] if s >= min_score]
        # 全被阈值筛掉时，至少给分数最高的几个（避免模型完全无工具可用）
        return top or [t for _, t in scored[:top_k]]

    def selector(self, query: str, top_k: int = 8, threshold: int = 8):
        """返回一个 (tools)->tools 的闭包：工具数超 threshold 时按 query 意图筛 top-k。

        供 run_agent 的 tool_selector 用，让 agent.py 不直接依赖嵌入器（解耦）。"""
        def select(tools: list[dict]) -> list[dict]:
            from .trace import tracer
            if len(tools) <= threshold:
                tracer.event("intent_route", triggered=False, total=len(tools),
                             threshold=threshold, selected=[t.get("name") for t in tools])
                return tools
            picked = self.search(query, tools, top_k=top_k)
            with tracer.span("intent_route", triggered=True, total=len(tools),
                             threshold=threshold, top_k=top_k, query=query) as s:
                try:
                    qv = self._encode(query)
                    scored = sorted(((sum(a*b for a,b in zip(qv, self.entries[t["name"]]["vec"]))
                        if t.get("name") in self.entries else 0.0, t.get("name")) for t in tools), key=lambda x:-x[0])
                    s.set(candidates=[{"tool":n,"score":round(sc,3)} for sc,n in scored[:top_k+3]],
                          selected=[t.get("name") for t in picked])
                except Exception: pass
            return picked
        return select
