"""会话管理：多会话持久化（把「聊天记录」从浏览器内存搬进本地磁盘）。

推理仍走无状态的 /v1/chat/completions；本模块只负责「会话」这层状态：
每个会话一个 JSON 文件存在 ~/.naga/conversations/{id}.json，含标题与完整消息历史。
这样刷新页面、重启服务、换设备（拷 ~/.naga）都不丢历史，也支持多会话切换。

一会话一文件（而非单个大 JSON）：新增/改一个会话只重写它自己，会话多了也不卡。
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

CONV_DIR = Path.home() / ".naga" / "conversations"


def _now() -> float:
    # 亚秒分辨率：同一秒内连续创建/更新的会话也能正确排序
    return round(time.time(), 3)


def _title_from(messages: list[dict]) -> str:
    """用第一条 user 消息的前若干字符当标题（content 可能是图文数组）。"""
    for m in messages:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
        c = (c or "").strip().replace("\n", " ")
        if c:
            return c[:40]
    return "新会话"


class ConversationStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else CONV_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, cid: str) -> Path:
        # 只允许我们生成的 hex id，杜绝路径穿越（../ 之类）
        if not cid or not all(c in "0123456789abcdef" for c in cid):
            raise ValueError("非法会话 id")
        return self.root / f"{cid}.json"

    def create(self, title: str | None = None, model: str | None = None) -> dict:
        cid = uuid.uuid4().hex[:16]
        conv = {"id": cid, "title": title or "新会话", "model": model,
                "created": _now(), "updated": _now(), "messages": []}
        self._write(conv)
        return conv

    def _write(self, conv: dict):
        self._path(conv["id"]).write_text(
            json.dumps(conv, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, cid: str) -> dict | None:
        p = self._path(cid)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def save(self, cid: str, messages: list[dict], title: str | None = None,
             model: str | None = None) -> dict | None:
        """整段替换某会话的消息历史（前端每轮结束后回存当前完整状态）。"""
        conv = self.get(cid)
        if conv is None:
            return None
        conv["messages"] = messages
        # 标题：显式传入 > 已有非默认标题 > 从首条 user 消息推导
        if title:
            conv["title"] = title[:60]
        elif conv.get("title") in (None, "", "新会话"):
            conv["title"] = _title_from(messages)
        if model:
            conv["model"] = model
        conv["updated"] = _now()
        self._write(conv)
        return conv

    def rename(self, cid: str, title: str) -> dict | None:
        conv = self.get(cid)
        if conv is None:
            return None
        conv["title"] = (title or "").strip()[:60] or conv["title"]
        conv["updated"] = _now()
        self._write(conv)
        return conv

    def delete(self, cid: str) -> bool:
        p = self._path(cid)
        if p.exists():
            p.unlink()
            return True
        return False

    def list(self) -> list[dict]:
        """会话摘要列表（不含完整消息），按最近更新倒序。"""
        out = []
        for f in self.root.glob("*.json"):
            try:
                c = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append({"id": c.get("id"), "title": c.get("title", "新会话"),
                        "updated": c.get("updated", 0), "model": c.get("model"),
                        "count": len(c.get("messages", []))})
        out.sort(key=lambda x: -x["updated"])
        return out
