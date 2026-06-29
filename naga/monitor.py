"""引擎监视器：收集推理过程的结构化事件，供 /monitor 实时查看。

每次请求 / 生成 / 前缀缓存命中 / 上下文注入 / 工具调用 / 模型加载，都 emit 一条
事件，进环形缓冲，并推给所有 SSE 订阅者。纯标准库、不依赖 naga 内部任何模块，
因此可被引擎各层安全引用（像 logging 一样到处撒点）。
"""

from __future__ import annotations

import threading
import time
from collections import deque


class Monitor:
    def __init__(self, maxlen: int = 800):
        self._buf: deque = deque(maxlen=maxlen)   # 最近事件环形缓冲
        self._subs: list[deque] = []              # 每个 SSE 连接一个队列
        self._lock = threading.Lock()
        self._seq = 0

    def emit(self, kind: str, **fields) -> dict:
        """记录一条事件。线程安全（生成在线程池里跑，SSE 在事件循环里读）。"""
        with self._lock:
            self._seq += 1
            ev = {"seq": self._seq, "t": round(time.time(), 3), "kind": kind, **fields}
            self._buf.append(ev)
            for q in self._subs:
                q.append(ev)
        return ev

    def recent(self, n: int = 100) -> list[dict]:
        with self._lock:
            return list(self._buf)[-n:]

    def subscribe(self) -> deque:
        q: deque = deque()
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: deque):
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)


monitor = Monitor()
