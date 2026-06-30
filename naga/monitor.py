"""引擎监视器：收集推理过程的结构化事件，供 /monitor 实时查看。

每次请求 / 生成 / 前缀缓存命中 / 上下文注入 / 工具调用 / 模型加载，都 emit 一条
事件，进环形缓冲，并推给所有 SSE 订阅者。纯标准库、不依赖 naga 内部任何模块，
因此可被引擎各层安全引用（像 logging 一样到处撒点）。

除了「实时事件流」（给人看），同一批事件还被 `Stats` 滚动聚合成「可优化指标」
（给机器/自己看）：decode tok/s 分位、TTFT 分布、前缀缓存复用率、各模型吞吐、
工具调用频次。这就是 Naga 的自观测—自跟踪闭环：每次推理都在更新自己的画像，
据此判断量化/前缀缓存/换模型有没有真的带来收益，并能通过 /metrics 标准接口导出。
"""

from __future__ import annotations

import threading
import time
from collections import deque


def _summary(samples) -> dict:
    """把一串样本压成一组分位统计（纯标准库，避免为分位数引入 numpy）。"""
    xs = list(samples)
    if not xs:
        return {"n": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "last": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(xs)

    def pct(p: float) -> float:
        if len(s) == 1:
            return s[0]
        idx = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
        return s[idx]

    return {
        "n": len(xs),
        "avg": round(sum(xs) / len(xs), 2),
        "p50": round(pct(50), 2),
        "p95": round(pct(95), 2),
        "last": round(xs[-1], 2),
        "min": round(s[0], 2),
        "max": round(s[-1], 2),
    }


class Stats:
    """把 monitor 事件滚动聚合成「优化决策用」的指标。

    回答这些问题：现在 decode 到底多快？TTFT 抖不抖？RadixAttention 真的在替我省
    prefill 吗？哪个工具被调得最多？哪个模型更划算？计数器是单调累计，分位数走
    最近 WINDOW 次的滑动窗口（既反映当前状态，又不会被历史拖死）。
    """

    WINDOW = 200  # 最近多少次生成参与分位数计算

    def __init__(self):
        self._lock = threading.Lock()
        self.started = time.time()
        self.totals = {
            "requests": 0, "generations": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "tool_calls": 0, "tool_errors": 0,
            "rag_injections": 0, "memory_injections": 0, "model_loads": 0,
        }
        self._decode: deque = deque(maxlen=self.WINDOW)   # decode tok/s 样本
        self._ttft: deque = deque(maxlen=self.WINDOW)     # ttft_ms 样本
        self._reuse: deque = deque(maxlen=self.WINDOW)    # 前缀缓存复用率样本
        self._prefix_matched = 0
        self._prefix_total = 0
        self.tools: dict[str, int] = {}                   # 工具名 -> 调用次数
        self.models: dict[str, dict] = {}                 # 模型 -> 吞吐画像

    def update(self, ev: dict):
        kind = ev.get("kind")
        with self._lock:
            if kind == "request":
                self.totals["requests"] += 1
            elif kind == "generation":
                self.totals["generations"] += 1
                self.totals["prompt_tokens"] += ev.get("prompt_tokens", 0)
                self.totals["completion_tokens"] += ev.get("completion_tokens", 0)
                d = ev.get("decode_tok_s") or 0.0
                t = ev.get("ttft_ms") or 0.0
                if d:
                    self._decode.append(d)
                if t:
                    self._ttft.append(t)
                m = self.models.setdefault(
                    ev.get("model", "?"),
                    {"generations": 0, "completion_tokens": 0, "decode": deque(maxlen=self.WINDOW)},
                )
                m["generations"] += 1
                m["completion_tokens"] += ev.get("completion_tokens", 0)
                if d:
                    m["decode"].append(d)
            elif kind == "prefix_cache":
                self._prefix_matched += ev.get("matched", 0)
                self._prefix_total += ev.get("total", 0)
                self._reuse.append(ev.get("reuse", 0.0))
            elif kind == "tool_call":
                self.totals["tool_calls"] += 1
                name = ev.get("name", "?")
                self.tools[name] = self.tools.get(name, 0) + 1
            elif kind == "tool_result":
                if str(ev.get("result", "")).startswith(("[工具执行出错]", "[错误]")):
                    self.totals["tool_errors"] += 1
            elif kind == "context":
                self.totals["rag_injections"] += ev.get("rag_docs", 0)
                self.totals["memory_injections"] += ev.get("memories", 0)
            elif kind == "model_load":
                self.totals["model_loads"] += 1

    def snapshot(self) -> dict:
        with self._lock:
            reuse_avg = (self._prefix_matched / self._prefix_total) if self._prefix_total else 0.0
            return {
                "uptime_s": round(time.time() - self.started, 1),
                "totals": dict(self.totals),
                "decode_tok_s": _summary(self._decode),
                "ttft_ms": _summary(self._ttft),
                "prefix_cache": {
                    "reuse_avg": round(reuse_avg, 3),
                    "matched_tokens": self._prefix_matched,
                    "total_tokens": self._prefix_total,
                    "saved_prefill_tokens": self._prefix_matched,  # 命中即省掉的 prefill 量
                    "reuse_recent": _summary(self._reuse),
                },
                "tools": dict(sorted(self.tools.items(), key=lambda kv: -kv[1])),
                "models": {
                    name: {
                        "generations": m["generations"],
                        "completion_tokens": m["completion_tokens"],
                        "decode_tok_s": _summary(m["decode"]),
                    }
                    for name, m in self.models.items()
                },
            }

    def prometheus(self) -> str:
        """Prometheus 文本曝光格式：可直接被 Prometheus/Grafana/OpenTelemetry 抓取。"""
        snap = self.snapshot()
        out: list[str] = []

        def line(name, value, labels=""):
            out.append(f"{name}{('{' + labels + '}') if labels else ''} {value}")

        out.append("# HELP naga_uptime_seconds Server uptime in seconds.")
        out.append("# TYPE naga_uptime_seconds gauge")
        line("naga_uptime_seconds", snap["uptime_s"])

        counter_help = {
            "requests": "Chat/completion requests received.",
            "generations": "Generation passes completed.",
            "prompt_tokens": "Prompt tokens processed.",
            "completion_tokens": "Completion tokens generated.",
            "tool_calls": "Tool calls dispatched.",
            "tool_errors": "Tool calls that returned an error.",
            "model_loads": "Models loaded/hot-swapped.",
        }
        for key, help_ in counter_help.items():
            out.append(f"# HELP naga_{key}_total {help_}")
            out.append(f"# TYPE naga_{key}_total counter")
            line(f"naga_{key}_total", snap["totals"].get(key, 0))

        out.append("# HELP naga_decode_tokens_per_second Decode throughput (sliding window).")
        out.append("# TYPE naga_decode_tokens_per_second gauge")
        for q in ("avg", "p50", "p95"):
            line("naga_decode_tokens_per_second", snap["decode_tok_s"][q], f'quantile="{q}"')

        out.append("# HELP naga_ttft_milliseconds Time to first token (sliding window).")
        out.append("# TYPE naga_ttft_milliseconds gauge")
        for q in ("avg", "p50", "p95"):
            line("naga_ttft_milliseconds", snap["ttft_ms"][q], f'quantile="{q}"')

        out.append("# HELP naga_prefix_cache_reuse_ratio Fraction of prompt tokens served from prefix cache.")
        out.append("# TYPE naga_prefix_cache_reuse_ratio gauge")
        line("naga_prefix_cache_reuse_ratio", snap["prefix_cache"]["reuse_avg"])

        out.append("# HELP naga_model_decode_tokens_per_second Per-model decode throughput (avg).")
        out.append("# TYPE naga_model_decode_tokens_per_second gauge")
        for name, m in snap["models"].items():
            safe = name.replace("\\", "_").replace('"', "_")
            line("naga_model_decode_tokens_per_second", m["decode_tok_s"]["avg"], f'model="{safe}"')

        return "\n".join(out) + "\n"


class Monitor:
    def __init__(self, maxlen: int = 800):
        self._buf: deque = deque(maxlen=maxlen)   # 最近事件环形缓冲
        self._subs: list[deque] = []              # 每个 SSE 连接一个队列
        self._lock = threading.Lock()
        self._seq = 0
        self.stats = Stats()                      # 同一批事件的滚动聚合（自观测/自跟踪）

    def emit(self, kind: str, **fields) -> dict:
        """记录一条事件。线程安全（生成在线程池里跑，SSE 在事件循环里读）。"""
        with self._lock:
            self._seq += 1
            ev = {"seq": self._seq, "t": round(time.time(), 3), "kind": kind, **fields}
            self._buf.append(ev)
            for q in self._subs:
                q.append(ev)
        # 聚合在 Monitor 锁之外完成（Stats 有自己的锁），避免锁嵌套
        self.stats.update(ev)
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
