"""自跟踪—自优化：把指标画像变成可执行建议 + 持久化历史（P19）。

`advise()` 读一份 Stats 快照，产出「哪里可以更快/更省/更稳」的具体建议——这就是
「自己观测并跟踪优化」闭环里「优化」那一环：引擎观测自己，然后告诉你下一步怎么调。
`MetricsHistory` 把快照定期落盘成 JSONL，重启后仍能看吞吐/延迟/缓存复用的长期趋势，
便于对比不同配置（如 Q4 vs bf16、开/关前缀缓存）的真实收益。
"""

from __future__ import annotations

import json
from pathlib import Path


def advise(snapshot: dict) -> list[dict]:
    """基于指标快照给出优化建议。每条：{level, area, message}。

    level ∈ good | info | suggest | warn。样本太少时不轻易下结论（避免噪声误导）。
    """
    tips: list[dict] = []
    dec = snapshot.get("decode_tok_s", {})
    ttft = snapshot.get("ttft_ms", {})
    pc = snapshot.get("prefix_cache", {})
    sch = snapshot.get("scheduler", {})
    tot = snapshot.get("totals", {})

    if dec.get("n", 0) >= 3 and dec.get("avg", 0) < 40:
        tips.append({"level": "suggest", "area": "throughput",
                     "message": f"decode 平均 {dec['avg']} tok/s 偏低；若在跑 bf16，试 "
                                f"--quantize --bits 4（约 1.8× 提速、~3× 省显存）。"})

    if ttft.get("n", 0) >= 3 and ttft.get("p95", 0) > 800:
        tips.append({"level": "suggest", "area": "latency",
                     "message": f"TTFT p95 {ttft['p95']}ms 偏高（长 prompt 的 prefill 开销）；"
                                f"确认前缀缓存开启，或减少注入的 RAG/记忆条数以缩短上下文。"})

    reuse = pc.get("reuse_avg", 0.0)
    if pc.get("total_tokens", 0) > 0 and reuse < 0.2 and tot.get("generations", 0) >= 5:
        tips.append({"level": "info", "area": "cache",
                     "message": f"前缀缓存复用率仅 {reuse:.0%}；多轮/RAG 本应更高，检查每轮的 "
                                f"system/历史前缀是否稳定（前缀一变动就会失配、退化成重算）。"})
    elif reuse >= 0.5:
        tips.append({"level": "good", "area": "cache",
                     "message": f"前缀缓存复用率 {reuse:.0%}，RadixAttention 正在有效省 prefill。"})

    if sch.get("max_queue_depth", 0) >= 3:
        wait_max = sch.get("wait_ms", {}).get("max", 0)
        tips.append({"level": "info", "area": "concurrency",
                     "message": f"峰值有 {sch['max_queue_depth']} 个请求排队、最长等待 {wait_max}ms；"
                                f"引擎串行执行，高并发下建议减小 max_tokens 或在上游分流。"})

    calls = tot.get("tool_calls", 0)
    errs = tot.get("tool_errors", 0)
    if calls > 0 and errs / calls > 0.3:
        tips.append({"level": "warn", "area": "tools",
                     "message": f"工具错误率 {errs}/{calls} 偏高；检查 MCP 服务器连通性与参数 schema。"})

    if not tips:
        tips.append({"level": "good", "area": "general", "message": "当前指标正常，暂无优化建议。"})
    return tips


def _record(snapshot: dict, ts: float) -> dict:
    """把完整快照投影成紧凑的趋势记录（只留对优化决策有用的标量）。"""
    return {
        "t": round(ts, 1),
        "uptime_s": snapshot.get("uptime_s", 0),
        "decode_avg": snapshot.get("decode_tok_s", {}).get("avg", 0),
        "decode_p95": snapshot.get("decode_tok_s", {}).get("p95", 0),
        "ttft_p50": snapshot.get("ttft_ms", {}).get("p50", 0),
        "ttft_p95": snapshot.get("ttft_ms", {}).get("p95", 0),
        "reuse_avg": snapshot.get("prefix_cache", {}).get("reuse_avg", 0),
        "generations": snapshot.get("totals", {}).get("generations", 0),
        "completion_tokens": snapshot.get("totals", {}).get("completion_tokens", 0),
    }


class MetricsHistory:
    """把指标快照追加成 JSONL 落盘；重启后仍能读回趋势。"""

    def __init__(self, path: Path, max_lines: int = 20000):
        self.path = Path(path)
        self.max_lines = max_lines

    def append(self, snapshot: dict, ts: float):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_record(snapshot, ts), ensure_ascii=False) + "\n")
        self._maybe_trim()

    def _maybe_trim(self):
        # 简单防膨胀：超过上限时截到后一半，避免文件无限增长。
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return
        if len(lines) > self.max_lines:
            keep = lines[-(self.max_lines // 2):]
            self.path.write_text("\n".join(keep) + "\n", encoding="utf-8")

    def recent(self, n: int = 200) -> list[dict]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        out = []
        for line in lines[-n:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
