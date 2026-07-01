"""优化顾问 + 指标历史（naga.optimize）的回归测试。"""

from naga.optimize import MetricsHistory, advise


def _snap(**over):
    base = {
        "uptime_s": 100.0,
        "totals": {"generations": 10, "completion_tokens": 500, "tool_calls": 0, "tool_errors": 0},
        "decode_tok_s": {"n": 10, "avg": 70.0, "p50": 70, "p95": 80},
        "ttft_ms": {"n": 10, "avg": 120, "p50": 110, "p95": 150},
        "prefix_cache": {"reuse_avg": 0.6, "total_tokens": 1000},
        "scheduler": {"max_queue_depth": 1, "wait_ms": {"max": 5}},
    }
    for k, v in over.items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) else v
    return base


def test_advise_low_decode_suggests_quant():
    tips = advise(_snap(decode_tok_s={"n": 5, "avg": 30.0, "p50": 30, "p95": 35}))
    assert any(t["area"] == "throughput" and "quantize" in t["message"] for t in tips)


def test_advise_high_ttft():
    tips = advise(_snap(ttft_ms={"n": 5, "avg": 900, "p50": 850, "p95": 1200}))
    assert any(t["area"] == "latency" for t in tips)


def test_advise_low_prefix_reuse():
    tips = advise(_snap(prefix_cache={"reuse_avg": 0.05, "total_tokens": 500}))
    assert any(t["area"] == "cache" and t["level"] == "info" for t in tips)


def test_advise_good_prefix_reuse():
    tips = advise(_snap(prefix_cache={"reuse_avg": 0.7, "total_tokens": 500}))
    assert any(t["area"] == "cache" and t["level"] == "good" for t in tips)


def test_advise_queue_depth_and_tool_errors():
    tips = advise(_snap(scheduler={"max_queue_depth": 4, "wait_ms": {"max": 900}},
                        totals={"generations": 10, "completion_tokens": 5,
                                "tool_calls": 10, "tool_errors": 5}))
    assert any(t["area"] == "concurrency" for t in tips)
    assert any(t["area"] == "tools" and t["level"] == "warn" for t in tips)


def test_advise_healthy_has_good_note():
    tips = advise(_snap())
    assert all(t["level"] in ("good", "info") for t in tips)


def test_metrics_history_append_and_recent(tmp_path):
    h = MetricsHistory(tmp_path / "m.jsonl")
    assert h.recent() == []                       # 不存在时返回空
    h.append(_snap(decode_tok_s={"avg": 70.0, "p95": 80, "n": 3}), ts=1000.0)
    h.append(_snap(decode_tok_s={"avg": 75.0, "p95": 82, "n": 3}), ts=1060.0)
    rec = h.recent()
    assert len(rec) == 2
    assert rec[0]["decode_avg"] == 70.0 and rec[1]["decode_avg"] == 75.0
    assert rec[1]["t"] == 1060.0


def test_metrics_history_trim(tmp_path):
    h = MetricsHistory(tmp_path / "m.jsonl", max_lines=10)
    for i in range(25):
        h.append(_snap(totals={"generations": i, "completion_tokens": i}), ts=float(i))
    rec = h.recent(1000)
    assert len(rec) <= 10                          # 触发截断，文件不无限膨胀
    assert rec[-1]["generations"] == 24            # 保留的是最近的
