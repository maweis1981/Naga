"""自观测指标聚合（naga.monitor.Stats）的回归测试。"""

from naga.monitor import Monitor, _summary


def test_summary_empty():
    s = _summary([])
    assert s == {"n": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "last": 0.0, "min": 0.0, "max": 0.0}


def test_summary_percentiles():
    s = _summary([10, 20, 30, 40, 50])
    assert s["n"] == 5
    assert s["avg"] == 30.0
    assert s["min"] == 10 and s["max"] == 50
    assert s["last"] == 50
    assert s["p50"] == 30
    assert s["p95"] == 50


def test_stats_aggregates_generation_events():
    mon = Monitor()
    for tok, ttft in [(75, 120), (60, 140), (80, 110)]:
        mon.emit("request", model="m", prompt_tokens=10)
        mon.emit("generation", model="m", prompt_tokens=10, completion_tokens=5,
                 ttft_ms=ttft, decode_tok_s=tok)
    snap = mon.stats.snapshot()
    assert snap["totals"]["requests"] == 3
    assert snap["totals"]["generations"] == 3
    assert snap["totals"]["completion_tokens"] == 15
    assert snap["decode_tok_s"]["n"] == 3
    assert snap["decode_tok_s"]["max"] == 80
    assert snap["models"]["m"]["generations"] == 3


def test_stats_prefix_cache_and_tools():
    mon = Monitor()
    mon.emit("prefix_cache", matched=80, total=100, reuse=0.8)
    mon.emit("prefix_cache", matched=40, total=100, reuse=0.4)
    mon.emit("tool_call", name="search")
    mon.emit("tool_call", name="search")
    mon.emit("tool_call", name="add")
    mon.emit("tool_result", name="add", result="[工具执行出错] boom")
    mon.emit("context", rag_docs=2, memories=1)
    snap = mon.stats.snapshot()
    assert snap["prefix_cache"]["reuse_avg"] == 0.6        # 120 matched / 200 total
    assert snap["prefix_cache"]["saved_prefill_tokens"] == 120
    assert snap["totals"]["tool_calls"] == 3
    assert snap["totals"]["tool_errors"] == 1
    assert snap["tools"]["search"] == 2
    assert snap["totals"]["rag_injections"] == 2
    assert snap["totals"]["memory_injections"] == 1


def test_prometheus_exposition_format():
    mon = Monitor()
    mon.emit("generation", model="qwen", prompt_tokens=10, completion_tokens=5,
             ttft_ms=100, decode_tok_s=50)
    text = mon.stats.prometheus()
    assert "# TYPE naga_generations_total counter" in text
    assert "naga_generations_total 1" in text
    assert 'naga_decode_tokens_per_second{quantile="avg"} 50' in text
    assert 'naga_model_decode_tokens_per_second{model="qwen"}' in text
    # 每个非注释行都应是 "name value" 形态
    for line in text.splitlines():
        if line and not line.startswith("#"):
            assert len(line.rsplit(" ", 1)) == 2
