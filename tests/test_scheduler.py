"""引擎调度器（naga.scheduler）回归测试：单线程串行执行 + 排队指标 + 异常透传。"""

import threading
import time

from naga.monitor import monitor
from naga.scheduler import EngineScheduler


def test_single_worker_serializes_execution():
    sched = EngineScheduler()
    in_flight = 0
    max_in_flight = 0
    lock = threading.Lock()

    def make_factory():
        def factory():
            nonlocal in_flight, max_in_flight
            with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            time.sleep(0.01)           # 占住引擎，制造重叠机会
            with lock:
                in_flight -= 1
            yield "x"
        return factory

    jobs = [sched.submit(make_factory()) for _ in range(8)]
    for j in jobs:                     # drain 全部，等它们跑完
        assert list(j.results()) == ["x"]

    assert max_in_flight == 1          # 单工作线程保证任意时刻只有一个任务在执行


def test_results_stream_in_order():
    sched = EngineScheduler()
    job = sched.submit(lambda: iter(["a", "b", "c"]))
    assert list(job.results()) == ["a", "b", "c"]


def test_error_propagates_to_consumer():
    sched = EngineScheduler()

    def boom():
        yield "ok"
        raise ValueError("kaboom")

    job = sched.submit(boom)
    got = []
    try:
        for x in job.results():
            got.append(x)
        assert False, "should have raised"
    except ValueError as e:
        assert str(e) == "kaboom"
    assert got == ["ok"]               # 出错前的产出仍正常到达


def test_submit_emits_schedule_metric():
    sched = EngineScheduler()
    before = monitor.stats.snapshot()["totals"]["queued"]
    job = sched.submit(lambda: iter(["z"]))
    list(job.results())
    snap = monitor.stats.snapshot()
    assert snap["totals"]["queued"] == before + 1
    assert snap["scheduler"]["wait_ms"]["n"] >= 1
