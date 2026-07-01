"""引擎调度器：单后台工作线程串行执行所有生成任务（P3 并发调度的串行基座）。

为什么需要：model / RadixCache / 分词器都是单实例、**非线程安全**。FastAPI 的流式响应
是同步生成器、被 Starlette 放到线程池里逐个 next() 地拉取，于是并发请求会同时进入
`engine.stream`，互相污染 KV 缓存、产出错误结果。

设计：所有生成都提交给**一个专属工作线程**排队执行。请求提交一个「生成器工厂」，工作线程
一次只跑一个、把产出逐条塞进该任务的输出队列；异步端点只从队列里 await 取结果，从不跨事件
循环持锁。这样：
  - 正确性——引擎同一时刻只被一个任务使用；
  - 解耦——生成不受客户端读取速度拖累（产出先进队列缓冲）；
  - 可观测——排队深度 / 等待时延 emit 成 schedule 事件喂给 Stats。
真正的 token 级连续批处理（多序列同批前向）是后续工作，但正确的串行调度是它的前提。
"""

from __future__ import annotations

import queue
import threading
import time

from .monitor import monitor


class Job:
    """一次生成任务：`factory()` 返回一个可迭代对象，其产出被逐条转发到 `out`。"""

    def __init__(self, factory):
        self.factory = factory
        self.out: queue.Queue = queue.Queue()     # ("item"|"error"|"done", value)
        self.enqueued = time.perf_counter()

    def results(self):
        """阻塞式地逐条取回结果（在线程池里 drain；出错则抛出原异常）。"""
        while True:
            kind, val = self.out.get()
            if kind == "done":
                return
            if kind == "error":
                raise val
            yield val


class EngineScheduler:
    def __init__(self):
        self._jobs: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True, name="naga-engine")
        self._worker.start()

    @property
    def queue_depth(self) -> int:
        return self._jobs.qsize()

    def _run(self):
        while True:
            job: Job = self._jobs.get()
            wait_ms = (time.perf_counter() - job.enqueued) * 1000
            monitor.emit("schedule", wait_ms=round(wait_ms, 1),
                         queue_depth=self._jobs.qsize() + 1)   # +1 含正在处理的这个
            try:
                for item in job.factory():
                    job.out.put(("item", item))
            except Exception as e:                              # 把异常透传给消费者
                job.out.put(("error", e))
            finally:
                job.out.put(("done", None))

    def submit(self, factory) -> Job:
        """提交一个生成器工厂，立即返回 Job；调用方从 job.results() 取回产出。"""
        job = Job(factory)
        self._jobs.put(job)
        return job


scheduler = EngineScheduler()
