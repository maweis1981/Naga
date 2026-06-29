"""本地硬件资源采样：CPU / 统一内存 / MLX 显存 / 本进程占用。

面向 Apple Silicon：统一内存是 CPU 与 GPU 共享的，所以最该盯的是
  - 系统统一内存用量（逼近物理上限就会 swap/卡顿）
  - MLX 实际占用（模型权重 + KV-Cache + 激活，逼近 GPU 推荐工作集就危险）
GPU 利用率% 在 macOS 无 sudo 拿不到，故以"MLX 显存占用 / 推荐工作集"作为
引擎对 GPU 压力的代理指标。psutil 属系统工具层（同 fastapi），不碰推理逻辑。
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import psutil

_proc = psutil.Process(os.getpid())
_POWER_FILE = Path.home() / ".naga" / "power.json"   # powermon 守护写入


def _perf_split() -> tuple[int, int]:
    """Apple Silicon 的 性能核 / 能效核 数量。"""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.logicalcpu", "hw.perflevel1.logicalcpu"],
            capture_output=True, text=True, timeout=2).stdout.split()
        return int(out[0]), int(out[1])
    except Exception:
        return psutil.cpu_count() or 0, 0


def _device_info() -> dict:
    try:
        return mx.device_info()
    except Exception:
        try:
            return mx.metal.device_info()
        except Exception:
            return {}


_P, _E = _perf_split()
_DEV = _device_info()
_GPU_MAX = int(_DEV.get("max_recommended_working_set_size", 0))
_DEV_NAME = _DEV.get("device_name", "GPU")

# 预热：psutil 的 cpu_percent 首次调用返回 0，先打一次基线
psutil.cpu_percent(percpu=True)
_proc.cpu_percent()


def static() -> dict:
    """开机不变的硬件信息（页面只取一次）。"""
    vm = psutil.virtual_memory()
    return {
        "device": _DEV_NAME,
        "p_cores": _P, "e_cores": _E, "cpu_total": psutil.cpu_count(),
        "mem_total_gb": round(vm.total / 1e9, 1),
        "gpu_max_gb": round(_GPU_MAX / 1e9, 1) if _GPU_MAX else round(vm.total / 1e9, 1),
    }


def sample() -> dict:
    """一次实时采样（页面定时轮询）。"""
    vm = psutil.virtual_memory()
    cores = psutil.cpu_percent(percpu=True)
    try:
        swap = psutil.swap_memory().used / 1e9
    except Exception:
        swap = 0.0
    return {
        "cpu_overall": round(sum(cores) / len(cores), 1) if cores else 0.0,
        "cores": [round(c, 1) for c in cores],
        "loadavg": [round(x, 2) for x in psutil.getloadavg()],
        "mem_used_gb": round((vm.total - vm.available) / 1e9, 2),
        "mem_total_gb": round(vm.total / 1e9, 1),
        "mem_pct": vm.percent,
        "swap_gb": round(swap, 2),
        "proc_cpu": round(_proc.cpu_percent(), 1),
        "proc_rss_gb": round(_proc.memory_info().rss / 1e9, 2),
        # MLX 显存：当前占用 / 历史峰值 / 内部可复用缓存
        "mlx_active_gb": round(mx.get_active_memory() / 1e9, 2),
        "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 2),
        "mlx_cache_gb": round(mx.get_cache_memory() / 1e9, 2),
        "gpu_max_gb": round(_GPU_MAX / 1e9, 1) if _GPU_MAX else round(vm.total / 1e9, 1),
    }


def power() -> dict | None:
    """GPU 功耗/利用率/热压力（来自 powermon 守护）。未启用或数据过期则返回 None。"""
    try:
        d = json.loads(_POWER_FILE.read_text())
        if time.time() - d.get("t", 0) < 5:        # 5 秒内才算有效
            return d
    except Exception:
        pass
    return None
