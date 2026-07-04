"""张量后端抽象：集中管理底层张量框架，让引擎不直接绑定某一个实现。

当前默认后端是 **MLX**（Apple Silicon / Metal GPU）。所有引擎代码通过
    from .backend import mx, nn        # 顶层模块
    from ..backend import mx, nn       # naga/models/ 子模块
拿到张量库，而不是直接 `import mlx.core`。这样将来接入其它后端（PyTorch/CUDA、
Linux CPU）时，只需在本文件里切换实现，引擎其余代码不必逐处改 import。

选择：环境变量 NAGA_BACKEND=mlx（默认 auto，自动选可用后端）。torch 后端为将来
预留（骨架已就位，实现见 issue #22）——现在请求它会抛清晰的 NotImplementedError。
"""
from __future__ import annotations

import os

BACKEND = os.environ.get("NAGA_BACKEND", "auto").lower()


def _select():
    # torch 后端：骨架预留，尚未实现
    if BACKEND == "torch":
        raise NotImplementedError(
            "PyTorch/CUDA 后端尚未实现——后端抽象骨架已就位（naga/backend.py）。"
            "当前仅支持 MLX（Apple Silicon）。Linux/CUDA 支持见 issue #22。")
    # mlx / auto：优先 MLX
    if BACKEND in ("mlx", "auto"):
        try:
            import mlx.core as _mx
            import mlx.nn as _nn
            return "mlx", _mx, _nn
        except Exception as e:  # pragma: no cover
            if BACKEND == "mlx":
                raise RuntimeError(f"NAGA_BACKEND=mlx 但 MLX 不可用: {e}")
    raise RuntimeError(  # pragma: no cover
        "未找到可用的张量后端。Naga 当前需要 MLX（Apple Silicon）；"
        "Linux/CUDA 支持在规划中（issue #22）。")


name, mx, nn = _select()


def is_mlx() -> bool:
    return name == "mlx"


def info() -> dict:
    return {"backend": name}
