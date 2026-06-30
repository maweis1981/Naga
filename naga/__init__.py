"""Naga —— 从零自研、面向 Apple Silicon 的多模态推理引擎（基于 MLX）。"""

__version__ = "0.1.0"


def __getattr__(name):
    # 惰性导出 Agent SDK，避免在「只用引擎」时牵连导入 agent/generate 链路。
    if name in ("Agent", "tool", "AgentResult"):
        from . import sdk
        return getattr(sdk, name)
    raise AttributeError(f"module 'naga' has no attribute {name!r}")
