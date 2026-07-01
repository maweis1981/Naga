"""Naga Agent SDK：把推理引擎 + 工具 + 工具调用循环封装成可编程的 Agent（P16）。

对标 OpenAI / Claude 的 Agent SDK，但工具在本地、模型也在本地（MLX）。最小用法：

    from naga.sdk import Agent, tool

    @tool
    def add(a: int, b: int) -> str:
        \"\"\"把两个整数相加。\"\"\"
        return str(a + b)

    agent = Agent(model="Qwen/Qwen2.5-3B-Instruct", tools=[add])
    print(agent.run("2 加 3 等于几？").text)      # -> 触发 add(2,3)，再用自然语言作答

要点：
  - `@tool` 把普通函数变成工具：函数名→工具名、首行 docstring→描述、类型注解→JSON Schema。
  - 工具调用复用引擎已有的「约束解码」路径（见 agent.run_agent），小模型也能产出合法调用。
  - 可同时挂本地函数工具与 MCP 服务器工具（`mcp=` 传入 MCPManager）。
  - `run()` 返回结构化结果（最终文本 + 工具调用轨迹 + 消息历史）；`stream()` 逐步产出事件。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

# Python 类型注解 -> JSON Schema 基础类型
_JSON_TYPES = {int: "integer", float: "number", str: "string", bool: "boolean",
               list: "array", dict: "object"}


def _schema_from_fn(fn: Callable) -> dict:
    """从函数签名推导工具参数的 JSON Schema（无注解的参数按 string 处理）。"""
    props: dict[str, dict] = {}
    required: list[str] = []
    for pname, p in inspect.signature(fn).parameters.items():
        if pname == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        jtype = _JSON_TYPES.get(p.annotation, "string")
        props[pname] = {"type": jtype}
        if p.default is inspect.Parameter.empty:
            required.append(pname)
    return {"type": "object", "properties": props, "required": required}


def tool(fn: Callable | None = None, *, name: str | None = None,
         description: str | None = None):
    """装饰器：把普通函数标注成 Agent 可调用的工具。可裸用 `@tool`，也可带参覆盖名字/描述。"""

    def wrap(f: Callable) -> Callable:
        if name:
            f._naga_tool_name = name           # type: ignore[attr-defined]
        if description:
            f._naga_tool_desc = description     # type: ignore[attr-defined]
        return f

    return wrap(fn) if fn is not None else wrap


class FunctionToolset:
    """把一组本地 Python 函数暴露成 MCPManager 同款接口（`tools()` / `call()`）。

    这样它能无缝插进现有的 `run_agent` 循环，和 MCP 服务器工具走完全一致的路径。
    """

    def __init__(self, fns: list[Callable]):
        self._fns: dict[str, Callable] = {}
        self._specs: list[dict] = []
        for fn in fns:
            tname = getattr(fn, "_naga_tool_name", None) or fn.__name__
            doc = getattr(fn, "_naga_tool_desc", None) \
                or (fn.__doc__ or "").strip().split("\n")[0] or tname
            schema = getattr(fn, "_naga_tool_schema", None) or _schema_from_fn(fn)
            self._fns[tname] = fn
            self._specs.append({"name": tname, "description": doc, "schema": schema})

    def tools(self) -> list[dict]:
        return list(self._specs)

    def call(self, name: str, arguments: dict) -> str:
        fn = self._fns.get(name)
        if fn is None:
            return f"[错误] 找不到工具 {name}"
        try:
            return str(fn(**(arguments or {})))
        except Exception as e:
            return f"[工具执行出错] {e}"


class _CompositeToolset:
    """把多个工具来源（本地函数 + 多个 MCP 服务器）合并成一个统一视图。"""

    def __init__(self, providers: list):
        self.providers = [p for p in providers if p is not None]

    def tools(self) -> list[dict]:
        out: list[dict] = []
        for p in self.providers:
            out.extend(p.tools())
        return out

    def call(self, name: str, arguments: dict) -> str:
        for p in self.providers:
            if any(t["name"] == name for t in p.tools()):
                return p.call(name, arguments)
        return f"[错误] 找不到工具 {name}"


@dataclass
class AgentResult:
    """一次 `Agent.run` 的结构化结果。"""
    text: str                                   # 最终自然语言回答
    steps: list[dict] = field(default_factory=list)   # 工具调用/结果轨迹
    messages: list[dict] = field(default_factory=list)  # 含最终回答的消息历史
    usage: dict = field(default_factory=dict)   # 用量：tool_calls/denied/steps/completion_tokens

    def __repr__(self) -> str:
        return f"AgentResult(text={self.text!r}, steps={len(self.steps)}, usage={self.usage})"


class Agent:
    """可编程的本地 Agent：引擎 + 工具 + 工具调用循环。

    参数：
      model        模型 id（未传 engine 时按需构建 Engine；可配 quantize/bits）
      engine       直接复用已加载的 Engine（与 model 二选一，优先 engine）
      tools        本地 Python 函数工具列表（配合 `@tool`）
      mcp          可选的 MCPManager（让 Agent 同时能用 MCP 服务器工具）
      system       系统提示词
      max_steps    工具调用最大轮数
      tool_choice  'auto' | 'required' | 'none'（语义见 agent.run_agent）
      其余 kwargs（temp/top_p/top_k/max_tokens）透传给生成。
    """

    def __init__(self, model: str | None = None, *, engine: Any = None,
                 tools: list[Callable] | None = None, mcp: Any = None,
                 name: str | None = None, description: str | None = None,
                 system: str | None = None, max_steps: int = 5,
                 tool_choice: str = "auto",
                 allowed_tools: list[str] | None = None,
                 blocked_tools: list[str] | None = None,
                 can_use_tool: Callable | None = None,
                 on_tool_call: Callable | None = None,
                 on_tool_result: Callable | None = None,
                 quantize: bool = False, bits: int = 4,
                 **params):
        if engine is None:
            from .engine import Engine
            engine = Engine(model or "Qwen/Qwen2.5-3B-Instruct",
                            quantize=quantize, bits=bits)
        self.engine = engine
        self._toolset = _CompositeToolset(
            [FunctionToolset(tools) if tools else None, mcp]
        )
        self.name = name
        self.description = description
        self.system = system
        self.max_steps = max_steps
        self.tool_choice = tool_choice
        # 权限：allowed/blocked 名单 + 自定义 can_use_tool 组合成一个权限函数
        self.allowed_tools = allowed_tools
        self.blocked_tools = blocked_tools
        self.can_use_tool = can_use_tool
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.params = params

    def _permission(self):
        """把 allowed/blocked 名单与 can_use_tool 组合成传给 run_agent 的权限函数。"""
        allowed, blocked, custom = self.allowed_tools, self.blocked_tools, self.can_use_tool
        if allowed is None and blocked is None and custom is None:
            return None

        def check(name, args):
            if blocked and name in blocked:
                return f"工具 {name} 在禁用名单中"
            if allowed is not None and name not in allowed:
                return f"工具 {name} 不在允许名单中"
            return custom(name, args) if custom else True

        return check

    @property
    def tools(self) -> list[dict]:
        return self._toolset.tools()

    def _messages(self, user: str, history: list[dict] | None) -> list[dict]:
        msgs: list[dict] = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        if history:
            msgs.extend(history)
        msgs.append({"role": "user", "content": user})
        return msgs

    def stream(self, user: str, history: list[dict] | None = None) -> Iterator[tuple[str, Any]]:
        """逐步产出 (kind, data)：kind ∈ {'delta','tool_call','tool_result','final'}。"""
        from .agent import run_agent
        msgs = self._messages(user, history)
        yield from run_agent(self.engine, self._toolset, msgs,
                             max_steps=self.max_steps, tool_choice=self.tool_choice,
                             permission=self._permission(),
                             on_tool_call=self.on_tool_call,
                             on_tool_result=self.on_tool_result,
                             **self.params)

    def run(self, user: str, history: list[dict] | None = None) -> AgentResult:
        """跑完整个工具调用循环，返回结构化结果（含用量统计）。"""
        steps: list[dict] = []
        text = ""
        tool_calls = denied = 0
        for kind, data in self.stream(user, history):
            if kind == "tool_call":
                steps.append({"type": "tool_call", **data})
                tool_calls += 1
            elif kind == "tool_result":
                steps.append({"type": "tool_result", **data})
                if data.get("denied"):
                    denied += 1
            elif kind == "final":
                text = data
        messages = self._messages(user, history) + [{"role": "assistant", "content": text}]
        # 用量统计（对标 SDK 的 message.usage）：轮数/工具次数/拒绝数 + 完成文本 token
        try:
            completion_tokens = len(self.engine.tok.encode(text)) if text else 0
        except Exception:
            completion_tokens = 0
        usage = {"tool_calls": tool_calls, "denied": denied,
                 "steps": sum(1 for s in steps if s["type"] == "tool_call"),
                 "completion_tokens": completion_tokens}
        return AgentResult(text=text, steps=steps, messages=messages, usage=usage)

    def as_tool(self, name: str | None = None, description: str | None = None) -> Callable:
        """把本 Agent 封装成一个可被**主 Agent 调用**的工具，实现子代理委派（对标 SDK 的 subagents）。

        主 Agent 调用该工具 -> 用 `task` 参数运行本子 Agent -> 返回其最终答案文本。
        子 Agent 有自己的 system/工具/权限，专注子任务。用法：
            researcher = Agent(..., name="researcher", description="查资料")
            main = Agent(..., tools=[researcher.as_tool()])
        """
        tname = name or self.name or "subagent"
        desc = description or self.description or f"把子任务委派给 {tname} 子代理处理"

        def delegate(task: str) -> str:
            return self.run(task).text

        delegate._naga_tool_name = tname            # type: ignore[attr-defined]
        delegate._naga_tool_desc = desc              # type: ignore[attr-defined]
        delegate._naga_tool_schema = {               # type: ignore[attr-defined]
            "type": "object",
            "properties": {"task": {"type": "string", "description": "交给子代理完成的任务描述"}},
            "required": ["task"],
        }
        return delegate
