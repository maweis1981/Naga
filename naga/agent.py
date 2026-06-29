"""工具调用 Agent 循环（P10）。

把可用工具描述进系统提示，让模型在需要时输出
  <tool_call>{"name":"...","arguments":{...}}</tool_call>
我们解析出来、通过 MCP 执行、把结果喂回去，循环直到模型给出最终答案。

注意：工具调用对模型能力要求较高，小模型（如 0.5B）常常调不稳。
要实际好用，建议切换到 7B 级别的模型。
"""

from __future__ import annotations

import json
import re

from .constrain import ToolCallConstraint
from .generate import generate_constrained

TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def build_tool_prompt(tools: list[dict]) -> str:
    lines = ["你可以调用以下工具来获取信息或执行操作："]
    for t in tools:
        lines.append(f"- {t['name']}：{t['description']}　参数: "
                     f"{json.dumps(t['schema'].get('properties', {}), ensure_ascii=False)}")
    lines.append(
        '当需要工具时，只输出一行：<tool_call>{"name":"工具名","arguments":{参数}}</tool_call> '
        '然后停止，等我把结果给你；拿到结果后再用自然语言回答用户。不需要工具就直接回答。'
    )
    return "\n".join(lines)


def parse_tool_call(text: str):
    # 优先匹配 <tool_call>{...}</tool_call> 标签
    m = TOOL_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and "name" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    # 容错：模型常常不加标签、直接吐 JSON —— 扫描第一个含 name/arguments 的对象
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                return obj
    return None


def constrained_tool_call(engine, messages, tools, max_tokens: int = 128):
    """P14：用约束解码生成一个**保证合法**的工具调用（JSON 可解析 + 工具名真实）。

    复用 engine 的模型与分词器，绕过自由生成，直接在 ToolCallConstraint 下采样。
    """
    names = [t["name"] for t in tools]
    ids = engine.tok.encode(engine.tok.apply_chat_template(messages))
    con = ToolCallConstraint(names)
    out_ids = list(generate_constrained(
        engine.model, ids, lambda i: engine.tok.decode(i), con,
        max_tokens, engine.tok.eos_ids))
    return parse_tool_call(engine.tok.decode(out_ids))


def _valid_call(call, names) -> bool:
    return (isinstance(call, dict) and call.get("name") in names
            and isinstance(call.get("arguments"), dict))


def _looks_like_tool_attempt(text: str, names: list[str]) -> bool:
    """模型像是想调工具却没调对：出现了工具名、tool_call 标签或裸 JSON/函数调用形态。"""
    low = text.lower()
    return ("tool_call" in low or "{" in text or "(" in text
            or any(n in text for n in names))


def run_agent(engine, mcp, messages, max_steps: int = 5,
              tool_choice: str = "auto", **params):
    """生成器，逐步 yield (kind, data)：kind ∈ {'tool_call','tool_result','final'}。

    tool_choice（仿 OpenAI 语义，约束解码是落地手段）：
      'auto'（默认）—— 模型自己决定。它像要调工具却格式/名字错时，用约束解码修复。
      'required'   —— 强制第一步必产出一个合法工具调用（应用已决定要用工具时）。
                      即便 0.5B 小模型也稳，因为非法 token 在采样层被屏蔽。
      'none'       —— 不修复、不强制，纯自由生成（基线）。"""
    tools = mcp.tools()
    names = [t["name"] for t in tools]
    msgs = list(messages)
    if tools:
        msgs = [{"role": "system", "content": build_tool_prompt(tools)}] + msgs

    for step in range(max_steps):
        text = ""
        if tool_choice == "required" and step == 0 and tools:
            call = constrained_tool_call(engine, msgs, tools)   # 强制：保证合法调用
        else:
            text = "".join(ch.delta for ch in engine.stream(msgs, **params) if not ch.done)
            call = parse_tool_call(text)
            # auto：模型像在尝试调用、但自由解析没拿到合法调用 -> 约束修复
            if tool_choice == "auto" and tools and not _valid_call(call, names) \
                    and _looks_like_tool_attempt(text, names):
                call = constrained_tool_call(engine, msgs, tools)
        if not _valid_call(call, names):
            yield ("final", text)
            return
        yield ("tool_call", call)
        result = mcp.call(call["name"], call.get("arguments", {}))
        yield ("tool_result", {"name": call["name"], "result": result})
        # 回填历史：约束/强制路径下 text 为空，用序列化的调用作为助手发言
        assistant_text = text or f"<tool_call>{json.dumps(call, ensure_ascii=False)}</tool_call>"
        msgs = msgs + [
            {"role": "assistant", "content": assistant_text},
            {"role": "user", "content": f"<tool_response>{result}</tool_response>\n请根据该结果回答用户的问题。"},
        ]
    yield ("final", text)
