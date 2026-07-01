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


def _looks_call(obj) -> bool:
    return isinstance(obj, dict) and "name" in obj


def parse_tool_calls(text: str) -> list[dict]:
    """从模型输出里解析出**所有**工具调用（支持单轮并行多工具，对齐 OpenAI tool_calls 数组）。

    容错解析三种形态：
      1) 多个 <tool_call>{...}</tool_call> 标签；
      2) 顶层 JSON 数组 [{...}, {...}]；
      3) 裸拼接的多个对象 {...}{...} —— 用 raw_decode 从每个 '{' 起点逐个抽取，
         跳过重叠（已消费区间），既拿全又不重复。
    去重（按 name+arguments），保持出现顺序。
    """
    calls: list[dict] = []

    # 1) 所有 <tool_call> 标签
    for m in TOOL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if _looks_call(obj):
            calls.append(obj)

    # 2/3) 扫描裸 JSON：对象要求含 name+arguments；数组则展开其中的调用
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in "{[":
            try:
                obj, end = dec.raw_decode(text[i:])
            except json.JSONDecodeError:
                i += 1
                continue
            if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                calls.append(obj)
            elif isinstance(obj, list):
                calls.extend(o for o in obj if isinstance(o, dict) and "name" in o and "arguments" in o)
            i += end                      # 跳过整个已解析的 JSON，避免重复/嵌套误抓
            continue
        i += 1

    # 去重（保持顺序）
    seen, out = set(), []
    for c in calls:
        key = (c.get("name"), json.dumps(c.get("arguments", {}), sort_keys=True, ensure_ascii=False))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def parse_tool_call(text: str):
    """向后兼容：返回第一个工具调用（单工具老路径仍用它）。"""
    calls = parse_tool_calls(text)
    return calls[0] if calls else None


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


def _peek_stream(engine, msgs, params):
    """流式生成，但先"窥视"开头判断是不是工具调用尝试。

    返回 (looks_tool, buffered, gen)：buffered 是已消费的开头 delta 列表，gen 是尚未消费
    的剩余生成器。工具调用会以 `<` / `{` 或 tool_call 开头——不像工具调用就可以放心一路
    流式吐给用户（普通回答有打字机效果）；像工具调用才需要收全再解析。"""
    gen = engine.stream(msgs, **params)
    buffered: list[str] = []
    looks_tool = False
    for ch in gen:
        if ch.done:
            break
        if ch.delta:
            buffered.append(ch.delta)
        head = "".join(buffered).lstrip()
        if head:                                  # 拿到第一段可见文本即可判定
            looks_tool = head[0] in "<{" or head.lower().startswith("tool_call")
            break
    return looks_tool, buffered, gen


def run_agent(engine, mcp, messages, max_steps: int = 5,
              tool_choice: str = "auto", **params):
    """生成器，逐步 yield (kind, data)：kind ∈ {'delta','tool_call','tool_result','final'}。

    'delta' 是最终答案的逐 token 增量（含工具执行后的作答也流式）；'final' 在结尾再发一次
    完整答案，方便只要聚合结果的消费者（如 Agent SDK）。普通回答只有第一段被短暂缓冲用于
    判别是否工具调用，其余一路流式——修掉了"开了 MCP 后普通对话整段返回、不流式"的问题。

    tool_choice（仿 OpenAI 语义）：'auto' 模型自决（格式错用约束解码修复）；'required' 强制
    首步产出合法工具调用；'none' 纯自由生成。"""
    tools = mcp.tools()
    names = [t["name"] for t in tools]
    msgs = list(messages)
    if tools:
        msgs = [{"role": "system", "content": build_tool_prompt(tools)}] + msgs

    def stream_answer(buffered, gen):
        """把已缓冲开头 + 剩余生成逐 token yield 成 delta，末尾发一次 final。"""
        parts: list[str] = []
        for d in buffered:
            if d:
                parts.append(d)
                yield ("delta", d)
        for ch in gen:
            if ch.done:
                break
            if ch.delta:
                parts.append(ch.delta)
                yield ("delta", ch.delta)
        yield ("final", "".join(parts))

    text = ""
    for step in range(max_steps):
        text = ""
        if tool_choice == "required" and step == 0 and tools:
            call = constrained_tool_call(engine, msgs, tools)   # 强制：保证合法调用
            calls = [call] if _valid_call(call, names) else []
        else:
            looks_tool, buffered, gen = _peek_stream(engine, msgs, params)
            if not looks_tool:                    # 普通回答：直接流式吐出，结束
                yield from stream_answer(buffered, gen)
                return
            # 像工具调用：收全剩余、解析出**所有**工具调用（单轮可含多个）
            text = "".join(buffered) + "".join(ch.delta for ch in gen if not ch.done)
            calls = [c for c in parse_tool_calls(text) if _valid_call(c, names)]
            # auto：像在尝试调用却一个都没解析出合法的 -> 约束解码修复出一个
            if tool_choice == "auto" and tools and not calls \
                    and _looks_like_tool_attempt(text, names):
                fixed = constrained_tool_call(engine, msgs, tools)
                if _valid_call(fixed, names):
                    calls = [fixed]
        if not calls:                             # 没有有效工具调用 -> 当最终答案
            if text:
                yield ("delta", text)
            yield ("final", text)
            return

        # 单轮执行**全部**工具调用（并行语义），逐个 emit，再一次性回填历史
        results = []
        for call in calls:
            yield ("tool_call", call)
            result = mcp.call(call["name"], call.get("arguments", {}))
            yield ("tool_result", {"name": call["name"], "result": result})
            results.append((call, result))
        # 回填：一条 assistant（含所有调用）+ 一条 user（汇总所有工具结果）
        assistant_text = text or "\n".join(
            f"<tool_call>{json.dumps(c, ensure_ascii=False)}</tool_call>" for c, _ in results)
        combined = "\n".join(
            f"<tool_response name=\"{c['name']}\">{r}</tool_response>" for c, r in results)
        msgs = msgs + [
            {"role": "assistant", "content": assistant_text},
            {"role": "user", "content": f"{combined}\n请综合以上工具结果回答用户的问题。"},
        ]
    if text:
        yield ("delta", text)
    yield ("final", text)
