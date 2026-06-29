"""A/B：工具调用 无约束 vs 约束解码。用 0.5B 小模型（最易翻车）跑多个请求，
统计"合法工具调用率"：能 JSON 解析 + 工具名真实存在 + arguments 是对象。"""
from __future__ import annotations
import json
from naga.loader import load_model
from naga.tokenizer import ChatTokenizer
from naga.generate import generate, generate_constrained
from naga.constrain import ToolCallConstraint
from naga.agent import build_tool_prompt, parse_tool_call

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

TOOLS = [
    {"name": "get_weather", "description": "查询某城市天气",
     "schema": {"properties": {"city": {"type": "string"}}}},
    {"name": "add", "description": "两数相加",
     "schema": {"properties": {"a": {"type": "number"}, "b": {"type": "number"}}}},
    {"name": "search_web", "description": "联网搜索关键词",
     "schema": {"properties": {"query": {"type": "string"}}}},
]
NAMES = [t["name"] for t in TOOLS]

REQUESTS = [
    "What's the weather in Tokyo?",
    "Add 17 and 25 for me.",
    "Search the web for the latest iPhone price.",
    "How hot is it in London right now?",
    "What is 128 plus 256?",
    "Find information about the Great Wall.",
]


def valid_call(obj) -> bool:
    return (isinstance(obj, dict) and obj.get("name") in NAMES
            and isinstance(obj.get("arguments"), dict))


def main():
    model, args, path = load_model(MODEL)
    tok = ChatTokenizer(path)
    sys = build_tool_prompt(TOOLS)

    def prompt_ids(user):
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
        return tok.encode(tok.apply_chat_template(msgs))

    print(f"{'请求':<42}{'无约束':>10}{'约束':>8}")
    n_un = n_co = 0
    for req in REQUESTS:
        pids = prompt_ids(req)

        # 无约束：自由生成，再正则/JSON 解析
        text = tok.decode(list(generate(model, pids, 80, tok.eos_ids, temp=0.0)))
        un_obj = parse_tool_call(text)
        un_ok = valid_call(un_obj)
        n_un += un_ok

        # 约束：只能产出合法工具调用
        con = ToolCallConstraint(NAMES)
        out = tok.decode(list(generate_constrained(
            model, pids, lambda ids: tok.decode(ids), con, 80, tok.eos_ids)))
        co_obj = parse_tool_call(out)
        co_ok = valid_call(co_obj)
        n_co += co_ok

        print(f"{req[:40]:<42}{('✓' if un_ok else '✗'):>9}{('✓' if co_ok else '✗'):>8}")
        if not un_ok:
            print(f"    无约束实际输出: {text.strip()[:70]!r}")
        print(f"    约束输出: {out.strip()[:70]!r}")

    print(f"\n合法工具调用率：无约束 {n_un}/{len(REQUESTS)}  |  约束 {n_co}/{len(REQUESTS)}")


if __name__ == "__main__":
    main()
