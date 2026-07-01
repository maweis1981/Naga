#!/usr/bin/env python3
"""Naga Agent SDK 演示：用本地函数工具跑一个 Agent（对标 OpenAI/Claude Agent SDK）。

运行（首次会下载模型）：
    .venv/bin/python examples/agent_sdk_demo.py

工具调用对模型能力有要求，建议用 3B 及以上模型。可用环境变量切换：
    NAGA_DEMO_MODEL=Qwen/Qwen2.5-3B-Instruct .venv/bin/python examples/agent_sdk_demo.py
"""

import os

from naga.sdk import Agent, tool


@tool
def add(a: int, b: int) -> str:
    """把两个整数相加并返回结果。"""
    return str(a + b)


@tool(description="查询某城市的当前天气（演示用，返回固定值）")
def get_weather(city: str) -> str:
    return f"{city}：晴，24°C"


def main():
    model = os.environ.get("NAGA_DEMO_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    agent = Agent(
        model=model,
        tools=[add, get_weather],
        system="你是一个会使用工具的助手，需要时调用工具，然后用自然语言回答。",
        quantize=True, bits=4,            # 省显存 + 提速
        max_tokens=256,
    )

    print("可用工具：", [t["name"] for t in agent.tools])
    for q in ["帮我算一下 128 加 256 等于多少？", "北京现在天气怎么样？"]:
        print(f"\n>>> {q}")
        result = agent.run(q)
        for step in result.steps:
            if step["type"] == "tool_call":
                print(f"  🔧 {step['name']}({step.get('arguments', {})})")
            else:
                print(f"  ↩ {step['result']}")
        print("  =", result.text)


if __name__ == "__main__":
    main()
