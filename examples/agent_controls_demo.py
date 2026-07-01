#!/usr/bin/env python3
"""Naga Agent 控制面演示：权限 / Hooks / 用量 / 子代理委派。

运行（首次会下载模型）：
    .venv/bin/python examples/agent_controls_demo.py
"""
import os

from naga.sdk import Agent, tool


@tool
def get_secret(key: str) -> str:
    """读取一个机密配置值（演示：应被权限拒绝）。"""
    return f"SECRET[{key}]=hunter2"


@tool
def add(a: int, b: int) -> str:
    """两个整数相加。"""
    return str(a + b)


def main():
    model = os.environ.get("NAGA_DEMO_MODEL", "Qwen/Qwen2.5-3B-Instruct")

    # 1) 权限 + Hooks：拒绝 get_secret，审计每次尝试
    audit = []
    agent = Agent(
        model=model, tools=[get_secret, add],
        system="你是助手，需要时调用工具。",
        can_use_tool=lambda name, args: "安全策略：禁止读取机密" if name == "get_secret" else True,
        on_tool_call=lambda name, args: audit.append((name, args)),
        max_steps=3, max_tokens=200,
    )
    res = agent.run("帮我读取 key 为 db_password 的机密配置。")
    print("=== 权限 + Hooks ===")
    print("  审计尝试:", audit)
    print("  用量:", res.usage)
    print("  答复:", res.text)

    # 2) 子代理委派：主代理把算术交给 calculator 子代理
    calc = Agent(
        engine=agent.engine, tools=[add], name="calculator",
        description="专门做算术计算的子代理，传入 task 描述要算什么",
        system="你是计算器，用 add 工具算数，然后只回答数字。", max_steps=3, max_tokens=150,
    )
    main_agent = Agent(
        engine=agent.engine, tools=[calc.as_tool()],
        system="你是助手。遇到算术就委派给 calculator 子代理。", max_steps=3, max_tokens=200,
    )
    res2 = main_agent.run("请帮我算 4567 加 8901 等于多少。")
    print("\n=== 子代理委派 ===")
    print("  步骤:", [(s["type"], s.get("name")) for s in res2.steps])
    print("  答复:", res2.text)


if __name__ == "__main__":
    main()
