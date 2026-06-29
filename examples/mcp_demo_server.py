#!/usr/bin/env python3
"""一个最小的演示用 MCP 服务器（stdio JSON-RPC）。

提供两个工具：add（两数相加）、current_time（当前时间）。
仅用于验证 Naga 的 MCP 客户端和工具调用循环，不依赖任何外部包。

配置示例（~/.naga/mcp.json）：
  {"mcpServers": {"demo": {"command": "python3",
     "args": ["/绝对路径/naga/examples/mcp_demo_server.py"]}}}
"""

import datetime
import json
import sys

TOOLS = [
    {"name": "add", "description": "计算两个数字相加的结果",
     "inputSchema": {"type": "object",
                     "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                     "required": ["a", "b"]}},
    {"name": "current_time", "description": "获取服务器当前的日期和时间",
     "inputSchema": {"type": "object", "properties": {}}},
]


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid = msg.get("id")
        method = msg.get("method")

        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "demo", "version": "0.1"}}})
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            p = msg.get("params", {})
            name, args = p.get("name"), p.get("arguments", {})
            if name == "add":
                text = str(args.get("a", 0) + args.get("b", 0))
            elif name == "current_time":
                text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            else:
                text = f"unknown tool: {name}"
            send({"jsonrpc": "2.0", "id": mid,
                  "result": {"content": [{"type": "text", "text": text}]}})
        elif mid is not None:
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32601, "message": "method not found"}})


if __name__ == "__main__":
    main()
