"""MCP（Model Context Protocol）客户端 + 多服务器管理（P10）。

MCP 是一套让"工具服务器"以标准方式暴露能力的协议。本地服务器走 stdio：
我们 spawn 一个子进程，用换行分隔的 JSON-RPC 2.0 跟它通信。

握手流程：
  initialize 请求 → 服务器回能力 → 发 notifications/initialized 通知
  → tools/list 拿工具清单 → tools/call 调用工具。

配置存 ~/.naga/mcp.json（格式同 Claude Desktop 的 mcpServers）。
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

MCP_FILE = Path.home() / ".naga" / "mcp.json"


class MCPClient:
    """连接单个 stdio MCP 服务器。"""

    def __init__(self, name: str, command: str, args=None, env=None):
        self.name = name
        full_env = {**os.environ, **(env or {})}
        self.proc = subprocess.Popen(
            [command, *(args or [])],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, env=full_env,
        )
        self._id = 0
        self.tools: list[dict] = []

    def _rpc(self, method: str, params=None):
        self._id += 1
        mid = self._id
        msg = {"jsonrpc": "2.0", "id": mid, "method": method}
        if params is not None:
            msg["params"] = params
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        # 读到 id 匹配的响应为止（中途的通知/无关消息忽略）
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"MCP 服务 {self.name} 无响应或已退出")
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("id") == mid:
                if "error" in obj:
                    raise RuntimeError(obj["error"].get("message", "MCP error"))
                return obj.get("result")

    def _notify(self, method: str, params=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def initialize(self) -> list[dict]:
        self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "naga", "version": "0.1"},
        })
        self._notify("notifications/initialized")
        res = self._rpc("tools/list") or {}
        self.tools = res.get("tools", [])
        return self.tools

    def call(self, tool: str, arguments: dict) -> str:
        res = self._rpc("tools/call", {"name": tool, "arguments": arguments}) or {}
        parts = [c.get("text", "") for c in res.get("content", []) if c.get("type") == "text"]
        return "\n".join(parts) if parts else json.dumps(res, ensure_ascii=False)

    def close(self):
        try:
            self.proc.terminate()
        except Exception:
            pass


class MCPManager:
    def __init__(self):
        self.clients: dict[str, MCPClient] = {}
        self.errors: dict[str, str] = {}
        self.config = self._load()

    def _load(self) -> dict:
        if MCP_FILE.exists():
            try:
                return json.loads(MCP_FILE.read_text())
            except Exception:
                pass
        return {"mcpServers": {}}

    def _save(self):
        MCP_FILE.parent.mkdir(exist_ok=True)
        MCP_FILE.write_text(json.dumps(self.config, ensure_ascii=False, indent=2))

    def connect_all(self):
        for name, spec in self.config.get("mcpServers", {}).items():
            self._connect(name, spec)

    def _connect(self, name: str, spec: dict):
        try:
            c = MCPClient(name, spec["command"], spec.get("args", []), spec.get("env"))
            c.initialize()
            self.clients[name] = c
            self.errors.pop(name, None)
        except Exception as e:
            self.errors[name] = str(e)

    def add_server(self, name: str, command: str, args=None, env=None) -> str | None:
        spec = {"command": command, "args": args or []}
        if env:
            spec["env"] = env
        self.config.setdefault("mcpServers", {})[name] = spec
        self._save()
        if name in self.clients:
            self.clients[name].close()
            del self.clients[name]
        self._connect(name, spec)
        return self.errors.get(name)

    def remove_server(self, name: str):
        self.config.get("mcpServers", {}).pop(name, None)
        self._save()
        if name in self.clients:
            self.clients[name].close()
            del self.clients[name]
        self.errors.pop(name, None)

    def tools(self) -> list[dict]:
        out = []
        for name, c in self.clients.items():
            for t in c.tools:
                out.append({"server": name, "name": t["name"],
                            "description": t.get("description", ""),
                            "schema": t.get("inputSchema", {})})
        return out

    def call(self, tool_name: str, arguments: dict) -> str:
        for c in self.clients.values():
            if any(t["name"] == tool_name for t in c.tools):
                try:
                    return c.call(tool_name, arguments)
                except Exception as e:
                    return f"[工具执行出错] {e}"
        return f"[错误] 找不到工具 {tool_name}"

    def state(self) -> dict:
        servers = []
        for name, spec in self.config.get("mcpServers", {}).items():
            connected = name in self.clients
            servers.append({
                "name": name, "command": spec["command"], "args": spec.get("args", []),
                "connected": connected, "error": self.errors.get(name),
                "tools": [t["name"] for t in self.clients[name].tools] if connected else [],
            })
        return {"servers": servers, "tools": self.tools()}
