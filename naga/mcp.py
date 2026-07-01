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
import queue
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

MCP_FILE = Path.home() / ".naga" / "mcp.json"

# 单次 RPC 的默认超时（秒）。stdio MCP 服务器若卡死/慢响应，超时让请求快速失败，
# 不至于把 HTTP worker 线程永久挂住。可用环境变量覆盖。
DEFAULT_TIMEOUT = float(os.environ.get("NAGA_MCP_TIMEOUT", "60"))

_EOF = object()  # 读取线程在子进程 stdout 关闭时投入的哨兵


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
        # 后台读取线程把每行解析后投入队列，_rpc 带超时地等待匹配 id 的响应。
        # 这样既避免了「select + 缓冲 readline」的坑，又能给阻塞读加上超时。
        self._inbox: queue.Queue = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        try:
            for line in self.proc.stdout:           # 迭代到子进程 stdout 关闭为止
                line = line.strip()
                if not line:
                    continue
                try:
                    self._inbox.put(json.loads(line))
                except json.JSONDecodeError:
                    continue
        finally:
            self._inbox.put(_EOF)

    def _rpc(self, method: str, params=None, timeout: float = DEFAULT_TIMEOUT):
        self._id += 1
        mid = self._id
        msg = {"jsonrpc": "2.0", "id": mid, "method": method}
        if params is not None:
            msg["params"] = params
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError) as e:
            raise RuntimeError(f"MCP 服务 {self.name} 已退出（无法写入）: {e}")
        # 读到 id 匹配的响应为止（中途的通知/无关消息忽略），带总超时。
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"MCP 服务 {self.name} 调用 {method} 超时（{timeout:.0f}s）")
            try:
                obj = self._inbox.get(timeout=remaining)
            except queue.Empty:
                raise RuntimeError(f"MCP 服务 {self.name} 调用 {method} 超时（{timeout:.0f}s）")
            if obj is _EOF:
                raise RuntimeError(f"MCP 服务 {self.name} 无响应或已退出")
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


class MCPHttpClient:
    """连接一个 Streamable-HTTP MCP 服务器（JSON-RPC over HTTP POST）。

    现代 MCP 除 stdio 外还有 HTTP 传输：客户端把 JSON-RPC 请求 POST 到单个端点，
    服务器以 application/json（单响应）或 text/event-stream（SSE）回复——两种都处理。
    与 stdio 客户端同接口（initialize / tools / call / close），故能被 MCPManager 统一管理。
    这让 Naga 不止能接本地子进程，还能接远程/托管的 MCP 服务器。
    """

    def __init__(self, name: str, url: str, headers: dict | None = None,
                 timeout: float = DEFAULT_TIMEOUT):
        self.name = name
        self.url = url
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **(headers or {}),
        }
        self.session_id: str | None = None
        self._id = 0
        self.tools: list[dict] = []

    def _post(self, payload: dict, expect_response: bool = True):
        data = json.dumps(payload).encode()
        headers = dict(self.headers)
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except Exception as e:
            raise RuntimeError(f"MCP HTTP {self.name} 请求失败: {e}")
        with resp:
            sid = resp.headers.get("Mcp-Session-Id")
            if sid:
                self.session_id = sid
            if not expect_response:
                return None
            ctype = resp.headers.get("Content-Type", "")
            if "text/event-stream" in ctype:               # SSE：读到带 id 的 data 帧为止
                for raw in resp:
                    line = raw.decode().strip()
                    if line.startswith("data:"):
                        try:
                            obj = json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            continue
                        if isinstance(obj, dict) and "id" in obj:
                            return obj
                return None
            body = resp.read()
            return json.loads(body) if body else None

    def _rpc(self, method: str, params=None):
        self._id += 1
        msg = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            msg["params"] = params
        obj = self._post(msg)
        if not obj:
            raise RuntimeError(f"MCP HTTP {self.name} 无响应")
        if "error" in obj:
            raise RuntimeError(obj["error"].get("message", "MCP error"))
        return obj.get("result")

    def _notify(self, method: str, params=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._post(msg, expect_response=False)

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
            if spec.get("url"):                            # HTTP 传输
                c = MCPHttpClient(name, spec["url"], spec.get("headers"))
            else:                                          # stdio 传输
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

    def add_http_server(self, name: str, url: str, headers: dict | None = None) -> str | None:
        spec: dict = {"url": url}
        if headers:
            spec["headers"] = headers
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
                "name": name,
                "transport": "http" if spec.get("url") else "stdio",
                "command": spec.get("command"), "url": spec.get("url"),
                "args": spec.get("args", []),
                "connected": connected, "error": self.errors.get(name),
                "tools": [t["name"] for t in self.clients[name].tools] if connected else [],
            })
        return {"servers": servers, "tools": self.tools()}
