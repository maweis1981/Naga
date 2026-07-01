"""MCP 客户端（naga.mcp）的回归测试：stdio 正常调用 + 超时不阻塞；HTTP 传输 JSON/SSE。"""

import http.server
import json
import threading
import time

import pytest

from naga.mcp import MCPClient, MCPHttpClient


def test_mcp_initialize_and_call(fake_mcp_server):
    cmd, args = fake_mcp_server
    c = MCPClient("fake", cmd, args)
    try:
        tools = c.initialize()
        assert [t["name"] for t in tools] == ["echo"]
        assert c.call("echo", {"text": "hi"}) == "ECHO: hi"
    finally:
        c.close()


def test_mcp_call_times_out_fast(fake_mcp_server):
    cmd, args = fake_mcp_server
    c = MCPClient("hang", cmd, args + ["--hang"])
    try:
        c.initialize()
        t0 = time.monotonic()
        with pytest.raises(RuntimeError, match="超时"):
            c._rpc("tools/call", {"name": "echo", "arguments": {"text": "x"}}, timeout=1.0)
        elapsed = time.monotonic() - t0
        assert elapsed < 5.0          # 必须快速失败，而不是挂到天荒地老
    finally:
        c.close()


def _make_http_mcp(sse=False):
    """在后台线程起一个最小 HTTP MCP 服务器；sse=True 时用 text/event-stream 回复。"""

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            msg = json.loads(self.rfile.read(n) or b"{}")
            mid, method = msg.get("id"), msg.get("method")
            if mid is None:                       # 通知：202 无响应体
                self.send_response(202)
                self.end_headers()
                return
            if method == "initialize":
                res = {"protocolVersion": "2024-11-05", "capabilities": {}}
            elif method == "tools/list":
                res = {"tools": [{"name": "ping", "description": "ping",
                                  "inputSchema": {"type": "object", "properties": {}}}]}
            elif method == "tools/call":
                res = {"content": [{"type": "text",
                                    "text": "pong:" + msg["params"]["arguments"].get("x", "")}]}
            else:
                res = {}
            env = {"jsonrpc": "2.0", "id": mid, "result": res}
            if sse:
                body = ("data: " + json.dumps(env) + "\n\n").encode()
                ctype = "text/event-stream"
            else:
                body = json.dumps(env).encode()
                ctype = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Mcp-Session-Id", "test-session")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/mcp"


def test_http_mcp_client_json():
    srv, url = _make_http_mcp(sse=False)
    try:
        c = MCPHttpClient("http", url)
        assert [t["name"] for t in c.initialize()] == ["ping"]
        assert c.session_id == "test-session"      # 会话 id 从响应头捕获
        assert c.call("ping", {"x": "hi"}) == "pong:hi"
    finally:
        srv.shutdown()


def test_http_mcp_client_sse():
    srv, url = _make_http_mcp(sse=True)
    try:
        c = MCPHttpClient("http", url)
        assert [t["name"] for t in c.initialize()] == ["ping"]
        assert c.call("ping", {"x": "sse"}) == "pong:sse"   # 解析 SSE data 帧
    finally:
        srv.shutdown()


def test_manager_routes_url_to_http(tmp_path, monkeypatch):
    import naga.mcp as mcp_mod
    monkeypatch.setattr(mcp_mod, "MCP_FILE", tmp_path / "mcp.json")
    srv, url = _make_http_mcp(sse=False)
    try:
        mgr = mcp_mod.MCPManager()
        err = mgr.add_http_server("remote", url)
        assert err is None
        st = mgr.state()
        server = st["servers"][0]
        assert server["transport"] == "http" and server["connected"]
        assert "ping" in server["tools"]
        assert mgr.call("ping", {"x": "z"}) == "pong:z"
    finally:
        srv.shutdown()
