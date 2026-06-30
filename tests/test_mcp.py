"""MCP 客户端（naga.mcp）的回归测试：正常调用 + 超时不再永久阻塞。"""

import time

import pytest

from naga.mcp import MCPClient


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
