"""共享测试夹具：用「假引擎 / 假管理器 / 假 MCP 服务器」在不下载模型、不动 GPU 的
前提下，端到端地验证服务层、Agent SDK、MCP 客户端与自观测指标的逻辑。"""

from __future__ import annotations

import json
import sys
import textwrap

import pytest

from naga.engine import Chunk


class FakeTok:
    eos_ids = {0}

    def apply_chat_template(self, messages):
        return "\n".join(str(m.get("content", "")) for m in messages)

    def encode(self, s):
        return list(range(len(str(s).split())))

    def decode(self, ids):
        return ""


class _Args:
    max_position_embeddings = 32768


class PlainEngine:
    """只产文本：固定吐 'Hello, world' + 用量。"""

    model_id = "naga-test"

    def __init__(self):
        self.tok = FakeTok()
        self.args = _Args()

    def stream(self, messages, **kw):
        for d in ["Hello", ", ", "world"]:
            yield Chunk(delta=d)
        yield Chunk(done=True, prompt_tokens=7, completion_tokens=3)

    def batch_generate(self, messages_list, **kw):
        # 回显每条输入的最后一段文本，便于断言不串台
        out = []
        for m in messages_list:
            last = str(m[-1].get("content", "")) if m else ""
            out.append({"text": f"echo:{last}", "prompt_tokens": 5, "completion_tokens": 2})
        return out


class ToolEngine:
    """会调工具：首轮吐一个 add 工具调用，拿到 tool_response 后给最终答案。"""

    model_id = "naga-test"

    def __init__(self):
        self.tok = FakeTok()
        self.args = _Args()

    def stream(self, messages, **kw):
        joined = " ".join(str(m.get("content", "")) for m in messages)
        if "<tool_response" in joined:
            for d in ["答案", "是", "5"]:
                yield Chunk(delta=d)
        else:
            yield Chunk(delta='<tool_call>{"name":"add","arguments":{"a":2,"b":3}}</tool_call>')
        yield Chunk(done=True, prompt_tokens=7, completion_tokens=3)


class _Coll:
    def search(self, q):
        return []


class _MCP:
    def __init__(self, tools=None):
        self._tools = tools or []

    def tools(self):
        return self._tools


class FakeManager:
    def __init__(self, engine, **settings):
        self._engine = engine
        self.active = engine.model_id
        self.docs = _Coll()
        self.memory = _Coll()
        self.mcp = _MCP()
        self.settings = {
            "rag_enabled": False, "memory_enabled": False, "mcp_enabled": False,
            "system_prompt": "", "max_tokens": 64, "temperature": 0.0, "top_p": 1.0,
            **settings,
        }

    def get(self, model):
        return self._engine

    def state(self):
        return {
            "active": self.active, "loaded": [self.active],
            "available": [{"id": self.active, "type": "text", "size_gb": 0.5}],
        }


@pytest.fixture
def plain_engine():
    return PlainEngine()


@pytest.fixture
def tool_engine():
    return ToolEngine()


@pytest.fixture
def client(plain_engine):
    """挂上假管理器的 FastAPI TestClient。"""
    from fastapi.testclient import TestClient

    from naga import server
    server.manager = FakeManager(plain_engine)
    return TestClient(server.app)


@pytest.fixture
def tool_client(tool_engine):
    """引擎会吐 add 工具调用的 TestClient（测 OpenAI tools/tool_calls 路径）。"""
    from fastapi.testclient import TestClient

    from naga import server
    server.manager = FakeManager(tool_engine)
    return TestClient(server.app)


@pytest.fixture
def fake_mcp_server(tmp_path):
    """写一个最小 stdio MCP 服务器脚本，返回 (python, [script_path])。"""
    script = tmp_path / "fake_mcp.py"
    script.write_text(textwrap.dedent('''
        import sys, json, time
        def send(o): sys.stdout.write(json.dumps(o)+"\\n"); sys.stdout.flush()
        HANG = "--hang" in sys.argv
        for line in sys.stdin:
            line = line.strip()
            if not line: continue
            msg = json.loads(line)
            mid, method = msg.get("id"), msg.get("method")
            if method == "initialize":
                send({"jsonrpc":"2.0","id":mid,"result":{"protocolVersion":"2024-11-05","capabilities":{}}})
            elif method == "notifications/initialized":
                pass
            elif method == "tools/list":
                send({"jsonrpc":"2.0","id":mid,"result":{"tools":[{"name":"echo","description":"echo","inputSchema":{"type":"object","properties":{"text":{"type":"string"}}}}]}})
            elif method == "tools/call":
                if HANG: time.sleep(30)
                send({"jsonrpc":"2.0","id":mid,"result":{"content":[{"type":"text","text":"ECHO: "+msg["params"]["arguments"].get("text","")}]}})
    '''))
    return sys.executable, [str(script)]
