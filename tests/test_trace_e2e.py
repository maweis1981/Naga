"""端到端回归：用户问余额 → 意图路由触发并选中查余额工具 → 工具被真实调用 →
trace 完整记录 intent_route（触发+选中+分数）与 tool.call（参数+返回）→ agent 据此作答。

假引擎/假 MCP/假嵌入器，不下载模型、确定性、秒级，锁住"意图路由 + 追踪"整条链路。"""
import math
from naga.agent import run_agent
from naga.engine import Chunk
from naga.toolindex import ToolIndex
from naga.trace import tracer

VOCAB = ["credit", "balance", "余额", "weather", "add", "list", "workflow", "task", "model", "time"]


class FakeEmb:
    def encode(self, t):
        t = str(t).lower()
        v = [1.0 if w in t else 0.0 for w in VOCAB]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


TOOLS = [
    {"name": "get_credit_balance", "description": "查询 Floniks credit balance 余额", "schema": {}},
    {"name": "get_weather", "description": "weather for a city", "schema": {}},
    {"name": "add", "description": "add two numbers", "schema": {}},
    {"name": "list_workflows", "description": "list workflow", "schema": {}},
    {"name": "get_task", "description": "task status", "schema": {}},
    {"name": "list_models", "description": "list model", "schema": {}},
    {"name": "current_time", "description": "current time", "schema": {}},
    {"name": "create_workflow", "description": "create workflow", "schema": {}},
    {"name": "execute_workflow", "description": "run workflow", "schema": {}},
    {"name": "list_templates", "description": "list templates", "schema": {}},
]


class Tok:
    eos_ids = (0,)
    def apply_chat_template(self, m): return ""
    def encode(self, s): return [0]
    def decode(self, i): return ""


class Eng:
    model_id = "t"
    tok = Tok()
    def stream(self, msgs, **kw):
        j = " ".join(str(m.get("content", "")) for m in msgs)
        if "<tool_response" in j:
            yield Chunk(delta="你的余额是 "); yield Chunk(delta="132535")
        else:
            yield Chunk(delta='{"name":"get_credit_balance","arguments":{}}')
        yield Chunk(done=True)


class MCP:
    def tools(self): return TOOLS
    def call(self, name, args):
        return '{"onetime_purchase_credit":132535}' if name == "get_credit_balance" else "x"


def test_floniks_balance_intent_route_and_trace(tmp_path):
    idx = ToolIndex(tmp_path / "ti.json", embedder=FakeEmb())
    idx.sync(TOOLS)
    query = "看一下我的 floniks 余额 credit balance"
    selector = idx.selector(query, top_k=5, threshold=8)   # 10 > 8 → 触发意图路由

    trace = tracer.start(query)
    with tracer.bound(trace):
        events = list(run_agent(Eng(), MCP(),
                                [{"role": "user", "content": query}],
                                tool_selector=selector))
    d = trace.to_dict()
    names = [s["name"] for s in d["spans"]]
    assert "intent_route" in names and "tool.call" in names

    ir = next(s for s in d["spans"] if s["name"] == "intent_route")
    assert ir["attrs"]["triggered"] is True
    assert "get_credit_balance" in ir["attrs"]["selected"]
    # 候选里查余额工具分数最高
    assert ir["attrs"]["candidates"][0]["tool"] == "get_credit_balance"

    tc = next(s for s in d["spans"] if s["name"] == "tool.call")
    assert tc["attrs"]["name"] == "get_credit_balance"
    assert "132535" in tc["attrs"]["result"]

    final = "".join(data for kind, data in events if kind == "final")
    assert "132535" in final
