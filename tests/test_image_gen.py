"""端到端回归：对话中调用 Floniks 生成图片（异步多步任务）。

链路：模型 single_task 发起任务 → 拿到 task_id → 轮询 get_task（processing→done）
→ 拿到 image_url → 作答。断言 trace 完整记录每一步 tool.call（参数+返回），
且最终答复带出图片 URL。用假引擎/假 MCP，不下载模型、确定性、进回归。"""
import math
from naga.agent import run_agent
from naga.engine import Chunk
from naga.toolindex import ToolIndex
from naga.trace import tracer

VOCAB = ["image", "picture", "生成", "图片", "task", "result", "credit", "list", "workflow", "weather"]


class FakeEmb:
    def encode(self, t):
        t = str(t).lower()
        v = [1.0 if w in t else 0.0 for w in VOCAB]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


TOOLS = [
    {"name": "single_task", "description": "生成图片 generate an image picture from a prompt", "schema": {}},
    {"name": "get_task", "description": "查询 task 任务状态，返回生成的 image 图片结果 result", "schema": {}},
    {"name": "get_credit_balance", "description": "credit balance 余额", "schema": {}},
    {"name": "list_workflows", "description": "list workflow", "schema": {}},
    {"name": "get_weather", "description": "weather city", "schema": {}},
    {"name": "add", "description": "add numbers", "schema": {}},
    {"name": "list_models", "description": "list models", "schema": {}},
    {"name": "list_templates", "description": "list templates", "schema": {}},
    {"name": "create_workflow", "description": "create workflow", "schema": {}},
    {"name": "current_time", "description": "current time", "schema": {}},
]


class Tok:
    eos_ids = (0,)
    def apply_chat_template(self, m): return ""
    def encode(self, s): return [0]
    def decode(self, i): return ""


class Eng:
    """模拟会异步编排的模型：先 single_task，再多轮 get_task，拿到图后作答。"""
    model_id = "t"
    tok = Tok()

    def stream(self, msgs, **kw):
        j = " ".join(str(m.get("content", "")) for m in msgs)
        if "image_url" in j:
            yield Chunk(delta="图片已生成："); yield Chunk(delta="![img](https://cdn.floniks.com/out/abc.png)")
        elif "task_id" in j or "processing" in j:
            yield Chunk(delta='{"name":"get_task","arguments":{"task_id":"t_abc"}}')
        else:
            yield Chunk(delta='{"name":"single_task","arguments":{"prompt":"a cat"}}')
        yield Chunk(done=True)


class MCP:
    def __init__(self):
        self._polls = 0
    def tools(self): return TOOLS
    def call(self, name, args):
        if name == "single_task":
            return '{"task_id":"t_abc","status":"queued"}'
        if name == "get_task":
            self._polls += 1
            return '{"status":"processing"}' if self._polls < 2 else \
                   '{"status":"done","image_url":"https://cdn.floniks.com/out/abc.png"}'
        return "x"


def test_floniks_image_generation(tmp_path):
    idx = ToolIndex(tmp_path / "ti.json", embedder=FakeEmb())
    idx.sync(TOOLS)
    query = "帮我用 Floniks 生成一张图片 image picture 图片"
    selector = idx.selector(query, top_k=6, threshold=8)

    trace = tracer.start(query)
    with tracer.bound(trace):
        events = list(run_agent(Eng(), MCP(),
                                [{"role": "user", "content": query}],
                                tool_selector=selector, max_steps=6))
    d = trace.to_dict()

    ir = next(s for s in d["spans"] if s["name"] == "intent_route")
    assert ir["attrs"]["triggered"] is True
    assert "single_task" in ir["attrs"]["selected"]
    assert "get_task" in ir["attrs"]["selected"]

    tool_calls = [s for s in d["spans"] if s["name"] == "tool.call"]
    called = [s["attrs"]["name"] for s in tool_calls]
    assert called[0] == "single_task"
    assert called.count("get_task") >= 2

    results = " ".join(s["attrs"].get("result", "") for s in tool_calls)
    assert "t_abc" in results and "image_url" in results and "abc.png" in results

    final = "".join(data for kind, data in events if kind == "final")
    assert "abc.png" in final
