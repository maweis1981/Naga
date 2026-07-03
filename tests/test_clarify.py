"""参数澄清（naga.clarify）测试：enum 非法、动态枚举瞎编、合法不打扰、run_agent 集成。"""
from naga.clarify import needs_clarify, _dynamic_source


class TS:
    def call(self, name, args):
        if name == "list_model_aliases":
            return ('[{"id":"nano_banana_2","name":"Nano Banana 2","type":"text_to_image"},'
                    '{"id":"seedream_t2i","name":"Seedream","type":"text_to_image"},'
                    '{"id":"img2img_x","name":"I2I","type":"image_to_image"}]')
        return "[]"


def test_dynamic_source_parsed():
    assert _dynamic_source({"description": "the id returned by list_model_aliases, not X"}) == "list_model_aliases"
    assert _dynamic_source({"description": "just a string"}) is None


def test_enum_invalid_value_clarifies():
    spec = {"name": "t", "schema": {"required": ["size"],
            "properties": {"size": {"type": "string", "enum": ["1024", "2048"]}}}}
    c = needs_clarify(spec, {"size": "999"}, None)
    assert c and c["param"] == "size"
    assert {"value": "1024", "label": "1024"} in c["options"]


def test_dynamic_hallucinated_value_clarifies():
    spec = {"name": "single_task", "schema": {"required": ["modelId"],
            "properties": {"modelId": {"description": "the alias id returned by list_model_aliases"}}}}
    c = needs_clarify(spec, {"modelId": "portrait"}, TS())
    assert c and c["param"] == "modelId" and c["source"] == "list_model_aliases"
    vals = [o["value"] for o in c["options"]]
    assert "nano_banana_2" in vals and "seedream_t2i" in vals


def test_dynamic_filters_by_modeltype():
    spec = {"name": "single_task", "schema": {"required": ["modelId"],
            "properties": {"modelId": {"description": "returned by list_model_aliases"}}}}
    c = needs_clarify(spec, {"modelId": "bad", "modelType": "text_to_image"}, TS())
    vals = [o["value"] for o in c["options"]]
    assert "img2img_x" not in vals


def test_valid_value_no_clarify():
    spec = {"name": "single_task", "schema": {"required": ["modelId"],
            "properties": {"modelId": {"description": "returned by list_model_aliases"}}}}
    assert needs_clarify(spec, {"modelId": "nano_banana_2"}, TS()) is None


def test_run_agent_emits_clarify():
    from naga.agent import run_agent
    from naga.engine import Chunk

    class Tok:
        eos_ids = (0,)
        def apply_chat_template(self, m): return ""
        def encode(self, s): return [0]
        def decode(self, i): return ""

    class Eng:
        model_id = "t"; tok = Tok()
        def stream(self, msgs, **kw):
            yield Chunk(delta='{"name":"single_task","arguments":{"modelId":"portrait","prompt":"x"}}')
            yield Chunk(done=True)

    class MCP:
        def tools(self):
            return [{"name": "single_task", "description": "生成图",
                     "schema": {"required": ["modelId"],
                                "properties": {"modelId": {"description": "returned by list_model_aliases"}}}}]
        def call(self, n, a):
            return TS().call(n, a) if n == "list_model_aliases" else "IMG"

    events = list(run_agent(Eng(), MCP(), [{"role": "user", "content": "生成图片"}]))
    kinds = [k for k, _ in events]
    assert "clarify" in kinds
    assert "tool_result" not in kinds
    clar = next(d for k, d in events if k == "clarify")
    assert clar["param"] == "modelId" and any(o["value"] == "nano_banana_2" for o in clar["options"])


def test_tool_call_endpoint():
    from fastapi.testclient import TestClient
    from naga import server

    class ToolsetStub:
        def call(self, n, a):
            return f"RESULT:{n}:{a.get('modelId')}"

    class Mgr:
        agent_toolset = ToolsetStub()
        mcp = ToolsetStub()

    server.manager = Mgr()
    c = TestClient(server.app)
    r = c.post("/tool/call", json={"name": "single_task", "arguments": {"modelId": "nano_banana_2"}})
    assert r.status_code == 200
    assert r.json()["result"] == "RESULT:single_task:nano_banana_2"
    assert c.post("/tool/call", json={}).status_code == 400
