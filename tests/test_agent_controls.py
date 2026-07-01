"""Agent 控制面测试：权限（can_use_tool / allowed / blocked）、Hooks、用量统计。"""

from naga.agent import apply_permission, run_agent
from naga.engine import Chunk
from naga.sdk import Agent, tool


# ── apply_permission 契约 ──────────────────────────────────────────
def test_apply_permission_variants():
    assert apply_permission(None, "x", {"a": 1}) == (True, {"a": 1}, "")
    assert apply_permission(lambda n, a: True, "x", {}) == (True, {}, "")
    assert apply_permission(lambda n, a: False, "x", {})[0] is False
    ok, args, reason = apply_permission(lambda n, a: "nope", "x", {})
    assert ok is False and reason == "nope"
    # 改写参数
    ok, args, reason = apply_permission(lambda n, a: {"input": {"a": 2}}, "x", {"a": 1})
    assert ok is True and args == {"a": 2}
    # 结构化拒绝
    ok, args, reason = apply_permission(lambda n, a: {"allow": False, "reason": "denied"}, "x", {})
    assert ok is False and reason == "denied"
    # 回调抛异常 -> 安全拒绝
    def boom(n, a):
        raise ValueError("x")
    assert apply_permission(boom, "x", {})[0] is False


# ── 假引擎/MCP：先调 secret_tool，拿到结果后作答 ───────────────────
class _Tok:
    eos_ids = (0,)

    def apply_chat_template(self, m):
        return ""

    def encode(self, s):
        return list(range(len(str(s))))

    def decode(self, i):
        return ""


class ToolEng:
    model_id = "t"
    tok = _Tok()

    def stream(self, msgs, **kw):
        joined = " ".join(str(m.get("content", "")) for m in msgs)
        if "<tool_response" in joined:
            for w in ["done"]:
                yield Chunk(delta=w)
        else:
            yield Chunk(delta='{"name":"secret_tool","arguments":{"q":"data"}}')
        yield Chunk(done=True)


class _MCP:
    def tools(self):
        return [{"name": "secret_tool", "description": "", "schema": {}}]

    def call(self, name, args):
        return f"EXECUTED:{args.get('q')}"


def test_permission_denies_tool():
    calls = list(run_agent(ToolEng(), _MCP(), [{"role": "user", "content": "go"}],
                           permission=lambda n, a: "not allowed"))
    results = [d for k, d in calls if k == "tool_result"]
    assert results[0]["denied"] is True
    assert "not allowed" in results[0]["result"]
    assert "EXECUTED" not in results[0]["result"]     # 工具没被真正执行


def test_permission_allows_and_rewrites_args():
    seen = {}

    def cap(n, a):
        seen["args"] = a
        return "42"

    mcp = _MCP()
    mcp.call = cap
    events = list(run_agent(ToolEng(), mcp, [{"role": "user", "content": "go"}],
                            permission=lambda n, a: {"input": {"q": "SANITIZED"}}))
    assert seen["args"] == {"q": "SANITIZED"}          # 权限改写了参数后才执行


def test_hooks_fire_pre_and_post():
    pre, post = [], []
    list(run_agent(ToolEng(), _MCP(), [{"role": "user", "content": "go"}],
                   on_tool_call=lambda n, a: pre.append(n),
                   on_tool_result=lambda n, r: post.append((n, r))))
    assert pre == ["secret_tool"]
    assert post and post[0][0] == "secret_tool" and "EXECUTED" in post[0][1]


# ── Agent SDK 层：allowed/blocked + usage ─────────────────────────
class _AgentTok(_Tok):
    pass


class SdkToolEng:
    model_id = "t"
    tok = _AgentTok()

    def stream(self, msgs, **kw):
        joined = " ".join(str(m.get("content", "")) for m in msgs)
        if "<tool_response" in joined:
            yield Chunk(delta="ok")
        else:
            yield Chunk(delta='{"name":"add","arguments":{"a":1,"b":2}}')
        yield Chunk(done=True)


def _add_tool():
    @tool
    def add(a: int, b: int) -> str:
        return str(a + b)
    return add


def test_agent_blocked_tools():
    agent = Agent(engine=SdkToolEng(), tools=[_add_tool()], blocked_tools=["add"])
    res = agent.run("add 1 and 2")
    denied = [s for s in res.steps if s["type"] == "tool_result" and s.get("denied")]
    assert denied and res.usage["denied"] == 1


def test_agent_allowed_tools_gate():
    agent = Agent(engine=SdkToolEng(), tools=[_add_tool()], allowed_tools=["other"])
    res = agent.run("add 1 and 2")
    assert res.usage["denied"] == 1                    # add 不在允许名单


def test_agent_usage_populated():
    agent = Agent(engine=SdkToolEng(), tools=[_add_tool()])
    res = agent.run("add 1 and 2")
    assert res.usage["tool_calls"] == 1
    assert res.usage["denied"] == 0
    assert "completion_tokens" in res.usage


# ── 子代理委派（subagents） ────────────────────────────────────────
class SubagentEng:
    """子代理引擎：不调工具，直接给出答案 'SUBRESULT'。"""
    model_id = "t"
    tok = _AgentTok()

    def stream(self, msgs, **kw):
        yield Chunk(delta="SUBRESULT")
        yield Chunk(done=True)


class MainEng:
    """主代理引擎：首轮委派给 researcher 子代理，拿到结果后作答。"""
    model_id = "t"
    tok = _AgentTok()

    def stream(self, msgs, **kw):
        joined = " ".join(str(m.get("content", "")) for m in msgs)
        if "<tool_response" in joined:
            yield Chunk(delta="final based on sub")
        else:
            yield Chunk(delta='{"name":"researcher","arguments":{"task":"go find it"}}')
        yield Chunk(done=True)


def test_as_tool_metadata():
    sub = Agent(engine=SubagentEng(), name="researcher", description="查资料的子代理")
    fn = sub.as_tool()
    assert fn._naga_tool_name == "researcher"
    assert "task" in fn._naga_tool_schema["properties"]
    assert fn("anything") == "SUBRESULT"          # 直接调用即运行子代理


def test_subagent_delegation_through_main_agent():
    sub = Agent(engine=SubagentEng(), name="researcher", description="查资料")
    main = Agent(engine=MainEng(), tools=[sub.as_tool()])
    res = main.run("please research X")
    # 主代理调用了 researcher 工具，且拿到子代理结果 SUBRESULT
    calls = [s for s in res.steps if s["type"] == "tool_call"]
    assert calls and calls[0]["name"] == "researcher"
    results = [s for s in res.steps if s["type"] == "tool_result"]
    assert results[0]["result"] == "SUBRESULT"
    assert res.text == "final based on sub"
