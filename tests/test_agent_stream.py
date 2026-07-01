"""run_agent 流式行为测试：普通回答逐 token（delta）、工具调用仍走缓冲解析。"""

from naga.agent import run_agent
from naga.engine import Chunk


class _MCP:
    def __init__(self, tools):
        self._t = tools

    def tools(self):
        return self._t

    def call(self, name, args):
        return "42"


class PlainEng:
    """普通回答：逐 token 吐 'The answer is here'。"""
    model_id = "t"

    class _Tok:
        def apply_chat_template(self, m):
            return ""

        def encode(self, s):
            return [0]

        def decode(self, i):
            return ""
        eos_ids = (0,)

    tok = _Tok()

    def stream(self, msgs, **kw):
        for w in ["The ", "answer ", "is ", "here"]:
            yield Chunk(delta=w)
        yield Chunk(done=True)


def test_plain_answer_streams_token_by_token():
    events = list(run_agent(PlainEng(), _MCP([]), [{"role": "user", "content": "hi"}]))
    kinds = [k for k, _ in events]
    deltas = [d for k, d in events if k == "delta"]
    assert kinds.count("delta") >= 2                 # 逐 token，而不是一整块
    assert "".join(deltas) == "The answer is here"
    assert events[-1][0] == "final"                  # 末尾仍发完整 final（SDK 用）
    assert events[-1][1] == "The answer is here"


class ToolThenAnswerEng:
    """第一轮吐工具调用，拿到 tool_response 后逐 token 作答。"""
    model_id = "t"
    tok = PlainEng._Tok()

    def stream(self, msgs, **kw):
        joined = " ".join(str(m.get("content", "")) for m in msgs)
        if "<tool_response>" in joined:
            for w in ["Result ", "is ", "42"]:
                yield Chunk(delta=w)
        else:
            yield Chunk(delta='<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>')
        yield Chunk(done=True)


def test_tool_call_then_streamed_answer():
    tools = [{"name": "add", "description": "add", "schema": {}}]
    events = list(run_agent(ToolThenAnswerEng(), _MCP(tools),
                            [{"role": "user", "content": "add"}]))
    kinds = [k for k, _ in events]
    assert "tool_call" in kinds and "tool_result" in kinds
    assert kinds.count("delta") >= 2                 # 工具后的作答也逐 token
    final = events[-1]
    assert final[0] == "final" and "42" in final[1]
