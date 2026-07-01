"""OpenAI 标准函数调用（请求带 tools → 返回结构化 tool_calls）的回归测试。"""

import json

from naga.server import _normalize_messages, _openai_tools_to_specs, _tool_call_obj

ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers.",
        "parameters": {"type": "object",
                       "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                       "required": ["a", "b"]},
    },
}


def test_openai_tools_to_specs():
    specs = _openai_tools_to_specs([ADD_TOOL])
    assert specs == [{"name": "add", "description": "Add two integers.",
                      "schema": ADD_TOOL["function"]["parameters"]}]


def test_normalize_messages_tool_calls_and_tool_role():
    msgs = _normalize_messages([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"function": {"name": "add", "arguments": '{"a":2,"b":3}'}}]},
        {"role": "tool", "content": "5"},
    ])
    assert msgs[0]["content"] == "hi"
    assert "<tool_call>" in msgs[1]["content"] and "add" in msgs[1]["content"]
    assert msgs[2]["content"] == "<tool_response>5</tool_response>"


def test_normalize_preserves_multimodal_list():
    content = [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "x"}}]
    msgs = _normalize_messages([{"role": "user", "content": content}])
    assert msgs[0]["content"] is content     # 原样透传


def test_tool_call_obj_shape():
    tc = _tool_call_obj({"name": "add", "arguments": {"a": 2, "b": 3}})
    assert tc["type"] == "function" and tc["id"].startswith("call_")
    assert tc["function"]["name"] == "add"
    assert json.loads(tc["function"]["arguments"]) == {"a": 2, "b": 3}


def test_chat_completion_returns_tool_calls(tool_client):
    r = tool_client.post("/v1/chat/completions", json={
        "model": "naga-test", "messages": [{"role": "user", "content": "2+3?"}],
        "tools": [ADD_TOOL], "tool_choice": "auto",
    })
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tcs = choice["message"]["tool_calls"]
    assert len(tcs) == 1
    assert tcs[0]["function"]["name"] == "add"
    assert json.loads(tcs[0]["function"]["arguments"]) == {"a": 2, "b": 3}


def test_chat_completion_tool_calls_streaming(tool_client):
    r = tool_client.post("/v1/chat/completions", json={
        "model": "naga-test", "messages": [{"role": "user", "content": "2+3?"}],
        "tools": [ADD_TOOL], "stream": True,
    })
    frames = [json.loads(l[6:]) for l in r.text.splitlines()
              if l.startswith("data: ") and l[6:] != "[DONE]"]
    tc_deltas = [c for f in frames for c in f.get("choices", []) if "tool_calls" in c.get("delta", {})]
    assert tc_deltas, "should stream a tool_calls delta"
    assert tc_deltas[0]["delta"]["tool_calls"][0]["function"]["name"] == "add"
    finishes = [c.get("finish_reason") for f in frames for c in f.get("choices", [])]
    assert "tool_calls" in finishes


def test_tool_choice_none_skips_tool_path(tool_client):
    r = tool_client.post("/v1/chat/completions", json={
        "model": "naga-test", "messages": [{"role": "user", "content": "hi"}],
        "tools": [ADD_TOOL], "tool_choice": "none",
    })
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]


def test_native_function_call_roundtrip_final_answer(tool_client):
    """第二个 round-trip：客户端回填 assistant.tool_calls + tool 结果后，应产出最终答案而非再次调工具。

    这是 Open WebUI native 函数调用真正可用的关键路径（工具结果 -> grounded 回答）。"""
    r = tool_client.post("/v1/chat/completions", json={
        "model": "naga-test",
        "messages": [
            {"role": "user", "content": "2+3?"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "add", "arguments": '{"a":2,"b":3}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "5"},
        ],
        "tools": [ADD_TOOL], "tool_choice": "auto",
    })
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "stop"                 # 收尾，不再调工具
    assert not choice["message"].get("tool_calls")
    assert "5" in (choice["message"].get("content") or "")   # grounded 于工具结果
