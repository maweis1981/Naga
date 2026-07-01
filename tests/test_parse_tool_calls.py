"""多工具解析 parse_tool_calls 的回归测试（单轮并行多工具）。"""

from naga.agent import parse_tool_call, parse_tool_calls


def test_single_bare_object():
    calls = parse_tool_calls('{"name":"add","arguments":{"a":1,"b":2}}')
    assert calls == [{"name": "add", "arguments": {"a": 1, "b": 2}}]


def test_multiple_bare_objects():
    text = '{"name":"current_time","arguments":{}}{"name":"add","arguments":{"a":1990,"b":35}}'
    calls = parse_tool_calls(text)
    assert [c["name"] for c in calls] == ["current_time", "add"]
    assert calls[1]["arguments"] == {"a": 1990, "b": 35}


def test_top_level_json_array():
    text = '[{"name":"a","arguments":{}},{"name":"b","arguments":{"x":1}}]'
    calls = parse_tool_calls(text)
    assert [c["name"] for c in calls] == ["a", "b"]


def test_multiple_tool_call_tags():
    text = ('<tool_call>{"name":"a","arguments":{}}</tool_call>\n'
            'some text\n'
            '<tool_call>{"name":"b","arguments":{"y":2}}</tool_call>')
    calls = parse_tool_calls(text)
    assert [c["name"] for c in calls] == ["a", "b"]


def test_dedup_identical_calls():
    text = '{"name":"a","arguments":{"x":1}}{"name":"a","arguments":{"x":1}}'
    calls = parse_tool_calls(text)
    assert len(calls) == 1                       # 完全相同的调用去重


def test_dedup_keeps_different_args():
    text = '{"name":"a","arguments":{"x":1}}{"name":"a","arguments":{"x":2}}'
    calls = parse_tool_calls(text)
    assert len(calls) == 2                       # 参数不同不算重复


def test_none_when_no_calls():
    assert parse_tool_calls("just a normal answer, no tools.") == []


def test_parse_tool_call_shim_returns_first():
    text = '{"name":"first","arguments":{}}{"name":"second","arguments":{}}'
    assert parse_tool_call(text) == {"name": "first", "arguments": {}}
    assert parse_tool_call("nothing here") is None


def test_ignores_non_tool_json():
    text = 'Here is data: {"foo": 1, "bar": 2}. No tool call.'
    assert parse_tool_calls(text) == []          # 缺 name/arguments 不算工具调用
