"""MCP 元数据读取 + build_tool_prompt 结构化呈现（读全 title/annotations/required/enum/参数说明）。"""
from naga.agent import build_tool_prompt


def test_build_tool_prompt_structured():
    tools = [{
        "name": "single_task", "title": "Single Task (Any Model)",
        "description": "生成图片",
        "schema": {"type": "object", "required": ["modelId", "prompt"],
                   "properties": {
                       "modelId": {"type": "string",
                                   "description": "Model alias id from list_model_aliases"},
                       "prompt": {"type": "string", "description": "文本提示"},
                       "size": {"type": "string", "enum": ["1024", "2048"]}}},
    }]
    p = build_tool_prompt(tools)
    assert "single_task" in p and "生成图片" in p
    assert "Single Task (Any Model)" in p                 # title 呈现
    assert "必填" in p and "可选" in p                     # required 区分
    assert "list_model_aliases" in p                      # 参数 description 呈现（怎么调的关键说明）
    assert "1024" in p and "2048" in p                    # enum 可选值呈现
    assert "modelId" in p and "prompt" in p


def test_build_tool_prompt_no_params():
    p = build_tool_prompt([{"name": "current_time", "description": "查时间", "schema": {}}])
    assert "current_time" in p and "无参数" in p


def test_mcp_tools_exposes_title_and_annotations():
    from naga.mcp import MCPManager
    mgr = MCPManager()

    class FakeClient:
        tools = [{"name": "t1", "title": "Tool One", "description": "d",
                  "annotations": {"readOnlyHint": True},
                  "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}}}]

    mgr.clients = {"srv": FakeClient()}
    out = mgr.tools()
    assert out[0]["title"] == "Tool One"
    assert out[0]["annotations"] == {"readOnlyHint": True}
    assert out[0]["schema"]["properties"]["x"]["type"] == "string"
