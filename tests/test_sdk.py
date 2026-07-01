"""Agent SDK（naga.sdk）的回归测试：schema 推导、工具集、Agent 循环。"""

from naga.sdk import Agent, FunctionToolset, _schema_from_fn, tool


def test_schema_from_type_hints():
    def f(a: int, b: str, c: float = 1.0, d: bool = False):
        ...
    schema = _schema_from_fn(f)
    assert schema["properties"]["a"]["type"] == "integer"
    assert schema["properties"]["b"]["type"] == "string"
    assert schema["properties"]["c"]["type"] == "number"
    assert schema["properties"]["d"]["type"] == "boolean"
    assert schema["required"] == ["a", "b"]          # 有默认值的不算 required


def test_tool_decorator_metadata():
    @tool
    def add(a: int, b: int) -> str:
        """把两个整数相加。"""
        return str(a + b)

    @tool(name="weather", description="查天气")
    def w(city: str) -> str:
        return "sunny"

    ts = FunctionToolset([add, w])
    specs = {t["name"]: t for t in ts.tools()}
    assert specs["add"]["description"] == "把两个整数相加。"
    assert "weather" in specs and specs["weather"]["description"] == "查天气"


def test_functiontoolset_call_and_errors():
    @tool
    def add(a: int, b: int) -> str:
        return str(a + b)

    ts = FunctionToolset([add])
    assert ts.call("add", {"a": 2, "b": 3}) == "5"
    assert ts.call("missing", {}).startswith("[错误]")
    assert ts.call("add", {"a": 1}).startswith("[工具执行出错]")   # 缺参 -> 捕获异常


def test_agent_run_full_loop(tool_engine):
    @tool
    def add(a: int, b: int) -> str:
        return str(a + b)

    agent = Agent(engine=tool_engine, tools=[add], system="你是助手")
    res = agent.run("2 加 3 等于几？")
    assert res.text == "答案是5"
    kinds = [s["type"] for s in res.steps]
    assert kinds == ["tool_call", "tool_result"]
    assert res.steps[0]["name"] == "add"
    assert res.steps[1]["result"] == "5"
    # 消息历史以最终回答收尾
    assert res.messages[-1] == {"role": "assistant", "content": "答案是5"}


def test_agent_without_tools_just_generates(plain_engine):
    agent = Agent(engine=plain_engine)
    res = agent.run("hi")
    assert res.text == "Hello, world"
    assert res.steps == []
