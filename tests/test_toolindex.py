"""工具索引 + 意图路由（naga.toolindex）测试：用假嵌入器，不下载模型。"""

import math

from naga.toolindex import ToolIndex, _sig

# 固定小词表的 bag-of-words 假嵌入器：向量按关键词命中构造并归一化。
# 这样"天气/paris"的 query 会和 get_weather 工具向量更接近。
VOCAB = ["weather", "city", "paris", "add", "sum", "number", "workflow", "list", "credit"]


class FakeEmbedder:
    model_id = "fake"

    def encode(self, text):
        t = str(text).lower()
        v = [1.0 if w in t else 0.0 for w in VOCAB]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


TOOLS = [
    {"name": "get_weather", "description": "Get the weather for a city like Paris", "server": "s1"},
    {"name": "add", "description": "Add two numbers, return the sum", "server": "s1"},
    {"name": "list_workflows", "description": "List the user's workflows", "server": "s2"},
    {"name": "get_credit", "description": "Get account credit balance", "server": "s2"},
]


def test_sync_embeds_and_persists(tmp_path):
    idx = ToolIndex(tmp_path / "ti.json", embedder=FakeEmbedder())
    n = idx.sync(TOOLS)
    assert n == 4                                   # 首次全部嵌入
    assert (tmp_path / "ti.json").exists()
    assert set(idx.entries) == {"get_weather", "add", "list_workflows", "get_credit"}
    # 再次 sync 无变化 → 不重嵌
    assert idx.sync(TOOLS) == 0


def test_sync_incremental_and_removal(tmp_path):
    idx = ToolIndex(tmp_path / "ti.json", embedder=FakeEmbedder())
    idx.sync(TOOLS)
    changed = TOOLS[:2] + [{"name": "add", "description": "CHANGED desc", "server": "s1"}]
    # add 描述变了(重嵌) + 少了两个工具(移除)
    n = idx.sync(changed)
    assert n >= 1
    assert "list_workflows" not in idx.entries and "get_credit" not in idx.entries


def test_search_matches_intent(tmp_path):
    idx = ToolIndex(tmp_path / "ti.json", embedder=FakeEmbedder())
    idx.sync(TOOLS)
    top = idx.search("what is the weather in Paris", TOOLS, top_k=2)
    assert top[0]["name"] == "get_weather"          # 意图匹配到天气工具
    top2 = idx.search("add these numbers", TOOLS, top_k=2)
    assert top2[0]["name"] == "add"


def test_search_returns_all_when_few_tools(tmp_path):
    idx = ToolIndex(tmp_path / "ti.json", embedder=FakeEmbedder())
    idx.sync(TOOLS)
    # top_k >= 工具数 → 全给（不必筛）
    assert len(idx.search("weather", TOOLS, top_k=8)) == 4


def test_selector_threshold(tmp_path):
    idx = ToolIndex(tmp_path / "ti.json", embedder=FakeEmbedder())
    idx.sync(TOOLS)
    # 阈值=2：4 个工具超阈值 → 按意图筛（min_score 会剔除不相关的，只留高相关）
    sel = idx.selector("weather in paris", top_k=2, threshold=2)
    picked = sel(TOOLS)
    assert 1 <= len(picked) < 4 and picked[0]["name"] == "get_weather"
    # 阈值=10：不超 → 原样全给
    sel2 = idx.selector("weather", top_k=2, threshold=10)
    assert len(sel2(TOOLS)) == 4


def test_run_agent_uses_selector():
    """run_agent 接入 tool_selector：只把筛后的工具给模型。"""
    from naga.agent import run_agent
    from naga.engine import Chunk

    class _Tok:
        eos_ids = (0,)
        def apply_chat_template(self, m): return ""
        def encode(self, s): return [0]
        def decode(self, i): return ""

    seen_prompt = {}

    class Eng:
        model_id = "t"; tok = _Tok()
        def stream(self, msgs, **kw):
            seen_prompt["sys"] = msgs[0]["content"] if msgs else ""
            yield Chunk(delta="hi"); yield Chunk(done=True)

    class MCP:
        def tools(self): return TOOLS
        def call(self, n, a): return "x"

    # selector 只保留 get_weather
    sel = lambda tools: [t for t in tools if t["name"] == "get_weather"]
    list(run_agent(Eng(), MCP(), [{"role": "user", "content": "weather?"}], tool_selector=sel))
    assert "get_weather" in seen_prompt["sys"]
    assert "list_workflows" not in seen_prompt["sys"]   # 被路由筛掉，没进 prompt
