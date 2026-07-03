"""回归：模型在工具调用 JSON 前带前缀（如 '>'）或空白时，工具仍被识别并执行。"""
from naga.agent import run_agent
from naga.engine import Chunk

class Tok:
    eos_ids=(0,)
    def apply_chat_template(self,m): return ""
    def encode(self,s): return [0]
    def decode(self,i): return ""

def _mcp():
    class M:
        def tools(self): return [{"name":"weather_poster","description":"生成天气海报","schema":{}}]
        def call(self,n,a): return "IMG_URL"
    return M()

def _run(first_chunk):
    class Eng:
        model_id="t"; tok=Tok()
        def stream(self, msgs, **kw):
            j=" ".join(str(m.get("content","")) for m in msgs)
            if "<tool_response" in j:
                yield Chunk(delta="海报已生成")
            else:
                for c in first_chunk: yield Chunk(delta=c)
            yield Chunk(done=True)
    return list(run_agent(Eng(), _mcp(), [{"role":"user","content":"做张海报"}]))

def _executed(events):
    return any(k=="tool_call" for k,_ in events) and any(k=="tool_result" for k,_ in events)

def test_json_with_gt_prefix_executes():
    # 复现 bug：'>' 前缀
    ev=_run(['>', '{"name":"weather_poster","arguments":{"city":"重庆"}}'])
    assert _executed(ev), "带 '>' 前缀的工具调用应被执行"

def test_json_with_leading_space_executes():
    ev=_run(['  {"name":"weather_poster","arguments":{"city":"上海"}}'])
    assert _executed(ev)

def test_plain_answer_still_streams():
    # 普通回答不应被误判成工具调用
    ev=_run(['你好，', '今天', '天气不错'])
    assert not any(k=="tool_call" for k,_ in ev)
    assert any(k=="delta" for k,_ in ev)

def test_bare_json_executes():
    ev=_run(['{"name":"weather_poster","arguments":{"city":"北京"}}'])
    assert _executed(ev)
