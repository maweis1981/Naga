from naga.trace import Tracer
def test_tree(tmp_path):
    tr = Tracer(tmp_path); t = tr.start("q")
    with tr.bound(t):
        with tr.span("intent_route") as s: s.set(triggered=True, selected=["a"])
        with tr.span("llm.generate"):
            with tr.span("tool.call", name="a") as ts: ts.set(result="5")
    tr.finish(t, output="ok"); d = t.to_dict()
    assert [s["name"] for s in d["spans"]] == ["intent_route", "llm.generate"]
    assert d["spans"][1]["children"][0]["name"] == "tool.call"
def test_noop(tmp_path):
    tr = Tracer(tmp_path)
    with tr.span("x") as s: s.set(a=1)
    assert tr.current is None
def test_persist(tmp_path):
    tr = Tracer(tmp_path); t = tr.start("a")
    with tr.bound(t): tr.event("s")
    tr.finish(t, output="done"); rec = tr.recent()
    assert len(rec) == 1 and tr.get(rec[0]["id"])["output"] == "done"
def test_bad_id(tmp_path):
    tr = Tracer(tmp_path); assert tr.get("../x") is None and tr.get("deadbeef") is None
