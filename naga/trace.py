"""请求级追踪（Trace/Span）：把一次请求的完整解析链路记成一棵树（LangSmith 式）。"""
from __future__ import annotations
import contextlib, json, threading, time, uuid
from pathlib import Path
TRACE_DIR = Path.home() / ".naga" / "traces"
MAX_TRACES = 500
def _ms(dt): return round(dt * 1000, 1)
class Span:
    def __init__(self, name, attrs):
        self.name = name; self.attrs = dict(attrs); self.children = []
        self._t0 = time.perf_counter(); self.dur_ms = 0.0
    def set(self, **kw): self.attrs.update(kw); return self
    def _close(self): self.dur_ms = _ms(time.perf_counter() - self._t0)
    def to_dict(self):
        return {"name": self.name, "dur_ms": self.dur_ms, "attrs": self.attrs,
                "children": [c.to_dict() for c in self.children]}
class _NullSpan:
    def set(self, **kw): return self
_NULL = _NullSpan()
class Trace:
    def __init__(self, input_text, meta=None):
        self.id = uuid.uuid4().hex[:16]; self.input = input_text; self.meta = meta or {}
        self.created = round(time.time(), 3); self._t0 = time.perf_counter()
        self.dur_ms = 0.0; self.output = ""; self.root = []; self.stack = []
    def to_dict(self):
        return {"id": self.id, "input": self.input, "output": self.output,
                "created": self.created, "dur_ms": self.dur_ms, "meta": self.meta,
                "spans": [s.to_dict() for s in self.root]}
class Tracer:
    def __init__(self, root=None):
        self.dir = Path(root) if root else TRACE_DIR; self._local = threading.local()
    @property
    def current(self): return getattr(self._local, "cur", None)
    def start(self, input_text, meta=None): return Trace(input_text, meta)
    def bind(self, trace): self._local.cur = trace
    @contextlib.contextmanager
    def bound(self, trace):
        prev = self.current; self.bind(trace)
        try: yield trace
        finally: self.bind(prev)
    @contextlib.contextmanager
    def span(self, span_name, **attrs):
        t = self.current
        if t is None:
            yield _NULL; return
        s = Span(span_name, attrs); (t.stack[-1].children if t.stack else t.root).append(s)
        t.stack.append(s)
        try: yield s
        finally: s._close(); t.stack.pop()
    def event(self, span_name, **attrs):
        with self.span(span_name, **attrs): pass
    def finish(self, trace, output=""):
        if trace is None: return trace
        trace.output = output; trace.dur_ms = _ms(time.perf_counter() - trace._t0)
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / (trace.id + ".json")).write_text(
                json.dumps(trace.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            files = sorted(self.dir.glob("*.json"), key=lambda f: f.stat().st_mtime)
            for f in files[:-MAX_TRACES]:
                try: f.unlink()
                except Exception: pass
        except Exception: pass
        return trace
    def get(self, tid):
        if not tid or not all(c in "0123456789abcdef" for c in tid): return None
        p = self.dir / (tid + ".json")
        if not p.exists(): return None
        try: return json.loads(p.read_text(encoding="utf-8"))
        except Exception: return None
    def recent(self, n=50):
        files = sorted(self.dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        out = []
        for f in files[:n]:
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                out.append({"id": d["id"], "input": d.get("input", "")[:80],
                            "created": d.get("created"), "dur_ms": d.get("dur_ms"),
                            "spans": len(d.get("spans", []))})
            except Exception: continue
        return out
tracer = Tracer()
