"""OpenAI 兼容的 HTTP 服务 + 模型管理 + 设置（P2 / P6）。

对外（OpenAI 兼容，可直接被 Open WebUI 等前端当后端连接）：
  GET  /v1/models             列出本地可用模型（标注当前活跃 / 已加载）
  POST /v1/chat/completions   对话补全（流式 SSE + 非流式；按 model 字段路由）
  POST /v1/embeddings         文本嵌入（复用手写 BERT 嵌入器，供前端 RAG 用）

管理（自用）：
  GET  /admin/state           当前状态：活跃模型 / 已加载 / 可用 / 下载进度 / 设置
  POST /admin/load   {model}  加载并切换到某模型
  POST /admin/download {repo} 后台从 HuggingFace 下载新模型
  POST /admin/settings {...}  保存默认参数 / 系统提示词

页面：
  GET  /          聊天界面
  GET  /settings  模型管理 + 系统设置界面

启动：python -m naga.server [--model 默认模型] [--port 8000]
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from starlette.concurrency import run_in_threadpool

from .monitor import monitor
from .scheduler import scheduler

app = FastAPI(title="Naga")
manager: ModelManager | None = None
ADMIN_TOKEN: str | None = None

WEBUI = Path(__file__).parent / "webui"
_SENTINEL = object()   # run_in_threadpool(next, it, _SENTINEL) 的迭代结束标记

from .optimize import MetricsHistory

metrics_hist = MetricsHistory(Path.home() / ".naga" / "metrics.jsonl")


def _cors_origins() -> tuple[list[str], str]:
    # 默认只放行本机页面与常见本地开发地址，避免把管理界面暴露给任意站点跨域调用。
    origins = [
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
    ]
    extra = [o.strip() for o in os.environ.get("NAGA_CORS_ORIGINS", "").split(",") if o.strip()]
    return origins + extra, r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"


_allow_origins, _allow_origin_regex = _cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_origin_regex=_allow_origin_regex,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in {"localhost"}


def _require_admin(req: Request):
    host = req.client.host if req.client else None
    if _is_loopback(host):
        return
    token = (req.headers.get("x-admin-token") or "").strip()
    if ADMIN_TOKEN and token == ADMIN_TOKEN:
        return
    if ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="invalid admin token")
    raise HTTPException(
        status_code=403,
        detail="remote admin is disabled unless NAGA_ADMIN_TOKEN is set",
    )


def _manager_module():
    from . import manager as manager_mod
    return manager_mod


# ---------- 页面 ----------

@app.get("/")
def index():
    return HTMLResponse((WEBUI / "index.html").read_text(encoding="utf-8"))


@app.get("/settings")
def settings_page():
    return HTMLResponse((WEBUI / "settings.html").read_text(encoding="utf-8"))


@app.get("/monitor")
def monitor_page():
    """引擎监视器页面：实时查看每次推理的 prefill/decode/前缀缓存/工具调用。"""
    return HTMLResponse((WEBUI / "monitor.html").read_text(encoding="utf-8"))


@app.get("/monitor/events")
def monitor_events(n: int = 200):
    """最近 n 条事件（页面初次加载时拉取历史）。"""
    return {"events": monitor.recent(n)}


@app.get("/monitor/hw")
def monitor_hw():
    """本地硬件资源实时采样：CPU / 统一内存 / MLX 显存 / 本进程。"""
    from . import hwstats
    return {"static": hwstats.static(), "sample": hwstats.sample(), "power": hwstats.power()}


@app.get("/health")
def health():
    """健康探针：供 Open WebUI / 反向代理 / 编排器判断服务是否就绪。"""
    active = manager.active if manager else None
    return {"status": "ok", "service": "naga", "active_model": active}


@app.get("/metrics")
def metrics():
    """自观测指标（JSON）：decode tok/s 分位、TTFT 分布、前缀缓存复用率、各模型吞吐、工具频次。

    这是 Naga 的「自跟踪—自优化」闭环对外的结构化出口：每次推理都在更新这份画像，
    据此可判断量化 / 前缀缓存 / 换模型是否真的带来收益。"""
    snap = monitor.stats.snapshot()
    if manager is not None:                       # 叠加一帧硬件采样，便于关联吞吐与功耗/显存
        try:
            from . import hwstats
            snap["hardware"] = {"sample": hwstats.sample(), "power": hwstats.power()}
        except Exception:
            pass
    return snap


@app.get("/metrics/prometheus")
def metrics_prometheus():
    """Prometheus 文本曝光格式：可直接被 Prometheus / Grafana / OpenTelemetry Collector 抓取。"""
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(monitor.stats.prometheus(),
                             media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/metrics/advice")
def metrics_advice():
    """优化顾问：基于当前指标画像给出可执行的调优建议（自跟踪—自优化闭环的"优化"环）。"""
    from .optimize import advise
    return {"advice": advise(monitor.stats.snapshot())}


@app.get("/metrics/history")
def metrics_history(n: int = 200):
    """指标历史趋势（重启后仍在）：decode/TTFT/前缀复用随时间的变化，便于对比配置收益。"""
    return {"history": metrics_hist.recent(n)}


@app.get("/monitor/stream")
async def monitor_stream():
    """SSE 实时事件流：新事件一产生就推给页面。"""
    import asyncio

    q = monitor.subscribe()

    async def gen():
        try:
            for ev in monitor.recent(50):                 # 先补发最近历史
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            while True:
                if q:
                    yield f"data: {json.dumps(q.popleft(), ensure_ascii=False)}\n\n"
                else:
                    await asyncio.sleep(0.15)
        finally:
            monitor.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------- OpenAI 兼容 ----------

@app.get("/v1/models")
def list_models():
    st = manager.state()
    created = int(time.time())
    data = [{
        "id": m["id"], "object": "model", "created": created, "owned_by": "naga",
        "type": m["type"], "size_gb": m["size_gb"],
        "active": m["id"] == st["active"], "loaded": m["id"] in st["loaded"],
    } for m in st["available"]]
    return {"object": "list", "data": data}


def _last_user_text(messages: list[dict]) -> str:
    """取最后一条 user 消息的文本（content 可能是字符串或图文数组）。"""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        c = m["content"]
        if isinstance(c, str):
            return c
        return " ".join(p.get("text", "") for p in c if p.get("type") == "text")
    return ""


def _params(body: dict) -> dict:
    s = manager.settings
    return dict(
        max_tokens=int(body.get("max_tokens") or s["max_tokens"]),
        temp=float(body.get("temperature", s["temperature"])),
        top_p=float(body.get("top_p", s["top_p"])),
        top_k=int(body.get("top_k", 0)),
    )


def _openai_tools_to_specs(tools: list[dict]) -> list[dict]:
    """OpenAI `tools`（[{type:function, function:{name,description,parameters}}]）→ 内部工具规格。"""
    specs = []
    for t in tools or []:
        fn = t.get("function") if t.get("type") == "function" else t
        if fn and fn.get("name"):
            specs.append({"name": fn["name"], "description": fn.get("description", ""),
                          "schema": fn.get("parameters", {})})
    return specs


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """把 OpenAI 的 assistant.tool_calls / tool 角色 / content=None 规整成模板能渲染的字符串。

    多模态的数组型 content 原样透传（给 VLM 用）。"""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):                 # 多模态图文数组，原样保留
            out.append(m)
            continue
        if content is None and m.get("tool_calls"):
            content = "\n".join(
                f"<tool_call>{json.dumps(tc.get('function', tc), ensure_ascii=False)}</tool_call>"
                for tc in m["tool_calls"])
        elif m.get("role") == "tool":                 # 客户端回传的工具结果
            content = f"<tool_response>{content}</tool_response>"
        out.append({**m, "content": content if content is not None else ""})
    return out


def _decide_tool_call(engine, messages, specs, tool_choice, params):
    """判定客户端 tools 是否触发调用；返回 ('tool', [call, ...]) 或 ('text', 已生成文本)。

    单轮可产出**多个**工具调用（auto 路径解析全部，对齐 OpenAI tool_calls 数组）。
    tool_choice：'required' 或 {'type':'function','function':{'name':X}} → 强制单个（约束解码保证合法）；
    'auto' → 自由生成，解析出所有合法调用；一个都没有但像在尝试时用约束解码补一个。"""
    from .agent import (_looks_like_tool_attempt, _valid_call, build_tool_prompt,
                        constrained_tool_call, parse_tool_calls)
    names = [s["name"] for s in specs]
    # 把工具清单注入系统提示，模型才有上下文去决定/生成调用（否则约束解码会走空）
    tmsgs = [{"role": "system", "content": build_tool_prompt(specs)}] + messages
    if isinstance(tool_choice, dict):
        want = tool_choice.get("function", {}).get("name")
        forced = [s for s in specs if s["name"] == want] or specs
        return ("tool", [constrained_tool_call(engine, tmsgs, forced)])
    if tool_choice == "required":
        return ("tool", [constrained_tool_call(engine, tmsgs, specs)])
    # auto：先自由生成，解析出所有合法工具调用
    text = "".join(ch.delta for ch in engine.stream(tmsgs, **params) if not ch.done)
    calls = [c for c in parse_tool_calls(text) if _valid_call(c, names)]
    if calls:
        return ("tool", calls)
    if _looks_like_tool_attempt(text, names):
        fixed = constrained_tool_call(engine, tmsgs, specs)
        if _valid_call(fixed, names):
            return ("tool", [fixed])
    return ("text", text)


def _tool_call_obj(call: dict) -> dict:
    """内部 call → OpenAI tool_calls 元素（arguments 是 JSON 字符串）。"""
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": call.get("name", ""),
                         "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)}}


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()
    messages = list(body["messages"])

    query = _last_user_text(messages)

    n_docs = n_mem = 0
    # RAG：检索相关文档片段，注入为上下文
    if manager.settings.get("rag_enabled", True):
        docs = manager.docs.search(query)
        if docs:
            n_docs = len(docs)
            ctx = "以下是从知识库检索到的相关资料，回答时优先依据它们：\n" + \
                  "\n".join(f"【{d['name']}】{d['text']}" for d in docs)
            messages = [{"role": "system", "content": ctx}] + messages

    # 语义检索相关记忆，注入为一条 system 上下文
    if manager.settings.get("memory_enabled", True):
        hits = manager.memory.search(query)
        if hits:
            n_mem = len(hits)
            mem = "以下是关于用户的已知信息（记忆），回答时酌情参考：\n" + \
                  "\n".join(f"- {h['text']}" for h in hits)
            messages = [{"role": "system", "content": mem}] + messages

    if n_docs or n_mem:                          # 监视：本次注入了多少 RAG/记忆上下文
        monitor.emit("context", rag_docs=n_docs, memories=n_mem)

    # 注入全局系统提示词（请求里没带 system 时）
    sp = (manager.settings.get("system_prompt") or "").strip()
    if sp and not any(m.get("role") == "system" for m in messages):
        messages = [{"role": "system", "content": sp}] + messages

    messages = _normalize_messages(messages)     # 规整 tool_calls / tool 角色 / None content
    engine = manager.get(body.get("model"))     # 按 model 字段路由 / 回退到活跃模型
    model_name = engine.model_id

    # 上下文长度保护：prompt + 预留输出不得超过模型窗口，否则返回标准 OpenAI 错误
    ctx_err = _context_length_error(engine, messages, body)
    if ctx_err is not None:
        return ctx_err

    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    stream = bool(body.get("stream", False))
    # OpenAI 语义：stream_options.include_usage=true 时，在 [DONE] 前补发一帧 usage
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

    # OpenAI 标准函数调用：请求带 tools 时，返回结构化 tool_calls 交给客户端执行（区别于内部 MCP agent）
    tool_choice = body.get("tool_choice", "auto")
    client_specs = _openai_tools_to_specs(body.get("tools", []))
    if client_specs and tool_choice != "none":
        prompt_tokens = _count_tokens(engine, messages)

        def decide():
            yield _decide_tool_call(engine, messages, client_specs, tool_choice, _params(body))

        kind, data = (await run_in_threadpool(lambda: list(scheduler.submit(decide).results())))[0]

        calls = [c for c in (data or []) if c and c.get("name")] if kind == "tool" else []
        if calls:
            for c in calls:
                monitor.emit("tool_call", name=c["name"], arguments=c.get("arguments", {}))
            tcs = [_tool_call_obj(c) for c in calls]   # 单轮可含多个（并行工具调用）
            comp_tokens = len(engine.tok.encode(json.dumps(calls, ensure_ascii=False)))
            msg = {"role": "assistant", "content": None, "tool_calls": tcs}
            if stream:
                async def sse_tc():
                    yield f"data: {json.dumps(_chunk(cid, created, model_name, {'role': 'assistant'}))}\n\n"
                    delta_tcs = [{'index': i, **tc} for i, tc in enumerate(tcs)]
                    yield f"data: {json.dumps(_chunk(cid, created, model_name, {'tool_calls': delta_tcs}), ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps(_chunk(cid, created, model_name, {}, 'tool_calls'))}\n\n"
                    if include_usage:
                        yield f"data: {json.dumps(_usage_chunk(cid, created, model_name, _usage(prompt_tokens, comp_tokens)))}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(sse_tc(), media_type="text/event-stream")
            return JSONResponse({
                "id": cid, "object": "chat.completion", "created": created, "model": model_name,
                "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
                "usage": _usage(prompt_tokens, comp_tokens),
            })

        # 未触发工具（auto 且模型直接作答）：把已生成文本当普通回答返回
        text = data if kind == "text" else ""
        comp_tokens = len(engine.tok.encode(text)) if text else 0
        if stream:
            async def sse_txt():
                yield f"data: {json.dumps(_chunk(cid, created, model_name, {'role': 'assistant'}))}\n\n"
                if text:
                    yield f"data: {json.dumps(_chunk(cid, created, model_name, {'content': text}), ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps(_chunk(cid, created, model_name, {}, 'stop'))}\n\n"
                if include_usage:
                    yield f"data: {json.dumps(_usage_chunk(cid, created, model_name, _usage(prompt_tokens, comp_tokens)))}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse_txt(), media_type="text/event-stream")
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": created, "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": _usage(prompt_tokens, comp_tokens),
        })

    # MCP 工具调用：开启且有可用工具时，走 Agent 循环
    if manager.settings.get("mcp_enabled") and manager.mcp.tools():
        from .agent import run_agent

        prompt_tokens = _count_tokens(engine, messages)

        def pieces():
            # 逐 token 转发 delta（普通回答/工具后作答都有打字机效果）；tool_call/tool_result
            # 转成可读标记；final 跳过（内容已由 delta 流出，避免重复）。
            for kind, data in run_agent(engine, manager.mcp, messages, **_params(body)):
                if kind == "delta":
                    yield data
                elif kind == "tool_call":
                    monitor.emit("tool_call", name=data["name"],
                                 arguments=data.get("arguments", {}))
                    yield f"\n🔧 调用 `{data['name']}`（{json.dumps(data.get('arguments', {}), ensure_ascii=False)}）\n"
                elif kind == "tool_result":
                    monitor.emit("tool_result", name=data["name"],
                                 result=str(data["result"])[:300])
                    yield f"↩ {data['result']}\n\n"
                # kind == "final": 已由 delta 流出，忽略

        if stream:
            async def sse_tool():
                yield f"data: {json.dumps(_chunk(cid, created, model_name, {'role': 'assistant'}))}\n\n"
                collected: list[str] = []
                job = scheduler.submit(pieces)             # 串行化整段 Agent 循环
                it = job.results()
                while True:
                    p = await run_in_threadpool(next, it, _SENTINEL)
                    if p is _SENTINEL:
                        break
                    collected.append(p)
                    yield f"data: {json.dumps(_chunk(cid, created, model_name, {'content': p}), ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps(_chunk(cid, created, model_name, {}, 'stop'))}\n\n"
                if include_usage:
                    completion_tokens = len(engine.tok.encode("".join(collected)))
                    usage = _usage(prompt_tokens, completion_tokens)
                    yield f"data: {json.dumps(_usage_chunk(cid, created, model_name, usage))}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse_tool(), media_type="text/event-stream")

        text = "".join(await run_in_threadpool(lambda: list(scheduler.submit(pieces).results())))
        completion_tokens = len(engine.tok.encode(text)) if text else 0
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": created, "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": _usage(prompt_tokens, completion_tokens),
        })

    if stream:
        async def sse():
            yield f"data: {json.dumps(_chunk(cid, created, model_name, {'role': 'assistant'}))}\n\n"
            usage = _usage(0, 0)
            job = scheduler.submit(lambda: engine.stream(messages, **_params(body)))
            it = job.results()
            while True:                                    # 串行化生成，避免并发污染 KV/前缀缓存
                ch = await run_in_threadpool(next, it, _SENTINEL)
                if ch is _SENTINEL:
                    break
                if ch.done:
                    usage = _usage(ch.prompt_tokens, ch.completion_tokens)
                    yield f"data: {json.dumps(_chunk(cid, created, model_name, {}, 'stop'))}\n\n"
                else:
                    obj = _chunk(cid, created, model_name, {"content": ch.delta})
                    yield f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
            if include_usage:
                yield f"data: {json.dumps(_usage_chunk(cid, created, model_name, usage))}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")

    def _collect():
        text_, usage_ = "", _usage(0, 0)
        for ch in scheduler.submit(lambda: engine.stream(messages, **_params(body))).results():
            if ch.done:
                usage_ = _usage(ch.prompt_tokens, ch.completion_tokens)
            else:
                text_ += ch.delta
        return text_, usage_

    text, usage = await run_in_threadpool(_collect)
    return JSONResponse({
        "id": cid, "object": "chat.completion", "created": created, "model": model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": usage,
    })


def _chunk(cid, created, model, delta, finish=None):
    return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def _usage(prompt_tokens: int, completion_tokens: int) -> dict:
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens}


def _usage_chunk(cid, created, model, usage):
    """OpenAI 流式用量帧：choices 为空、只带 usage，紧跟在 stop 之后、[DONE] 之前。

    Open WebUI 等前端在 stream_options.include_usage=true 时依赖这一帧做 token 计量。"""
    return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [], "usage": usage}


def _count_tokens(engine, messages: list[dict]) -> int:
    """估算一组 messages 套上对话模板后的 token 数（工具调用路径补 usage 用）。"""
    try:
        return len(engine.tok.encode(engine.tok.apply_chat_template(messages)))
    except Exception:
        return 0


def _context_length_error(engine, messages: list[dict], body: dict):
    """prompt + 预留输出超过模型上下文窗口时，返回标准 OpenAI 400 错误；否则 None。

    对标 OpenAI 的 context_length_exceeded：与其让底层产出乱码/晦涩错误，不如明确拒绝。"""
    max_ctx = getattr(getattr(engine, "args", None), "max_position_embeddings", 0)
    if not max_ctx:
        return None
    prompt_tokens = _count_tokens(engine, messages)
    want_out = int(body.get("max_tokens") or 0)
    if prompt_tokens + want_out > max_ctx:
        return JSONResponse(status_code=400, content={"error": {
            "message": (f"This model's maximum context length is {max_ctx} tokens. "
                        f"However, your messages resulted in {prompt_tokens} tokens"
                        + (f" and you requested {want_out} completion tokens" if want_out else "")
                        + ". Please shorten the conversation."),
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "param": "messages",
        }})
    return None


@app.post("/batch")
async def batch(req: Request):
    """批量补全：一次前向服务多条互不相关的对话，聚合吞吐更高（P3 批量解码）。

    请求体：{"model": ..., "inputs": [[{role,content},...], ...], "max_tokens": ..., ...}
    返回：{"completions": [{index, message, usage}, ...]}。不注入 RAG/记忆，是低层吞吐端点。"""
    body = await req.json()
    inputs = body.get("inputs")
    if not inputs or not isinstance(inputs, list):
        return JSONResponse({"error": {"message": "inputs (list of message lists) is required"}},
                            status_code=400)
    engine = manager.get(body.get("model"))
    msgs_list = [_normalize_messages(list(x)) for x in inputs]
    params = _params(body)

    def job():
        yield engine.batch_generate(msgs_list, **params)

    results = (await run_in_threadpool(lambda: list(scheduler.submit(job).results())))[0]
    return {
        "object": "list", "model": engine.model_id,
        "completions": [
            {"index": i, "message": {"role": "assistant", "content": r["text"]},
             "finish_reason": "stop", "usage": _usage(r["prompt_tokens"], r["completion_tokens"])}
            for i, r in enumerate(results)
        ],
    }


@app.post("/v1/embeddings")
async def embeddings(req: Request):
    """OpenAI 兼容文本嵌入。input 可为单条字符串或字符串数组。"""
    body = await req.json()
    inp = body.get("input")
    if inp is None or inp == "":
        return JSONResponse({"error": {"message": "input is required"}}, status_code=400)
    texts = [inp] if isinstance(inp, str) else [str(t) for t in inp]

    from .embed import get_embedder
    emb = get_embedder()
    vecs = emb.encode(texts)                      # 传入 list，必返回 list[list[float]]
    data = [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)]
    return {
        "object": "list", "data": data,
        "model": body.get("model") or emb.model_id,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


# ---------- 管理 ----------

@app.get("/admin/state")
def admin_state(req: Request):
    _require_admin(req)
    return manager.state()


@app.post("/admin/load")
async def admin_load(req: Request):
    _require_admin(req)
    mid = (await req.json())["model"]
    try:
        manager.ensure(mid)                       # 可能较慢（需加载/下载）
        return {"ok": True, "active": manager.active}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/admin/download")
async def admin_download(req: Request):
    _require_admin(req)
    repo = (await req.json())["repo"].strip()
    manager.download(repo)
    return {"ok": True, "status": manager.dl_status}


@app.post("/admin/reveal")
async def admin_reveal(req: Request):
    """在系统文件管理器里打开某模型所在目录（macOS Finder / Linux / Windows）。"""
    import subprocess
    import sys

    _require_admin(req)
    if not _is_loopback(req.client.host if req.client else None):
        raise HTTPException(status_code=403, detail="reveal is only available from localhost")

    mid = (await req.json())["model"]
    path = _manager_module().model_local_path(mid)
    if not path or not path.exists():
        return JSONResponse({"ok": False, "error": "找不到本地路径"}, status_code=404)

    target = str(path)
    if sys.platform == "darwin":
        subprocess.Popen(["open", target])
    elif sys.platform.startswith("linux"):
        subprocess.Popen(["xdg-open", target])
    elif sys.platform.startswith("win"):
        subprocess.Popen(["explorer", target])
    return {"ok": True, "path": target}


@app.get("/admin/memory")
def memory_list(req: Request):
    _require_admin(req)
    return {"items": manager.memory.list()}


@app.post("/admin/memory/add")
async def memory_add(req: Request):
    _require_admin(req)
    item = manager.memory.add((await req.json()).get("text", ""))
    return {"ok": bool(item), "item": item}


@app.post("/admin/memory/delete")
async def memory_delete(req: Request):
    _require_admin(req)
    ok = manager.memory.delete((await req.json())["id"])
    return {"ok": ok}


@app.get("/admin/docs")
def docs_state(req: Request):
    _require_admin(req)
    return manager.docs.state()


@app.post("/admin/docs/add")
async def docs_add(req: Request):
    _require_admin(req)
    return manager.docs.add((await req.json())["path"])


@app.post("/admin/docs/remove")
async def docs_remove(req: Request):
    _require_admin(req)
    return {"ok": manager.docs.remove((await req.json())["path"])}


@app.get("/admin/mcp")
def mcp_state(req: Request):
    _require_admin(req)
    return manager.mcp.state()


@app.post("/admin/mcp/add")
async def mcp_add(req: Request):
    _require_admin(req)
    b = await req.json()
    if b.get("url"):                              # HTTP 传输：连远程/托管 MCP 服务器
        err = manager.mcp.add_http_server(b["name"], b["url"], b.get("headers"))
    else:                                         # stdio 传输：本地子进程
        err = manager.mcp.add_server(b["name"], b["command"], b.get("args", []), b.get("env"))
    return {"ok": err is None, "error": err}


@app.post("/admin/mcp/remove")
async def mcp_remove(req: Request):
    _require_admin(req)
    manager.mcp.remove_server((await req.json())["name"])
    return {"ok": True}


@app.post("/admin/settings")
async def admin_settings(req: Request):
    _require_admin(req)
    body = await req.json()
    keys = ("temperature", "top_p", "max_tokens", "system_prompt",
            "memory_enabled", "rag_enabled", "mcp_enabled")
    manager.settings = {**manager.settings, **{k: body[k] for k in keys if k in body}}
    _manager_module().save_settings(manager.settings)
    return {"ok": True, "settings": manager.settings}


DEFAULT_MODEL = "llava-hf/llava-interleave-qwen-0.5b-hf"


def main():
    ap = argparse.ArgumentParser(description="Naga 服务")
    ap.add_argument("--model", default=None,
                    help="启动时加载的模型；不指定则恢复上次用的模型，再退回内置默认")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--admin-token", default=None,
                    help="远程访问 /admin/* 时需要的令牌；也可用环境变量 NAGA_ADMIN_TOKEN")
    ap.add_argument("--quantize", action="store_true", help="对文本模型做 Q4/Q8 量化加载（省显存、提速）")
    ap.add_argument("--bits", type=int, default=4, help="量化位宽（4 或 8），需配合 --quantize")
    args = ap.parse_args()

    global ADMIN_TOKEN, manager
    ADMIN_TOKEN = args.admin_token or os.environ.get("NAGA_ADMIN_TOKEN")
    mgr_mod = _manager_module()
    # 模型选择优先级：命令行显式 > 上次活跃（持久化）> 内置默认
    model = args.model or mgr_mod.load_last_model() or DEFAULT_MODEL
    restored = (not args.model) and model != DEFAULT_MODEL
    q = "（Q%d 量化）" % args.bits if args.quantize else ""
    print(f"⏳ 启动，加载 {model}{q}{'（恢复上次）' if restored else ''} ...", flush=True)
    manager = mgr_mod.ModelManager(
        default_model=model, quantize=args.quantize, bits=args.bits
    )
    print(f"✓ 就绪 http://{args.host}:{args.port}   设置页 /settings", flush=True)

    _start_metrics_snapshotter()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def _start_metrics_snapshotter(interval_s: int = 60):
    """后台线程：定期把指标快照落盘成历史趋势（仅在有生成活动时写，避免刷空数据）。"""
    import threading
    import time as _time

    def loop():
        last_gen = -1
        while True:
            _time.sleep(interval_s)
            snap = monitor.stats.snapshot()
            gens = snap["totals"]["generations"]
            if gens != last_gen:                      # 只在有新生成时记录一帧
                last_gen = gens
                try:
                    metrics_hist.append(snap, _time.time())
                except Exception:
                    pass

    threading.Thread(target=loop, daemon=True, name="naga-metrics").start()


if __name__ == "__main__":
    main()
