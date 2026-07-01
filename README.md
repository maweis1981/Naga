# Naga

**Naga** is a high-performance, from-scratch runtime for LLM inference, serving, and agentic workloads on Apple Silicon.

Built directly on top of [MLX](https://github.com/ml-explore/mlx) tensor operators, Naga implements *everything above the operator layer by itself* — the Transformer forward pass, KV-cache, sampling, quantization, prefix caching, constrained decoding, an OpenAI-compatible server, a WebUI, semantic memory, RAG, and an MCP tool-calling agent. No `transformers`, `llama.cpp`, `vLLM`, `mlx-lm`, or `ollama` under the hood.

It is built for developers who want a lightweight, hackable engine to run, serve, and optimize local LLM applications — and to actually understand every layer of how an inference engine works.

## Features

- **Hand-written multimodal inference** — Qwen2/Qwen2.5 text models and LLaVA-style vision models (self-written SigLIP ViT + projector), with a KV-cached two-phase (prefill/decode) generation loop.
- **Weight quantization (INT4 / INT8)** — self-written `QuantizedLinear` + `QuantizedEmbedding` on `mx.quantized_matmul`. ~1.8× faster decode, ~3× lower memory.
- **RadixAttention prefix caching** — a radix tree that reuses KV across requests; flat per-turn latency for multi-turn chat, RAG, and agent loops.
- **Batched decoding** — multiple sequences (of different lengths, via left-padding + per-sequence RoPE positions and pad-masking) decoded in one batched forward for ~1.6× aggregate throughput at B=6, exposed via `POST /batch` (`{"inputs": [[{role,content}...], ...]}` → per-input completions). Numerically exact vs serial in fp32 (see `tests/test_batched.py`); A/B in `scratch_batch.py`.
- **Constrained decoding** — a self-written JSON grammar automaton that guarantees valid, schema-correct tool calls even from tiny models.
- **OpenAI-compatible serving** — `/v1/chat/completions` with SSE streaming, standard function calling (pass `tools` → get structured `tool_calls` back, honoring `tool_choice: auto | required | {function}` via constrained decoding), context-length guard (clean `context_length_exceeded` 400 like OpenAI), plus a self-built single-page WebUI.
- **Local memory + RAG** — hand-written BERT embeddings, semantic retrieval, and document (txt/md/pdf) chunking & retrieval.
- **MCP agent** — MCP client over **stdio** (local subprocess, with per-call timeouts so a stalled server can't hang a request) **or Streamable HTTP** (remote/hosted servers, JSON or SSE responses), plus a tool-calling loop (`tool_choice: auto | required | none`). Config in `~/.naga/mcp.json`: `{"mcpServers": {"local": {"command": "python3", "args": [...]}, "remote": {"url": "https://host/mcp", "headers": {"Authorization": "Bearer ..."}}}}`.
- **Agent SDK** — an importable `Agent` class + `@tool` decorator (`naga.sdk`) that turns plain Python functions into tools (JSON schema auto-derived from type hints) and runs the constrained tool-calling loop locally, composable with MCP servers. See `examples/agent_sdk_demo.py`.
- **Live monitor dashboard** — `/monitor` streams every inference (prefill/decode tok/s, prefix-cache reuse, tool calls) plus a hardware panel (CPU per-core, unified memory, MLX VRAM, and — with `naga.powermon` — GPU utilization/power/thermal).
- **Self-observability, tracking & optimization** — the same event stream is rolled up into optimization-grade metrics: `GET /metrics` (JSON: decode tok/s p50/p95, TTFT distribution, prefix-cache reuse ratio, per-model throughput, queue depth, tool frequency), `GET /metrics/prometheus` (Prometheus exposition for Grafana/OpenTelemetry), `GET /metrics/advice` (an **optimization advisor** that reads the live profile and suggests concrete tuning — quantize, cut context, check prefix stability, etc.), `GET /metrics/history` (persisted throughput/latency/cache trends that survive restarts, for A/B-ing configs), plus a `GET /health` probe for Open WebUI and orchestrators.
- **Benchmark & profiling tools** — reproducible A/B harnesses for quantization, attention, prefix caching, and constrained decoding.

## Requirements

- macOS on Apple Silicon (M-series)
- Python 3.10+

## Open Source Readiness

The repository now includes the basics expected from an open source project:

- Apache-2.0 `LICENSE`
- packaging metadata in `pyproject.toml`
- `.gitignore` for local artifacts
- `CONTRIBUTING.md`
- `SECURITY.md`

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e .
```

If you are working in an offline environment, make sure the virtualenv already has `setuptools` available before running the editable install.

If you prefer not to install as a package yet, the minimum runtime dependencies are:

```bash
.venv/bin/pip install -U mlx tokenizers huggingface_hub numpy pillow fastapi uvicorn psutil pypdf
```

## Development & Tests

Install dev extras and run the suite (no model download or GPU needed — it uses stub
engines, a fake stdio MCP server, and FastAPI's TestClient):

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

The tests cover the self-observability aggregation (`/metrics` + Prometheus), the
OpenAI-compatible server (streaming `usage`, `/v1/models`, `/health`), the Agent SDK
(`@tool` schema derivation + tool-calling loop), and the MCP client's call/timeout paths.

## Benchmarking

`bench/benchmark_serving.py` is a standard LLM-serving benchmark (metrics aligned with
vLLM's `benchmark_serving` / LLMPerf): **TTFT** (time to first token), **TPOT** (time per
output token), **ITL** (inter-token latency), **E2E** latency, and request/output token
throughput — each reported as mean / median / p99. It drives the OpenAI streaming API
(`stream_options.include_usage` for exact token counts) and is stdlib-only.

```bash
naga serve --model Qwen/Qwen2.5-0.5B-Instruct           # in one terminal
# for a pure-inference measurement, turn off RAG/memory/MCP context injection:
curl -s localhost:8000/admin/settings -H 'Content-Type: application/json' \
  -d '{"rag_enabled":false,"memory_enabled":false,"mcp_enabled":false}'
python bench/benchmark_serving.py --url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-0.5B-Instruct --num-prompts 24 --concurrency 8 --max-tokens 128
```

Example (Qwen2.5-0.5B bf16, M-series): at concurrency 1, TTFT ≈ 14 ms, TPOT ≈ 8 ms
(~120 tok/s decode). Because the scheduler is single-worker serial, raising concurrency
keeps aggregate output throughput flat (~115 tok/s) while TTFT grows with queue depth —
the case the batched-decode path (`POST /batch`) is built to improve.

Add `--batch` to also drive `/batch` (server-side batched forward) and print a serial-vs-batch
A/B. Measured (16 prompts, 64 tokens each): serial streaming ~117 tok/s vs `/batch` (B=16)
**~536 tok/s = 4.6× aggregate throughput** — the batched forward amortizes each weight read
across the whole batch (decode is memory-bandwidth-bound), so the win grows with batch size.

## Quick Start

**Single-shot generation** (downloads `Qwen/Qwen2.5-0.5B-Instruct` on first run):

```bash
.venv/bin/python -m naga.cli "Explain KV-cache in three sentences."
# or, after `pip install -e .`
naga-cli "Explain KV-cache in three sentences."
```

**Quantized + larger model:**

```bash
.venv/bin/python -m naga.cli --model Qwen/Qwen2.5-3B-Instruct --quantize --bits 4 "你好"
```

**Serve an OpenAI-compatible API + WebUI:**

```bash
.venv/bin/python -m naga serve            # then open http://localhost:8000
# or: naga serve
```

If you bind the server to anything other than `127.0.0.1`, set an admin token first:

```bash
export NAGA_ADMIN_TOKEN="change-me"
.venv/bin/python -m naga serve --host 0.0.0.0
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"naga","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

**Unified CLI** (talks to the running server):

```bash
.venv/bin/python -m naga chat "你好"      # one-shot
.venv/bin/python -m naga chat -i          # interactive
.venv/bin/python -m naga models           # list local models
.venv/bin/python -m naga use Qwen/Qwen2.5-3B-Instruct
# or the installed console script:
naga chat "你好"
```

**Available console scripts after `pip install -e .`:**

```bash
naga        # unified thin client: serve/chat/models/use/pull/...
naga-cli    # direct local single-shot inference
naga-serve  # OpenAI-compatible server + WebUI
naga-vlm    # direct multimodal CLI
```

Runtime data (settings, memory, documents, MCP servers) lives under `~/.naga/`.

## Security Notes

- `Naga` is intended for local use by default.
- `/admin/*` endpoints are treated as privileged operations.
- Remote admin access is blocked unless `NAGA_ADMIN_TOKEN` or `--admin-token` is set.
- CORS defaults are limited to localhost-style origins; use `NAGA_CORS_ORIGINS` if you need extra trusted origins.

## Project Status

This repository is already feature-rich, but it is still closer to an advanced hacking-friendly runtime than a polished packaged release. The new `pyproject.toml` makes it installable with standard Python tooling, which is the easiest way to keep the CLI, server, and WebUI entry points aligned.

## Performance

Qwen2.5-3B on an Apple M2 Max (32 GB), greedy decode, same prompt:

| Engine | decode tok/s | peak memory |
|---|---|---|
| Naga (bf16) | 41 | 6.3 GB |
| Naga (Q8) | 60 | 3.4 GB |
| **Naga (Q4)** | **75** | **1.9 GB** |
| ollama (Q4_K_M) | 107 | — |

Naga's hand-written engine reaches ~70% of ollama's heavily-optimized Q4 decode speed, with prefix caching delivering up to **7.9× faster** time-to-first-token on multi-turn conversations. See `scratch_*.py` for the reproducible benchmarks.

## Architecture & Roadmap

Everything above the MLX tensor-operator layer is implemented from scratch.

| Stage | Content | Status |
|---|---|---|
| **P0** | Engine core: Transformer forward + single-shot CLI | ✅ |
| **P1** | KV-cache + sampling (temperature / top-p) + streaming | ✅ |
| **P2** | OpenAI-compatible HTTP API (`/v1/chat/completions` + SSE) | ✅ |
| **P3** | Concurrent scheduling (serial scheduler ✅; batched decode ✅ — 1.6× throughput at B=6; mid-flight continuous admission ⬜) | 🚧 |
| **P4** | Multimodal vision (self-written SigLIP ViT + projector) | ✅ |
| **P5** | WebUI (self-built single-page streaming chat) | ✅ |
| **P6** | Model management (scan / hot-swap / download) + settings | ✅ |
| **P7** | Unified CLI (chat/models/use/pull/serve) | ✅ |
| **P8** | Local memory (hand-written BERT embeddings + semantic retrieval) | ✅ |
| **P9** | Document management + RAG (txt/md/pdf chunk embedding & retrieval) | ✅ |
| **P10** | MCP client + multi-server config + tool-calling agent loop | ✅ |
| **P11** | Weight quantization (INT4/INT8, self-written quantized layers) | ✅ |
| **P12** | Fused attention fast path (optional `--fast-attn`) | ✅ |
| **P13** | RadixAttention prefix KV cache | ✅ |
| **P14** | Constrained decoding (JSON grammar + tool-call constraint) | ✅ |
| **P15** | Self-observability: rolled-up metrics (`/metrics` JSON + Prometheus) + `/health` | ✅ |
| **P19** | Optimization advisor (`/metrics/advice`) + persisted metric history (`/metrics/history`) | ✅ |
| **P16** | Agent SDK (`naga.sdk`: `Agent` + `@tool`, local functions + MCP) | ✅ |

```
naga/
├── config.py / loader.py / tokenizer.py   # model args, weight loading, tokenization
├── models/qwen2.py                        # hand-written Qwen2 forward (Attention/RoPE/GQA/SwiGLU)
├── models/siglip.py / llava.py / bert.py  # vision encoder, VLM, embedding model
├── generate.py                            # autoregressive / cached / constrained generation
├── cache.py / radix.py                    # KV-cache + RadixAttention prefix cache
├── quantize.py / constrain.py             # INT4/INT8 quantization, constrained decoding
├── engine.py / server.py / webui/         # engine wrapper, OpenAI API, WebUI
├── memory.py / docstore.py / embed.py     # semantic memory + RAG
└── mcp.py / agent.py                       # MCP client + tool-calling agent
```

## License

Naga is licensed under the Apache License 2.0.
