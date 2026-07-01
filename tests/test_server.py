"""服务层（naga.server）的回归测试：健康探针 / 指标 / OpenAI 兼容 + 流式用量。"""

import json


def _parse_sse(text):
    """从 SSE 文本里抽出所有 data 帧（去掉 [DONE]）。"""
    frames = []
    for line in text.splitlines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            frames.append(json.loads(line[6:]))
    return frames


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["service"] == "naga"
    assert body["active_model"] == "naga-test"


def test_metrics_json(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    snap = r.json()
    assert "totals" in snap and "decode_tok_s" in snap and "prefix_cache" in snap


def test_metrics_prometheus(client):
    r = client.get("/metrics/prometheus")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "naga_uptime_seconds" in r.text


def test_v1_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    m = data["data"][0]
    assert m["id"] == "naga-test" and m["owned_by"] == "naga"
    assert m["created"] > 0                       # 真实时间戳，不再是 0


def test_chat_completion_non_stream_usage(client):
    r = client.post("/v1/chat/completions",
                    json={"model": "naga-test", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Hello, world"
    assert body["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


def test_chat_completion_stream_with_usage(client):
    r = client.post("/v1/chat/completions",
                    json={"model": "naga-test", "messages": [{"role": "user", "content": "hi"}],
                          "stream": True, "stream_options": {"include_usage": True}})
    frames = _parse_sse(r.text)
    content = "".join(c.get("delta", {}).get("content", "") or ""
                      for f in frames for c in f.get("choices", []))
    assert content == "Hello, world"
    usage_frames = [f for f in frames if f.get("usage")]
    assert len(usage_frames) == 1
    assert usage_frames[0]["choices"] == []       # 用量帧 choices 必须为空（OpenAI 语义）
    assert usage_frames[0]["usage"]["total_tokens"] == 10


def test_chat_completion_stream_without_usage_option(client):
    r = client.post("/v1/chat/completions",
                    json={"model": "naga-test", "messages": [{"role": "user", "content": "hi"}],
                          "stream": True})
    frames = _parse_sse(r.text)
    assert all(not f.get("usage") for f in frames)   # 没要 usage 就不发用量帧


def test_embeddings_validation(client):
    r = client.post("/v1/embeddings", json={"input": ""})
    assert r.status_code == 400


def test_batch_endpoint(client):
    r = client.post("/batch", json={"model": "naga-test", "inputs": [
        [{"role": "user", "content": "one"}],
        [{"role": "user", "content": "two"}],
        [{"role": "user", "content": "three"}],
    ]})
    assert r.status_code == 200
    body = r.json()
    comps = body["completions"]
    assert [c["index"] for c in comps] == [0, 1, 2]
    assert comps[0]["message"]["content"] == "echo:one"      # 不串台
    assert comps[2]["message"]["content"] == "echo:three"
    assert comps[0]["usage"]["total_tokens"] == 7


def test_batch_endpoint_requires_inputs(client):
    r = client.post("/batch", json={"model": "naga-test"})
    assert r.status_code == 400


def test_metrics_advice(client):
    r = client.get("/metrics/advice")
    assert r.status_code == 200
    advice = r.json()["advice"]
    assert isinstance(advice, list) and advice
    assert all("level" in t and "area" in t and "message" in t for t in advice)


def test_metrics_history_endpoint(client):
    r = client.get("/metrics/history")
    assert r.status_code == 200
    assert isinstance(r.json()["history"], list)
