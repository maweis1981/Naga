"""会话存储 ConversationStore 的回归测试。"""

import pytest

from naga.conversations import ConversationStore, _title_from


def test_create_and_get(tmp_path):
    s = ConversationStore(tmp_path)
    conv = s.create()
    assert conv["id"] and conv["title"] == "新会话" and conv["messages"] == []
    got = s.get(conv["id"])
    assert got["id"] == conv["id"]


def test_save_sets_messages_and_autotitle(tmp_path):
    s = ConversationStore(tmp_path)
    conv = s.create()
    msgs = [{"role": "user", "content": "什么是 KV 缓存？"},
            {"role": "assistant", "content": "..."}]
    saved = s.save(conv["id"], msgs)
    assert saved["messages"] == msgs
    assert saved["title"] == "什么是 KV 缓存？"        # 自动从首条 user 消息取标题
    assert saved["updated"] >= saved["created"]


def test_title_from_multimodal():
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "x"}},
        {"type": "text", "text": "描述这张图"}]}]
    assert _title_from(msgs) == "描述这张图"


def test_save_preserves_custom_title(tmp_path):
    s = ConversationStore(tmp_path)
    conv = s.create()
    s.rename(conv["id"], "我的自定义标题")
    saved = s.save(conv["id"], [{"role": "user", "content": "hi"}])
    assert saved["title"] == "我的自定义标题"          # 不被自动标题覆盖


def test_list_sorted_by_updated(tmp_path):
    import time
    s = ConversationStore(tmp_path)
    a = s.create()
    time.sleep(0.01)
    b = s.create()
    s.save(b["id"], [{"role": "user", "content": "later"}])   # b 更晚更新
    lst = s.list()
    assert [c["id"] for c in lst][0] == b["id"]        # 最近更新在前
    assert all("messages" not in c for c in lst)       # 列表不含完整消息
    assert lst[0]["count"] == 1


def test_delete(tmp_path):
    s = ConversationStore(tmp_path)
    conv = s.create()
    assert s.delete(conv["id"]) is True
    assert s.get(conv["id"]) is None
    assert s.delete(conv["id"]) is False               # 再删返回 False


def test_rejects_path_traversal_id(tmp_path):
    s = ConversationStore(tmp_path)
    with pytest.raises(ValueError):
        s.get("../etc/passwd")


def test_get_missing_returns_none(tmp_path):
    s = ConversationStore(tmp_path)
    assert s.get("abc123") is None
    assert s.save("abc123", []) is None


# ── HTTP API 测试 ──────────────────────────────────────────────────
def _api_client(tmp_path):
    from fastapi.testclient import TestClient
    from naga import server
    server.conversations = ConversationStore(tmp_path)
    return TestClient(server.app)


def test_api_crud_flow(tmp_path):
    c = _api_client(tmp_path)
    # 建
    r = c.post("/api/chats", json={})
    cid = r.json()["id"]
    assert r.status_code == 200 and cid
    # 存消息
    r = c.put(f"/api/chats/{cid}", json={"messages": [{"role": "user", "content": "你好世界"}]})
    assert r.json()["title"] == "你好世界"
    # 取
    r = c.get(f"/api/chats/{cid}")
    assert r.json()["messages"][0]["content"] == "你好世界"
    # 列表
    r = c.get("/api/chats")
    assert any(x["id"] == cid for x in r.json()["chats"])
    # 改名
    r = c.patch(f"/api/chats/{cid}", json={"title": "改过的标题"})
    assert r.json()["title"] == "改过的标题"
    # 删
    assert c.delete(f"/api/chats/{cid}").json()["ok"] is True
    assert c.get(f"/api/chats/{cid}").status_code == 404


def test_api_bad_id_returns_400(tmp_path):
    c = _api_client(tmp_path)
    assert c.get("/api/chats/not-hex-id!").status_code == 400


def test_api_missing_returns_404(tmp_path):
    c = _api_client(tmp_path)
    assert c.get("/api/chats/deadbeef").status_code == 404
