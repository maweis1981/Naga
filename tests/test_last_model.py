"""记住上次活跃模型（naga.manager 持久化）的回归测试。"""

import naga.manager as m


def test_save_and_load_last_model(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "NAGA_DIR", tmp_path)
    monkeypatch.setattr(m, "STATE_FILE", tmp_path / "state.json")
    assert m.load_last_model() is None                 # 无记录
    m.save_last_model("Qwen/Qwen2.5-3B-Instruct")
    assert m.load_last_model() == "Qwen/Qwen2.5-3B-Instruct"
    m.save_last_model("Qwen/Qwen2.5-0.5B-Instruct")    # 覆盖
    assert m.load_last_model() == "Qwen/Qwen2.5-0.5B-Instruct"


def test_save_last_model_preserves_other_state(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "NAGA_DIR", tmp_path)
    monkeypatch.setattr(m, "STATE_FILE", tmp_path / "state.json")
    (tmp_path / "state.json").write_text('{"other":"keep"}')
    m.save_last_model("X/Y")
    import json
    data = json.loads((tmp_path / "state.json").read_text())
    assert data["last_model"] == "X/Y"
    assert data["other"] == "keep"                     # 不破坏其它字段


def test_load_last_model_tolerates_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "STATE_FILE", tmp_path / "state.json")
    (tmp_path / "state.json").write_text("{not json")
    assert m.load_last_model() is None                 # 损坏文件不抛异常
