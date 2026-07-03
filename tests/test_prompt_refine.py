"""海报 prompt 后处理：删 no-text、强制补渲染文字要求 + 专业关键词。"""
from naga.builtins import refine_poster_prompt, tool_weather_poster
import naga.builtins as B

W = {"condition": "多云", "temperature_c": 39.0, "today_high": 39.2, "today_low": 25.7, "humidity": 31}


def test_removes_no_text():
    qwen = "A vertical 4:5 flat illustration of Beijing skyline. No text is used."
    out = refine_poster_prompt(qwen, "北京", W)
    assert "No text" not in out and "no text" not in out.lower()


def test_removes_chinese_no_text():
    out = refine_poster_prompt("北京天际线海报，不要文字。", "北京", W)
    assert "不要文字" not in out


def test_forces_text_requirement():
    out = refine_poster_prompt("A simple poster.", "北京", W)
    assert "clearly render this text" in out
    assert "39.0°C" in out and "北京" in out and "Humidity 31%" in out


def test_adds_quality_keywords():
    out = refine_poster_prompt("A poster.", "北京", W)
    assert "8k" in out and "Cinematic" in out and "Behance" in out


def test_keeps_qwen_content():
    out = refine_poster_prompt("Beijing skyline under cloudy skies with recognizable landmarks.", "北京", W)
    assert "recognizable landmarks" in out            # 保留 qwen 的创意内容


def test_poster_applies_refine(monkeypatch):
    monkeypatch.setattr(B, "tool_weather", lambda c: {"city": c, **W})
    monkeypatch.setattr(B, "_floniks_generate", lambda prompt, alias="nano_banana_2": prompt)  # 回传 prompt 便于检查
    B.set_prompt_llm(lambda city, w: "Beijing skyline. No text.")   # 模拟 qwen 输出带 no text
    try:
        out = tool_weather_poster("北京")
        assert "No text" not in out["prompt"] and "clearly render this text" in out["prompt"]
        assert "8k" in out["prompt"]
    finally:
        B.set_prompt_llm(None)
