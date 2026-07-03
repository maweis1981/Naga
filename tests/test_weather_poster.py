import json
import naga.builtins as B
from naga.builtins import _build_poster_prompt, tool_weather_poster, BuiltinToolset

def test_prompt_uses_weather_and_no_text():
    w = {"condition":"晴","temperature_c":26.7,"today_low":25,"today_high":39,"humidity":71}
    p = _build_poster_prompt("北京", w)
    assert "北京" in p and "4:5" in p and "26.7" in p
    assert "No text" in p                                  # 文字留后期叠加
    assert "golden sunlight" in p                          # 晴天氛围
    p2 = _build_poster_prompt("上海", {"condition":"小雨","temperature_c":18,"today_low":15,"today_high":20,"humidity":90})
    assert "rain" in p2 and "golden sunlight" not in p2    # 雨天氛围不同

def test_weather_poster_orchestration(monkeypatch):
    # mock 天气 + mock 生图，只验证工作流编排与返回结构
    monkeypatch.setattr(B, "tool_weather", lambda c: {"city":c,"condition":"晴","temperature_c":26.7,
                        "today_low":25,"today_high":39,"humidity":71,"country":"中国"})
    monkeypatch.setattr(B, "_floniks_generate", lambda prompt, alias="nano_banana_2": "https://cdn.x/out/poster.png")
    out = tool_weather_poster("北京")
    assert out["image_url"].endswith("poster.png")
    assert out["markdown"] == "![北京今日天气播报](https://cdn.x/out/poster.png)"
    assert out["condition"] == "晴" and out["city"] == "北京"

def test_weather_poster_weather_fail(monkeypatch):
    monkeypatch.setattr(B, "tool_weather", lambda c: {"error":"找不到城市"})
    assert "error" in tool_weather_poster("nowhere")

def test_weather_poster_gen_fail_returns_prompt(monkeypatch):
    monkeypatch.setattr(B, "tool_weather", lambda c: {"city":c,"condition":"晴","temperature_c":26,
                        "today_low":25,"today_high":39,"humidity":71})
    def boom(prompt, alias="nano_banana_2"): raise RuntimeError("floniks down")
    monkeypatch.setattr(B, "_floniks_generate", boom)
    out = tool_weather_poster("北京")
    assert "error" in out and "prompt" in out              # 失败也回传 prompt，便于排查

def test_weather_poster_registered():
    assert "weather_poster" in {t["name"] for t in BuiltinToolset().tools()}
