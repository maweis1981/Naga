"""天气多源容错：主源失败自动切备用源；都失败给清晰错误。"""
import naga.builtins as B


def test_openmeteo_primary(monkeypatch):
    def fake(url, timeout=10):
        if "geocoding" in url:
            return {"results":[{"name":"北京","country":"中国","latitude":39.9,"longitude":116.4}]}
        return {"current":{"temperature_2m":25,"weather_code":0,"relative_humidity_2m":40,"wind_speed_10m":10},
                "daily":{"temperature_2m_max":[30],"temperature_2m_min":[20]}}
    monkeypatch.setattr(B, "_get_json", fake)
    out = B.tool_weather("北京")
    assert out["source"] == "open-meteo" and out["temperature_c"] == 25


def test_fallback_to_wttr(monkeypatch):
    def fake(url, timeout=10):
        if "open-meteo.com/v1/forecast" in url:
            raise RuntimeError("SSL handshake timed out")   # 主源天气接口挂
        if "geocoding" in url:
            return {"results":[{"name":"Singapore","latitude":1.3,"longitude":103.8}]}
        if "wttr.in" in url:
            return {"current_condition":[{"temp_C":"31","humidity":"71","windspeedKmph":"9",
                        "weatherDesc":[{"value":"Partly cloudy"}]}],
                    "weather":[{"maxtempC":"33","mintempC":"27"}],
                    "nearest_area":[{"areaName":[{"value":"Singapore"}],"country":[{"value":"Singapore"}]}]}
        return {}
    monkeypatch.setattr(B, "_get_json", fake)
    out = B.tool_weather("Singapore")
    assert out["source"] == "wttr.in"                       # 自动切到备用源
    assert out["temperature_c"] == 31.0 and out["condition"] == "多云"


def test_all_sources_fail(monkeypatch):
    def fake(url, timeout=10): raise RuntimeError("network down")
    monkeypatch.setattr(B, "_get_json", fake)
    out = B.tool_weather("北京")
    assert "error" in out and "多个数据源" in out["error"]   # 清晰错误，非笼统超时
