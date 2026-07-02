import json
import naga.builtins as B
from naga.builtins import BuiltinToolset, CompositeToolset, tool_calendar_info, tool_current_time

def test_current_time_fields():
    r = tool_current_time("")
    for k in ("datetime","date","time","weekday","timezone","timestamp"): assert k in r

def test_current_time_timezone():
    assert "UTC+8" in tool_current_time("Asia/Shanghai")["utc_offset"]

def test_calendar_ganzhi_zodiac():
    a = tool_calendar_info("2024-02-10"); assert a["year_ganzhi"]=="甲辰" and a["zodiac"]=="龙"
    b = tool_calendar_info("2025-06-01"); assert b["year_ganzhi"]=="乙巳" and b["zodiac"]=="蛇"

def test_calendar_bad_date():
    assert "error" in tool_calendar_info("2024/02/10")

def test_toolset_tools_and_call():
    ts = BuiltinToolset()
    assert {t["name"] for t in ts.tools()} == {"current_time","calendar_info","weather","locate_by_ip"}
    out = json.loads(ts.call("calendar_info", {"date":"2024-02-10"}))
    assert out["zodiac"]=="龙"
    assert ts.call("nope",{}).startswith("[错误]")

def test_weather_mocked(monkeypatch):
    def fake(url, timeout=10):
        if "geocoding" in url:
            return {"results":[{"name":"北京","country":"中国","latitude":39.9,"longitude":116.4}]}
        return {"current":{"temperature_2m":25,"weather_code":0,"relative_humidity_2m":40,"wind_speed_10m":10},
                "daily":{"temperature_2m_max":[30],"temperature_2m_min":[20]}}
    monkeypatch.setattr(B,"_get_json",fake)
    out = json.loads(BuiltinToolset().call("weather",{"city":"北京"}))
    assert out["city"]=="北京" and out["condition"]=="晴" and out["temperature_c"]==25

def test_locate_mocked(monkeypatch):
    monkeypatch.setattr(B,"_get_json",lambda url,timeout=10:{"status":"success","query":"1.2.3.4",
        "city":"Shanghai","regionName":"Shanghai","country":"China","timezone":"Asia/Shanghai","lat":31.2,"lon":121.4})
    out = json.loads(BuiltinToolset().call("locate_by_ip",{}))
    assert out["city"]=="Shanghai" and out["timezone"]=="Asia/Shanghai"

def test_composite():
    class Fake:
        def tools(self): return [{"name":"x","description":"","schema":{}}]
        def call(self,n,a): return "X"
    c = CompositeToolset([BuiltinToolset(), Fake()])
    names = [t["name"] for t in c.tools()]
    assert "current_time" in names and "x" in names
    assert c.call("x",{})=="X" and json.loads(c.call("current_time",{}))["date"]
