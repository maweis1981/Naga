"""内置预置工具：无需配置 MCP 即可用的一批基础工具（时间/万年历/天气/IP 定位）。

与 MCP 工具同接口（tools()/call()），可直接并入 agent 循环。纯标准库；本地工具
（时间、干支、生肖）精确计算，联网工具（天气、IP 定位）失败时优雅降级为错误信息。

注意：服务端工具用「IP 近似定位」，非浏览器 GPS——精确定位需前端 navigator.geolocation。
"""
from __future__ import annotations
import json, urllib.request, urllib.parse
from datetime import datetime, date
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

WEEK = ["星期一","星期二","星期三","星期四","星期五","星期六","星期日"]
GAN = "甲乙丙丁戊己庚辛壬癸"
ZHI = "子丑寅卯辰巳午未申酉戌亥"
ZODIAC = ["鼠","牛","虎","兔","龙","蛇","马","羊","猴","鸡","狗","猪"]
WMO = {0:"晴",1:"晴间多云",2:"多云",3:"阴",45:"雾",48:"雾凇",51:"小毛毛雨",53:"毛毛雨",55:"大毛毛雨",
       61:"小雨",63:"中雨",65:"大雨",71:"小雪",73:"中雪",75:"大雪",80:"阵雨",81:"强阵雨",82:"暴雨",
       95:"雷阵雨",96:"雷阵雨伴冰雹",99:"强雷暴冰雹"}

def _get_json(url, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent":"naga/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def _ganzhi_year(y): return GAN[(y-4)%10] + ZHI[(y-4)%12]

def tool_current_time(timezone: str = ""):
    tz = None
    if timezone and ZoneInfo:
        try: tz = ZoneInfo(timezone)
        except Exception: tz = None
    now = datetime.now(tz) if tz else datetime.now().astimezone()
    off = now.utcoffset(); total = (off.total_seconds()/3600) if off else 0.0
    hh = int(total); mm = int(round(abs(total-hh)*60))
    return {"datetime": now.strftime("%Y-%m-%d %H:%M:%S"), "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"), "weekday": WEEK[now.weekday()],
            "timezone": str(now.tzinfo), "utc_offset": f"UTC{hh:+d}"+(f":{mm:02d}" if mm else ""),
            "timestamp": int(now.timestamp())}

def tool_calendar_info(date_str: str = ""):
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else date.today()
    except Exception:
        return {"error": "date 格式应为 YYYY-MM-DD"}
    y = d.year
    return {"date": d.isoformat(), "weekday": WEEK[d.weekday()],
            "day_of_year": d.timetuple().tm_yday, "week_of_year": int(d.strftime("%W")),
            "year_ganzhi": _ganzhi_year(y), "zodiac": ZODIAC[(y-4)%12], "is_weekend": d.weekday()>=5}

def tool_weather(city: str = ""):
    if not city: return {"error":"请提供城市名 city"}
    try:
        g = _get_json("https://geocoding-api.open-meteo.com/v1/search?count=1&language=zh&name="+urllib.parse.quote(city))
        res = (g or {}).get("results") or []
        if not res: return {"error": f"找不到城市: {city}"}
        loc = res[0]; lat, lon = loc["latitude"], loc["longitude"]
        w = _get_json(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&daily=temperature_2m_max,temperature_2m_min&timezone=auto")
        cur = w.get("current",{}); daily = w.get("daily",{})
        return {"city": loc.get("name"), "country": loc.get("country",""),
                "temperature_c": cur.get("temperature_2m"), "condition": WMO.get(cur.get("weather_code"),"未知"),
                "humidity": cur.get("relative_humidity_2m"), "wind_kmh": cur.get("wind_speed_10m"),
                "today_high": (daily.get("temperature_2m_max") or [None])[0],
                "today_low": (daily.get("temperature_2m_min") or [None])[0]}
    except Exception as e:
        return {"error": f"天气查询失败: {e}"}

def tool_locate_by_ip(ip: str = ""):
    try:
        d = _get_json("http://ip-api.com/json/"+urllib.parse.quote(ip)+"?lang=zh-CN&fields=status,country,regionName,city,timezone,lat,lon,query")
        if d.get("status")!="success": return {"error":"定位失败","raw":d}
        return {"ip": d.get("query"), "city": d.get("city"), "region": d.get("regionName"),
                "country": d.get("country"), "timezone": d.get("timezone"),
                "lat": d.get("lat"), "lon": d.get("lon"), "note":"基于 IP 的近似定位（非 GPS）"}
    except Exception as e:
        return {"error": f"IP 定位失败: {e}"}

_SPECS = [
    {"name":"current_time","arg":"timezone","fn":tool_current_time,
     "description":"获取当前日期、时间、星期、时区。可选 timezone（如 Asia/Shanghai）",
     "schema":{"type":"object","properties":{"timezone":{"type":"string","description":"IANA 时区名，留空为本机时区"}}}},
    {"name":"calendar_info","arg":"date","fn":tool_calendar_info,
     "description":"万年历：查某日的公历详情、星期、干支纪年、生肖。可选 date（YYYY-MM-DD，默认今天）",
     "schema":{"type":"object","properties":{"date":{"type":"string","description":"YYYY-MM-DD，留空为今天"}}}},
    {"name":"weather","arg":"city","fn":tool_weather,
     "description":"查询某城市的实时天气（温度/天气/湿度/风速/今日高低温）。参数 city",
     "schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},
    {"name":"locate_by_ip","arg":"ip","fn":tool_locate_by_ip,
     "description":"按 IP 近似定位所在城市、地区、时区、经纬度。可选 ip（留空为服务器出口 IP）",
     "schema":{"type":"object","properties":{"ip":{"type":"string"}}}},
]

class BuiltinToolset:
    def __init__(self, enabled=None):
        self._by = {s["name"]: s for s in _SPECS}
        self.enabled = set(enabled) if enabled else set(self._by)
    def tools(self):
        return [{"name":s["name"],"description":s["description"],"schema":s["schema"],"server":"builtin"}
                for s in _SPECS if s["name"] in self.enabled]
    def call(self, name, arguments):
        s = self._by.get(name)
        if s is None or name not in self.enabled:
            return f"[错误] 未知内置工具 {name}"
        args = arguments or {}
        val = args.get(s["arg"], "")
        if val == "" and args:
            val = next(iter(args.values()))
        try:
            res = s["fn"](val)
        except Exception as e:
            res = {"error": str(e)}
        return json.dumps(res, ensure_ascii=False)

class CompositeToolset:
    """把内置工具集与 MCP 等多个来源合并成统一的 tools()/call() 视图。"""
    def __init__(self, providers):
        self.providers = [p for p in providers if p is not None]
    def tools(self):
        out=[]
        for p in self.providers:
            try: out.extend(p.tools())
            except Exception: pass
        return out
    def call(self, name, arguments):
        for p in self.providers:
            try:
                if any(t["name"]==name for t in p.tools()):
                    return p.call(name, arguments)
            except Exception: pass
        return f"[错误] 找不到工具 {name}"
