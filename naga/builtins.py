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

def _weather_openmeteo(city):
    g = _get_json("https://geocoding-api.open-meteo.com/v1/search?count=1&language=zh&name="+urllib.parse.quote(city))
    res = (g or {}).get("results") or []
    if not res: raise RuntimeError("geocoding 无结果")
    loc = res[0]; lat, lon = loc["latitude"], loc["longitude"]
    w = _get_json(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&daily=temperature_2m_max,temperature_2m_min&timezone=auto", timeout=8)
    cur = w.get("current",{}); daily = w.get("daily",{})
    return {"city": loc.get("name"), "country": loc.get("country",""), "source": "open-meteo",
            "temperature_c": cur.get("temperature_2m"), "condition": WMO.get(cur.get("weather_code"),"未知"),
            "humidity": cur.get("relative_humidity_2m"), "wind_kmh": cur.get("wind_speed_10m"),
            "today_high": (daily.get("temperature_2m_max") or [None])[0],
            "today_low": (daily.get("temperature_2m_min") or [None])[0]}

_WTTR_CN = {"Sunny":"晴","Clear":"晴","Partly cloudy":"多云","Cloudy":"多云","Overcast":"阴",
            "Mist":"雾","Fog":"雾","Patchy rain possible":"小雨","Light rain":"小雨","Moderate rain":"中雨",
            "Heavy rain":"大雨","Light snow":"小雪","Moderate snow":"中雪","Heavy snow":"大雪",
            "Thundery outbreaks possible":"雷阵雨","Patchy light rain":"小雨"}

def _weather_wttr(city):
    d = _get_json("https://wttr.in/"+urllib.parse.quote(city)+"?format=j1", timeout=10)
    cur = d["current_condition"][0]; today = d["weather"][0]
    desc = (cur.get("weatherDesc") or [{}])[0].get("value","").strip()
    area = (d.get("nearest_area") or [{}])[0]
    name = ((area.get("areaName") or [{}])[0].get("value") or city)
    return {"city": name, "country": ((area.get("country") or [{}])[0].get("value","")), "source": "wttr.in",
            "temperature_c": float(cur.get("temp_C")), "condition": _WTTR_CN.get(desc, desc or "未知"),
            "humidity": int(cur.get("humidity")), "wind_kmh": float(cur.get("windspeedKmph") or 0),
            "today_high": float(today.get("maxtempC")), "today_low": float(today.get("mintempC"))}

def tool_weather(city: str = ""):
    if not city: return {"error":"请提供城市名 city"}
    errors = []
    for src in (_weather_openmeteo, _weather_wttr):     # 主源失败自动切备用源
        try:
            return src(city)
        except Exception as e:
            errors.append(f"{src.__name__}: {e}")
    return {"error": "天气服务暂时不可达（已尝试多个数据源）", "detail": errors}

def tool_locate_by_ip(ip: str = ""):
    try:
        d = _get_json("http://ip-api.com/json/"+urllib.parse.quote(ip)+"?lang=zh-CN&fields=status,country,regionName,city,timezone,lat,lon,query")
        if d.get("status")!="success": return {"error":"定位失败","raw":d}
        return {"ip": d.get("query"), "city": d.get("city"), "region": d.get("regionName"),
                "country": d.get("country"), "timezone": d.get("timezone"),
                "lat": d.get("lat"), "lon": d.get("lon"), "note":"基于 IP 的近似定位（非 GPS）"}
    except Exception as e:
        return {"error": f"IP 定位失败: {e}"}


# ── 工作流工具：天气播报海报（naga 引擎内部编排：天气→prompt→生图→轮询）──
_PROMPT_LLM = None
def set_prompt_llm(fn):
    """注入用 LLM 生成海报 prompt 的函数 fn(city, weather)->str；未注入则用内置模板兜底。"""
    global _PROMPT_LLM
    _PROMPT_LLM = fn

POSTER_MOODS = {
    "晴":"bright warm golden sunlight, clear blue sky, cheerful uplifting mood, sun rays",
    "晴间多云":"soft golden light with a few clouds, warm pleasant mood",
    "多云":"soft diffused daylight, scattered fluffy clouds, calm gentle mood",
    "阴":"overcast soft grey light, calm moody atmosphere",
    "雾":"soft misty fog, dreamy low-visibility atmosphere",
    "小雨":"gentle rain, wet reflective streets, cozy umbrellas, fresh clean air",
    "中雨":"steady rain, glossy wet pavement, dramatic soft clouds",
    "大雨":"heavy rain, dramatic dark clouds, moody wet city",
    "雷阵雨":"dramatic storm clouds, distant lightning, dynamic energetic sky",
    "小雪":"soft falling snow, serene white winter, cozy warm window lights",
    "中雪":"steady snowfall, peaceful snowy cityscape, cool blue tones",
    "大雪":"heavy snow, wintry blizzard mood, warm glowing lights",
}

def _build_poster_prompt(city, w):
    """根据真实天气动态构造社交媒体海报级生图 prompt（文生图；文字后期叠加）。"""
    cond = w.get("condition", "")
    mood = POSTER_MOODS.get(cond, "natural pleasant daylight, clean atmosphere")
    return (
        f"A modern flat-illustration weather-forecast poster for social media, vertical 4:5, "
        f"depicting {city} city skyline with recognizable landmarks under {cond} weather. "
        f"{mood}. Clean minimal design, generous negative space at the top for a headline, "
        f"a large friendly weather icon representing '{cond}', and a lower info band area. "
        f"Temperature about {w.get('temperature_c')}°C (today {w.get('today_low')}°C to {w.get('today_high')}°C), "
        f"humidity {w.get('humidity')}%. Soft pastel gradient background, vibrant but tasteful palette, "
        f"subtle long shadows, high detail, professional editorial poster, trending on Behance, 8k, "
        f"crisp vector style. No text, no letters, no words."
    )

def refine_poster_prompt(prompt, city, w):
    """引擎兜底后处理 qwen 写的 prompt：删掉擅自加的 no-text、强制补上渲染天气文字要求 +
    专业出图关键词。保证成品是「带清晰天气文字的高质量海报」，弥补小模型不听指令的问题。"""
    import re
    p = (prompt or "").strip()
    # 1) 删掉模型擅自加的「no text / 无文字」类语句（它常违背指令）
    p = re.sub(r"[^.。;；\n]*\b[Nn]o\s+(text|letters|words|typography)\b[^.。;；\n]*[.。;；]?", "", p)
    p = re.sub(r"[^.。;；\n]*(不要文字|无文字|不含文字|避免文字)[^.。;；\n]*[.。;；]?", "", p)
    p = re.sub(r"\s{2,}", " ", p).strip().rstrip(",，").strip()
    cond = w.get("condition", ""); tc = w.get("temperature_c"); hi = w.get("today_high"); lo = w.get("today_low"); hu = w.get("humidity")
    # 2) 强制补上「必须渲染这些文字」（拼写正确、清晰易读）
    text_req = (f' IMPORTANT: clearly render this text ON the poster, correctly spelled and legible — '
                f'title "{city}", a big temperature "{tc}°C", weather "{cond}", '
                f'a line "H {hi}° L {lo}°", and "Humidity {hu}%".')
    if "IMPORTANT: clearly render" not in p:
        p += text_req
    # 3) 强制补齐专业出图关键词（小模型常写不全）
    quality = (" Cinematic soft lighting, tasteful color palette, subtle long shadows, "
               "clean minimal composition, high detail, professional editorial poster, "
               "trending on Behance, 8k, crisp vector illustration, vertical 4:5.")
    p += quality
    return p


def _floniks_config():
    """从 ~/.naga/mcp.json 读一个 HTTP 型 MCP（floniks 优先）的 url + headers。"""
    import json as _j
    from pathlib import Path as _P
    f = _P.home()/".naga"/"mcp.json"
    if not f.exists(): return None
    try:
        servers = _j.loads(f.read_text()).get("mcpServers", {})
        # 优先 floniks，其次任意含 single_task 能力的 HTTP MCP
        for name, spec in servers.items():
            if spec.get("url") and "floniks" in name.lower():
                return spec["url"], spec.get("headers", {})
        for name, spec in servers.items():
            if spec.get("url"):
                return spec["url"], spec.get("headers", {})
    except Exception:
        pass
    return None

def _floniks_generate(prompt, model_alias="nano_banana_2", max_polls=2):
    """调 MCP single_task 发起文生图 + 轮询 get_task 取图，返回 image_url（失败抛异常）。"""
    import re
    cfg = _floniks_config()
    if not cfg:
        raise RuntimeError("未配置 HTTP 型 MCP（~/.naga/mcp.json 无 url 服务器）")
    url, headers = cfg
    from .mcp import MCPHttpClient
    c = MCPHttpClient("floniks", url, headers)
    c.initialize()
    res = c.call("single_task", {"modelId": model_alias, "modelType": "text_to_image", "prompt": prompt})
    m = re.search(r'"task_id"\s*:\s*"([^"]+)"', res)
    if not m:
        raise RuntimeError("发起生图失败: " + str(res)[:200])
    tid = m.group(1)
    for _ in range(max_polls):
        r = c.call("get_task", {"id": tid, "wait_seconds": 50})
        urls = re.findall(r'https?://[^\s"\\]+\.(?:png|jpg|jpeg|webp)', r)
        if urls:
            return urls[0]
        if '"terminal": true' in r or '"terminal":true' in r:
            raise RuntimeError("任务终止但无图: " + str(r)[:200])
    raise RuntimeError("生图超时，task_id=" + tid + "（可稍后用 get_task 查询）")

def tool_weather_poster(city: str = "", model_alias: str = "nano_banana_2"):
    """一站式：取真实天气 → 构造 prompt → 生成天气播报海报，返回图片 URL + markdown。"""
    if not city:
        return {"error": "请提供城市名 city"}
    w = tool_weather(city)
    if "error" in w:
        return w
    prompt = None
    if _PROMPT_LLM:                                   # 用 qwen 创作 prompt（更智能）
        try:
            prompt = _PROMPT_LLM(city, w)
        except Exception:
            prompt = None
    if not prompt:                                    # LLM 未注入/失败 → 模板兜底
        prompt = _build_poster_prompt(city, w)
    prompt = refine_poster_prompt(prompt, city, w)    # 引擎兜底：删no-text、补文字要求+专业关键词
    try:
        img = _floniks_generate(prompt, model_alias)
    except Exception as e:
        return {"error": f"生图失败: {e}", "weather": w, "prompt": prompt}
    return {"city": w.get("city"), "condition": w.get("condition"),
            "temperature_c": w.get("temperature_c"),
            "today_low": w.get("today_low"), "today_high": w.get("today_high"),
            "prompt": prompt, "image_url": img,
            "markdown": f"![{city}今日天气播报]({img})"}

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
    {"name":"weather_poster","arg":"city","fn":tool_weather_poster,
     "description":"生成某城市今日天气播报海报（发社交网络用）：自动取实时天气→构造高质量生图prompt→调MCP文生图→返回图片URL。参数 city",
     "schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},
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
