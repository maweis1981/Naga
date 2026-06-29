"""GPU 功耗/利用率/热压力采样守护（需要 root）。

Apple Silicon 上，GPU 利用率、功耗、热压力只有 `powermetrics` 能拿到，而它要 root。
本守护解析 powermetrics 的连续输出，把最新一拍写进 ~/.naga/power.json（带时间戳），
Naga 服务端只读这个文件、判断是否新鲜——这样服务本身无需以 root 运行。

启动（在 Naga 目录下）：
    sudo .venv/bin/python -m naga.powermon
或后台：
    sudo nohup .venv/bin/python -m naga.powermon >/dev/null 2>&1 &

坑：sudo 下 HOME 会变成 /var/root，必须用 SUDO_USER 定位到真实用户的 ~/.naga。
"""

from __future__ import annotations

import json
import os
import pwd
import re
import subprocess
import sys
import time
from pathlib import Path


def out_path() -> Path:
    """真实用户（而非 root）的 ~/.naga/power.json。"""
    user = os.environ.get("SUDO_USER") or os.environ.get("USER") or ""
    try:
        home = Path(pwd.getpwnam(user).pw_dir)
    except Exception:
        home = Path.home()
    return home / ".naga" / "power.json"


PATS = {
    "gpu_power_mw": re.compile(r"GPU Power:\s+([\d.]+)\s*mW"),
    "gpu_util": re.compile(r"GPU HW active residency:\s+([\d.]+)%"),
    "gpu_mhz": re.compile(r"GPU HW active frequency:\s+([\d.]+)\s*MHz"),
    "cpu_power_mw": re.compile(r"CPU Power:\s+([\d.]+)\s*mW"),
    "ane_power_mw": re.compile(r"ANE Power:\s+([\d.]+)\s*mW"),
    "combined_power_mw": re.compile(r"Combined Power[^:]*:\s+([\d.]+)\s*mW"),
    "thermal": re.compile(r"pressure level:\s+(\w+)", re.I),
}


def parse_line(line: str, cur: dict) -> str | None:
    """把一行喂进解析器，更新 cur；返回匹配到的字段名（用于判断何时落盘）。"""
    for key, pat in PATS.items():
        m = pat.search(line)
        if m:
            cur[key] = m.group(1) if key == "thermal" else float(m.group(1))
            return key
    return None


def record(cur: dict) -> dict:
    """把累积字段整理成一条精简记录（mW→W）。"""
    rec: dict = {"t": round(time.time(), 1)}
    if "gpu_util" in cur:        rec["gpu_util"] = round(cur["gpu_util"], 1)
    if "gpu_mhz" in cur:         rec["gpu_mhz"] = round(cur["gpu_mhz"])
    if "gpu_power_mw" in cur:    rec["gpu_w"] = round(cur["gpu_power_mw"] / 1000, 2)
    if "cpu_power_mw" in cur:    rec["cpu_w"] = round(cur["cpu_power_mw"] / 1000, 2)
    if "ane_power_mw" in cur:    rec["ane_w"] = round(cur["ane_power_mw"] / 1000, 2)
    if "combined_power_mw" in cur: rec["total_w"] = round(cur["combined_power_mw"] / 1000, 2)
    if "thermal" in cur:         rec["thermal"] = cur["thermal"]
    return rec


def _write(out, rec: dict):
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False))
    os.replace(tmp, out)
    try:
        os.chmod(out, 0o644)                 # 让以普通用户运行的服务可读
    except Exception:
        pass


def main():
    if os.geteuid() != 0:
        sys.exit("powermon 需要 root：sudo .venv/bin/python -m naga.powermon")

    out = out_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    _write(out, {"t": round(time.time(), 1), "starting": True})   # 启动心跳，便于确认路径
    print(f"⚡ powermon 启动，写入 {out}（每秒一拍，Ctrl-C 停止）", flush=True)

    # 轮询模式：每秒独立跑一次 powermetrics 取单拍 —— 自包含、必定输出、即退出，
    # 不依赖长连管道的缓冲行为（流式 -n 0 容易卡住或不输出）。
    cmd = ["powermetrics", "--samplers", "gpu_power,cpu_power,thermal", "-n", "1", "-i", "300"]
    n = 0
    try:
        while True:
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                cur: dict = {}
                for line in res.stdout.splitlines():
                    parse_line(line, cur)
                rec = record(cur)
                if len(rec) <= 1:            # 只有时间戳 = 一个字段都没解析到
                    rec["error"] = "powermetrics 输出未解析到字段"
                    if n == 0:
                        print("⚠️ 未解析到字段，powermetrics 原始输出前 20 行：", flush=True)
                        print("\n".join(res.stdout.splitlines()[:20]) or res.stderr[:500], flush=True)
                _write(out, rec)
            except subprocess.TimeoutExpired:
                _write(out, {"t": round(time.time(), 1), "error": "powermetrics 超时"})
            n += 1
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
