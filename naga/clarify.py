"""工具调用参数澄清：LLM 填的参数缺失/非法/瞎编时，产出可让用户选择的候选（选项 B）。

依据 MCP 元数据推理：
  - 参数有 enum 且值不在其中 → 用 enum 作为可选项；
  - 参数说明形如「the id returned by X」（动态枚举，schema 没写死 enum）→ 引擎自动调工具 X
    取真实候选值（如 modelId → list_model_aliases 的真实 alias），校验 LLM 填的值是否真实存在，
    不存在就产出澄清选项让用户点选。
无法给出真实候选时不澄清（交回常规流程），避免打扰。
"""
from __future__ import annotations

import json
import re

_SOURCE_RE = re.compile(r"returned by (\w+)", re.I)


def _dynamic_source(pinfo: dict) -> str | None:
    m = _SOURCE_RE.search(str(pinfo.get("description", "")))
    return m.group(1) if m else None


def _fetch_candidates(tool_name: str, toolset, want_type: str | None = None) -> list[dict]:
    if toolset is None:
        return []
    try:
        data = json.loads(toolset.call(tool_name, {}))
    except Exception:
        return []
    items = data if isinstance(data, list) else (
        data.get("aliases") or data.get("items") or data.get("data")
        or data.get("models") or [])
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        v = it.get("id") or it.get("alias")
        if not v:
            continue
        typ = str(it.get("type") or it.get("modelType") or "")
        if want_type and typ and want_type not in typ:
            continue
        out.append({"value": v, "label": it.get("name", v), "type": typ})
    return out


def needs_clarify(spec: dict, args: dict, toolset) -> dict | None:
    schema = spec.get("schema") or {}
    props = schema.get("properties", {})
    required = schema.get("required", []) or []
    args = args or {}
    for pname in required:
        pinfo = props.get(pname, {}) if isinstance(props.get(pname), dict) else {}
        val = args.get(pname)
        enum = pinfo.get("enum")
        if enum:
            if val is None or val == "" or val not in enum:
                return {"tool": spec.get("name"), "param": pname,
                        "question": (f"参数「{pname}」的值 {val!r} 无效，请选择" if val
                                     else f"请选择参数「{pname}」"),
                        "options": [{"value": e, "label": str(e)} for e in enum]}
            continue
        src = _dynamic_source(pinfo)
        if src:
            want = args.get("modelType") if pname == "modelId" else None
            opts = _fetch_candidates(src, toolset, want_type=want)
            if opts and (val is None or val == "" or val not in [o["value"] for o in opts]):
                return {"tool": spec.get("name"), "param": pname,
                        "question": (f"「{val}」不是有效的 {pname}，请选择" if val
                                     else f"请选择 {pname}"),
                        "options": opts, "source": src}
    return None
