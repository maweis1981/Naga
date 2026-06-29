"""Naga 统一命令行（P7）。

把正在运行的服务当后端，命令行只是个"瘦客户端"，通过 HTTP 调用它的
OpenAI 接口和管理接口。这样 CLI 和 WebUI 共享同一个已加载的模型。

用法：
    python -m naga serve                 启动服务
    python -m naga chat "你好"           单轮对话（流式）
    python -m naga chat -i               交互式多轮对话
    python -m naga chat -m 图.jpg "看图"  带图片对话
    python -m naga models                列出本地模型
    python -m naga use Qwen/Qwen2.5-3B-Instruct   切换模型
    python -m naga pull Qwen/Qwen2.5-3B-Instruct  下载新模型
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8000"


def _req(url: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    try:
        return urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers))
    except urllib.error.URLError:
        sys.exit("✗ 连不上服务。先在另一个终端运行：python -m naga serve")


def _image_data_url(path: str) -> str:
    mime = "image/png" if path.lower().endswith("png") else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


def stream_chat(url: str, model, messages) -> str:
    body = {"model": model or "naga", "stream": True, "messages": messages}
    resp = _req(url + "/v1/chat/completions", body)
    print("Naga > ", end="", flush=True)
    acc = ""
    for raw in resp:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        delta = json.loads(payload)["choices"][0]["delta"].get("content")
        if delta:
            print(delta, end="", flush=True)
            acc += delta
    print()
    return acc


def cmd_chat(args):
    # 带图时按 OpenAI 多模态格式组装第一条消息
    def user_msg(text):
        if args.image:
            return {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _image_data_url(args.image)}},
                {"type": "text", "text": text}]}
        return {"role": "user", "content": text}

    if args.interactive:
        print("交互模式（/exit 退出，/clear 清空上下文）")
        messages = []
        while True:
            try:
                q = input("\n你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q in ("/exit", "/quit"):
                break
            if q == "/clear":
                messages = []
                print("（已清空）")
                continue
            if not q:
                continue
            messages.append(user_msg(q))
            args.image = None     # 图片只在首轮带
            reply = stream_chat(args.url, args.model, messages)
            messages.append({"role": "assistant", "content": reply})
    else:
        if not args.prompt:
            sys.exit("用法：naga chat \"你的问题\"  或  naga chat -i")
        stream_chat(args.url, args.model, [user_msg(args.prompt)])


def cmd_models(args):
    data = json.load(_req(args.url + "/v1/models"))["data"]
    for m in data:
        mark = "●" if m["active"] else " "
        loaded = " (已加载)" if m["loaded"] else ""
        print(f" {mark} {m['id']}  [{m['type']}] {m['size_gb']}GB{loaded}")


def cmd_use(args):
    r = json.load(_req(args.url + "/admin/load", {"model": args.model}))
    print("✓ 已切换到:", r.get("active")) if r.get("ok") else sys.exit("✗ " + str(r.get("error")))


def cmd_pull(args):
    _req(args.url + "/admin/download", {"repo": args.repo})
    print(f"⏳ 开始后台下载 {args.repo}，用 `naga models` 或设置页查看进度")


def cmd_remember(args):
    r = json.load(_req(args.url + "/admin/memory/add", {"text": args.text}))
    print("✓ 已记住:", r["item"]["text"]) if r.get("ok") else print("✗ 内容为空")


def cmd_memory(args):
    items = json.load(_req(args.url + "/admin/memory"))["items"]
    if not items:
        print("（还没有记忆。用 `naga remember \"...\"` 添加）")
    for it in items:
        print(f"  [{it['id']}] {it['text']}")


def cmd_forget(args):
    ok = json.load(_req(args.url + "/admin/memory/delete", {"id": args.id}))["ok"]
    print("✓ 已删除" if ok else "✗ 找不到该条")


def cmd_ingest(args):
    import os
    r = json.load(_req(args.url + "/admin/docs/add", {"path": os.path.abspath(args.path)}))
    if not r.get("ok") and "error" in r:
        sys.exit("✗ " + r["error"])
    files = r.get("files", [])
    ok = sum(1 for f in files if f.get("ok"))
    total = sum(f.get("chunks", 0) for f in files if f.get("ok"))
    print(f"✓ 已索引 {ok}/{len(files)} 个文件，共 {total} 个文本块")
    for f in files:
        if not f.get("ok"):
            print(f"  ✗ {f['name']}: {f.get('error')}")


def cmd_docs(args):
    st = json.load(_req(args.url + "/admin/docs"))
    if st["sources"]:
        print("已挂载目录源:")
        for s in st["sources"]:
            print("  📂", s)
    if not st["docs"]:
        print("（还没有文档。用 `naga ingest <文件或目录>` 添加）")
        return
    print(f"文档（共 {st['total_chunks']} 块）:")
    for d in st["docs"]:
        print(f"  📄 {d['name']}  ({d['chunks']} 块)  {d['path']}")


def cmd_mcp(args):
    st = json.load(_req(args.url + "/admin/mcp"))
    if not st["servers"]:
        print("（还没有配置 MCP 服务器。在设置页添加，或编辑 ~/.naga/mcp.json）")
        return
    for s in st["servers"]:
        flag = "✓" if s["connected"] else "✗"
        print(f" {flag} {s['name']}  ({s['command']} {' '.join(s['args'])})")
        if s.get("error"):
            print(f"     错误: {s['error']}")
        for t in s["tools"]:
            print(f"     · {t}")


def cmd_serve(args):
    from .server import main as serve_main
    sys.argv = ["naga-server", "--port", str(args.port), "--host", args.host]
    if args.model:
        sys.argv += ["--model", args.model]
    if getattr(args, "quantize", False):
        sys.argv += ["--quantize", "--bits", str(args.bits)]
    serve_main()


def main():
    ap = argparse.ArgumentParser(prog="naga", description="Naga 命令行")
    ap.add_argument("--url", default=DEFAULT_URL, help="服务地址")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chat", help="对话")
    c.add_argument("prompt", nargs="?")
    c.add_argument("-i", "--interactive", action="store_true", help="交互式多轮")
    c.add_argument("-m", "--image", help="图片路径（多模态模型）")
    c.add_argument("--model", help="指定模型，默认用当前活跃模型")
    c.set_defaults(func=cmd_chat)

    sub.add_parser("models", help="列出本地模型").set_defaults(func=cmd_models)

    u = sub.add_parser("use", help="切换活跃模型")
    u.add_argument("model")
    u.set_defaults(func=cmd_use)

    p = sub.add_parser("pull", help="下载模型")
    p.add_argument("repo")
    p.set_defaults(func=cmd_pull)

    r = sub.add_parser("remember", help="添加一条记忆")
    r.add_argument("text")
    r.set_defaults(func=cmd_remember)

    sub.add_parser("memory", help="列出所有记忆").set_defaults(func=cmd_memory)

    fg = sub.add_parser("forget", help="删除一条记忆")
    fg.add_argument("id")
    fg.set_defaults(func=cmd_forget)

    ig = sub.add_parser("ingest", help="导入文档（文件或目录）到知识库")
    ig.add_argument("path")
    ig.set_defaults(func=cmd_ingest)

    sub.add_parser("docs", help="列出知识库文档").set_defaults(func=cmd_docs)

    sub.add_parser("mcp", help="列出 MCP 服务器与工具").set_defaults(func=cmd_mcp)

    s = sub.add_parser("serve", help="启动服务")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--model", help="启动时加载的默认模型")
    s.add_argument("--quantize", action="store_true", help="对文本模型做 Q4/Q8 量化加载")
    s.add_argument("--bits", type=int, default=4, help="量化位宽（4 或 8）")
    s.set_defaults(func=cmd_serve)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
