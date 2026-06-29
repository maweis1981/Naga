"""模型管理器 + 全局设置。

职责：
  - 扫描本地已下载的模型（HuggingFace 缓存里凡有 config.json 的都算）；
  - 按需加载 / 热切换当前活跃模型，切换时卸载旧模型释放统一内存；
  - 后台线程下载新模型，不阻塞请求；
  - 读写持久化的全局设置（默认参数、系统提示词）。

为省内存，同一时刻只常驻 1 个模型（max_loaded）。个人本地够用，
要同时挂多个把 max_loaded 调大即可。
"""

from __future__ import annotations

import gc
import json
import threading
from pathlib import Path

from .engine import Engine, VlmEngine
from .loader import ensure_local

HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
NAGA_DIR = Path.home() / ".naga"
SETTINGS_FILE = NAGA_DIR / "settings.json"

DEFAULT_SETTINGS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
    "system_prompt": "",
    "memory_enabled": True,
    "rag_enabled": True,
    "mcp_enabled": False,
}


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return {**DEFAULT_SETTINGS, **json.loads(SETTINGS_FILE.read_text())}
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)


def save_settings(s: dict):
    NAGA_DIR.mkdir(exist_ok=True)
    clean = {k: s[k] for k in DEFAULT_SETTINGS if k in s}
    SETTINGS_FILE.write_text(json.dumps({**DEFAULT_SETTINGS, **clean}, ensure_ascii=False, indent=2))


def _repo_from_dir(d: Path) -> str:
    # models--Qwen--Qwen2.5-0.5B-Instruct -> Qwen/Qwen2.5-0.5B-Instruct
    return d.name[len("models--"):].replace("--", "/")


def discover_local() -> list[dict]:
    """扫描 HF 缓存，列出所有"有 config.json"的已下载模型。"""
    out = []
    if not HF_HUB.exists():
        return out
    for d in sorted(HF_HUB.glob("models--*")):
        snaps = sorted(d.glob("snapshots/*"))
        if not snaps:
            continue
        cfg = snaps[-1] / "config.json"
        if not cfg.exists():
            continue
        try:
            mt = json.loads(cfg.read_text()).get("model_type", "?")
        except Exception:
            mt = "?"
        blobs = d / "blobs"
        size = sum(f.stat().st_size for f in blobs.glob("*") if f.is_file()) if blobs.exists() else 0
        out.append({
            "id": _repo_from_dir(d), "type": mt,
            "size_gb": round(size / 1e9, 2),
            "path": str(snaps[-1]),     # 该模型文件所在目录（快照）
        })
    return out


def model_local_path(model_id: str) -> Path | None:
    for m in discover_local():
        if m["id"] == model_id:
            return Path(m["path"])
    return None


def _build_engine(model_id: str, quantize: bool = False, bits: int = 4):
    path = ensure_local(model_id)
    cfg = json.loads((path / "config.json").read_text())
    if cfg.get("model_type") == "llava":
        return VlmEngine(model_id)                       # 多模态暂不量化
    return Engine(model_id, quantize=quantize, bits=bits)


class ModelManager:
    def __init__(self, default_model: str | None = None, max_loaded: int = 1,
                 quantize: bool = False, bits: int = 4):
        self.engines: dict[str, object] = {}
        self.active: str | None = None
        self.max_loaded = max_loaded
        self.quantize = quantize          # 对文本模型统一应用的量化设置
        self.bits = bits
        self.settings = load_settings()
        self._dl_status = "idle"
        from .docstore import DocumentStore
        from .memory import MemoryStore
        self.memory = MemoryStore()         # 记忆库（嵌入器首次用到时才加载）
        self.docs = DocumentStore()         # 文档库（RAG）
        from .mcp import MCPManager
        self.mcp = MCPManager()             # MCP 工具服务器
        self.mcp.connect_all()              # 连接已配置的服务器（无配置则空转）
        if default_model:
            self.ensure(default_model)

    def ensure(self, model_id: str):
        """确保 model_id 已加载并设为活跃；超过上限时卸载旧模型。"""
        if model_id in self.engines:
            self.active = model_id
            return self.engines[model_id]

        if len(self.engines) >= self.max_loaded:
            self.engines.clear()           # 卸载所有旧模型
            gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass

        engine = _build_engine(model_id, self.quantize, self.bits)
        self.engines[model_id] = engine
        self.active = model_id
        return engine

    def get(self, model_id: str | None = None):
        """取引擎：指定且已知就切过去，否则用当前活跃模型。"""
        known = {m["id"] for m in discover_local()}
        if model_id and model_id in known:
            return self.ensure(model_id)
        if self.active is None:
            raise RuntimeError("还没有加载任何模型")
        return self.engines[self.active]

    def download(self, repo_id: str):
        """后台线程下载，立即返回；进度用 dl_status() 查。"""
        def run():
            try:
                self._dl_status = f"下载中：{repo_id}"
                ensure_local(repo_id)
                self._dl_status = f"完成：{repo_id}"
            except Exception as e:
                self._dl_status = f"失败：{repo_id} — {e}"
        threading.Thread(target=run, daemon=True).start()

    @property
    def dl_status(self) -> str:
        return self._dl_status

    def state(self) -> dict:
        return {
            "active": self.active,
            "loaded": list(self.engines.keys()),
            "available": discover_local(),
            "download_status": self._dl_status,
            "settings": self.settings,
        }
