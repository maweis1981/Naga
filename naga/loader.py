"""把磁盘上的权重灌进我们手写的模型。

流程：定位模型目录（本地路径或自动从 HuggingFace 下载）-> 读 config
-> 建模型骨架 -> 用 mx.load 直接读 safetensors -> load_weights 对号入座。
"""

from __future__ import annotations

from pathlib import Path

from .backend import mx

from .config import ModelArgs, load_config
from .models import Model


def ensure_local(model_id: str) -> Path:
    """本地存在就直接用；否则当作 HF 仓库名下载。"""
    p = Path(model_id)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        model_id,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "merges*", "vocab*"],
    )
    return Path(local)


def load_model(model_id: str, quantize: bool = False,
               group_size: int = 64, bits: int = 4, fast_attn: bool = False):
    path = ensure_local(model_id)
    args: ModelArgs = load_config(path)
    args.fast_attn = fast_attn   # 运行期开关，注入到每层注意力

    model = Model(args)

    # 一个或多个 .safetensors 分片，全部读进一个字典
    weights: dict[str, mx.array] = {}
    for shard in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(shard)))

    # 权重 key（model.layers.0.self_attn.q_proj.weight ...）会自动映射到
    # 同名的模块属性。strict=False 容忍 tied 模型里缺失的 lm_head。
    model.load_weights(list(weights.items()), strict=False)

    # P11：可选地把所有线性层量化成 INT4/INT8（省显存、提速）
    if quantize:
        from .quantize import quantize_module
        n = quantize_module(model, group_size, bits)
        print(f"🗜  已量化 {n} 个线性层 -> {bits}-bit (group_size={group_size})")

    mx.eval(model.parameters())  # 立即把权重读进内存（MLX 默认惰性求值）

    return model, args, path


def _sanitize_vlm(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """修正卷积权重布局：PyTorch [O,I,kh,kw] -> MLX 的 NHWC [O,kh,kw,I]。"""
    out = {}
    for k, v in weights.items():
        if k.endswith("patch_embedding.weight") and v.ndim == 4:
            v = v.transpose(0, 2, 3, 1)
        out[k] = v
    return out


def load_vlm(model_id: str):
    """加载 LLaVA 式多模态模型（视觉塔 + 投影 + LLM）。"""
    from .models.llava import LlavaModel, load_llava_config

    path = ensure_local(model_id)
    va, ta, image_token = load_llava_config(path)
    model = LlavaModel(va, ta, image_token)

    weights: dict[str, mx.array] = {}
    for shard in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    weights = _sanitize_vlm(weights)

    # 加载并报告对不上的 key（debug 用：名字写错会在这里暴露）
    model_keys = set(dict(_tree_flatten(model.parameters())).keys())
    file_keys = set(weights.keys())
    missing = sorted(model_keys - file_keys)
    unexpected = sorted(file_keys - model_keys)
    if missing:
        print(f"⚠️ 模型需要但权重缺失 {len(missing)} 个，例如: {missing[:4]}")
    if unexpected:
        print(f"⚠️ 权重多出 {len(unexpected)} 个（未用），例如: {unexpected[:4]}")

    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return model, (va, ta, image_token), path


def _tree_flatten(tree, prefix=""):
    """把嵌套的参数树压平成 [(dotted_key, array)]，用于核对权重名。"""
    items = []
    if isinstance(tree, dict):
        for k, v in tree.items():
            items += _tree_flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            items += _tree_flatten(v, f"{prefix}.{i}")
    else:
        items.append((prefix, tree))
    return items
