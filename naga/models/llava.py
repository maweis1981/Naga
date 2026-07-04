"""LLaVA 顶层组装：视觉塔 + 投影层 + （复用的）Qwen2 LLM。

把图片变成向量后，最关键的一步是"拼接"（merge）：在文本 token 序列里
有一个图像占位符 <image>，我们把它替换成视觉塔产出的 729 个视觉向量，
让图文向量排成一条序列，再交给 LLM 一起做注意力。位置编码仍是普通 1D，
所以下游的 Qwen2 前向**原样复用**。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..backend import mx, nn

from ..config import ModelArgs
from .qwen2 import Model as LanguageModel
from .siglip import VisionArgs, VisionTower


def gelu(x: mx.array) -> mx.array:
    # 精确 GELU（投影层用），projector_hidden_act=gelu
    return 0.5 * x * (1.0 + mx.erf(x * 0.7071067811865476))


class Projector(nn.Module):
    """两层 MLP，把视觉维度(1152)对齐到文本维度(1024)。"""

    def __init__(self, vision_dim: int, text_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(vision_dim, text_dim)
        self.linear_2 = nn.Linear(text_dim, text_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(gelu(self.linear_1(x)))


class LlavaModel(nn.Module):
    def __init__(self, vision_args: VisionArgs, text_args: ModelArgs, image_token_index: int):
        super().__init__()
        self.vision_tower = VisionTower(vision_args)
        self.multi_modal_projector = Projector(vision_args.hidden_size, text_args.hidden_size)
        self.language_model = LanguageModel(text_args)
        self.image_token_index = image_token_index

    def get_image_features(self, pixel_values: mx.array) -> mx.array:
        feats = self.vision_tower(pixel_values)             # [B, 729, 1152]
        return self.multi_modal_projector(feats)            # [B, 729, 1024]

    def _merge(self, h: mx.array, input_ids: mx.array, img: mx.array) -> mx.array:
        # B=1 简化：把唯一的 <image> 占位符替换成 729 个视觉向量，序列随之变长
        ids = input_ids[0].tolist()
        pos = ids.index(self.image_token_index)
        return mx.concatenate([h[:, :pos], img, h[:, pos + 1:]], axis=1)

    def __call__(self, input_ids: mx.array, pixel_values=None, caches=None) -> mx.array:
        h = self.language_model.model.embed_tokens(input_ids)   # [B, L, 1024]
        if pixel_values is not None:
            h = self._merge(h, input_ids, self.get_image_features(pixel_values))
        return self.language_model(inputs_embeds=h, caches=caches)

    def make_caches(self):
        return self.language_model.make_caches()


def load_llava_config(model_path: Path):
    cfg = json.loads((Path(model_path) / "config.json").read_text())
    vc, tc = cfg["vision_config"], cfg["text_config"]

    va = VisionArgs(
        hidden_size=vc["hidden_size"],
        intermediate_size=vc["intermediate_size"],
        num_hidden_layers=vc["num_hidden_layers"],
        num_attention_heads=vc["num_attention_heads"],
        image_size=vc["image_size"],
        patch_size=vc["patch_size"],
        layer_norm_eps=vc.get("layer_norm_eps", 1e-6),
    )
    ta = ModelArgs(
        hidden_size=tc["hidden_size"],
        intermediate_size=tc["intermediate_size"],
        num_hidden_layers=tc["num_hidden_layers"],
        num_attention_heads=tc["num_attention_heads"],
        num_key_value_heads=tc["num_key_value_heads"],
        vocab_size=tc["vocab_size"],
        rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
        rope_theta=tc.get("rope_theta", 1_000_000.0),
        head_dim=tc.get("head_dim", tc["hidden_size"] // tc["num_attention_heads"]),
        tie_word_embeddings=tc.get("tie_word_embeddings", False),
    )
    return va, ta, cfg["image_token_index"]
