"""图像预处理：把任意来源的图片变成 SigLIP 要的张量。

SigLIP 的处理很简单：缩放到 384×384、像素归一到 [0,1]、再用 mean=std=0.5
标准化到 [-1,1]。输出 NHWC（[1,384,384,3]）以匹配 MLX 的 Conv2d。
"""

from __future__ import annotations

import base64
import io

from .backend import mx
import numpy as np
from PIL import Image


def load_image(src) -> Image.Image:
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    s = str(src)
    if s.startswith(("http://", "https://")):
        import urllib.request
        data = urllib.request.urlopen(s).read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    if s.startswith("data:"):
        b64 = s.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return Image.open(s).convert("RGB")


def preprocess(src, size: int = 384) -> mx.array:
    img = load_image(src).resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0     # [H, W, 3] -> [0,1]
    arr = (arr - 0.5) / 0.5                              # -> [-1, 1]
    return mx.array(arr)[None]                           # [1, H, W, 3]
