"""后端抽象骨架：backend 提供 mx/nn，Mac 上为 MLX，行为不变。"""
def test_backend_provides_mx_nn():
    from naga.backend import mx, nn, name, info
    assert name in ("mlx", "torch")
    assert hasattr(mx, "array") and hasattr(nn, "Module")
    a = mx.array([1, 2, 3])                    # 基本张量可用
    assert list(a.shape) == [3]
    assert info()["backend"] == name


def test_backend_is_mlx_on_apple():
    from naga.backend import is_mlx
    assert is_mlx()                            # 本机是 Apple Silicon → MLX


def test_modules_import_via_backend():
    import naga.generate, naga.cache, naga.radix, naga.loader, naga.quantize  # noqa
    from naga.models import qwen2, bert, siglip, llava  # noqa
    # 抽查：确认没有残留的直接 mlx import
    import pathlib, re
    root = pathlib.Path(naga.generate.__file__).parent
    leaked = []
    for p in list(root.glob("*.py")) + list((root / "models").glob("*.py")):
        if p.name == "backend.py":
            continue
        if re.search(r"^import mlx", p.read_text(encoding="utf-8"), re.M):
            leaked.append(p.name)
    assert not leaked, f"仍直接 import mlx: {leaked}"
