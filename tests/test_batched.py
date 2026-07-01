"""批量解码（naga.generate.batched_generate）的数值正确性测试。

用一个 fp32 的小随机模型（不下载、不碰 bf16 舍入）验证：批内每条序列（含不同长度、
左填充）逐 token 与它单独 serial 生成完全一致——证明位置/掩码/填充的逻辑正确。
真实 bf16 模型上的细微差异是硬件浮点非结合性（与前缀缓存同类），非逻辑错误。
"""

from collections import defaultdict

import mlx.core as mx

from naga.config import ModelArgs
from naga.generate import batched_generate, generate
from naga.models.qwen2 import Model


def _tiny_model():
    mx.random.seed(0)
    args = ModelArgs(hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                     num_attention_heads=4, num_key_value_heads=2, vocab_size=200,
                     rms_norm_eps=1e-6, rope_theta=1e6, head_dim=16, tie_word_embeddings=True)
    model = Model(args)
    mx.eval(model.parameters())
    return model


def test_batched_equals_serial_fp32_varied_lengths():
    model = _tiny_model()
    prompts = [[3, 7, 1, 9, 2, 5, 11], [10, 4, 8], [1, 2, 3, 4, 5]]   # 不同长度
    n = 15
    serial = [list(generate(model, p, n, (), temp=0.0)) for p in prompts]
    bat = defaultdict(list)
    for b, t in batched_generate(model, prompts, n, (), temp=0.0):
        bat[b].append(t)
    for i in range(len(prompts)):
        assert serial[i] == bat[i], (i, serial[i], bat[i])


def test_batched_single_sequence_matches_serial():
    model = _tiny_model()
    p = [5, 9, 2, 7, 1]
    serial = list(generate(model, p, 10, (), temp=0.0))
    bat = [t for _, t in batched_generate(model, [p], 10, (), temp=0.0)]
    assert serial == bat


def test_batched_respects_per_sequence_eos():
    model = _tiny_model()
    # 用某序列自然产出的第一个 token 当它的 eos，验证该序列提前停、其他序列继续
    p0, p1 = [3, 7, 1, 9], [10, 4, 8, 2, 6]
    first0 = next(t for b, t in batched_generate(model, [p0], 3, (), temp=0.0) if b == 0)
    got = defaultdict(list)
    for b, t in batched_generate(model, [p0, p1], 8, (first0,), temp=0.0):
        got[b].append(t)
    assert first0 not in got[0]                     # eos 不产出
    assert len(got[1]) >= 1                          # 另一序列不受影响、继续生成
