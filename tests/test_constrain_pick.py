"""约束解码取样器 _constrained_pick 的回归测试：约束未完成时 EOS 必须被屏蔽而非停摆。"""

import mlx.core as mx

from naga.generate import _constrained_pick


class FakeConstraint:
    """只接受字符 'x' 一步即完成的玩具约束。"""

    def __init__(self):
        self.done = False

    def complete(self):
        return self.done

    def snapshot(self):
        return self.done

    def restore(self, s):
        self.done = s

    def step(self, ch):
        if ch == "x":
            self.done = True
            return True
        return False


def _decode(ids):
    return "".join("x" if i == 5 else "y" for i in ids)


def test_eos_banned_while_constraint_incomplete():
    # argmax 是 eos(id=0)，但约束还没完成 -> 必须屏蔽 eos，继续找到合法 token(id=5)
    logits = mx.array([10.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0])
    idx = _constrained_pick(logits, [], "", _decode, FakeConstraint(), eos_ids=(0,))
    assert idx == 5                     # 不再因 eos 而返回 None


def test_eos_accepted_once_complete():
    # 约束已完成时，argmax 的 eos 应被接受（允许合法收尾）
    con = FakeConstraint()
    con.done = True
    logits = mx.array([10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    idx = _constrained_pick(logits, [], "", _decode, con, eos_ids=(0,))
    assert idx == 0
