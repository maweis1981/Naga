"""P14：约束解码（Constrained Decoding）。

痛点：工具调用让小模型自由生成 JSON，常残缺/瞎编工具名/漏字段，解析失败。
思路：把"合法输出"建成一个状态机（自动机）。每个解码步，根据已生成文本所处的
状态，只允许那些"能让输出继续合法"的 token——非法 token 的 logit 直接屏蔽。
模型于是**物理上无法**写出非法结构。

本文件是状态机层（纯 Python，不碰算子）：
  - JsonGrammar     —— 增量 JSON 文法自动机，逐字回答"这个字符合法吗"。
  - ToolCallConstraint —— 在 JSON 之上，强制 {"name":<工具名枚举>,"arguments":<对象>}。
采样侧的接线在 generate.py 的 generate_constrained。
"""

from __future__ import annotations

_WS = " \t\n\r"
_DIGITS = "0123456789"
_NUMCHARS = "0123456789.eE+-"


class JsonGrammar:
    """增量 JSON 值校验：逐字 step，非法即拒（拒时不改状态由调用方快照保证）。

    用一个"期望状态 expect + 容器栈 stack"的下推自动机描述 JSON 语法：
    遇到 { 压入对象、[ 压入数组，值结束后据栈顶决定下一步该接 , 还是闭合。
    """

    def __init__(self):
        self.stack: list[str] = []   # 'obj' / 'arr'
        self.expect = "val"          # 当前期望
        self.lit_rem = ""            # 字面量 true/false/null 的剩余待匹配

    def snapshot(self):
        return (list(self.stack), self.expect, self.lit_rem)

    def restore(self, snap):
        self.stack, self.expect, self.lit_rem = list(snap[0]), snap[1], snap[2]

    def complete(self) -> bool:
        # 顶层值已完整收尾：栈空且处在结束态
        return not self.stack and self.expect == "end"

    def _after_value(self):
        if not self.stack:
            self.expect = "end"
        elif self.stack[-1] == "obj":
            self.expect = "objcomma"
        else:
            self.expect = "arrcomma"

    def step(self, c: str) -> bool:
        e = self.expect

        # 结构位允许空白
        if e in ("val", "val0", "key0", "colon", "objcomma", "arrcomma", "keyc", "end") and c in _WS:
            return True

        if e in ("val", "val0"):
            if c == "{":
                self.stack.append("obj"); self.expect = "key0"; return True
            if c == "[":
                self.stack.append("arr"); self.expect = "val0"; return True
            if c == '"':
                self.expect = "str"; return True
            if c == "-" or c in _DIGITS:
                self.expect = "num"; return True
            if c in "tfn":
                self.lit_rem = {"t": "rue", "f": "alse", "n": "ull"}[c]; self.expect = "lit"; return True
            if e == "val0" and c == "]":          # 空数组
                self.stack.pop(); self._after_value(); return True
            return False

        if e == "key0":                            # { 之后：键 或 }
            if c == '"':
                self.expect = "key"; return True
            if c == "}":
                self.stack.pop(); self._after_value(); return True
            return False

        if e == "keyc":                            # , 之后：必须是键
            if c == '"':
                self.expect = "key"; return True
            return False

        if e == "key":
            if c == "\\":
                self.expect = "key_esc"; return True
            if c == '"':
                self.expect = "colon"; return True
            return True                            # 键里允许任意普通字符
        if e == "key_esc":
            self.expect = "key"; return True

        if e == "str":
            if c == "\\":
                self.expect = "str_esc"; return True
            if c == '"':
                self._after_value(); return True
            return True
        if e == "str_esc":
            self.expect = "str"; return True

        if e == "colon":
            if c == ":":
                self.expect = "val"; return True
            return False

        if e == "num":
            if c in _NUMCHARS:
                return True
            self._after_value()                    # 数字遇非数字字符即结束，重新处理该字符
            return self.step(c)

        if e == "lit":
            if self.lit_rem and c == self.lit_rem[0]:
                self.lit_rem = self.lit_rem[1:]
                if not self.lit_rem:
                    self._after_value()
                return True
            return False

        if e == "objcomma":                        # 对象里一个值之后：, 或 }
            if c == ",":
                self.expect = "keyc"; return True
            if c == "}":
                self.stack.pop(); self._after_value(); return True
            return False

        if e == "arrcomma":                        # 数组里一个值之后：, 或 ]
            if c == ",":
                self.expect = "val"; return True
            if c == "]":
                self.stack.pop(); self._after_value(); return True
            return False

        if e == "end":
            return False                           # 顶层值已结束，只剩空白（上面已放行）

        return False


class ToolCallConstraint:
    """强制输出 {"name": "<合法工具名>", "arguments": <JSON对象>} 的约束。

    分阶段：先逐字强制 {"name": " 脚手架；name 阶段约束到工具名枚举（trie）；
    再强制 ", "arguments": ；arguments 走 JsonGrammar（必须是对象）；最后强制 }。
    """

    def __init__(self, tool_names: list[str]):
        self.names = list(tool_names)
        self.phase = "lit"          # lit: 吐固定脚手架; name; args; tail; done
        self.buf = ""               # 当前阶段已吐文本
        self.target = '{"name": "'  # 当前要逐字吐的固定串
        self.name_so_far = ""
        self.json = JsonGrammar()

    def snapshot(self):
        return (self.phase, self.buf, self.target, self.name_so_far, self.json.snapshot())

    def restore(self, s):
        self.phase, self.buf, self.target, self.name_so_far = s[0], s[1], s[2], s[3]
        self.json.restore(s[4])

    def complete(self) -> bool:
        return self.phase == "done"

    def _name_extends(self, prefix: str) -> bool:
        return any(n.startswith(prefix) for n in self.names)

    def step(self, c: str) -> bool:
        p = self.phase

        if p == "lit":                              # 逐字吐固定脚手架
            if self.buf == self.target:
                pass
            if c == self.target[len(self.buf)]:
                self.buf += c
                if self.buf == self.target:
                    self.phase = "name"; self.name_so_far = ""
                return True
            return False

        if p == "name":                             # 约束到工具名枚举
            if c == '"':
                if self.name_so_far in self.names:
                    self.phase = "tail"; self.buf = ""; self.target = ', "arguments": '
                    return True
                return False                        # 名字没填完，不许收尾
            if self._name_extends(self.name_so_far + c):
                self.name_so_far += c; return True
            return False

        if p == "tail":                             # 逐字吐 , "arguments":
            if c == self.target[len(self.buf)]:
                self.buf += c
                if self.buf == self.target:
                    self.phase = "args"
                return True
            return False

        if p == "args":                             # arguments 必须是 JSON 对象
            # 还没进对象时，只允许空白或 '{'
            if not self.json.stack and self.json.expect == "val" and c not in _WS and c != "{":
                return False
            ok = self.json.step(c)
            if ok and self.json.complete():
                self.phase = "close"                # arguments 对象闭合，接着要外层 }
            return ok

        if p == "close":                            # 闭合整个工具调用的外层 }
            if c in _WS:
                return True
            if c == "}":
                self.phase = "done"; return True
            return False

        if p == "done":
            return False
        return False
