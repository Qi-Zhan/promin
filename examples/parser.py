from dataclasses import dataclass
from typing import Iterable, List, Optional


# -----------------------------
# Tokenizer
# -----------------------------


@dataclass(frozen=True)
class Token:
    kind: str
    text: str

    def __repr__(self) -> str:
        return f"Token({self.kind}, {self.text!r})"


def tokenize(source: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    while i < len(source):
        ch = source[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit():
            start = i
            while i < len(source) and (source[i].isdigit() or source[i] == "."):
                i += 1
            tokens.append(Token("NUM", source[start:i]))
            continue
        if ch.isalpha() or ch == "_":
            start = i
            while i < len(source) and (source[i].isalnum() or source[i] == "_"):
                i += 1
            tokens.append(Token("IDENT", source[start:i]))
            continue
        if ch in "+-*/^=().?:![]":
            tokens.append(Token("OP", ch))
            i += 1
            continue
        raise SyntaxError(f"Unexpected character: {ch!r}")
    tokens.append(Token("EOF", ""))
    return tokens


# -----------------------------
# AST nodes
# -----------------------------


class Expr:
    def to_string(self) -> str:
        return str(self)


@dataclass
class Number(Expr):
    value: str

    def __str__(self) -> str:
        return self.value


@dataclass
class Identifier(Expr):
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class Prefix(Expr):
    op: str
    expr: Expr

    def __str__(self) -> str:
        return f"({self.op} {self.expr})"


@dataclass
class Postfix(Expr):
    op: str
    expr: Expr

    def __str__(self) -> str:
        return f"({self.op} {self.expr})"


@dataclass
class Infix(Expr):
    op: str
    left: Expr
    right: Expr

    def __str__(self) -> str:
        return f"({self.op} {self.left} {self.right})"


@dataclass
class Ternary(Expr):
    cond: Expr
    then_expr: Expr
    else_expr: Expr

    def __str__(self) -> str:
        return f"(? {self.cond} {self.then_expr} {self.else_expr})"


@dataclass
class Index(Expr):
    target: Expr
    index: Expr

    def __str__(self) -> str:
        return f"([ {self.target} {self.index})"


# -----------------------------
# Pratt parser
# -----------------------------


class Parser:
    def __init__(self, tokens: Iterable[Token]) -> None:
        self.tokens = list(tokens)
        self.pos = 0

    def current(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.current()
        self.pos += 1
        return tok

    def expect(self, kind: str, text: Optional[str] = None) -> Token:
        tok = self.current()
        if tok.kind != kind or (text is not None and tok.text != text):
            raise SyntaxError(f"Expected {kind} {text or ''}, got {tok}")
        return self.advance()

    def parse(self) -> Expr:
        expr = self.parse_expression(0)
        self.expect("EOF")
        return expr

    def parse_expression(self, min_bp: int) -> Expr:
        """Parse an expression with precedence climbing based on left binding power."""
        tok = self.advance()
        left = self.nud(tok)

        while True:
            tok = self.current()
            lbp = self.lbp(tok)
            if lbp <= min_bp:
                break
            tok = self.advance()
            left = self.led(tok, left)
        return left

    def nud(self, tok: Token) -> Expr:
        """Null denotation: parse a token that starts an expression."""
        if tok.kind == "NUM":
            return Number(tok.text)
        if tok.kind == "IDENT":
            return Identifier(tok.text)
        if tok.kind == "OP" and tok.text in ("+", "-"):
            right = self.parse_expression(55)
            return Prefix(tok.text, right)
        if tok.kind == "OP" and tok.text == "(":
            expr = self.parse_expression(0)
            self.expect("OP", ")")
            return expr
        raise SyntaxError(f"Unexpected token in nud: {tok}")

    def led(self, tok: Token, left: Expr) -> Expr:
        """Left denotation: parse a token that continues an expression."""
        if tok.kind != "OP":
            raise SyntaxError(f"Unexpected token in led: {tok}")
        op = tok.text

        if op == "?":
            then_expr = self.parse_expression(0)
            self.expect("OP", ":")
            else_expr = self.parse_expression(19)
            return Ternary(left, then_expr, else_expr)

        if op == "=":
            right = self.parse_expression(9)
            return Infix("=", left, right)

        if op in ("+", "-"):
            right = self.parse_expression(31)
            return Infix(op, left, right)

        if op in ("*", "/"):
            right = self.parse_expression(41)
            return Infix(op, left, right)

        if op == ".":
            right = self.parse_expression(69)
            return Infix(op, left, right)

        if op == "!":
            return Postfix(op, left)

        if op == "[":
            index = self.parse_expression(0)
            self.expect("OP", "]")
            return Index(left, index)

        raise SyntaxError(f"Unknown operator: {op}")

    def lbp(self, tok: Token) -> int:
        """Left binding power: determines operator precedence in the led() method."""
        if tok.kind != "OP":
            return 0
        if tok.text == "=":
            return 10
        if tok.text == "?":
            return 20
        if tok.text in ("+", "-"):
            return 30
        if tok.text in ("*", "/"):
            return 40
        if tok.text == "!":
            return 60
        if tok.text == ".":
            return 70
        if tok.text == "[":
            return 80
        return 0


def expr(source: str) -> Expr:
    return Parser(tokenize(source)).parse()


def tests() -> None:
    s = expr("1")
    assert s.to_string() == "1"
    s = expr("1 + 2 * 3")
    assert s.to_string() == "(+ 1 (* 2 3))"
    s = expr("a + b * c * d + e")
    assert s.to_string() == "(+ (+ a (* (* b c) d)) e)"
    s = expr("f . g . h")
    assert s.to_string() == "(. f (. g h))"
    s = expr(" 1 + 2 + f . g . h * 3 * 4")
    assert s.to_string() == "(+ (+ 1 2) (* (* (. f (. g h)) 3) 4))"
    s = expr("--1 * 2")
    assert s.to_string() == "(* (- (- 1)) 2)"
    s = expr("--f . g")
    assert s.to_string() == "(- (- (. f g)))"
    s = expr("-9!")
    assert s.to_string() == "(- (! 9))"
    s = expr("f . g !")
    assert s.to_string() == "(! (. f g))"
    s = expr("(((0)))")
    assert s.to_string() == "0"
    s = expr("x[0][1]")
    assert s.to_string() == "([ ([ x 0) 1)"
    s = expr(
        "a ? b :\n" "         c ? d\n" "         : e",
    )
    assert s.to_string() == "(? a b (? c d e))"
    s = expr("a = 0 ? b : c = d")
    assert s.to_string() == "(= a (= (? 0 b c) d))"
    print("ok")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        tests()
    else:
        if len(sys.argv) > 1:
            source_code = sys.argv[1]
        else:
            source_code = sys.stdin.readline()
        ast = expr(source_code)
        print(ast.to_string())
