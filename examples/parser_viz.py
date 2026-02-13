"""
Pratt Parser – step-by-step visualization with Manim.

Shows three layers:
  1. Token bar   – highlights each token as self.pos advances
  2. BP check    – displays the binding-power comparison that drives Pratt parsing
  3. AST tree    – nodes and edges fade in as the parser creates them

Usage
-----
    manim -pql examples/parser_viz.py PrattParserScene

Change EXPRESSION inside the class to visualize a different input.
"""

import numpy as np
from manim import *

from parser import (
    Expr,
    Identifier,
    Index,
    Infix,
    Number,
    Parser,
    Postfix,
    Prefix,
    Ternary,
    Token,
    tokenize,
)


# =================================================================
# Tracing parser – records events for animation replay
# =================================================================


class TracingParser(Parser):
    """Parser subclass that records (advance / bp / create) events."""

    def __init__(self, tokens):
        super().__init__(tokens)
        self.events: list[tuple] = []
        self._id_counter = 0
        self.children_map: dict[int, list[int]] = {}
        self.labels: dict[int, str] = {}

    # -- helpers --------------------------------------------------

    def _new_node(self, label: str, kids: list[int]) -> int:
        self._id_counter += 1
        nid = self._id_counter
        self.labels[nid] = label
        self.children_map[nid] = kids
        return nid

    # -- overrides ------------------------------------------------

    def advance(self) -> Token:
        idx, tok = self.pos, self.current()
        self.pos += 1
        self.events.append(("advance", idx, tok))
        return tok

    def parse_expression(self, min_bp: int) -> Expr:
        tok = self.advance()
        left = self.nud(tok)

        while True:
            tok = self.current()
            lbp_val = self.lbp(tok)
            cont = lbp_val > min_bp
            tok_label = tok.text if tok.text else "EOF"
            self.events.append(("bp", tok_label, lbp_val, min_bp, cont))
            if not cont:
                break
            tok = self.advance()
            left = self.led(tok, left)
        return left

    def nud(self, tok: Token) -> Expr:
        if tok.kind == "NUM":
            nid = self._new_node(tok.text, [])
            node = Number(tok.text)
            node._nid = nid  # type: ignore[attr-defined]
            self.events.append(("create", nid, tok.text, []))
            return node

        if tok.kind == "IDENT":
            nid = self._new_node(tok.text, [])
            node = Identifier(tok.text)
            node._nid = nid  # type: ignore[attr-defined]
            self.events.append(("create", nid, tok.text, []))
            return node

        if tok.kind == "OP" and tok.text in ("+", "-"):
            right = self.parse_expression(55)
            nid = self._new_node(tok.text, [right._nid])  # type: ignore[attr-defined]
            node = Prefix(tok.text, right)
            node._nid = nid  # type: ignore[attr-defined]
            self.events.append(("create", nid, tok.text, [right._nid]))  # type: ignore[attr-defined]
            return node

        if tok.kind == "OP" and tok.text == "(":
            inner = self.parse_expression(0)
            self.expect("OP", ")")
            return inner

        raise SyntaxError(f"Unexpected: {tok}")

    def led(self, tok: Token, left: Expr) -> Expr:
        op = tok.text

        def _infix(rbp: int) -> Expr:
            right = self.parse_expression(rbp)
            kids = [left._nid, right._nid]  # type: ignore[attr-defined]
            nid = self._new_node(op, kids)
            node = Infix(op, left, right)
            node._nid = nid  # type: ignore[attr-defined]
            self.events.append(("create", nid, op, kids))
            return node

        if op == "?":
            then_expr = self.parse_expression(0)
            self.expect("OP", ":")
            else_expr = self.parse_expression(19)
            kids = [left._nid, then_expr._nid, else_expr._nid]  # type: ignore[attr-defined]
            nid = self._new_node("?", kids)
            node = Ternary(left, then_expr, else_expr)
            node._nid = nid  # type: ignore[attr-defined]
            self.events.append(("create", nid, "?", kids))
            return node

        if op == "=":
            return _infix(9)
        if op in ("+", "-"):
            return _infix(31)
        if op in ("*", "/"):
            return _infix(41)
        if op == ".":
            return _infix(69)

        if op == "!":
            kids = [left._nid]  # type: ignore[attr-defined]
            nid = self._new_node("!", kids)
            node = Postfix("!", left)
            node._nid = nid  # type: ignore[attr-defined]
            self.events.append(("create", nid, "!", kids))
            return node

        if op == "[":
            idx_expr = self.parse_expression(0)
            self.expect("OP", "]")
            kids = [left._nid, idx_expr._nid]  # type: ignore[attr-defined]
            nid = self._new_node("[", kids)
            node = Index(left, idx_expr)
            node._nid = nid  # type: ignore[attr-defined]
            self.events.append(("create", nid, "[", kids))
            return node

        raise SyntaxError(f"Unknown op: {op}")


# =================================================================
# Tree layout
# =================================================================


def compute_tree_layout(
    root: int,
    children_map: dict[int, list[int]],
    h_gap: float = 1.6,
    v_gap: float = 1.2,
) -> dict[int, np.ndarray]:
    """Place each node so that leaves are sequential L→R and parents are centered."""

    # 1. compute depth of every node
    depth: dict[int, int] = {}

    def _set_depth(nid: int, d: int):
        depth[nid] = d
        for c in children_map.get(nid, []):
            _set_depth(c, d + 1)

    _set_depth(root, 0)

    # 2. assign x positions (leaves get sequential slots)
    x_pos: dict[int, float] = {}
    _seq = [0.0]

    def _assign_x(nid: int):
        kids = children_map.get(nid, [])
        if not kids:
            x_pos[nid] = _seq[0]
            _seq[0] += 1
            return
        for c in kids:
            _assign_x(c)
        x_pos[nid] = sum(x_pos[c] for c in kids) / len(kids)

    _assign_x(root)

    # 3. center horizontally and auto-shrink when too wide
    all_x = list(x_pos.values())
    cx = (min(all_x) + max(all_x)) / 2
    span = (max(all_x) - min(all_x)) * h_gap
    if span > 11:
        h_gap *= 11 / span

    return {
        nid: np.array([(x_pos[nid] - cx) * h_gap, -depth[nid] * v_gap, 0.0])
        for nid in x_pos
    }


# =================================================================
# Manim scene
# =================================================================

NODE_RADIUS = 0.30
TOKEN_COLORS = {"NUM": BLUE_C, "IDENT": GREEN_C, "OP": GOLD_C}


class PrattParserScene(Scene):
    EXPRESSION = "1 + 2 + f . g . h * 3 * 4"

    def construct(self):
        src = self.EXPRESSION

        # ── trace the parse ──────────────────────────
        tokens = tokenize(src)
        vis_tokens = tokens[:-1]  # drop EOF for on-screen display

        tp = TracingParser(tokens)
        result = tp.parse()
        events = tp.events
        root_id = result._nid  # type: ignore[attr-defined]

        # ── compute tree layout ──────────────────────
        raw_pos = compute_tree_layout(root_id, tp.children_map)
        tree_origin = np.array([0.0, -0.8, 0.0])
        screen_pos = {nid: p + tree_origin for nid, p in raw_pos.items()}

        # ── expression title ─────────────────────────
        title = Text(src, font="Menlo", font_size=36)
        title.to_edge(UP, buff=0.45)

        # ── token bar ────────────────────────────────
        bar = VGroup()
        for tok in vis_tokens:
            col = TOKEN_COLORS.get(tok.kind, WHITE)
            tx = Text(tok.text, font="Menlo", font_size=22)
            bx = RoundedRectangle(
                corner_radius=0.08,
                width=max(tx.width + 0.35, 0.65),
                height=0.5,
                color=col,
                fill_opacity=0.12,
                stroke_width=2,
            )
            bar.add(VGroup(bx, tx))

        bar.arrange(RIGHT, buff=0.22)
        bar.next_to(title, DOWN, buff=0.45)

        # ── position arrow (▼ below current token) ──
        pos_arrow = (
            Triangle(fill_color=RED_C, fill_opacity=1, stroke_width=0)
            .scale(0.1)
            .rotate(PI)
        )
        pos_arrow.next_to(bar[0], DOWN, buff=0.12)

        # fixed anchor for the bp-check text
        bp_anchor = bar.get_bottom() + DOWN * 0.55

        # ── intro ────────────────────────────────────
        self.play(FadeIn(title, shift=DOWN * 0.15), run_time=0.5)
        self.play(
            *[FadeIn(b, shift=UP * 0.15) for b in bar],
            run_time=0.6,
        )
        self.play(FadeIn(pos_arrow, shift=DOWN * 0.08), run_time=0.25)
        self.wait(0.3)

        # ── replay events ────────────────────────────
        node_mobs: dict[int, VGroup] = {}
        consumed: set[int] = set()
        current_bp_mob: VMobject | None = None

        def _clear_bp():
            nonlocal current_bp_mob
            if current_bp_mob is not None:
                self.play(FadeOut(current_bp_mob), run_time=0.1)
                current_bp_mob = None

        for ev in events:
            kind = ev[0]

            # ---- advance: move arrow, highlight token ----
            if kind == "advance":
                _, idx, _tok = ev
                if idx >= len(vis_tokens):
                    continue
                _clear_bp()
                target = bar[idx]
                anims: list = [pos_arrow.animate.next_to(target, DOWN, buff=0.12)]
                if idx not in consumed:
                    consumed.add(idx)
                    anims.append(target[0].animate.set_fill(YELLOW_C, opacity=0.35))
                self.play(*anims, run_time=0.3)

            # ---- bp check: show binding-power comparison ----
            elif kind == "bp":
                _, tok_txt, lbp_val, min_bp, cont = ev
                sym = ">" if cont else "≤"
                tag = "→ led" if cont else "→ break"
                col = GREEN_C if cont else RED_C
                msg = f"bp({tok_txt})={lbp_val} {sym} min_bp={min_bp}  {tag}"
                new_t = Text(msg, font="Menlo", font_size=18, color=col)
                new_t.move_to(bp_anchor)

                anims = []
                if current_bp_mob is not None:
                    anims.append(FadeOut(current_bp_mob, run_time=0.08))
                anims.append(FadeIn(new_t, run_time=0.15))
                self.play(*anims)
                current_bp_mob = new_t
                self.wait(0.12)

            # ---- create: add AST node + edges ----
            elif kind == "create":
                _, nid, label, kids = ev
                _clear_bp()

                pos = screen_pos[nid]
                is_leaf = len(kids) == 0
                col = GREEN_C if is_leaf else BLUE_C

                circ = Circle(
                    radius=NODE_RADIUS,
                    color=col,
                    fill_opacity=0.22,
                    stroke_width=2.5,
                )
                txt = Text(label, font="Menlo", font_size=20, color=WHITE)
                grp = VGroup(circ, txt).move_to(pos)
                node_mobs[nid] = grp

                # edges from this node to each child
                edges = []
                for kid_id in kids:
                    child_center = node_mobs[kid_id].get_center()
                    d = child_center - pos
                    d_len = float(np.linalg.norm(d))
                    if d_len < 1e-6:
                        continue
                    dn = d / d_len
                    edge = Line(
                        pos + dn * NODE_RADIUS,
                        child_center - dn * NODE_RADIUS,
                        color=GREY_B,
                        stroke_width=1.8,
                    )
                    edge.set_z_index(-1)
                    edges.append(edge)

                self.play(
                    FadeIn(grp, scale=0.6),
                    *[Create(e) for e in edges],
                    run_time=0.45,
                )
