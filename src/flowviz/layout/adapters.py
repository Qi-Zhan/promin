from __future__ import annotations

from typing import Any, Callable

from .types import Anchor, LinksLayoutContext, Position


# Temporary adapter for migration from old (ctx -> LayoutResult-like) style.
def old_links_layout_adapter(old_fn: Callable[[Any], Any]) -> Callable[[list[Anchor], Anchor, LinksLayoutContext], list[Anchor]]:
    def _adapt(targets: list[Anchor], origin: Anchor, ctx: LinksLayoutContext) -> list[Anchor]:
        children = []
        for i, t in enumerate(targets):
            children.append(
                {
                    "node_id": t.id,
                    "field": t.meta.get("field"),
                    "direction": t.meta.get("hint", "auto"),
                    "index": i,
                }
            )

        legacy_ctx = type(
            "LegacyLayoutContext",
            (),
            {
                "parent_id": ctx.parent_id,
                "children": children,
                "params": dict(ctx.params),
                "gap_x": ctx.gap_x,
                "gap_y": ctx.gap_y,
            },
        )()
        result = old_fn(legacy_ctx)
        positions = getattr(result, "positions", None)
        if positions is None and isinstance(result, dict):
            positions = result
        if not isinstance(positions, dict):
            raise TypeError("old layout adapter expected result with a dict-like 'positions'")

        out: list[Anchor] = []
        for t in targets:
            dx, dy = positions.get(t.id, (0.0, -ctx.gap_y))
            out.append(t.with_pos(Position(origin.pos.x + float(dx), origin.pos.y + float(dy))))
        return out

    return _adapt
