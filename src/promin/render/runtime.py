from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
from manim import DOWN, UP, FadeIn, Scene, Text, YELLOW_C

from .layout_registry import _custom_layout_bootstrap_code
from .scene import _ManimStateRenderer
from .snapshot_view import _contrast_text_color
from .types import RenderConfig

logger = logging.getLogger(__name__)


def render_states(
    states: list,
    path: str,
    fps: int = 30,
    title: str = "",
    config: RenderConfig | None = None,
) -> Path:
    cfg = config or RenderConfig()
    out = Path(path).resolve()
    logger.info("render_states: %d states -> %s", len(states), out)
    format_ext = out.suffix.lower().lstrip(".") or "mp4"
    if format_ext == "mov":
        format_ext = "mp4"
    output_name = out.stem if out.suffix else out.name

    scene_src = _generate_scene_source(states, title=title, config=cfg)

    with tempfile.TemporaryDirectory(prefix="promin_") as tmp:
        script = Path(tmp) / "_promin_scene.py"
        script.write_text(scene_src, encoding="utf-8")

        quality_flag = f"-q{cfg.quality}"
        cmd = [
            sys.executable,
            "-m",
            "manim",
            "render",
            quality_flag,
            "--format",
            format_ext,
            "--fps",
            str(fps),
            "--media_dir",
            tmp,
            "-o",
            output_name,
            str(script),
            "ProminScene",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("render_states: manim stderr: %s", result.stderr)
            raise RuntimeError(f"manim render failed (exit {result.returncode})")

        candidates = list(Path(tmp).rglob(f"{output_name}.{format_ext}"))
        if not candidates:
            candidates = list(Path(tmp).rglob(out.name))
        if candidates:
            import shutil

            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(candidates[0]), str(out))
            logger.info("render_states: moved output to %s", out)
        else:
            raise RuntimeError(
                f"manim render succeeded but output file was not found for {out.name}"
            )

    return out


def _generate_scene_source(
    states: list, title: str = "", config: RenderConfig | None = None
) -> str:
    cfg = config or RenderConfig()

    frames = []
    for s in states:
        frames.append(
            {
                "snapshot": s.snapshot,
                "loc": repr(s.current_loc) if s.current_loc else None,
            }
        )
    frames_json = json.dumps(frames, default=str)
    layout_bootstrap = _custom_layout_bootstrap_code()

    bg = cfg.background_color or "#000000"
    if cfg.text_color == "auto":
        text_color = _contrast_text_color(bg)
    else:
        text_color = cfg.text_color or "#888888"
    title_color = cfg.title_color or YELLOW_C

    bg_line = ""
    if cfg.background_color:
        bg_line = f'self.camera.background_color = "{cfg.background_color}"'

    header = [
        "from __future__ import annotations",
        "import json",
        "import importlib",
        "import promin as pm",
        "from manim import *",
        "from promin.render import (",
        "    _ManimStateRenderer,",
        "    RenderConfig,",
        "    register_layout,",
        ")",
        "",
        f"FRAMES = json.loads({frames_json!r})",
    ]
    if layout_bootstrap:
        header.extend(["", layout_bootstrap])
    header.append("")

    scene_src = textwrap.dedent(
        f"""\
        class ProminScene(Scene):
            def construct(self):
                {bg_line}
                title_text = {title!r}
                title_color = {title_color!r}
                if title_text:
                    t = Text(title_text, font=\"Menlo\", font_size=30, color=title_color)
                    t.to_edge(UP, buff=0.3)
                    self.play(FadeIn(t, shift=DOWN * 0.15), run_time=0.3)

                cfg = RenderConfig(
                    background_color={cfg.background_color!r},
                    node_color={cfg.node_color!r},
                    edge_color={cfg.edge_color!r},
                    title_color={cfg.title_color!r},
                    text_color={text_color!r},
                    quality={cfg.quality!r},
                )
                renderer = _ManimStateRenderer(self, config=cfg)
                n = len(FRAMES)
                for i, frame in enumerate(FRAMES):
                    loc = frame.get(\"loc\")
                    loc_text = f\"S{{i}}  {{loc}}\" if loc else \"\"
                    counter = f\"{{i+1}}/{{n}}\"
                    renderer.show_state(
                        frame[\"snapshot\"], loc_text=loc_text, counter_text=counter,
                    )
                    self.wait(0.3)

                renderer.clear()
                self.wait(1.0)
        """
    )
    return "\n".join(header) + scene_src


def render_states_inline(
    scene: Scene,
    states: list,
    title: str = "",
    origin: np.ndarray | None = None,
    config: RenderConfig | None = None,
) -> None:
    cfg = config or RenderConfig()
    if origin is None:
        origin = np.array([0.0, 0.5, 0.0])

    logger.info("render_states_inline: %d states, origin=%s", len(states), origin.tolist())

    if cfg.background_color:
        scene.camera.background_color = cfg.background_color

    title_color = cfg.title_color or YELLOW_C
    if title:
        t = Text(title, font="Menlo", font_size=30, color=title_color)
        t.to_edge(UP, buff=0.3)
        scene.play(FadeIn(t, shift=DOWN * 0.15), run_time=0.3)

    renderer = _ManimStateRenderer(scene, origin=origin, config=cfg)
    n = len(states)
    for i, state in enumerate(states):
        loc_text = f"S{i}  {state.current_loc}" if state.current_loc else ""
        counter = f"{i + 1}/{n}"
        renderer.show_state(state.snapshot, loc_text=loc_text, counter_text=counter)
        scene.wait(0.3)

    renderer.clear()
    scene.wait(1.0)
