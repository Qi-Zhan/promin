from pathlib import Path

from promin import render
from promin.render import runtime


def test_render_states_requests_and_moves_gif(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_generate_scene_source", lambda *args, **kwargs: "")

    class _Result:
        returncode = 0
        stderr = ""

    def fake_run(cmd, capture_output=True, text=True):
        media_dir = Path(cmd[cmd.index("--media_dir") + 1])
        output_name = cmd[cmd.index("-o") + 1]
        fmt = cmd[cmd.index("--format") + 1]
        produced = media_dir / "videos" / "ProminScene" / f"{output_name}.{fmt}"
        produced.parent.mkdir(parents=True, exist_ok=True)
        produced.write_bytes(b"GIF89a")
        return _Result()

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)

    out = render.render_states([], str(tmp_path / "demo.gif"))

    assert out == (tmp_path / "demo.gif").resolve()
    assert out.exists()
    assert out.read_bytes() == b"GIF89a"
