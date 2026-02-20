from promin import trace


def test_render_video_path_keeps_mp4_suffix(monkeypatch):
    captured = {}

    def fake_render_states(states, path, fps=30, title="", config=None):
        captured["path"] = path
        return path

    monkeypatch.setattr(trace, "render_states", fake_render_states)

    sm = trace.StateMachine()
    sm.states = []
    sm.render(path="media/demo.mp4")

    assert captured["path"] == "media/demo.mp4"


def test_render_video_path_keeps_gif_suffix(monkeypatch):
    captured = {}

    def fake_render_states(states, path, fps=30, title="", config=None):
        captured["path"] = path
        return path

    monkeypatch.setattr(trace, "render_states", fake_render_states)

    sm = trace.StateMachine()
    sm.states = []
    sm.render(path="media/demo.gif")

    assert captured["path"] == "media/demo.gif"
