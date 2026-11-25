import json

import pytest

from ostinato import scenes


@pytest.mark.parametrize(
    ("scene", "posture", "seed"),
    [
        ("breathing and sideways sway", "lying", 7100),
        ("camera shake, faint wall", "lying", 7005),  # once gave a false rate
        ("swaying side to side", "upright", 7100),
        ("rocking", "lying", 7100),
    ],
)
def test_scene(scene, posture, seed):
    _, _, truth, shown = scenes.run((scene, posture, seed))
    rates = [b for b in shown if b is not None]
    if truth is None:
        assert not rates
    else:
        assert rates and max(abs(b - truth) for b in rates) < 1


def test_summary_counts_any_shown_rate():
    results = [("still", "lying", None, [None, 15.0]), ("still", "lying", None, [None]), ("breathing", "upright", 12.0, [None, 13.0])]
    out = scenes.summarize(results)
    assert out["still, lying"] == {"clips": 2, "given_a_rate": 1}
    assert out["breathing, upright"]["largest_error"] == pytest.approx(1.0)


def test_command_labels_results_synthetic(monkeypatch, capsys):
    monkeypatch.setattr(scenes, "FALSE", ("still",))
    monkeypatch.setattr(scenes, "BREATHING", ())
    monkeypatch.setattr(scenes, "SECONDS", 1)
    monkeypatch.setattr(scenes, "ProcessPoolExecutor", lambda workers: Inline())
    scenes.main(["--clips", "1"])
    assert json.loads(capsys.readouterr().out)["source"] == "synthetic scenes"


class Inline:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    map = staticmethod(map)
