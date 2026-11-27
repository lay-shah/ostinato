import pytest

from ostinato import placement, speed


def test_placement_rules():
    assert placement.view(0.8, 1.0).problem is None
    assert "overlap" in placement.view(0.3, 1.0).problem
    assert "breathing only" in placement.view(1.2, 0.8).problem
    assert "small" in placement.view(1.8, 4.0).problem
    near, far = placement.view(0.8, 1.0), placement.view(0.8, 2.0)
    assert near.pixels > far.pixels and near.elevation > far.elevation
    assert placement.view(0.8, 1.0, width=1280).pixels == pytest.approx(2 * near.pixels)
    # Straight down, the chest moves toward the camera and not across the image.
    assert placement.view(1.5, 0.001).pixels < 0.01


def test_placement_table_says_it_is_calculated(capsys):
    placement.main([])
    out = capsys.readouterr().out
    assert "Calculated, not measured" in out and "worth trying" in out
    with pytest.raises(SystemExit):
        placement.main(["--fov", "0"])


def test_speed_without_the_pose_model():
    stages = speed.measure(lambda frame: None, frames=3)
    assert all(ms > 0 for name, ms in stages.items() if name != "pose model")
    assert set(speed.ROVER) <= stages.keys() and "steadiness (rover, while settling)" in stages
    times = {"pose model": 60, "chest tracking": 39, "breathing estimate (once a second)": 10}
    assert speed.frame_rate(times) == 10
    times.update({speed.ROVER[0]: 30, speed.ROVER[1]: 1, "steadiness (rover, while settling)": 2})
    assert speed.frame_rate(times, rover=True) == pytest.approx(990 / 130)


def test_speed_report(capsys, monkeypatch):
    class Finder:
        def __call__(self, frame):
            return None

        def close(self):
            pass

    monkeypatch.setattr(speed, "PoseFinder", Finder)
    speed.main(["--frames", "2"])
    out = capsys.readouterr().out
    assert "Desktop app: about" in out and "Rover: about" in out and "markers, 1280 x 720" in out


def test_speed_rejects_a_missing_image():
    with pytest.raises(SystemExit):
        speed.main(["--image", "does-not-exist.jpg"])
