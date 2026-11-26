import re
from pathlib import Path

ROOT = Path(__file__).parents[1]

NAMES = {
    "LEFT_PWM": "Left motor PWM",
    "LEFT_DIR": "Left motor direction",
    "RIGHT_PWM": "Right motor PWM",
    "RIGHT_DIR": "Right motor direction",
    "LEFT_A": "Left encoder A",
    "LEFT_B": "Left encoder B",
    "RIGHT_A": "Right encoder A",
    "RIGHT_B": "Right encoder B",
    "FRONT_TRIG": "Front distance sensor trigger",
    "FRONT_ECHO": "Front distance sensor echo",
    "REAR_TRIG": "Rear distance sensor trigger",
    "REAR_ECHO": "Rear distance sensor echo",
    "ESTOP": "Emergency stop pole 2",
}


def test_wiring_guide_matches_the_board_program():
    sketch = (ROOT / "firmware" / "ostinato_board" / "ostinato_board.ino").read_text()
    pins = dict(re.findall(r"\b([A-Z_]+) = (A?\d+)\b", sketch))
    guide = (ROOT / "docs" / "WIRING.md").read_text()
    table = {what: pin for pin, what in re.findall(r"^\| (A?\d+) \| ([^|]+?) \|", guide, re.MULTILINE)}
    for name, words in NAMES.items():
        found = [pin for what, pin in table.items() if what.startswith(words)]
        assert found == [pins[name]], name
    assert len({pins[n] for n in NAMES}) == len(NAMES)  # no pin used twice
