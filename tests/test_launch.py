"""The double-click launchers in launch/ for a Mac."""

import importlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

LAUNCH = Path(__file__).parents[1] / "launch"
COMMANDS = sorted(LAUNCH.glob("*.command"))


def test_there_are_launchers():
    assert {p.name for p in COMMANDS} >= {"Breathing Trials.command", "Rover Rehearsal.command"}


@pytest.mark.parametrize("path", COMMANDS, ids=lambda p: p.stem)
def test_each_launcher_runs_a_real_module(path):
    text = path.read_text()
    assert text.startswith("#!/bin/bash\n") and os.access(path, os.X_OK)
    modules = re.findall(r"-m (ostinato[.\w]*)", text)
    scripts = set(re.findall(r"\.venv/bin/([\w-]+)", text)) - {"python"}
    assert modules or scripts
    for name in modules:
        importlib.import_module(name)
    project = (LAUNCH.parent / "pyproject.toml").read_text()
    section = project.split("[project.scripts]")[1].split("\n[")[0]
    declared = set(re.findall(r"^([\w-]+) = ", section, re.MULTILINE))
    assert scripts <= declared
    if shutil.which("bash"):
        subprocess.run(["bash", "-n", str(path)], check=True)
