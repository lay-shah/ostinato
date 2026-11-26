import shutil
import subprocess
from pathlib import Path

import pytest

FIRMWARE = Path(__file__).parents[1] / "firmware"


@pytest.fixture(scope="session")
def board_program(tmp_path_factory):
    """The motor board's code, compiled to run on this computer."""
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("no C++ compiler")
    program = tmp_path_factory.mktemp("board") / "check"
    flags = ["-std=c++11", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined"]
    subprocess.run([compiler, *flags, "-o", program, FIRMWARE / "check.cpp"], check=True)
    return program
