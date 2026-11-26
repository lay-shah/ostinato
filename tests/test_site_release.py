"""A release contains only the browser assets from the named commit."""

import hashlib
import json
import subprocess
from pathlib import Path
from runpy import run_path

import pytest

BUILD = run_path(str(Path(__file__).resolve().parents[1] / "tools/build_site.py"))


def test_release_manifest_and_rebuild(tmp_path):
    revision = BUILD["stage"](tmp_path)
    manifest = json.loads((tmp_path / "release.json").read_bytes())
    assert manifest["source_commit"] == revision
    assert set(manifest["sha256"]) == set(BUILD["ASSETS"])
    for name, checksum in manifest["sha256"].items():
        actual = (tmp_path / name).read_bytes()
        assert hashlib.sha256(actual).hexdigest() == checksum
        committed = subprocess.check_output(
            ["git", "show", f"{revision}:docs/{name}"], cwd=BUILD["ROOT"]
        )
        assert actual == committed
    (tmp_path / "index.html").write_text("outdated")
    BUILD["stage"](tmp_path)
    assert hashlib.sha256((tmp_path / "index.html").read_bytes()).hexdigest() == manifest["sha256"]["index.html"]


def test_release_refuses_unrelated_files(tmp_path):
    trial = tmp_path / "trial.json"
    trial.write_text("local")
    with pytest.raises(ValueError, match="previously staged"):
        BUILD["stage"](tmp_path)
    assert list(tmp_path.iterdir()) == [trial]
