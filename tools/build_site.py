"""Stage the public browser site from the checked-out Git commit."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    "index.html", "live.html", "PROJECT_REPORT.html", "live.js",
    "live-worker.js", "live-engine.py", "live-core.json",
)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def stage(destination):
    revision = git("rev-parse", "HEAD").decode().strip()
    assets = {name: git("show", f"{revision}:docs/{name}") for name in ASSETS}
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in assets.items()}
    manifest = {"source_commit": revision, "sha256": hashes}
    assets["release.json"] = (json.dumps(manifest, indent=2) + "\n").encode()

    destination.mkdir(parents=True, exist_ok=True)
    for entry in destination.iterdir():
        if entry.name not in assets or entry.is_symlink() or not entry.is_file():
            raise ValueError(f"Use an empty or previously staged site directory: {entry}")
    # Publish the manifest last, after staging all files on the same filesystem.
    with TemporaryDirectory(dir=destination) as temporary:
        for name, content in assets.items():
            (Path(temporary) / name).write_bytes(content)
        for name in assets:
            (Path(temporary) / name).replace(destination / name)
    return revision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        print(f"Staged {stage(args.destination)} in {args.destination}")
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")
