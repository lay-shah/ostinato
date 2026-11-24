"""Refresh the Python sources shipped to the browser. Run from any directory."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("breathing", "chest", "decision", "response")


def build(source):
    modules = {f"{name}.py": (source / f"{name}.py").read_text(encoding="utf-8") for name in MODULES}
    # The desktop entry point also opens an OpenCV window. Ship the identical
    # monitor class and its constants without importing the windowing code.
    app = (source / "app.py").read_text(encoding="utf-8")
    pieces = ["from collections import deque", "import numpy as np", "from .breathing import Reading, estimate"]
    for node in ast.parse(app).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"WINDOW", "EXTRA_VIEWS"} for t in node.targets):
            pieces.append(ast.get_source_segment(app, node))
        if isinstance(node, ast.ClassDef) and node.name == "Monitor":
            pieces.append(ast.get_source_segment(app, node))
    modules["app.py"] = "\n\n".join(pieces) + "\n"
    return json.dumps(modules, ensure_ascii=False, indent=2) + "\n"


if __name__ == "__main__":
    (ROOT / "docs/live-core.json").write_text(build(ROOT / "src/ostinato"), encoding="utf-8")
