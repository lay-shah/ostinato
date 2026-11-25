#!/bin/bash
# Double-click in Finder. Uses the Mac Terminal, which can ask for camera access.
cd "$(dirname "$0")/.." || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "Set up Ostinato first. See README.md."
  read -r -p "Press Enter to close. "
  exit 1
fi
.venv/bin/python -m ostinato.trial
read -r -p "Press Enter to close. "
