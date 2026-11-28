#!/bin/bash
# Double-click in Finder. Uses the Mac Terminal, which can ask for camera access.
# Checks a room's setup file, then compares what the webcam sees at each place with it.
cd "$(dirname "$0")/.." || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "Set up Ostinato first. See README.md."
  read -r -p "Press Enter to close. "
  exit 1
fi
echo "Drag the room's setup file into this window and press Enter,"
read -r -p "or just press Enter to try docs/room.example.json: " room
room="${room%"${room##*[![:space:]]}"}"  # Finder adds a space after a dragged file
room="${room//\\ / }"                    # and a backslash before each space in its name
room="${room:-docs/room.example.json}"
if .venv/bin/python -m ostinato.robot --setup "$room" --check; then
  .venv/bin/python -m ostinato.robot --setup "$room" --rehearse all
fi
read -r -p "Press Enter to close. "
