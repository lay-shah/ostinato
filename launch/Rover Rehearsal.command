#!/bin/bash
# Double-click in Finder. Uses the Mac Terminal, which can ask for camera access.
# Checks a room's setup file and draws it, then compares what the webcam sees at each
# place with it.
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
mkdir -p data
map="data/room-map-$(date +%Y%m%d-%H%M%S).png"
.venv/bin/python -m ostinato.robot --setup "$room" --check --map "$map"
passed=$?
open "$map" 2>/dev/null  # compare it with the room, problems marked, before going on
if [ $passed -eq 0 ]; then
  record="${map%.png}.txt"
  record="${record/room-map/rehearsal}"
  .venv/bin/python -u -m ostinato.robot --setup "$room" --rehearse all | tee "$record"
  echo "Saved what it saw to $record, and the map to $map."
else
  echo "Fix the problems in the setup file, then run this again."
fi
read -r -p "Press Enter to close. "
