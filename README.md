# Ostinato

Ostinato is a planned autonomous robot that helps responders find and assess casualties in disaster areas where communication networks are down. It is designed to detect people, measure breathing from chest movement, check for a response to a spoken prompt, and report "unable to assess" when its data is not trustworthy. All processing is intended to run on the device.

**Status:** early breathing detector for one person, tested on synthetic signals and video only. It has not been validated against counted breaths. It decides whether to report, wait, move, or get help. Navigation exists only in [simulation](docs/SIMULATION.md).

> Research prototype. Not a medical device, and not for use in real emergencies. No detected motion must never be reported as "not breathing" or "dead".

## First prototype

- Live breathing-rate measurement from a laptop camera, with a confidence score
- A simulated robot that chooses where to look next
- A ranked map explaining why each person was prioritized

Navigation is simulated in the first prototype. See [the plan](docs/PLAN.md).

## Setup

Requires Python 3.10, 3.11, or 3.12.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Run the breathing detector

```bash
ostinato
```

Sit still, 1 to 2 meters away, facing a fixed camera with your shoulders and chest visible and some background on at least one side. Patterned clothing works best. The first reading takes at least 25 seconds. If the app decides to move, reposition the camera and press M. Press R to restart the assessment or Q to quit.

Video is processed on the device and never saved. The signal score is a heuristic, not a probability of being correct. Rates outside 6 to 42 breaths/min are not measured.

## Run the validation workflow

Run the reproducible blocked-view demo with synthetic frames:

```bash
pytest -q tests/test_chest.py::test_paired_trial_from_synthetic_frames_to_decision
```

For a physical trial, an observer counts breaths during two 30-second windows. Keep individual trial files local. See the [paired trial guide](docs/VIEWPOINT_TRIALS.md) for setup, accuracy scoring and replay assumptions.

```bash
mkdir -p data  # ignored by git
python -m ostinato.viewpoints trial data/pair-01.json --views front-1m angled-2m
python -m ostinato.viewpoints report data/pair-01.json --tolerance-bpm 2
python -m ostinato.viewpoints replay data/pair-01.json
```

## See the simulated medic view

```bash
ostinato-sim scene 1001 --html medic.html
```

Open `medic.html` in any browser. It works offline and shows only what the simulated robot knows.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.
