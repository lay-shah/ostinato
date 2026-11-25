# Ostinato

Ostinato is a planned autonomous robot that helps responders find and assess casualties in disaster areas where communication networks are down. It is designed to detect people, measure breathing from chest movement, check for a response to a spoken prompt, and report "unable to assess" when its data is not trustworthy. All processing is intended to run on the device.

**Status:** a desktop app measures one person's breathing from a webcam, asks for a hand raise, and decides whether to report, wait, move, or get help. Navigation exists only in [simulation](docs/SIMULATION.md). Tested mostly on synthetic signals and video; **accuracy on people has not been measured**.

> Research prototype. Not a medical device, and not for use in real emergencies. No detected motion must never be reported as "not breathing" or "dead".

## Explore Ostinato

[Run the live camera assessment](https://sahilnpatel.github.io/ostinato/live.html), [try the synthetic signal demo](https://sahilnpatel.github.io/ostinato/), or [read the project report](https://sahilnpatel.github.io/ostinato/PROJECT_REPORT.html). The live browser page downloads its runtime and pose model on first use. It processes frames on your device using the Python chest tracker, breathing estimator, assessment and hand-response logic. The desktop app below can also run offline after installation.

To serve the browser version locally, run `python -m http.server --bind 127.0.0.1 --directory docs 8000` and open `http://localhost:8000/live.html`. After changing the Python core, refresh the browser bundle with `python docs/build_web_core.py`.

## First prototype

- Live breathing-rate measurement from a laptop camera, with a heuristic signal-quality score
- A simulated robot that chooses where to look next
- A ranked map explaining why each person was prioritized

Navigation is simulated in the first prototype. See [the plan](docs/PLAN.md), the [robot build plan](docs/ROBOT.md), and the [demo guide](docs/DEMO.md).

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

Sit still, 1 to 2 meters away, facing a fixed camera with your shoulders and chest visible and some background on at least one side. Keep fans, curtains, and people out of the background. Patterned clothing works best. The first reading takes at least 25 seconds. Before leaving a view or giving up, the app asks the person out loud to raise one hand, using the voice built into your computer, and watches for about 10 seconds. If nothing is seen, it tells you to reposition the camera and press M. Press V to ask at any time, or start with `--no-prompt` to ask only on V. A hand counts only if it was seen down, then held up and out to the side after the request, so touching the face should not count. A hand resting below the frame counts as down, but the raised wrist must be seen. It needs the head, both shoulders and that wrist in view, and cannot tell who raised a hand if people swap places. Not responding proves nothing. Press R to restart the assessment or Q to quit.

Video is processed on the device and never saved. The signal score is a heuristic, not a probability of being correct. Rates outside 6 to 42 breaths/min are not measured.

On a Mac, you can instead double-click `Ostinato.command` or `Ostinato Demo.command` in the `launch` folder.

### Try it without a camera

```bash
ostinato --demo
```

This runs the real tracker, estimator, decisions, and voice prompt on a drawn person: a blocked view, a spoken prompt, a move, a measured rate, and then motion that fades. It is labeled synthetic throughout. Add `--no-prompt` to keep it silent.

### Save a report

Press S to save the current assessment as an HTML page in `data/`, which git ignores. It holds times, rates, and decisions, never video, and only you can read the file. Use `--reports FOLDER` to save elsewhere.

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

To turn finished trials into one page with error, bias, limits of agreement, and coverage:

```bash
ostinato-accuracy data/pair-*.json --html data/accuracy.html
```

## See the simulated medic view

```bash
ostinato-sim scene 1001 --html data/medic.html
```

Open `data/medic.html` in any browser. It works offline and shows only what the simulated robot knows.

How the estimator decides, its benchmark, and its known weaknesses are in [ESTIMATOR.md](docs/ESTIMATOR.md).

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.
