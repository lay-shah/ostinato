# Ostinato

Ostinato is a planned autonomous robot that helps responders find and assess casualties in disaster areas where communication networks are down. It is designed to detect people, measure breathing from chest movement, check for a response to a spoken prompt, and report "unable to assess" when its data is not trustworthy. All processing is intended to run on the device.

**Status:** the browser and desktop prototypes estimate one person’s breathing rate from webcam video, check for a hand raise, and indicate whether to collect more video, change views, or ask for help. Navigation exists in [simulation](docs/SIMULATION.md). Testing has focused on synthetic signals and video. **No accuracy results from trials on people have been published.**

> Research prototype. Not a medical device, and not for use in real emergencies. No detected motion must never be reported as "not breathing" or "dead".

## Try it in the browser

Open the [live camera assessment](https://sahilnpatel.github.io/ostinato/live.html), [synthetic signal demo](https://sahilnpatel.github.io/ostinato/), or [project report](https://sahilnpatel.github.io/ostinato/PROJECT_REPORT.html). The site is public and asks for no account or sign-in.

- **Try sample** runs the assessment on generated motion without requesting your camera.
- **Start camera** requests camera access in your browser. Frame one person’s head, both shoulders, and chest with some stationary background visible. Hold the camera steady and allow at least 25 seconds of usable video. Use **Ask for response** to check for a hand raise, **Confirm new view** after moving the camera when asked, and **Stop** when done. Before leaving a view with persistently weak evidence, the camera session checks for a hand raise. If a local voice is unavailable, read the displayed prompt aloud. See the [demo guide](docs/DEMO.md) for details.

The browser downloads the runtime and, in camera mode, the pose model. Video processing runs in the browser. Ostinato does not upload or save camera frames. **Download assessment summary** saves a local text file with results and decisions after the assessment finishes. Treat reports from real sessions as private.

To preview from a checkout, serve the files on localhost. Open `http://localhost:8000/live.html` in your browser.

```bash
python3 -m http.server --bind 127.0.0.1 --directory docs 8000
```

## Setup

The desktop app requires Python 3.10, 3.11, or 3.12 and can run offline after installation. These setup commands use a POSIX shell.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

## Run the breathing detector

```bash
ostinato
```

Sit still, 1 to 2 meters away, facing a fixed camera with your shoulders and chest visible and some background on at least one side. Keep fans, curtains, and people out of the background. Patterned clothing works best. The first reading takes at least 25 seconds. Before leaving a view or giving up, the app asks the person out loud to raise one hand, using the voice built into your computer, and watches for about 10 seconds. If nothing is seen, it tells you to reposition the camera and press M. Press V to ask at any time, or start with `--no-prompt` to ask only on V. A hand counts only if it was seen down, then held up and out to the side after the request, so touching the face should not count. A hand resting below the frame counts as down, but the raised wrist must be seen. It needs the head, both shoulders and that wrist in view, and cannot tell who raised a hand if people swap places. Not responding proves nothing. Press R to restart the assessment or Q to quit.

Video is processed on the device and never saved. The signal score is a heuristic, not a probability of being correct. Rates outside 6 to 42 breaths/min are not measured.

On a Mac, you can instead double-click `Ostinato.command` or `Ostinato Demo.command` in the `launch` folder.

### Measure someone lying down

Put the camera at their side, level, about 0.8 m high and 1 m away, with the head, shoulders, and hips in view. A hand raised toward the ceiling counts as a response. Straight overhead does not work. Run `python -m ostinato.placement` for other positions, and `python -m ostinato.trial` to record counted breaths. This has been tested on synthetic video only. The [robot build plan](docs/ROBOT.md) uses these positions.

### Try it without a camera

```bash
ostinato --demo
```

This runs the real tracker, estimator, decisions, and voice prompt on a drawn person: a blocked view, a spoken prompt, a move, a measured rate, and then motion that fades. It is labeled synthetic throughout. Add `--no-prompt` to keep it silent.

### Save a report

Press S to save the current assessment as an HTML page in `data/`, which git ignores. It contains times, rates, and decisions and does not contain video. Keep it local unless participants agreed to sharing. Use `--reports FOLDER` to save elsewhere.

## Run the validation workflow

Run the reproducible blocked-view demo with synthetic frames:

```bash
pytest -q tests/test_chest.py::test_paired_trial_from_synthetic_frames_to_decision
```

For a physical trial, use a consenting healthy volunteer and a separate observer. The observer counts breaths during two timed 30-second windows and enters each count before seeing the estimate. Use anonymous view labels and keep individual trial files local. The [paired trial guide](docs/VIEWPOINT_TRIALS.md) covers setup, scoring, and replay assumptions.

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

See [CONTRIBUTING.md](CONTRIBUTING.md) for source changes and Pages deployment.
