# Ostinato

Ostinato is a planned autonomous robot that helps responders find and assess casualties in disaster areas where communication networks are down. It is designed to detect people, measure breathing from chest movement, check for a response to a spoken prompt, and report "unable to assess" when its data is not trustworthy. All processing is intended to run on the device.

**Status:** project setup. No detection or navigation features exist yet.

> Research prototype. Not a medical device, and not for use in real emergencies. No detected motion must never be reported as "not breathing" or "dead".

## First prototype

- Live breathing-rate measurement from a laptop camera, with a confidence score
- A simulated robot that chooses where to look next
- A ranked map explaining why each person was prioritized

Navigation is simulated in the first prototype. See [the plan](docs/PLAN.md).

## Setup

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.
