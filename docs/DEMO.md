# Demo guide

## One-minute version, no camera

Run `ostinato --demo`, or double-click `launch/Ostinato Demo.command` on a Mac. Turn the volume up.

| About when | On screen | Say |
|---|---|---|
| 0 s | Debris hides the chest; "Unable to assess" | "It cannot see the chest, so it does not guess." |
| 5 s | The app speaks and watches for a hand | "Before giving up on a view, it asks for a response. Silence proves nothing." |
| 17 s | MOVE the camera | "No response seen, so it looks for a better view. In this demo the move is simulated." |
| 20-45 s | Chest motion builds up | "The tracking, rate estimate, and decisions are the real code, on a drawn person." |
| 48 s | About 16 breaths/min | "Now it trusts the signal, and labels the result with its time." |
| 56 s | Motion fades; the result turns amber | "The motion changed, so the old number is no longer presented as current." |

Press S to save the report and Q to quit.

## Live version

1. Sit 1 to 2 m away, facing a fixed camera, with head, shoulders, and some background in view. Turn off fans and keep people out of the background.
2. Sit still for about 30 seconds until a rate is reported.
3. Cover your chest. The live rate disappears at once and the result turns amber.
4. Press R, then V. With hands down, raise one and hold it.
5. Press R and sway your whole body. It should refuse to call that breathing.

Rehearse steps 3 to 5 on the laptop and in the room you will use. Keep `--demo` ready in case lighting or the camera fails.

## The search simulation

```bash
ostinato-sim scene 1001 --html data/medic.html
```

Say that the map, people, and urgency cue are made up. The result to quote is the honest one: ranking people helps only if the robot has a real clue about who is urgent, and no such clue exists yet. See [SIMULATION.md](SIMULATION.md).

## Do not say

- "It detects whether someone is alive" or "not breathing". It never reports either.
- "Accurate to within half a breath". That is synthetic data. Accuracy on people has not been measured.
- "51 seconds instead of 109". That needs an urgency cue that does not exist.
- "Nobody else does this". Field-tested systems exist, including in the DARPA Triage Challenge.

## Likely questions

**How accurate is it?** Unknown on people. Counted-breath trials are next, and the tools for them are built. On synthetic breathing, mean errors were 0.2 to 0.4 breaths/min and it refused many windows ([ESTIMATOR.md](ESTIMATOR.md)).

**What fools it?** Steady sway in the breathing range, or wind on clothing. It has checks for moving backgrounds and rocking, tested on synthetic video only.

**Why refuse so often?** A wrong number can send help the wrong way. "Unable to assess" asks for a person instead.

**Does it send video anywhere?** The desktop app does not save or send video.

**Where is the robot?** Navigation is simulated. Today a person moves the camera.
