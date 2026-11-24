# Demo guide

## Browser walkthrough

Open the [live camera page](https://sahilnpatel.github.io/ostinato/live.html), then choose **Try sample**. Let the sample finish and select **Download assessment summary**. Select **Restart assessment**, then choose **Ask for response** as soon as it is available to see the scripted hand raise while the new assessment is still gathering evidence. **Stop** ends the session.

For a camera walkthrough, select **Start camera** and allow camera access. Place the camera on a stable surface, 1 to 2 m away from a consenting healthy volunteer. Include their head, both shoulders, chest, and some stationary background. Allow at least 25 seconds of usable video. If the view stays weak, the browser asks the person to raise a hand before requesting a new view or in-person help. The browser can speak using a local English voice if one is available. If the text prompt appears without sound, read it aloud as soon as it appears. The browser then starts watching for a raised hand automatically. Keep the head and shoulders in frame so the camera can see the raised wrist. If the assessment requests a new view, reposition the camera, wait for it to settle, then select **Confirm new view**. Use **Can’t move** when a second view is unavailable. When the assessment finishes, you can download a report with assessment results and decisions. **Stop** releases the camera.

A webcam walkthrough shows whether the interface and camera work under those conditions. Use the [paired webcam trial](VIEWPOINT_TRIALS.md#run-a-paired-webcam-trial) to compare estimates with independently counted breaths.

## Desktop demo, no camera

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

## Desktop webcam walkthrough

1. Sit 1 to 2 m away, facing a fixed camera, with head, shoulders, and some background in view. Turn off fans and keep people out of the background.
2. Sit still for about 30 seconds until a rate is reported.
3. Obstruct the camera’s view of the chest and watch for the live rate to drop out as tracking or signal quality degrades.
4. Press R, then V. With hands down, raise one and hold it.
5. Press R and sway your whole body. It should refuse to call that breathing.

Rehearse steps 3 to 5 on the laptop and in the room you will use. Keep `--demo` ready in case lighting or the camera fails.

## The search simulation

```bash
ostinato-sim scene 1001 --html data/medic.html
```

Say that the map, people, and urgency cue are made up. The result to quote is the honest one: ranking people helps only if the robot has a real clue about who is urgent, and no such clue exists yet. See [SIMULATION.md](SIMULATION.md).

## Interpreting the demo

- A declined assessment or an unseen hand raise leaves breathing status and responsiveness unknown.
- Reported benchmark errors describe accepted windows of synthetic motion. No accuracy results from trials on people have been published.
- Simulated gains from prioritization depend on the urgency cue assigned to the simulated scene.

## Likely questions

**How accurate is it?** Unknown on people. Counted-breath trials are next, and the tools for them are built. On synthetic breathing, mean errors were 0.2 to 0.4 breaths/min and it refused many windows ([ESTIMATOR.md](ESTIMATOR.md)).

**What fools it?** Steady sway in the breathing range, or wind on clothing. It has checks for moving backgrounds and rocking, tested on synthetic video only.

**Why refuse so often?** A wrong number can send help the wrong way. "Unable to assess" asks for a person instead.

**Does it send video anywhere?** Both versions process video on your device. The browser downloads software and model files. Reports contain assessment results and decisions, with no video.

**Where is the robot?** Navigation is simulated. Today a person moves the camera.
