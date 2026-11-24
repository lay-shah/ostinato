# Can a second viewpoint recover a usable reading?

Use this prototype to study camera measurements and viewpoint selection. A declined reading means the available video was insufficient. The signal score describes the motion signal. It has no established interpretation as a probability of breathing or clinical urgency.

## Run the simulation

After [desktop setup](../README.md#setup), run from the project root:

```sh
python -m ostinato.viewpoints simulate
```

Both policies start at position zero, travel at one coordinate unit per second, and spend 30 seconds observing at each of two marked locations. Nearest-first chooses by travel time. Visibility-first prefers a geometrically unobstructed view, breaking ties by travel time. Each policy sees only geometry before it moves. On a declined observation it tries the remaining view once. Each event records the viewpoint, elapsed time and reason. A null time to reading means both attempts failed.

| Scripted scene | Nearest-first | Visibility-first |
|---|---:|---:|
| Near view obstructed | 64 s | 34 s |
| Both clear | 31 s | 31 s |
| Clear view has weak signal | 31 s | 67 s |
| Neither usable | Declines after 64 s | Declines after 67 s |

These outcomes are assigned fixtures for checking the decision logic. The simulator does not detect obstructions, estimate real-world travel times, or model clinical priority.

## Run a paired webcam trial

Follow the [desktop setup](../README.md#setup) first. This workflow opens the camera from the desktop Python app. If your system asks for permission, allow camera access for the terminal or Python. Add `--camera 1` if the intended camera is device 1.

1. Recruit a consenting healthy volunteer and a separate observer. Follow the data-collection rules in [the project plan](PLAN.md).
2. Mark two camera positions and measure their distances from the volunteer. Start with a clear front view at 1 m and a clear angled view at 2 m. Keep the volunteer seated and the camera fixed during each window. Frame the head, both shoulders, chest, and some stationary background. Keep lighting and camera settings consistent.
3. Agree on a counting rule before starting. For example, count each inhalation that starts between START and STOP once and ask the volunteer to breathe normally. Seat the observer where they can see or hear breaths independently of the webcam and shield them from the estimate.
4. Run the trial commands. At each view, follow the terminal prompts, allow the framing check, count for the marked window, and enter the whole-number count. Repeat both views if the capture is interrupted. For repeated pairs, swap the order and include a partially obscured condition. Keep completed pairs even when both windows are declined.

```sh
mkdir -p data  # ignored by git
python -m ostinato.viewpoints trial data/pair-01.json --views front-1m angled-2m
python -m ostinato.viewpoints trial data/pair-02.json --views angled-2m front-1m
python -m ostinato.viewpoints report data/pair-01.json data/pair-02.json
```

Before each window the program spends up to 8 seconds saying whether it can track the chest. This is advice only and every view is recorded, so difficult views are not dropped.

Each view starts with a fresh tracker and has its own reference count. Use anonymous view labels. The command saves elapsed times, counts, estimated rates, signal scores, decline reasons, and the timing of per-frame readings. It does not write images or video. A completed pair needs two uninterrupted windows. Keep trial files local unless participants agreed to sharing.

## Read the report

Record reported windows divided by all windows (coverage), mean absolute error over reported windows, the number of first-view declines and how often a second view returned a rate. Inspect recovered readings against their own manual references before calling them accurate. The reference is the count multiplied by 60 and divided by the recorded window duration in seconds. One extra count in 30 seconds changes it by 2 breaths/min. Run `ostinato-accuracy data/pair-*.json --html data/accuracy.html` to also report signed bias and limits of agreement across measured windows. For a small sample or repeated windows from the same volunteer, treat these figures as descriptive.

Use the results to decide the next build. If changing position repeatedly produces a usable and accurate estimate, collect more varied scenes and test viewpoint selection on held-out trials. If it does not, inspect tracking and signal quality before building a mobile platform. Revisit the decision with a responder using a past incident and ask which observation would have changed their next action.

## Score recovered readings and replay the decision

New paired trials save version 3 files with the live monitor's reading for every frame. The report still accepts versions 1 and 2. Version 2 sampled about once a second and could miss tracking dropouts, so it cannot be replayed. Set an explicit experimental error tolerance to count recoveries within that distance of the independent manual rate. The default of 2 breaths/min corresponds to one counted breath in a 30-second window. It is a reporting choice without clinical validation.

```sh
python -m ostinato.viewpoints report data/pair-01.json --tolerance-bpm 2
python -m ostinato.viewpoints replay data/pair-01.json
ostinato-sim compare --scenes 10
```

Replay feeds the timed samples through the same per-person assessor in recorded and reversed order. Both policies get 30 seconds per view and zero modeled travel time. Results describe an offline counterfactual using the two recorded windows. Older files contain too little evidence to replay confirmation. Replay never substitutes the manual count for an estimator reading.

The simulation comparison reports mission time to a usable result averaged over people who received a rate, the unassessed fraction over all people, and time spent observing at failed views averaged over scenes. Unreachable people count as unassessed with zero observation time. Travel is excluded from failed observation time. Read the measured-only time beside the unassessed fraction: skipping hard cases can lower it. In the live app, a completed decision is labeled as a past result with its session timestamp, separate from the live camera evidence.
