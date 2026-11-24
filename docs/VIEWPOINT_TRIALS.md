# Can a second viewpoint recover a usable reading?

Use this prototype to study camera measurements and viewpoint selection. A declined reading means the available video was insufficient. The signal score describes the motion signal. It has no established interpretation as a probability of breathing or clinical urgency.

## Run the simulation

From the project root, run:

```sh
PYTHONPATH=src python -m ostinato.viewpoints simulate
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

Install the existing project dependencies using the repository README. Use a consenting healthy teammate and a separate observer. Mark two positions with measured distances. Keep the person, lighting and camera settings consistent. For repeated pairs, alternate the starting position and include a partially obscured condition. Follow the data-collection rules in [the project plan](PLAN.md).

```sh
mkdir -p data  # ignored by git
python -m ostinato.viewpoints trial data/pair-01.json --views front-1m angled-2m
python -m ostinato.viewpoints trial data/pair-02.json --views angled-2m front-1m
python -m ostinato.viewpoints report data/pair-01.json data/pair-02.json
```

An observer counts breaths between each START and STOP and enters the count before seeing the estimate. Each view starts with a fresh tracker and has its own reference count. Use anonymous view labels. The command saves timestamps as elapsed durations, counts, estimated rates, signal scores and decline reasons. It never writes images or video. A completed pair needs two uninterrupted windows. Keep trial files local unless participants agreed to sharing.

## Read the report

Record reported windows divided by all windows (coverage), mean absolute error over reported windows, the number of first-view declines and how often a second view returned a rate. Inspect recovered readings against their own manual references before calling them accurate. A hand count is a practical reference with its own uncertainty: one extra count over 30 seconds changes the reference by 2 breaths/min.

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
