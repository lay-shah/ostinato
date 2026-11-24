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
python -m ostinato.viewpoints trial pair-01.json --views front-1m angled-2m
python -m ostinato.viewpoints trial pair-02.json --views angled-2m front-1m
python -m ostinato.viewpoints report pair-01.json pair-02.json
```

An observer counts breaths between each START and STOP and enters the count before seeing the estimate. Each view starts with a fresh tracker and has its own reference count. Use anonymous view labels. The command saves timestamps as elapsed durations, counts, estimated rates, signal scores and decline reasons. It never writes images or video. A completed pair needs two uninterrupted windows. Keep trial files local unless participants agreed to sharing.

## Read the report

Record reported windows divided by all windows (coverage), mean absolute error over reported windows, the number of first-view declines and how often a second view returned a rate. Inspect recovered readings against their own manual references before calling them accurate. A hand count is a practical reference with its own uncertainty: one extra count over 30 seconds changes the reference by 2 breaths/min.

Use the results to decide the next build. If changing position repeatedly produces a usable and accurate estimate, collect more varied scenes and test viewpoint selection on held-out trials. If it does not, inspect tracking and signal quality before building a mobile platform. Revisit the decision with a responder using a past incident and ask which observation would have changed their next action.
