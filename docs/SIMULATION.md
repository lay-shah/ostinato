# Simulated multi-casualty search

All results here are simulated. They test planning logic, not sensors, rescue performance, or clinical value.

```bash
ostinato-sim compare                                   # 100 scenes, about 80 seconds
ostinato-sim scene 1001 --image map.png                # one scene, with a map
ostinato-sim scene 1001 --policy nearest-first
```

A robot moving at 1 m/s searches a 20 x 20 m map with walls and 4 to 7 people. It keeps clear of walls and casualties, and uses the real decision engine to wait, move, report, or ask for help. The planner sees the map, locations, an estimated chest direction, and a noisy urgency cue. It cannot see who is truly urgent or which views will fail.

Both policies only consider views the map shows as unobstructed. "Nearest" picks the closest person or view. "Ranked" trades the cue against travel time; "predicted" trades the estimated chest direction against it.

This complements the [paired viewpoint trials](VIEWPOINT_TRIALS.md), which study two views of one person and real paired webcam measurements.

## Result, seeds 0-99

The main measure is mean time to reach urgent people, averaged within each of the 94 scenes that have any. These seeds were run once on the final planner; tests use other seeds.

| Cue | Person / view choice | Time to reach | Change vs. nearest-first (95% CI) | Faster / same / slower scenes | People measured |
|---|---|---:|---:|---:|---:|
| any | nearest / nearest (nearest-first) | 109.0 s | | | 66.5% |
| any | nearest / predicted | 105.5 s | -3.5 s (-10.7 to 3.6) | 54 / 4 / 36 | 80.9% |
| useful | ranked / predicted | 51.2 s | -57.8 s (-69.6 to -46.4) | 78 / 4 / 12 | 80.9% |
| weak | ranked / predicted | 83.6 s | -25.4 s (-37.4 to -14.0) | 62 / 3 / 29 | 80.9% |
| useless | ranked / predicted | 113.2 s | 4.2 s (-9.1 to 17.2) | 45 / 1 / 48 | 80.9% |

- Ranking people accounts for the faster reach, and only if the urgency cue carries information. With a useless cue there is no gain.
- Predicting views raises the share measured, but shows no clear effect on reach time.
- With a useful cue, the whole search took longer: 252 s versus 238 s.

## What if the assumptions are wrong?

`ostinato-sim sweep` changes one assumption at a time (60 scenes, seeds 2000-2059, weak urgency cue). Bold values are the defaults. The planner is not told that the world changed.

| Assumption | Value | Measured: nearest-first | Measured: planner | Change in time to reach urgent people (95% CI) | Change in total time |
|---|---:|---:|---:|---:|---:|
| Chest-direction error (radians) | 0 | 66.8% | 81.3% | -29.7 s (-45.8 to -14.7) | +8.2 s |
| Chest-direction error (radians) | **0.35** | 66.8% | 80.7% | -29.3 s (-45.7 to -14.4) | +8.6 s |
| Chest-direction error (radians) | 0.8 | 66.8% | 78.5% | -27.9 s (-44.1 to -13.0) | +14.5 s |
| Chest-direction error (radians) | 1.6 | 66.8% | 71.3% | -24.8 s (-41.3 to -9.5) | +21.2 s |
| Chest-direction error (radians) | 3 | 66.8% | 67.1% | -23.6 s (-39.8 to -8.5) | +23.3 s |
| Chance a view is blocked | 0 | 70.4% | 82.5% | -31.6 s (-49.3 to -15.5) | +6.9 s |
| Chance a view is blocked | **0.2** | 66.8% | 80.7% | -29.3 s (-45.7 to -14.4) | +8.6 s |
| Chance a view is blocked | 0.5 | 49.5% | 62.5% | -23.0 s (-38.4 to -9.0) | +16.8 s |
| Chance a view is blocked | 0.8 | 21.5% | 30.2% | -16.3 s (-29.9 to -3.1) | +29.1 s |
| Chance motion is too weak | 0 | 76.7% | 92.4% | -33.8 s (-47.3 to -21.1) | -7.2 s |
| Chance motion is too weak | **0.1** | 66.8% | 80.7% | -29.3 s (-45.7 to -14.4) | +8.6 s |
| Chance motion is too weak | 0.3 | 52.6% | 65.3% | -26.9 s (-46.4 to -10.1) | +29.1 s |
| Chance motion is too weak | 0.6 | 32.0% | 38.1% | -14.9 s (-37.6 to +5.0) | +70.6 s |
| Usable angle from face-on (degrees) | 40 | 43.8% | 66.2% | -25.5 s (-40.0 to -12.2) | +14.6 s |
| Usable angle from face-on (degrees) | **75** | 66.8% | 80.7% | -29.3 s (-45.7 to -14.4) | +8.6 s |
| Usable angle from face-on (degrees) | 110 | 80.7% | 85.2% | -31.5 s (-49.2 to -15.1) | +8.0 s |

- Choosing views by chest direction helps only if that direction can be estimated. At 3 radians of error it is essentially random, and the measured share matches nearest-first (67.1% vs. 66.8%).
- The gain in reaching urgent people shrinks as views fail or motion is too weak, and is not clear when 60% of people cannot be measured.
- The full search takes longer in every world except when everyone is measurable.
- Reaching urgent people sooner still depends on an urgency cue that does not exist yet.

## Limits

People's locations are assumed known. View success and the urgency cue follow rules we made up, and the planner's chest-direction rule resembles the rule that generates outcomes, which favors it. Readings are always correct when given, and signals never drop out. People do not block views, and walls are fully mapped. The attention order is illustrative, not a triage method. No real urgency cue exists yet, so these gains cannot be claimed for a real robot.
