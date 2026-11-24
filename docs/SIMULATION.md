# Simulated multi-casualty search

All results here are simulated. They test planning logic, not sensors, rescue performance, or clinical value.

```bash
ostinato-sim compare                                   # 100 scenes, about 80 seconds
ostinato-sim scene 1001 --image map.png                # one scene, with a map
ostinato-sim scene 1001 --policy nearest-first
```

A robot moving at 1 m/s searches a 20 x 20 m map with walls and 4 to 7 people. It keeps clear of walls and casualties, and uses the real decision engine to wait, move, report, or ask for help. The planner sees the map, locations, an estimated chest direction, and a noisy urgency cue. It cannot see who is truly urgent or which views will fail.

Both policies only consider views the map shows as unobstructed. "Nearest" picks the closest person or view. "Ranked" trades the cue against travel time; "predicted" trades the estimated chest direction against it.

This complements the viewpoint trials on the `feature/viewpoint-trials` branch, which study two views of one person and real paired webcam measurements.

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

## Limits

People's locations are assumed known. View success and the urgency cue follow rules we made up, and the planner's chest-direction rule resembles the rule that generates outcomes, which favors it. Readings are always correct when given, and signals never drop out. People do not block views, and walls are fully mapped. The attention order is illustrative, not a triage method. No real urgency cue exists yet, so these gains cannot be claimed for a real robot.
