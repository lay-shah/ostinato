# Plan

| Milestone | Done when |
|---|---|
| 1. Breathing | Live rate and confidence from a laptop camera; declines to report on poor signal |
| 2. Decision engine | Assess, reposition, and "unable to assess" logic is tested |
| 3. Simulation | Robot chooses viewpoints and is compared with nearest-first search |
| 4. Medic view | Ranked map shows confidence and reasons |
| 5. Validation | Results against hand-counted breaths at two distances are reported |

## Measures

- Breathing-rate mean absolute error
- How often the system correctly declines to judge
- Time to reach high-priority simulated casualties versus nearest-first

## Rules

- Missing motion is never evidence of death.
- Process video on the device and do not save it by default.
- Use healthy, consenting volunteers. Ask about IRB review before collecting data for research or publication.
- Label simulated results as simulated.
