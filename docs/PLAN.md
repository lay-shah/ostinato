# Plan

| Milestone | Done when | Status |
|---|---|---|
| 1. Breathing | Live rate and confidence from a laptop camera; declines to report on poor signal | Built; accuracy on people not measured |
| 2. Decision engine | Assess, reposition, and "unable to assess" logic is tested | Built |
| 3. Simulation | Robot chooses viewpoints and is compared with nearest-first search | Built; needs a real urgency cue |
| 4. Medic view | Ranked map shows confidence and reasons | Built |
| 5. Validation | Results against hand-counted breaths at two distances are reported | Tools built; no trials yet |
| 6. Rover | Drives to a person, stops, and measures | [Planned](ROBOT.md); drive control simulated only |

## Measures

- Breathing-rate mean absolute error
- How often the system correctly declines to judge
- Time to reach high-priority simulated casualties versus nearest-first

## Rules

- Missing motion is never evidence of death.
- Process video on the device and do not save it by default.
- Use healthy, consenting volunteers. Ask about IRB review before collecting data for research or publication.
- Label simulated results as simulated.
