# Robot build plan

A new rover that drives to one person, **stops**, and measures from a camera on a mast. Nothing here has been built, and none of the software has run on hardware. Prices are rough, from November 2026 listings or estimates; check them before buying.

## Build in this order

| Step | Done when | Tool | Cost |
|---|---|---|---|
| 1. Camera position | Counted-breath trials on people lying down, at several heights and distances, show where a rate is given and how far off it is | `python -m ostinato.trial` | $0 |
| 2. Computer | The speed test's rover line gives 10 frames/s or more, with a photo of a person | `python -m ostinato.speed --image photo.jpg` | borrow first |
| 3. Chassis | Wheels far enough apart not to tip, with the real part weights | `python -m ostinato.tipping --part …` | $90-200 |
| 4. Mast | The camera is steady within 2 s of every push | `python -m ostinato.shake` | $30-50 |
| 5. Wiring | Every check passes with the wheels off the ground | [WIRING.md](WIRING.md), `python -m ostinato.bench` | $140-220 |
| 6. Driven by hand | Stops 1 m away, measures, moves to a second view, with the remote | `python -m ostinato.remote` | $0 |
| 7. Autonomy | Goes between marked viewpoints by itself and stops for obstacles | `python -m ostinato.robot` | $0-50 |

Do not buy the rover before step 1 works.

## Where to put the camera

`python -m ostinato.placement` calculates which side views are worth trying, assuming a 5 mm chest rise and a 70 degree camera.

| Lens height | Distance along the ground | Angle | Chest motion |
|---|---|---:|---:|
| 0.6 m | 0.8-1.25 m | 18-27 degrees | 1.7-2.3 px |
| **0.8 m** | **0.8-1.5 m** | **22-37 degrees** | **1.3-1.8 px** |
| 1.0 m | 1.0-1.5 m | 28-39 degrees | 1.2-1.4 px |

Start with a 0.8 m mast. It leaves the widest range of distances, and a 1 m gap keeps the rover clear of the person. Lower, the shoulders overlap. Higher, the chest mostly moves toward the camera and a raised hand may be missed. Keep the camera level from side to side: only motion up and down the image is measured.

These are calculations. Real chest motion varies by person, clothing, and position, and can be smaller than 5 mm.

## Parts

| Part | Requirement | Rough cost |
|---|---|---:|
| Computer | 10 frames/s on the speed test's rover line; Raspberry Pi 5 8 GB is a candidate, not yet tested | $175-200 |
| Chassis | Four wheels, encoders, about 3 kg payload; about 27 cm wide and long for a 0.8 m mast | $90-200 |
| Motor board, driver, distance sensors | Arduino-compatible board, PWM and direction driver, two HC-SR04 | $30-50 |
| Camera | 70 degrees across, 720p, fixed focus | $25-40 |
| Mast and pan-tilt | Stiff tube; 3D-printed mounts | $30-50 |
| Speaker | Heard at 2 m | $10-15 |
| Power | Battery with protection, fuses, separate supplies for motors and computer, two-pole emergency stop | $70-110 |
| Storage, cooling, wiring | | $40-60 |

About $470-725 in total. The 27 cm comes from `python -m ostinato.tipping` with guessed masses: a sudden stop going down a 10 degree slope decides it. Weigh the parts and run it again before choosing a chassis, and keep the battery low.

## Safety rules

- A person holds the emergency stop whenever the motors have power.
- Slower than 0.3 m/s near people. Stop at least 1 m away from every part of them, head and feet included. Never drive over or behind someone.
- Motors stop if the computer, remote, or a sensor goes quiet for 0.5 s.
- Never measure while moving. After stopping, it waits until the picture itself has been still for 2 s, not just the wheels. If it never steadies within 10 s, it tries the next view. A blank wall cannot be followed, so there it only waits 2 s.
- Check that it does not tip on a 10 degree slope or when stopping suddenly, with the mast on.
- Use healthy adults who agree to take part. This is not a medical device.

## How it stays safe

Each layer stops the wheels on its own, so one failing does not move the rover.

1. **Emergency stop switch.** One pole cuts motor power directly. The other tells the board, wired so that a cut wire reads as pressed. See [WIRING.md](WIRING.md).
2. **Motor board** (`firmware/ostinato_board`). Stops after 0.5 s without a good command, ignores damaged or repeated commands, waits for a stop command after any halt, cuts off a wheel that is pushed but does not turn for a second, and stops for good, until it is restarted, when the battery stays below its own cutoff for a second, even with the computer gone. Wheel speeds above 450 mm/s are refused.
3. **Guard** (`ostinato.rover`). Every command passes through it. It limits speed and acceleration, keeps 1 m from the person (from the whole body when the setup file says which way they lie), stops for obstacles, and stops when the remote, a sensor, the position estimate, or its own loop goes quiet for 0.5 s. Turning on the spot is the only movement allowed without knowing how far away the person is. An emergency stop holds until the operator clears it.
4. **Position** (`ostinato.localize`). When it is unsure where it is, it says nothing to the guard, so the rover stops. It may first turn once around in place to find a marker.
5. **Person check.** With the camera's height in the setup file, it stops if the person is more than 0.25 m from where the file says, or was not seen in place before it leaves a viewpoint. With the head direction too, it stops if their shoulders and hips show them lying more than 45 degrees from it, head and feet swapped included.
6. **Battery.** With `battery_low` in the setup file, the rover holds its wheels and asks for a person once the battery has stayed below it for about 2 s, so a short dip when starting off passes.
7. **Remote** (`ostinato.remote`). Signed heartbeats, 10 a second. Old ones played back are ignored.

## Software

| Command | What it does |
|---|---|
| `python -m ostinato.localize --print 1 2 3 4` | Markers to print and tape upright around the room |
| `python -m ostinato.marker_range`, also `--blur 2 --light 0.3 --noise 10` | How far a drawn marker is read, sharp or blurred, lit or dim, and how far off |
| `python -m ostinato.calibrate --print`, then `--square 25` | Measures the camera's lens with a printed board |
| `python -m ostinato.bench --port …` | First power-on checks, wheels off the ground; `--measure` finds counts per meter |
| `python -m ostinato.robot --setup room.json --check` | Checks a setup file: start too near the person, a viewpoint with no marker in view or one turned away, a marker on the path |
| `python -m ostinato.remote --make-key key.txt` | Makes the key shared by the remote and the rover |
| `python -m ostinato.robot --setup room.json --port … --key key.txt` | Runs the rover. It starts stopped. Before leaving each view it asks the person aloud to raise a hand, unless `--no-prompt` |
| `python -m ostinato.remote --robot <address> --key key.txt` | The operator's window: G lets it move, Space stops it |
| `… --log data/run.jsonl`, then `--replay data/run.jsonl` | Saves and summarizes a run, without pictures |
| `python -m ostinato.rover_demo` | The whole thing on a drawn person and room; also `launch/Rover Simulation.command` |

The setup file holds a person's measurements: where the rover starts, where the person and markers are, the viewpoints, and the camera. Start from [room.example.json](room.example.json), which passes the check, and replace every number with one measured in the room. The format is explained at the top of `src/ostinato/robot.py`. Run logs hold the breathing result, so keep them in `data/`, which is never committed. Without `--key`, the rover takes G and Space from its own window, which is only for testing on a stand.

## Knowing where it is

It combines the wheel encoders with the printed markers, which the measuring camera also sees, and keeps track of how sure it is. It looks for markers in every picture while it may move, and twice a second while stopped. Stopped, nothing changes but the noise, and the time is better spent measuring. Turn each marker to face the viewpoints that use it, and put the direction it faces in the setup file as `[x, y, facing]`: turned 50 degrees or more away, a blurred camera reads it reliably only to about 3 m. With facings, the setup check counts a marker turned more than 45 degrees from a view only within 3 m, and one turned more than 65 degrees not at all. A person still has to place it at the start and measure where the casualty and the markers are: it does not find people by itself.

## Tested so far, in simulation only

Simulated wheels, sightings, and boards are kinder than real ones.

- Against the compiled board code, 300 of 300 drives reached the first viewpoint, including with 30% of messages lost or damaged, and none came nearer than 1.2 m. Hiding markers beyond 4 m changed nothing in 120 more drives.
- With a box on the short way round to a viewpoint, it stopped for it, turned back, and reached the view the other way round, in 5 of 5 drives. It tries the other way once per view, then moves on.
- With no marker in sight, it turned once and stopped. After a cut cable, an emergency stop, or a board restart, the wheels never turned again.
- The person check caught every shift of 0.28 m or more, with no false alarms in 30 drives. Real pose estimates are rougher, so expect false stops.
- Drawn markers, 16 cm, read by a 720p 70 degree camera: facing it, reliably to 6 m when sharp and 5 m when blurred (2 px), with distances off by 1-3% and directions by about 0.001 rad. That is inside what the position estimate assumes (4% and 0.02 rad) to about 4 m. Turned 50 degrees away and blurred, only to about 3 m, and blurred in a dim room, 2 m. Without facings in the setup file, the check counts every marker within 4.9 m as seen, whichever way it faces.
- With the picture shaking for 4 s after it stopped, the rover held still and began measuring 2 s after the shaking ended. The steadiness check settled on a still, grainy picture and waited out a simulated 10 px wobble.
- Around a person lying down, with views on both sides: told which way the head is, it came no nearer than 1.24 m to any part of them in 20 drives. Told only where the chest is, it passed 0.15 m from the feet. Give the head direction.
- Seeing shoulders and hips through the camera, the rover stopped in 10 of 10 drives when the person lay 55 degrees or more from the setup file, head and feet swapped included, and in none up to 35 degrees. The simulated pose points are cleaner than a real pose model's.
- The board program, on simulated motors 15% weaker than it assumes, held speed within 5% and stopped within about 0.5 s of silence.

## Still unknown

- Breathing accuracy on people, seated or lying down
- Whether blankets, thick clothing, or injuries hide chest motion
- Frame rate on the robot's computer. On an Apple M4 laptop, finding markers nearly doubles the time for each picture, so a computer that runs the desktop app well may still be too slow for the rover
- How long the mast shakes after stopping
- How well markers are read, and the person is found, by a real camera in a real room
