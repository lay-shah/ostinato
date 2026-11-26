# Robot build plan

A new rover that drives to one person, **stops**, and measures from a camera on a mast. Nothing here has been built. Prices are rough, from November 2026 listings or estimates; check them before buying.

## Build in this order

| Step | Done when | Cost |
|---|---|---|
| 1. Camera position | Counted-breath trials on people lying down, at several heights and distances, show where a rate is given and how far off it is | $0 |
| 2. Computer | `python -m ostinato.speed --image photo.jpg` gives 10 frames/s or more on it | borrow first |
| 3. Mast | `python -m ostinato.shake` shows the camera is steady within 2 s of every push | $30-50 |
| 4. Remote-controlled rover | Drives, stops 1 m away, measures, moves to a second view | $200-400 |
| 5. Autonomy | Goes between marked viewpoints and stops for obstacles | $0-50 |

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
| Computer | 10 frames/s in the speed test; Raspberry Pi 5 8 GB is a candidate, not yet tested | $175-200 |
| Chassis | Four wheels, encoders, about 3 kg payload; wide enough that a 0.8 m mast does not tip | $90-200 |
| Motor driver, distance sensors | Stops on loss of signal | $30-50 |
| Camera | 70 degrees across, 720p, fixed focus | $25-40 |
| Mast and pan-tilt | Stiff tube; 3D-printed mounts | $30-50 |
| Speaker | Heard at 2 m | $10-15 |
| Power | Battery with protection, fuse, separate supplies for motors and computer, emergency stop | $70-110 |
| Storage, cooling, wiring | | $40-60 |

About $470-725 in total.

`python -m ostinato.tipping` calculates how far apart the wheels must be. With guessed part masses and a 0.8 m mast, about 27 cm both ways, decided by a sudden stop going down a 10 degree slope. Weigh the parts and run it again with `--part` before choosing a chassis. Keep the battery low.

## Safety rules

- A person holds the emergency stop whenever the motors have power.
- Slower than 0.3 m/s near people. Stop at least 1 m away. Never drive over or behind someone.
- Motors stop if the computer, radio, or a sensor goes quiet for 0.5 s.
- Never measure while moving.
- Check that it does not tip on a 10 degree slope or when stopping suddenly, with the mast on.
- Use healthy adults who agree to take part. This is not a medical device.

## Drive software

`firmware/motor_guard.h` is the last check, for the motor board, and `ostinato.motors` is the computer's side of the cable. The board stops when it hears no good command for 0.5 s, ignores damaged or repeated commands, and after any stop waits for a stop command before moving again. It is tested on a computer. It still needs pin, motor, encoder, and sensor code for the board you choose.

`python -m ostinato.rover` runs the drive controller on a simulated base. Every command passes through `Guard`, which limits speed, keeps 1 m from the person, stops for obstacles, and stops when the link, a sensor, or the control loop goes quiet for 0.5 s. An emergency stop holds until cleared. The mission drives around the person on a circle, stops, waits 2 s, and only then measures.

`python -m ostinato.rover_demo`, or `launch/Rover Simulation.command` on a Mac, shows a run in a window with the real tracker and decisions on a drawn person.

It has never run on a robot, and it does not replace a switch that cuts motor power.

## Knowing where it is

`ostinato.localize` combines the wheel encoders with printed markers that the same camera sees. Print them with `python -m ostinato.localize --print 1 2 3 4`, tape them upright, and measure where each one and the person are. Measure the camera's lens once with `python -m ostinato.calibrate`, which uses a printed board; without that, distances to markers rely on the lens angle on the box. It tracks its own uncertainty. When that passes 0.15 m, or the markers contradict it, the guard hears nothing more and the rover stops.

In simulation it reached its viewpoint in 99 of 100 drives and never came within 1.1 m of the person. With every marker hidden it stopped in 100 of 100. Simulated wheels and sightings are kinder than real ones. A person still has to place it at the start and measure where the casualty is: it does not find people by itself.

## Running it

`python -m ostinato.robot --setup room.json --port /dev/ttyACM0` runs the whole thing: camera, markers, wheels, guard, and the breathing check at each viewpoint. It needs `pip install -e ".[robot]"`. The setup file holds measurements a person takes; its format is at the top of `src/ostinato/robot.py`. Check it first with `python -m ostinato.robot --setup room.json --check`. That flags a start too near the person, a viewpoint that would see no marker, and a marker on the rover's path. It starts stopped. Drive it from a laptop with `python -m ostinato.remote --robot <address> --key key.txt` and `--key key.txt` on the rover; make the key once with `python -m ostinato.remote --make-key key.txt`. G on the remote lets it move, and Space stops it. If the remote's heartbeats stop for 0.5 s, because the window closed, the laptop slept, or Wi-Fi dropped, the rover stops. Messages are signed, and old ones played back are ignored. Without `--key` it takes G and Space from its own window, which is only for testing on a stand. `--log run.jsonl` saves where it thought it was and why it did what it did, twice a second, without pictures. `--replay run.jsonl` summarizes a saved run. Logs hold the breathing result, so keep them in `data/`, which is never committed.

With the camera's height in the setup file, the camera checks at each viewpoint that the person is where the file says, within 0.25 m. If not, or if it never saw them there, the rover stops for a person to check: someone who moved may be beside it, out of the picture. In simulation it caught every shift of 0.28 m or more, with no false alarms in 30 drives. Real pose estimates are rougher, so expect false stops.

When it is unsure where it is, it turns on the spot, at most once around, to find a marker. Turning in place is the only movement allowed without knowing how far away the person is.

Against the compiled board code with simulated wheels and markers, it reached its first viewpoint in 300 of 300 drives, including with 30% of messages lost or damaged, and was never nearer than 1.2 m. With no marker in sight it turned once and stopped. After a cut cable, an emergency stop, or a board restart, the wheels never turned again. It has not run on hardware.

## Still unknown

- Breathing accuracy on people, seated or lying down
- Whether blankets, thick clothing, or injuries hide chest motion
- Frame rate on the robot's computer
- How long the mast shakes after stopping
