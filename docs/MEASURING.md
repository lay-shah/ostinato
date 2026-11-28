# Measuring a room for the rover

The rover drives on a setup file that a person measures with a tape: where the person is and which way they lie, where the markers are and which way they face, the furniture, where the rover starts, and the viewpoints. This page says how to measure those so the signs come out right. Start from [room.example.json](room.example.json) and change its numbers.

## The two directions everything uses

Work from above, as on a map.

- **The middle of the person's chest is (0, 0).** Every place is measured from there, in meters.
- **Choose +x.** Any direction along the floor will do: toward a wall or the door is easiest. Lay a strip of tape from the chest that way.
- **+y is a quarter turn to the left of +x,** seen from above. Standing at the chest and facing +x, +y is on your left.
- **Angles are in degrees, from +x, turning toward +y.** 0 is +x, 90 is +y, 180 is −x, and 270 (or −90) is −y.

Getting +y on the wrong side mirrors the whole room: the check may still pass, but the rover would drive on the wrong side. The map below catches it.

## What to measure

| In the file | What it is | How |
|---|---|---|
| `"person": [0, 0, head]` | The chest, and the direction from the chest to the head | Lying along +y with the head that way: 90. To the nearest 15 degrees is plenty |
| `"markers": {"1": [x, y, facing]}` | Where each printed marker is, and the way its printed side faces | Measure along x, then along y. A marker on the wall at −x facing back into the room faces 0 |
| `"start": [x, y, heading]` | Where the rover is put down, and the way its front faces | Facing the person from the +x side: 180 |
| `"views": [30, 150]` | Directions from the chest to measure from, in order | Beside the chest, not past the head or feet: about 60 to 120 degrees from the head direction, either side |
| `"obstacles": [[x, y, r]]` | Furniture, as circles big enough to cover it | Middle and radius. Things not listed are found by the rover's distance sensors |
| `"camera": {"height": 0.8}` | The lens's height above the floor | Needed for the rover to check the person is where the file says |

Markers face the views that use them: turned more than 45 degrees away, a marker is only read close up, and more than 65 degrees, not at all.

## Check it against the room

1. `python -m ostinato.robot --setup room.json --check --map room.png` lists which markers each place reads, any problems, and draws the room. Hold the map beside the room: the person, the markers, and the furniture should be where they are, the head at the right end, and the rover's start on the right side.
2. `python -m ostinato.robot --setup room.json --rehearse all`, or double-click `launch/Rover Rehearsal.command`, then stand a webcam at the start and each view in turn. It says where the markers put the camera. A few centimeters is fine; more than 0.15 m means a measurement, or where the camera stands, is off.

The rover refuses to drive on a file the check finds problems in.
