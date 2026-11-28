# Wiring

How to connect the motor board program in `firmware/ostinato_board`. Nothing has been wired yet. The pins match the program; if you change one, change both, and a test checks they agree. Check every part's own datasheet first: these are typical values, not your parts'.

## Power

```
Battery ── fuse ── emergency stop (pole 1, normally closed) ── motor driver power
   │
   └── fuse ── 5 V regulator ── computer and motor board
```

- **The emergency stop cuts motor power directly.** Use a latching mushroom switch with two normally closed poles. Pole 1 carries the motor current, so it must be rated for the motors' stall current. If it is not, pole 1 drives a relay or contactor that is.
- **Pole 2 tells the board.** Wire it from pin A0 to ground. The board sees a pressed switch, a cut wire, or an unplugged plug the same way, as pressed, and stops.
- Keep motor and logic supplies separate, joined only at one ground point. Motor current on the logic ground makes the board reset or read encoders wrongly.
- Fuse each branch, close to the battery.

## Motor board pins

| Pin | Goes to | Notes |
|---:|---|---|
| 5 | Left motor PWM | |
| 4 | Left motor direction | |
| 6 | Right motor PWM | |
| 7 | Right motor direction | |
| 2 | Left encoder A | must take interrupts |
| 8 | Left encoder B | |
| 3 | Right encoder A | must take interrupts |
| 9 | Right encoder B | |
| 10 | Front distance sensor trigger | |
| 11 | Front distance sensor echo | see below for 3.3 V boards |
| 12 | Rear distance sensor trigger | |
| 13 | Rear distance sensor echo | see below for 3.3 V boards |
| A0 | Emergency stop pole 2, to ground | uses the internal pull-up |
| A1 | Battery, through a divider | see below |
| GND | Logic ground | one point only |
| USB | Computer | commands and reports, 115200 baud |

With four wheels, wire the two motors on each side together as one, and put the encoder on one of them.

The HC-SR04 sends a 5 V echo. On a board with 3.3 V pins, put a divider on each echo line, 1 kΩ from the sensor and 2 kΩ to ground, or the pin may be damaged.

The battery goes to A1 through a divider: 10 kΩ from the battery's positive side after its fuse, and 3.3 kΩ from A1 to ground. On a 5 V board that reads batteries up to 20 V. On a 3.3 V board, or above 20 V, choose resistors that keep A1 below the board's own voltage at full charge, and change `BATTERY_MV_PER_COUNT` to match. Put `battery_low` in the setup file so the rover stops before the battery is flat. The board also has its own cutoff, `BATTERY_CUTOFF_MV`, set for a three-cell lithium battery: below it for a second, it stops the motors until it is restarted, so the battery is not run flat even if the computer has stopped. Set it for your battery, a little below `battery_low`.

## Before the first drive

1. Check every connection with the motor power off.
2. With the emergency stop pressed, power up. The wheels must not move.
3. Put the rover on a stand with its wheels off the ground, and run `python -m ostinato.bench --port …`. Fix everything it reports. Check the battery voltage it prints against a meter.
4. Run `python -m ostinato.bench --port … --measure` on the floor to find counts per meter. Put that in the setup file, and set `MM_PER_TICK` in the board program to 1000 divided by it.
