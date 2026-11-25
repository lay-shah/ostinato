"""Whether a rover with a mast would tip: `python -m ostinato.tipping`.

A calculation for choosing a chassis, for a stiff body on hard ground. The masses are
guesses: weigh the parts and pass your own. It does not replace a tip test.
"""

import argparse
import math
from dataclasses import dataclass

G = 9.81
SLOPE = 10.0  # degrees, from the safety rules
SPEED = 0.3  # m/s
MARGIN = 1.5  # the center of mass must lean this much further than the worst case

# (name, kg, height in m)
PARTS = (
    ("chassis, motors, wheels", 1.6, 0.06),
    ("battery", 0.5, 0.08),
    ("computer, drivers, speaker", 0.5, 0.15),
    ("mast", 0.4, 0.45),
    ("camera and pan-tilt", 0.25, 0.80),
)


@dataclass(frozen=True)
class Body:
    mass: float
    height: float  # of the center of mass
    inertia: float  # about the center of mass, for point masses

    @classmethod
    def of(cls, parts):
        if not parts or not all(
            math.isfinite(m) and m > 0 and math.isfinite(h) and h >= 0 for _, m, h in parts
        ):
            raise ValueError("every part needs a mass above 0 and a height of 0 or more")
        mass = sum(m for _, m, _ in parts)
        height = sum(m * h for _, m, h in parts) / mass
        if height <= 0:
            raise ValueError("the center of mass is at the ground")
        return cls(mass, height, sum(m * (h - height) ** 2 for _, m, h in parts))

    def slope(self, base):
        """The steepest slope in degrees it can stand on. `base` is wheel to wheel."""
        return math.degrees(math.atan2(base / 2, self.height))

    def braking(self, base, slope=0.0):
        """The hardest stop in m/s^2, facing down a slope in degrees, before it tips."""
        a = math.radians(slope)
        return G * (base / 2 * math.cos(a) - self.height * math.sin(a)) / self.height

    def curb(self, base):
        """The speed at which wheels that stop dead tip it over."""
        half = base / 2
        arm = math.hypot(half, self.height)
        # Angular momentum about the wheel is kept; then it must lift to the tipping point.
        inertia = self.inertia + self.mass * arm**2
        spin = math.sqrt(2 * self.mass * G * (arm - self.height) / inertia)
        return spin * inertia / (self.mass * self.height)


def smallest_base(body, slope=SLOPE, speed=SPEED, margin=MARGIN, stop=0.1):
    """The smallest wheel-to-wheel distance, and what decides it."""
    if not (0 <= slope < 45 and speed > 0 and margin >= 1 and stop > 0):
        raise ValueError("slope must be 0 to 45 degrees, with speed and stop above 0 and margin at least 1")
    brake = speed / stop  # m/s^2, stopping this fast
    needs = []
    a = math.radians(slope)
    needs.append((2 * margin * body.height * math.tan(a), f"standing on a {slope:g} degree slope"))
    lean = body.height * (math.sin(a) + brake / G) / math.cos(a)
    needs.append((2 * margin * lean, f"stopping in {stop:g} s down a {slope:g} degree slope"))
    low, high = 0.0, 5.0
    for _ in range(60):
        middle = (low + high) / 2
        low, high = (low, middle) if body.curb(middle) >= margin * speed else (middle, high)
    needs.append((high, f"wheels hitting a curb at {speed:g} m/s"))
    return max(needs)


def part(text):
    try:
        name, mass, height = text.rsplit(",", 2)
        return name, float(mass), float(height)
    except ValueError:
        raise argparse.ArgumentTypeError("use name,kg,height in m") from None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--part", type=part, action="append", metavar="NAME,KG,HEIGHT", help="replaces the guesses"
    )
    parser.add_argument("--slope", type=float, default=SLOPE)
    parser.add_argument("--speed", type=float, default=SPEED)
    parser.add_argument(
        "--stop", type=float, default=0.1, help="seconds to stop from full speed (default: 0.1)"
    )
    args = parser.parse_args(argv)
    try:
        body = Body.of(args.part or PARTS)
        base, why = smallest_base(body, args.slope, args.speed, stop=args.stop)
    except ValueError as error:
        parser.error(str(error))
    if not args.part:
        print("These masses are guesses. Weigh the parts and use --part.")
    print(f"{body.mass:.2f} kg, center of mass {body.height:.2f} m up.")
    print(f"The wheels should be at least {100 * base:.0f} cm apart, front to back and side to side.")
    print(f"That is decided by {why}, with a margin of {MARGIN:g}.")
    for wide in (0.15, 0.20, 0.25, 0.30):
        print(
            f"  {100 * wide:.0f} cm: tips at {body.slope(wide):.0f} degrees, "
            f"or stopping harder than {max(0.0, body.braking(wide, args.slope)):.1f} m/s^2 down the slope, "
            f"or hitting a curb at {body.curb(wide):.2f} m/s"
        )


if __name__ == "__main__":
    main()
