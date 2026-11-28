"""A map of a setup file, from above, to check the measurements by eye.

`python -m ostinato.robot --setup room.json --check --map room.png`. A wrong sign or a
swapped x and y is hard to see in a list of numbers and easy to see here.
"""

import math

from PIL import Image, ImageDraw

from .rover import KEEP_OUT, RING, TRAVEL, plan
from .ui import AMBER, BG, BLUE, CARD, DIM, GREEN, INK, LINE, font

SIZE = 900  # px, the map's width
MARGIN = 0.8  # m of room shown beyond everything in the file
GREY = (120, 132, 144)


def bounds(setup):
    """The part of the room to show: (left, bottom, right, top) in meters."""
    xs, ys = [setup.start.x], [setup.start.y]
    for x, y in [*setup.body.ends, *setup.markers.values()]:
        xs.append(x)
        ys.append(y)
    for a in range(0, 360, 10):  # the whole way round, as the route may take it
        x, y = setup.body.around(math.radians(a), TRAVEL)
        xs.append(x)
        ys.append(y)
    for x, y, r in setup.obstacles:
        xs += [x - r, x + r]
        ys += [y - r, y + r]
    return min(xs) - MARGIN, min(ys) - MARGIN, max(xs) + MARGIN, max(ys) + MARGIN


def draw(setup, problems=(), notes=()):
    """The map, as a PIL image. `problems` and `notes` are listed below it."""
    left, bottom, right, top = bounds(setup)
    scale = SIZE / max(right - left, top - bottom)
    width, height = round((right - left) * scale), round((top - bottom) * scale)
    lines = [(f"Problem: {p}", AMBER) for p in problems] + [(f"Note: {n}", DIM) for n in notes]
    image = Image.new("RGB", (width, height + 70 + 26 * len(lines)), BG)
    d = ImageDraw.Draw(image)

    def at(x, y):
        return (x - left) * scale, (top - y) * scale

    def outline(clearance, **style):
        points = [at(*setup.body.around(a, clearance)) for a in [i * math.pi / 36 for i in range(73)]]
        d.line(points, **style)

    # One meter grid, labeled along the bottom and left.
    d.rectangle((0, 0, width, height), fill=CARD)
    for x in range(math.ceil(left), math.floor(right) + 1):
        d.line([at(x, bottom), at(x, top)], fill=LINE)
        d.text((at(x, bottom)[0] + 3, height - 18), f"{x} m", font=font(13), fill=DIM)
    for y in range(math.ceil(bottom), math.floor(top) + 1):
        d.line([at(left, y), at(right, y)], fill=LINE)
        d.text((4, at(left, y)[1] - 16), f"{y} m", font=font(13), fill=DIM)

    for x, y, r in setup.obstacles:
        (x0, y0), (x1, y1) = at(x - r, y + r), at(x + r, y - r)
        d.ellipse((x0, y0, x1, y1), fill=GREY)
    outline(TRAVEL, fill=LINE, width=1)
    outline(KEEP_OUT, fill=AMBER, width=2)

    # The person: a line from head to feet, or a dot when it is not known how they lie.
    (hx, hy), (fx, fy) = setup.body.ends
    d.line([at(hx, hy), at(fx, fy)], fill=(84, 138, 190), width=max(6, round(0.35 * scale)))
    cx, cy = at(*setup.person)
    r = max(5, 0.12 * scale)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(186, 140, 104))
    if setup.head is not None:
        x, y = at(hx + 0.3 * math.cos(setup.head), hy + 0.3 * math.sin(setup.head))
        d.text((x, y), "head", font=font(14), fill=INK, anchor="mm")

    # The planned way round, from the start through each view.
    where, route = setup.start, [at(setup.start.x, setup.start.y)]
    for bearing in setup.views:
        path, _ = plan(where, setup.body, bearing, setup.obstacles)
        route += [at(*p) for p in path]
        where = type(where)(*path[-1], 0.0)
    d.line(route, fill=BLUE, width=2)

    # Each view, with what its camera takes in.
    half = math.radians(setup.camera.fov) / 2
    for i, bearing in enumerate(setup.views, 1):
        x, y = setup.body.around(bearing, RING)
        facing = math.atan2(setup.person[1] - y, setup.person[0] - x)
        for side in (-half, half):
            end = at(x + 2.5 * math.cos(facing + side), y + 2.5 * math.sin(facing + side))
            d.line([at(x, y), end], fill=GREY)
        px, py = at(x, y)
        d.ellipse((px - 12, py - 12, px + 12, py + 12), fill=CARD, outline=BLUE, width=2)
        d.text((px, py), str(i), font=font(14), fill=BLUE, anchor="mm")

    # The start, pointing the way it faces.
    s = setup.start
    tip = at(s.x + 0.3 * math.cos(s.heading), s.y + 0.3 * math.sin(s.heading))
    back = [
        at(s.x + 0.15 * math.cos(s.heading + a), s.y + 0.15 * math.sin(s.heading + a)) for a in (2.4, -2.4)
    ]
    d.polygon([tip, *back], fill=GREEN)
    d.text((tip[0] + 6, tip[1]), "start", font=font(14), fill=GREEN)

    # Markers, with an arrow for the way their printed side faces.
    for n, (x, y) in setup.markers.items():
        mx, my = at(x, y)
        if n in setup.facings:
            a = setup.facings[n]
            d.line([(mx, my), at(x + 0.4 * math.cos(a), y + 0.4 * math.sin(a))], fill=INK, width=2)
        d.rectangle((mx - 10, my - 10, mx + 10, my + 10), fill=INK)
        d.text((mx, my), str(n), font=font(13), fill=BG, anchor="mm")

    key = "From above. Blue: the way round. Orange: 1 m from the person. Grey: furniture."
    d.text((12, height + 12), key, font=font(15), fill=DIM)
    d.text((12, height + 36), "Arrows: the way each marker and the start face.", font=font(15), fill=DIM)
    for i, (text, color) in enumerate(lines):
        d.text((12, height + 64 + 26 * i), text, font=font(15), fill=color)
    return image
