"""The live screen. Draws only; makes no decisions."""

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .breathing import MIN_SECONDS
from .decision import Action

W, H = 1080, 660
VIDEO = (24, 120, 664, 600)  # x0, y0, x1, y1
PANEL = (688, 1056)

BG, CARD, LINE = (16, 26, 36), (22, 35, 48), (56, 76, 94)
INK, DIM = (236, 242, 248), (179, 196, 212)
GREEN, AMBER, BLUE = (116, 211, 150), (250, 178, 84), (128, 188, 250)

HINTS = {
    "chest not visible": "Sit back: show your head and chest",
    "not enough chest detail": "Try patterned clothing or more light",
    "cannot check camera motion": "Show some background beside the person",
    "camera moving": "Keep the camera still",
    "too much movement": "Ask the person to stay still",
    "signal too weak": "Move the camera closer",
    "signal not rhythmic enough": "Hold still and keep the view clear",
    "rate not stable": "Hold still a little longer",
    "tracking interrupted": "Keep the chest in view",
    "stream interrupted": "Waiting for steady video",
    "methods disagree": "Hold still a little longer",
    "rate outside measurable range": "Cannot measure here; a person should check",
    "motion faded": "Motion dropped; a person should check",
    "need more data": "Hold still while it measures",
    "starting": "Finding the chest",
}


@lru_cache(maxsize=32)
def font(size):
    return ImageFont.load_default(size)  # bundled with Pillow, so nothing to install


def decision_text(decision):
    """Return (headline, detail). The headline says what a person should do."""
    if decision.action is Action.GET_HELP:
        when = f"{decision.completed_at_s:.0f}s"
        return "GET HELP: a person must check", f"Past result at {when}: {decision.status.value} ({decision.reason})"
    if decision.action is Action.REPORT:
        result = decision.status.value if decision.bpm is None else f"{decision.bpm:.1f} breaths/min"
        return f"PAST RESULT at {decision.completed_at_s:.0f}s: {result}", f"{decision.reason.capitalize()}; press R to reassess"
    if decision.action is Action.MOVE:
        return "MOVE the camera, then press M", f"This view did not work: {decision.reason}"
    return "ASSESSING: hold still", decision.reason.capitalize()


def prompt_status(check):
    if check is None or check.state == "idle":
        return "Not asked yet", DIM
    if check.active:
        return ("Watching for hand" if check.state == "watching" else "Asking now"), BLUE
    if check.outcome == "response":
        return "Hand raise seen", GREEN
    return "None seen (proves nothing)", AMBER


class _Painter:
    def __init__(self, image):
        self.d = ImageDraw.Draw(image)
        self.boxes = []  # every text box drawn, for tests

    def text(self, x, y, text, size, color, room, bold=False):
        """One line, shrunk if needed to stay inside `room` pixels."""
        stroke = 1 if bold else 0
        while size > 9 and self.d.textlength(text, font(size)) + 2 * stroke > room:
            size -= 1
        self.d.text((x, y), text, font=font(size), fill=color, stroke_width=stroke, stroke_fill=color)
        box = self.d.textbbox((x, y), text, font=font(size), stroke_width=stroke)
        self.boxes.append((text, box, x + room))

    def card(self, y0, y1, title):
        x0, x1 = PANEL
        self.d.rounded_rectangle((x0, y0, x1, y1), 12, fill=CARD, outline=LINE)
        self.text(x0 + 18, y0 + 13, title, 14, DIM, x1 - x0 - 36)


def outdated(decision, reading):
    """A finished rate that the camera no longer supports."""
    return decision.bpm is not None and reading.bpm is None


def _banner(p, decision, prompt, reading):
    headline, detail = prompt or decision_text(decision)
    old = not prompt and outdated(decision, reading)
    if old:
        detail = f"Not confirmed now ({reading.reason}). Check the person or press R."
    if prompt:
        color = BLUE
    elif decision.action in (Action.GET_HELP, Action.MOVE) or old:
        color = AMBER
    elif decision.action is Action.REPORT:
        color = GREEN
    else:
        color = INK
    p.d.rounded_rectangle((24, 20, W - 24, 104), 12, fill=CARD, outline=color, width=2)
    p.d.rounded_rectangle((24, 20, 34, 104), 5, fill=color)
    p.text(54, 30, headline, 30, color, W - 110, bold=True)
    p.text(54, 72, detail, 18, INK, W - 110)


def _live(p, monitor):
    r, x0, x1 = monitor.reading, *PANEL
    room = x1 - x0 - 36
    p.card(120, 296, "LIVE BREATHING")
    if r.bpm is not None:
        p.text(x0 + 18, 152, f"{r.bpm:.0f}", 78, GREEN, 104, bold=True)
        p.text(x0 + 126, 198, "breaths/min", 22, INK, room - 108)
        p.text(x0 + 18, 256, f"Signal score {r.confidence:.2f} (not a probability)", 16, DIM, room)
        return
    p.text(x0 + 18, 156, "Unable to assess", 32, AMBER, room, bold=True)
    p.text(x0 + 18, 204, r.reason.capitalize(), 18, INK, room)
    p.text(x0 + 18, 230, HINTS.get(r.reason, "Adjust the view"), 16, DIM, room)
    if r.reason == "need more data" and monitor.samples:
        done = min(1.0, (monitor.samples[-1][0] - monitor.samples[0][0]) / MIN_SECONDS)
        p.d.rounded_rectangle((x0 + 18, 268, x1 - 18, 278), 5, fill=LINE)
        if done > 0.03:
            p.d.rounded_rectangle((x0 + 18, 268, x0 + 18 + room * done, 278), 5, fill=BLUE)


def _wave(p, monitor, window):
    x0, x1 = PANEL
    p.card(310, 440, f"CHEST MOTION, LAST {window:.0f} S")
    left, right, top, bottom = x0 + 18, x1 - 18, 346, 426
    p.d.line((left, (top + bottom) // 2, right, (top + bottom) // 2), fill=LINE)
    if len(monitor.samples) < 2:
        return
    t, y = map(np.array, zip(*monitor.samples))
    good = np.isfinite(y)
    if good.sum() < 2:
        return
    lo, hi = np.percentile(y[good], [2, 98])
    px = left + (right - left) * np.clip((t - t[0]) / window, 0, 1)
    py = bottom - (bottom - top) * np.clip((y - lo) / max(hi - lo, 1e-6), 0, 1)
    run = []
    for x, v, ok in zip(px, py, good):  # break the line wherever tracking was lost
        if ok:
            run.append((float(x), float(v)))
        else:
            if len(run) > 1:
                p.d.line(run, fill=GREEN, width=2)
            run = []
    if len(run) > 1:
        p.d.line(run, fill=GREEN, width=2)


def _steps(p, decision, check, view, views):
    x0, x1 = PANEL
    p.card(454, 600, "THIS ASSESSMENT")
    status, color = prompt_status(check)
    result = decision.status.value if decision.final else "in progress"
    if decision.final and decision.bpm is not None:
        result = f"{decision.bpm:.1f}/min at {decision.completed_at_s:.0f}s (past)"
    rows = [("View", f"{view} of {views}", INK), ("Response", status, color), ("Result", result, INK)]
    for i, (name, value, c) in enumerate(rows):
        y = 494 + 34 * i
        p.text(x0 + 18, y, name, 17, DIM, 84)
        p.text(x0 + 108, y, value, 17, c, x1 - x0 - 126)


def _footer(p, decision):
    x = 24
    keys = [("V", "Ask", not decision.final), ("M", "Moved", decision.action is Action.MOVE),
            ("R", "Restart", True), ("Q", "Quit", True)]  # fmt: skip
    for key, label, on in keys:
        color = INK if on else LINE
        p.d.rounded_rectangle((x, 618, x + 28, 644), 6, outline=color)
        p.text(x + 8, 622, key, 16, color, 18)
        p.text(x + 38, 622, label, 16, color, 70)
        x += 116
    p.text(W - 24 - 466, 622, "Research prototype. Not a medical device. Video is never saved.", 16, DIM, 466)


def compose(frame, monitor, decision, check=None, view=1, views=1, window=30.0):
    """Return (screen image in BGR, painter). `frame` is not changed."""
    canvas = np.full((H, W, 3), BG[::-1], np.uint8)
    x0, y0, x1, y1 = VIDEO
    h, w = frame.shape[:2]
    scale = min((x1 - x0) / w, (y1 - y0) / h)
    size = max(1, round(w * scale)), max(1, round(h * scale))
    ox, oy = x0 + (x1 - x0 - size[0]) // 2, y0 + (y1 - y0 - size[1]) // 2
    canvas[y0:y1, x0:x1] = CARD[::-1]
    canvas[oy : oy + size[1], ox : ox + size[0]] = cv2.resize(frame, size)
    if monitor.tracker.box:
        a, b, c, d, _ = monitor.tracker.box
        corner = (int(ox + a * scale), int(oy + b * scale)), (int(ox + c * scale), int(oy + d * scale))
        color = GREEN if monitor.reading.bpm is not None else BLUE
        cv2.rectangle(canvas, *corner, color[::-1], 2, cv2.LINE_AA)
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    p = _Painter(image)
    _banner(p, decision, check.message() if check else None, monitor.reading)
    _live(p, monitor)
    _wave(p, monitor, window)
    _steps(p, decision, check, view, views)
    _footer(p, decision)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), p
