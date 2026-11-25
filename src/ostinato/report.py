"""A private record of one assessment: times, rates, and decisions. Never video."""

import os
from datetime import datetime
from html import escape
from pathlib import Path
from statistics import median

from .breathing import MIN_SECONDS

CSS = """
:root{--bg:#f3f5f7;--card:#fff;--ink:#16202a;--dim:#55616b;--line:#d3dadd;--ok:#1f6b3c;--warn:#9a4a00}
@media(prefers-color-scheme:dark){:root{--bg:#101a24;--card:#162330;--ink:#f0f3f4;--dim:#9eaab2;
--line:#344150;--ok:#74d396;--warn:#fab254}}
*{box-sizing:border-box}body{margin:0;padding:24px 16px;background:var(--bg);color:var(--ink);
font:16px/1.5 system-ui,sans-serif}main{max-width:820px;margin:auto}h1{font-size:2rem;line-height:1.2;margin:4px 0}
h2{font-size:1.2rem;margin:28px 0 8px}.warn{color:var(--warn)}.ok{color:var(--ok)}.dim{color:var(--dim)}
.box{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;overflow-x:auto}
table{border-collapse:collapse;width:100%}td{padding:6px 12px 6px 0;border-bottom:1px solid var(--line);
vertical-align:top}td:first-child{white-space:nowrap;font-variant-numeric:tabular-nums}svg{width:100%;display:block}
"""


class Recorder:
    """Collects what happened. Call `update` on every frame."""

    def __init__(self, label=None):
        self.label = label
        self.restart(0.0)

    def restart(self, t):
        self.start = t
        self.readings = []  # (seconds, rate or None, reason), about one a second
        self.events = []  # (seconds, text)
        self.decision = None
        self._last = {}

    def _event(self, key, value, t, text):
        if self._last.get(key) != value:
            self._last[key] = value
            if text:
                self.events.append((t - self.start, text))

    def update(self, t, reading, decision, check, view):
        self.decision = decision
        if not self.readings or t - self.start - self.readings[-1][0] >= 1.0:
            self.readings.append((t - self.start, reading.bpm, reading.reason))
        self._event("view", view, t, f"View {view}")
        prompt = check.outcome or check.state
        text = {
            "speaking": "Asked the person to raise a hand",
            "response": "Saw a hand raised",
            "none seen": "No response seen (proves nothing)",
        }.get(prompt)
        self._event("prompt", prompt, t, text)
        what = (decision.action, decision.status)
        said = f"Decision: {decision.action.value} ({decision.reason})"
        self._event("decision", what, t, said if decision.action.value != "wait" else None)


def _chart(readings):
    end = max(readings[-1][0], 1.0)
    parts = ['<svg viewBox="0 0 760 200" role="img" aria-label="Breathing rate over time">']
    for bpm in (10, 20, 30, 40):
        y = 170 - bpm * 3.5
        parts.append(
            f'<line x1="36" x2="750" y1="{y}" y2="{y}" stroke="var(--line)"/>'
            f'<text x="4" y="{y + 4}" font-size="12" fill="var(--dim)">{bpm}</text>'
        )
    run = []
    for t, bpm, _ in [*readings, (end, None, "")]:
        if bpm is not None:
            run.append(f"{36 + 714 * t / end:.1f},{170 - bpm * 3.5:.1f}")
            continue
        if len(run) == 1:
            x, y = run[0].split(",")
            parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="var(--ok)"/>')
        elif run:
            parts.append(f'<polyline points="{" ".join(run)}" fill="none" stroke="var(--ok)" stroke-width="3"/>')
        run = []
    parts.append('<text x="36" y="194" font-size="12" fill="var(--dim)">0 s</text>')
    parts.append(f'<text x="750" y="194" font-size="12" text-anchor="end" fill="var(--dim)">{end:.0f} s</text></svg>')
    return "".join(parts)


def _rate_note(recorder):
    """Explain a result without a rate, or mention one that was seen anyway."""
    if recorder.decision.bpm is not None:
        return ""
    rates = [bpm for _, bpm, _ in recorder.readings if bpm is not None]
    if rates:
        spread = f", from {min(rates):.0f} to {max(rates):.0f}" if max(rates) - min(rates) >= 1 else ""
        text = (
            f"A live breathing rate was shown in {len(rates)} one-second checks: typically {median(rates):.1f} "
            f"breaths/min{spread}. It was not part of the decision."
        )
    elif recorder.readings[-1][0] < MIN_SECONDS:
        text = (
            f"No breathing rate: it needs at least {MIN_SECONDS:.0f} seconds of clear video, and this record covers "
            f"{recorder.readings[-1][0]:.0f}. To measure one, keep still and watch for longer before saving."
        )
    else:
        return ""
    return f"<p>{text}</p>"


def render(recorder, when):
    d = recorder.decision
    if d is None or not recorder.readings:
        raise ValueError("nothing has been assessed yet")
    live = recorder.readings[-1]
    if d.bpm is not None:
        taken = d.completed_at_s - recorder.start
        headline, css = f"{d.bpm:.1f} breaths/min, measured at {taken:.0f} s", "ok"
        if live[1] is None:
            headline, css = f"{headline} (not confirmed when saved)", "warn"
    elif d.final and d.status.value == "responded":
        headline, css = "Responded: raised a hand", "ok"
    elif d.final:
        headline, css = "Unable to assess: a person must check", "warn"
    else:
        headline, css = "Assessment not finished", "warn"
    now = f"{live[1]:.1f} breaths/min" if live[1] is not None else f"unable to assess ({live[2]})"
    counts = {}
    for _, bpm, reason in recorder.readings:
        if bpm is None:
            counts[reason] = counts.get(reason, 0) + 1
    rows = "".join(f"<tr><td>{t:.0f} s</td><td>{escape(text)}</td></tr>" for t, text in recorder.events)
    why = "".join(
        f"<tr><td>{n} s</td><td>{escape(reason)}</td></tr>" for reason, n in sorted(counts.items(), key=lambda x: -x[1])
    )
    label = f'<p class="warn"><b>{escape(recorder.label)}</b></p>' if recorder.label else ""
    measured = sum(r[1] is not None for r in recorder.readings)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark">'
        f"<title>Ostinato assessment</title><style>{CSS}</style></head><body><main>{label}"
        f'<p class="dim">Ostinato assessment, {escape(when)}</p><h1 class="{css}">{escape(headline)}</h1>'
        f"<p>When saved, the live reading was: {escape(now)}.</p>"
        f"{_rate_note(recorder)}"
        "<p class='dim'>Research prototype, not a medical device. A missing rate or response says nothing about whether "
        "the person is breathing. Not validated against counted breaths.</p>"
        f"<h2>Rate over time</h2><div class='box'>{_chart(recorder.readings)}"
        f"<p class='dim'>A rate was shown in {measured} of {len(recorder.readings)} one-second checks. "
        "Gaps mean no trustworthy reading.</p></div>"
        f"<h2>What happened</h2><div class='box'><table>{rows}</table></div>"
        f"<h2>Why there was no reading</h2><div class='box'><table>{why or '<tr><td>None</td><td></td></tr>'}</table></div>"
        "</main></body></html>"
    )


def save(recorder, folder, now=None):
    """Write a new file only its owner can read, and return its path."""
    now = now or datetime.now()
    page = render(recorder, now.strftime("%B %d, %Y, %H:%M"))
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"assessment-{now.strftime('%Y%m%d-%H%M%S-%f')}.html"
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as f:
        f.write(page)
    return path
