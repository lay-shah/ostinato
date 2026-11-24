"""Offline medic page for one simulated search. Shows only what the robot knows."""

from html import escape

from .simulation import CELL, N, rank_for_medics

S = 16  # drawing units per cell

CSS = """
:root{--bg:#f7f7f4;--card:#fff;--ink:#1c1f22;--dim:#5c636a;--line:#d5d8da;--wall:#8b9298;
--check:#b45309;--outside:#b3261e;--within:#2e6b3e;--path:#2563a8}
@media(prefers-color-scheme:dark){:root{--bg:#15181a;--card:#1f2326;--ink:#eceeef;--dim:#a6adb3;
--line:#3a4045;--wall:#6b7278;--check:#f0a04b;--outside:#ff8a80;--within:#7fcf95;--path:#7db4ee}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.45 system-ui,sans-serif;padding:16px}
main{max-width:1100px;margin:auto}h1{font-size:1.4rem;margin:0}
.sim{border:2px solid var(--check);padding:8px 12px;margin:12px 0;font-weight:600}
.cols{display:grid;gap:16px;grid-template-columns:minmax(0,1fr) minmax(0,1fr)}
@media(max-width:760px){.cols{grid-template-columns:1fr}}
svg{width:100%;background:var(--card);border:1px solid var(--line);display:block}
ol{list-style:none;margin:0;padding:0}li{background:var(--card);border:1px solid var(--line);
border-left:6px solid;padding:10px 12px;margin-bottom:8px}
.check{border-left-color:var(--check)}.outside{border-left-color:var(--outside)}.within{border-left-color:var(--within)}
li b{font-size:1.05rem}li span{display:block}.dim{color:var(--dim);font-size:.9rem}
"""


def group(result):
    if result["bpm"] is None:
        return "check"
    return "within" if 10 <= result["bpm"] <= 30 else "outside"


LABELS = {"check": "Needs a human check", "outside": "Rate outside 10-30", "within": "Rate within 10-30"}


def _map(scene, out, order):
    size = N * S
    parts = [f'<svg viewBox="0 0 {size} {size}" role="img" aria-label="Search map with people numbered by rank">']
    for r in range(N):
        c = 0
        while c < N:
            if not scene.walls[r, c]:
                c += 1
                continue
            start = c
            while c < N and scene.walls[r, c]:
                c += 1
            parts.append(
                f'<rect x="{start * S}" y="{r * S}" width="{(c - start) * S}" height="{S}" fill="var(--wall)" shape-rendering="crispEdges"/>'
            )

    def xy(cell):
        return cell[1] * S + S // 2, cell[0] * S + S // 2

    track = " ".join("{},{}".format(*xy(c)) for c in out["track"])
    parts.append(f'<polyline points="{track}" fill="none" stroke="var(--path)" stroke-width="2" stroke-dasharray="6 4"/>')
    x, y = xy(scene.start)
    parts.append(f'<rect x="{x - 8}" y="{y - 8}" width="16" height="16" fill="var(--path)"/>')
    cells = {p.known.id: p.known.cell for p in scene.people}
    by_id = {r["person"]: r for r in out["results"]}
    for rank, row in enumerate(order, 1):
        x, y = xy(cells[row["person"]])
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="13" fill="var(--{group(by_id[row["person"]])})"/>'
            f'<text x="{x}" y="{y + 6}" text-anchor="middle" font-size="17" font-weight="700" '
            f'fill="var(--card)">{rank}</text>'
        )
    return "".join([*parts, "</svg>"])


def render(scene, out):
    order = rank_for_medics(out["results"])
    by_id = {r["person"]: r for r in out["results"]}
    cells = {p.known.id: p.known.cell for p in scene.people}
    items = []
    for rank, row in enumerate(order, 1):
        r = by_id[row["person"]]
        g = group(r)
        cell = cells[row["person"]]
        facts = [f"Person {r['person']}", f"{cell[1] * CELL:.0f} m east, {cell[0] * CELL:.0f} m south of the start"]
        if r["reached_s"] is None:
            facts.append("not reached")
        else:
            n = len(r["views"])
            facts += [f"reached at {r['reached_s']:.0f} s", f"{n} view{'s' if n != 1 else ''} tried"]
        score = "no signal score" if r["score"] is None else f"signal score {r['score']:.2f} (fixed in simulation)"
        items.append(
            f'<li class="{g}"><b>{rank}. {LABELS[g]}</b><span>{escape(row["why"])}</span>'
            f'<span class="dim">{escape("; ".join([*facts, score]))}</span></li>'
        )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Ostinato medic view (simulated)</title><style>{CSS}</style></head><body><main>"
        "<h1>Ostinato: who to check, and why</h1>"
        '<div class="sim">SIMULATED DATA. Research prototype, not a medical device or a triage method.</div>'
        "<p>\"Needs a human check\" means the robot could not measure. It says nothing about whether the person "
        "is breathing. Ranked first so nobody is overlooked.</p>"
        f'<div class="cols"><div>{_map(scene, out, order)}<p class="dim">Numbers match the list. Square: start. '
        f"Dashed line: robot path. Gray: walls.</p></div><ol>{''.join(items)}</ol></div></main></body></html>"
    )
