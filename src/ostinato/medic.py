"""Offline medic page for one simulated search. Shows only what the robot knows."""

from html import escape

from .simulation import CELL, N, rank_for_medics

S = 16  # drawing units per cell

CSS = """
:root{--bg:#fbfdff;--card:#fff;--ink:#172737;--dim:#55616b;--line:#d3dadd;--wall:#aab4ba;
--check:#9a4a00;--outside:#a3223d;--within:#1f6b3c;--path:#126dc1}
@media(prefers-color-scheme:dark){:root{--bg:#101a24;--card:#162330;--ink:#f0f3f4;--dim:#9eaab2;
--line:#343d43;--wall:#56626a;--check:#fab254;--outside:#ff8fa0;--within:#74d396;--path:#80bcfa}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.5 "Helvetica Neue",Helvetica,Arial,sans-serif;padding:24px 16px}
main{max-width:1120px;margin:auto}h1{font-size:1.75rem;line-height:1.2;margin:0}
.sim{color:var(--check);font-weight:650;margin:6px 0 20px}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:0 0 16px}
.stats div{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 16px}
.stats b{display:block;font-size:2rem;line-height:1.1;font-variant-numeric:tabular-nums}
.note{max-width:75ch;margin:0 0 20px}
.cols{display:grid;gap:20px;grid-template-columns:minmax(0,1fr) minmax(0,1fr);align-items:start}
.map{position:sticky;top:16px}@media(max-width:760px){.cols{grid-template-columns:1fr}.map{position:static}}
svg{width:100%;background:var(--card);border:1px solid var(--line);border-radius:12px;display:block}
ol{list-style:none;margin:0;padding:0;display:grid;gap:10px}
li{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;
display:grid;grid-template-columns:40px 1fr;gap:12px}
.n{width:36px;height:36px;border-radius:50%;display:grid;place-items:center;font-weight:700;color:var(--card)}
.check .n{background:var(--check)}.outside .n{background:var(--outside)}.within .n{background:var(--within)}
.c-check{color:var(--check)}.c-outside{color:var(--outside)}.c-within{color:var(--within)}
li b{display:block;font-size:1.1rem;line-height:1.3}li p{margin:2px 0}.dim{color:var(--dim);font-size:.9rem}
header{display:flex;align-items:center;justify-content:space-between;gap:1rem;border-bottom:1px solid var(--line);padding:0 0 1rem}
.brand,h1,h2{font-family:"Bodoni Moda","Bodoni 72",Georgia,serif;font-weight:400;letter-spacing:-.035em}
.brand{font-size:1.9rem}.brand i,nav a{color:var(--path)}nav{display:flex;gap:1rem}a{text-underline-offset:.22em}
h1{font-size:clamp(2.4rem,5vw,4.25rem);margin:1.2rem 0 .3rem}h2{font-size:1.75rem;margin:0 0 .7rem}
:focus-visible{outline:3px solid var(--path);outline-offset:3px}
li:target{outline:2px solid var(--path);outline-offset:2px}li{scroll-margin-top:1rem}
svg a:hover circle,svg a:focus-visible circle{stroke:var(--ink);stroke-width:3}
@media(max-width:480px){.stats{gap:.4rem}.stats div{padding:.6rem}.stats b{font-size:1.5rem}.stats .dim{font-size:.8rem}}
@media print{body{padding:0}.map{position:static}li{break-inside:avoid}}
"""


def group(result):
    if result["bpm"] is None:
        return "check"
    return "within" if 10 <= result["bpm"] <= 30 else "outside"


LABELS = {"check": "Needs a human check", "outside": "Rate outside 10-30", "within": "Rate within 10-30"}


def headline(result):
    return LABELS["check"] if result["bpm"] is None else f"{result['bpm']:.0f} breaths/min"


def detail(result):
    return f"Unable to assess: {result['reason']}" if result["bpm"] is None else LABELS[group(result)]


def _map(scene, out, order):
    size = N * S
    parts = [
        f'<svg viewBox="0 0 {size} {size}" role="group" aria-label="Search map with links to ranked assessments">'
    ]
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
    parts.append(
        f'<polyline points="{track}" fill="none" stroke="var(--path)" stroke-width="2" stroke-dasharray="6 4"/>'
    )
    x, y = xy(scene.start)
    parts.append(f'<rect x="{x - 8}" y="{y - 8}" width="16" height="16" fill="var(--path)"/>')
    cells = {p.known.id: p.known.cell for p in scene.people}
    by_id = {r["person"]: r for r in out["results"]}
    for rank, row in enumerate(order, 1):
        x, y = xy(cells[row["person"]])
        parts.append(
            f'<a href="#assessment-{rank}" aria-label="Read assessment {rank}">'
            f'<circle cx="{x}" cy="{y}" r="13" fill="var(--{group(by_id[row["person"]])})"/>'
            f'<text x="{x}" y="{y + 6}" text-anchor="middle" font-size="17" font-weight="700" '
            f'fill="var(--card)">{rank}</text></a>'
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
        facts = [
            f"Person {r['person']}",
            f"{cell[1] * CELL:.0f} m east, {cell[0] * CELL:.0f} m south of the start",
        ]
        if r["reached_s"] is None:
            facts.append("not reached")
        else:
            n = len(r["views"])
            facts += [f"Reached at {r['reached_s']:.0f} s", f"{n} view{'s' if n != 1 else ''} tried"]
        score = (
            "no signal score"
            if r["score"] is None
            else f"signal score {r['score']:.2f} (fixed in simulation)"
        )
        items.append(
            f'<li class="{g}" id="assessment-{rank}" tabindex="-1"><span class="n">{rank}</span><div><b class="c-{g}">{headline(r)}</b>'
            f'<p>{escape(detail(r))}</p><p class="dim">{escape(" · ".join([*facts, score]))}</p></div></li>'
        )
    groups = [group(r) for r in out["results"]]
    stats = "".join(
        f'<div><b class="c-{g}">{groups.count(g)}</b><span class="dim">{LABELS[g]}</span></div>'
        for g in LABELS
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="color-scheme" content="light dark">'
        f"<title>Ostinato medic view (simulated)</title><style>{CSS}</style></head><body><main>"
        '<header><span class="brand">Ostinato<i>.</i></span><nav aria-label="Report sections">'
        '<a href="#map">Map</a><a href="#assessments">Assessments</a></nav></header>'
        "<h1>Search assessment</h1>"
        '<p class="sim">SIMULATED DATA. Research prototype, not a medical device or a triage method.</p>'
        f'<div class="stats">{stats}</div>'
        '<p class="note">"Needs a human check" means the robot could not measure. It says nothing about whether '
        "the person is breathing. These people are listed first so nobody is overlooked.</p>"
        f'<div class="cols"><div class="map" id="map"><h2>Search map</h2>{_map(scene, out, order)}<p class="dim">Select a numbered marker to read its assessment. '
        f"Square: start. Dashed line: robot path. Gray: walls.</p></div><section id='assessments'><h2>Who to check first</h2><ol>{''.join(items)}</ol></section></div>"
        "</main></body></html>"
    )
