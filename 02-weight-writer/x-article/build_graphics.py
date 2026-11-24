#!/usr/bin/env python3
"""Render the X article graphics for post.md with headless Chrome.

Numbers are copied from the DATA object in ../explainer/weight-writer-explainer.html and from ../post.md.
Same palette, fonts, sizes and rendering as the upscale_net X article, so both read as one series.
"""
import subprocess, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "img"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
INK, SEC, MUTE, GRID, SURF = "#0b0b0b", "#52514e", "#85837d", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SANS = "-apple-system, 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif"
SERIF = "Georgia, 'Times New Roman', serif"
MONO = "'SF Mono', Menlo, Consolas, monospace"
FOOT = "Qwen2.5-0.5B, one run per setting"

# ---------- data (DATA in the explainer) ----------
FIRST_TRY = {  # percent of 205 training tasks solved on the first try; index = round
    "t08": [0, 11.7, 13.7, 15.6, 16.1, 17.1, 20.0, 21.5, 22.4, 22.4, 22.9, 24.4, 24.9, 25.4, 25.4, 25.4,
            26.3, 26.3, 26.8, 27.3, 27.8, 28.3, 29.3, 30.7, 31.2, 31.7, 31.7, 31.7, 31.7, 31.7, 32.2],
    "t13": [0, 24.4, 30.7, 34.6, 37.1, 38.0, 40.0, 40.5, 41.0, 41.0, 42.0],
    "tries64": [0, 25.9, 31.7, 34.6, 36.1, 36.1, 37.1, 38.0, 38.0, 39.5, 40.5],
    "passOnly": [0, 3.4, 3.4, 3.4, 3.4, 3.4, 3.4],
}
RANDOM_POOL, TEACHER = 85.4, 100
DISCOVERED = {  # distinct training functions found so far (of 205); index = round - 1
    "random": [112.0, 128.8, 140.9, 149.3, 155.6, 161.1, 166.3, 170.3, 174.5, 177.4],
    "t13": [50, 63, 71, 76, 78, 82, 83, 84, 84, 86],
    "tries64": [53, 65, 70, 74, 74, 76, 78, 78, 81, 83],
    "t08": [24, 28, 32, 33, 35, 40, 44, 46, 46, 47],
}
SELF_T = [0.838, 0.741, 0.792, 0.800, 0.975, 1.136, 1.283, 1.372, 1.373, 1.371, 1.380]  # selfexplore.runs[0].rounds[*].t
MENU = [0.5, 0.8, 1.1, 1.4, 1.7]

# sanity checks against the numbers quoted in post.md
assert FIRST_TRY["t08"][1] == 11.7 and FIRST_TRY["t08"][10] == 22.9 and FIRST_TRY["t08"][30] == 32.2
assert FIRST_TRY["t08"][25:30] == [31.7] * 5
assert FIRST_TRY["t13"][10] == 42.0 and FIRST_TRY["tries64"][10] == 40.5 and FIRST_TRY["passOnly"][6] == 3.4
assert [DISCOVERED[k][9] for k in ("t13", "tries64", "t08")] == [86, 83, 47]
assert [DISCOVERED[k][0] for k in ("t13", "tries64", "t08")] == [50, 53, 24]
assert len(SELF_T) == 11 and SELF_T[-1] == 1.380


def page(body, w, h, bg=SURF):
    return f"""<!doctype html><meta charset=utf-8><style>
html,body{{margin:0;padding:0;width:{w}px;height:{h}px;background:{bg};overflow:hidden;
font-family:{SANS};color:{INK};-webkit-font-smoothing:antialiased}}
.h{{font:700 34px/1.2 {SERIF};letter-spacing:-.3px}} .s{{font-size:21px;color:{SEC};line-height:1.35;margin-top:8px}}
.foot{{font-size:17px;color:{MUTE}}}
</style>{body}"""


def render(name, html, w, h):
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "p.html"; f.write_text(html)
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={td}/prof",
                        f"--screenshot={OUT / name}", f"--window-size={w},{h}", "--force-device-scale-factor=2",
                        f"file://{f}"], check=True, capture_output=True, timeout=90)
    print("rendered", name)


def line_chart(series, x0, x1, y0, y1, w, h, ticks_y, ticks_x, xlabel, labels, extra=None, m=(66, 220, 14, 62)):
    """series: (points, colour, dashed). labels: (name, value text, y or None, at_end).
    at_end=True puts the label in the right margin, False puts it just right of the line's last point."""
    L, R, T, B = m
    pw, ph = w - L - R, h - T - B
    X = lambda v: L + (v - x0) / (x1 - x0) * pw
    Y = lambda v: T + (y1 - v) / (y1 - y0) * ph
    o = [f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" font-family="{SANS}">']
    for t, lab in ticks_y:
        o.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="{GRID}"/>')
        o.append(f'<text x="{L-12}" y="{Y(t)+6:.1f}" text-anchor="end" font-size="18" fill="{SEC}">{lab}</text>')
    for t, lab in ticks_x:
        o.append(f'<text x="{X(t):.1f}" y="{T+ph+30}" text-anchor="middle" font-size="18" fill="{SEC}">{lab}</text>')
    o.append(f'<text x="{L+pw/2}" y="{T+ph+56}" text-anchor="middle" font-size="18" fill="{MUTE}">{xlabel}</text>')
    if extra:
        o.append(extra(X, Y))
    for pts, col, dash in series:
        d = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in pts)
        da = ' stroke-dasharray="7 6"' if dash else ""
        o.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="2.5"{da} stroke-linejoin="round" stroke-linecap="round"/>')
    for (pts, col, _), (txt, val, ly, at_end) in zip(series, labels):
        a, b = pts[-1]
        o.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="5.5" fill="{col}" stroke="{SURF}" stroke-width="2"/>')
        yy = Y(b) + 6 if ly is None else ly
        lx = L + pw + 14 if at_end else X(a) + 14
        o.append(f'<text x="{lx:.1f}" y="{yy:.1f}" font-size="19" font-weight="600" fill="{INK}">{val}'
                 f'<tspan dx="9" font-weight="400" fill="{SEC}">{txt}</tspan></text>')
    o.append("</svg>")
    return "".join(o)


def legend(items):
    return '<div style="display:flex;gap:26px;margin:14px 0 2px;font-size:19px;color:' + SEC + '">' + "".join(
        f'<span style="display:flex;align-items:center;gap:8px"><i style="width:22px;height:4px;border-radius:2px;background:{c};display:block"></i>{t}</span>'
        for t, c in items) + "</div>"


def dodge(items, gap=24):
    """items: {name: y in px}. Push labels apart, top to bottom, so none are closer than gap."""
    out, last = {}, None
    for k in sorted(items, key=items.get):
        y = items[k]
        if last is not None and y - last < gap:
            y = last + gap
        out[k] = y; last = y
    return out


# ---------- header: the XOR example from the prompt, truth table in, network out ----------
def header():
    w, h = 1500, 600
    bg, fill, ring, txt, faint = "#0f1420", "#4f9bf0", "#46506a", "#f4f3ef", "#8f96aa"
    neg = "#f08a5d"
    s = [f'<svg width="{w}" height="{h}" style="position:absolute;left:0;top:0" font-family="{SANS}">']
    # truth table
    tx, ty = 812, 190
    s.append(f'<text x="{tx}" y="{ty-46}" font-size="20" fill="{faint}" letter-spacing="1">x1 x2</text>')
    s.append(f'<text x="{tx+118}" y="{ty-46}" font-size="20" fill="{faint}">y</text>')
    s.append(f'<line x1="{tx}" x2="{tx+140}" y1="{ty-32}" y2="{ty-32}" stroke="{ring}" stroke-width="1.5"/>')
    for i, (inp, out) in enumerate([("0 0", 0), ("0 1", 1), ("1 0", 1), ("1 1", 0)]):
        y = ty + i * 58
        s.append(f'<text x="{tx+4}" y="{y}" font-family="{MONO}" font-size="30" fill="#c7cad4">{inp}</text>')
        if out:
            s.append(f'<circle cx="{tx+124}" cy="{y-10}" r="11" fill="{fill}"/>')
        else:
            s.append(f'<circle cx="{tx+124}" cy="{y-10}" r="10" fill="none" stroke="{ring}" stroke-width="2"/>')
        s.append(f'<text x="{tx+146}" y="{y}" font-family="{MONO}" font-size="22" fill="{faint}">{out}</text>')
    # arrow
    s.append(f'<line x1="988" x2="1036" y1="300" y2="300" stroke="{faint}" stroke-width="2.5"/>'
             f'<path d="M1036,291 L1050,300 L1036,309 z" fill="{faint}"/>')
    # network: W1 = [[1, 1], [1, 1]], b1 = [0, -1], W2 = [1, -2], b2 = 0
    W1, b1, W2, b2 = [[1, 1], [1, 1]], [0, -1], [1, -2], 0
    inp = [(1100, 210), (1100, 390)]
    hid = [(1262, 210), (1262, 390)]
    out = (1414, 300)

    def edge(p, q, wt, t):
        col = fill if wt > 0 else neg
        lx, ly = p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t
        lab = f"{wt}"
        return (f'<line x1="{p[0]}" y1="{p[1]}" x2="{q[0]}" y2="{q[1]}" stroke="{col}" stroke-width="{1.6 + 1.4*abs(wt)}" stroke-opacity=".8"/>',
                f'<rect x="{lx-19:.1f}" y="{ly-15:.1f}" width="38" height="30" rx="8" fill="{bg}" stroke="{col}" stroke-width="1.5"/>'
                f'<text x="{lx:.1f}" y="{ly+7:.1f}" text-anchor="middle" font-family="{MONO}" font-size="20" font-weight="600" fill="{txt}">{lab}</text>')
    lines, labs = [], []
    for i in range(2):          # hidden unit i
        for j in range(2):      # input j
            straight = i == j
            a, b = edge(inp[j], hid[i], W1[i][j], 0.5 if straight else 0.74)
            lines.append(a); labs.append(b)
    for i in range(2):
        a, b = edge(hid[i], out, W2[i], 0.5)
        lines.append(a); labs.append(b)
    s += lines
    for (x, y), name in zip(inp, ("x1", "x2")):
        s.append(f'<circle cx="{x}" cy="{y}" r="27" fill="{bg}" stroke="{ring}" stroke-width="2.5"/>'
                 f'<text x="{x}" y="{y+8}" text-anchor="middle" font-size="22" fill="{txt}">{name}</text>')
    for k, ((x, y), name) in enumerate(zip(hid, ("h1", "h2"))):
        s.append(f'<circle cx="{x}" cy="{y}" r="27" fill="{fill}"/>'
                 f'<text x="{x}" y="{y+8}" text-anchor="middle" font-size="22" font-weight="600" fill="{bg}">{name}</text>')
        by = y - 44 if k == 0 else y + 60
        s.append(f'<text x="{x}" y="{by}" text-anchor="middle" font-family="{MONO}" font-size="20" fill="{faint}">b {b1[k]}</text>')
    x, y = out
    s.append(f'<circle cx="{x}" cy="{y}" r="27" fill="{txt}"/>'
             f'<text x="{x}" y="{y+8}" text-anchor="middle" font-size="22" font-weight="600" fill="{bg}">y</text>'
             f'<text x="{x}" y="{y-44}" text-anchor="middle" font-family="{MONO}" font-size="20" fill="{faint}">b {b2}</text>'
             f'<text x="{x}" y="{y+60}" text-anchor="middle" font-family="{MONO}" font-size="20" fill="{faint}">&gt; 0.5</text>')
    s += labs
    s.append("</svg>")
    t = f"""<div style="position:absolute;left:70px;top:0;height:{h}px;width:660px;display:flex;flex-direction:column;justify-content:center;color:{txt}">
<div style="font:700 74px/1.06 {SERIF};letter-spacing:-1px">Weight Writer</div>
<div style="font:400 31px/1.32 {SERIF};color:#c7cad4;margin-top:24px">Self-training from a checker works, but it is not recursive self-improvement</div>
<div style="font-size:19px;line-height:1.45;color:{faint};margin-top:34px">A truth table in, the weights of a network out: XOR, as in the prompt.<br>Every weight is an integer from -2 to 2. Orange: negative.</div>
<div style="font-size:16px;color:#5d6478;margin-top:18px">{FOOT}</div></div>"""
    render("01-header.png", page("".join(s) + t, w, h, bg), w, h)


# ---------- table helper ----------
def table(head, rows, bold):
    """head: column titles (first is the row label). rows: list of label + cells. bold: set of (row, col)."""
    th = f"text-align:right;font-weight:600;padding:0 0 14px 22px;font-size:20px;color:{SEC};vertical-align:bottom;line-height:1.25"
    td = f"text-align:right;padding:10px 0 10px 22px;font-size:23px;font-variant-numeric:tabular-nums;border-top:1px solid {GRID}"
    out = f'<table style="border-collapse:collapse;width:100%;margin-top:24px"><tr><th style="{th};text-align:left;padding-left:0">{head[0]}</th>'
    out += "".join(f'<th style="{th}">{c}</th>' for c in head[1:]) + "</tr>"
    for r, row in enumerate(rows):
        out += f'<tr><td style="padding:10px 0;font-size:21px;border-top:1px solid {GRID}">{row[0]}</td>'
        for c, v in enumerate(row[1:], 1):
            out += f'<td style="{td}{";font-weight:700" if (r, c) in bold else ""}">{v}</td>'
        out += "</tr>"
    return out + "</table>"


# ---------- table 1: solved on the first try ----------
def results():
    rows = [("Base model", "0%", "0%"),
            ("Unverified control, round 3", "0%", "0%"),
            ("Pass-only, round 6", "3.4%", "0%"),
            ("Hindsight, temperature 0.8, round 10", "22.9%", "0%"),
            ("Hindsight, temperature 0.8, round 30", "32.2%", "0%"),
            ("Hindsight, temperature 1.3, round 10", "42.0%", "0%"),
            ("Hindsight, 64 tries, round 10", "40.5%", "0%"),
            ("Random weights, one training run", "85.4%", "1.6%"),
            ("Teacher, 150 passes (solutions given)", "100%", "7.9%")]
    w, h = 1000, 690
    body = f"""<div style="padding:34px 40px">
<div class=h>The loops learn the training tasks, but not new ones</div>
<div class=s>Solved on the first try. Best loop: 42.0%. Random weights plus training: 85.4%.</div>
{table(["", "Training tasks<br>(205)", "Held-out tasks<br>(63)"], rows, {(5, 1)})}
<div class=foot style="margin-top:14px">{FOOT}</div></div>"""
    render("02-results.png", page(body, w, h), w, h)


# ---------- figure 1: first try by round ----------
def first_try():
    w, h = 1000, 640
    H, T, B = 452, 14, 62
    names = [("t08", "Temperature 0.8", BLUE), ("t13", "Temperature 1.3", ORANGE),
             ("tries64", "64 tries", AQUA), ("passOnly", "Pass-only", YELLOW)]
    ph = H - T - B
    Yp = lambda v: T + (100 - v) / 100 * ph
    ser = [([(i, v) for i, v in enumerate(FIRST_TRY[k])], c, False) for k, _, c in names]
    # the two 10-round lines end at 42.0 and 40.5, 6 px apart: push their labels apart
    pos = dodge({"t13": Yp(42.0) + 6, "tries64": Yp(40.5) + 6}, gap=24)
    shift = (pos["t13"] + pos["tries64"]) / 2 - (Yp(42.0) + Yp(40.5)) / 2 - 6
    pos = {k: v - shift for k, v in pos.items()}
    labels = [("Temperature 0.8", "32.2", None, True), ("Temperature 1.3", "42.0", pos["t13"], False),
              ("64 tries", "40.5", pos["tries64"], False), ("Pass-only", "3.4", None, False)]

    def refs(X, Y):
        o = ""
        for v, val, name, sub in ((TEACHER, "100", "Teacher", "solutions given"), (RANDOM_POOL, "85.4", "Random weights", "plus training")):
            o += f'<line x1="{X(0):.1f}" x2="{X(30):.1f}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{MUTE}" stroke-width="2" stroke-dasharray="7 6"/>'
            o += (f'<text x="{X(30)+14:.1f}" y="{Y(v)+6:.1f}" font-size="19" font-weight="600" fill="{INK}">{val}'
                  f'<tspan dx="9" font-weight="400" fill="{SEC}">{name}</tspan></text>'
                  f'<text x="{X(30)+14:.1f}" y="{Y(v)+28:.1f}" font-size="17" fill="{MUTE}">{sub}</text>')
        return o
    svg = line_chart(ser, 0, 30, 0, 100, w - 40, H, [(t, f"{t}%") for t in (0, 20, 40, 60, 80, 100)],
                     [(t, str(t)) for t in (0, 5, 10, 15, 20, 25, 30)], "Round", labels, refs, m=(66, 220, T, B))
    body = f"""<div style="padding:34px 20px 0 40px"><div class=h>The loops flatten far below random weights</div>
<div class=s>Training tasks solved on the first try, by round. Dashed: one-off training runs, for reference.</div>
{legend([(t, c) for _, t, c in names])}{svg}<div class=foot style="margin-top:2px">{FOOT}</div></div>"""
    render("03-first-try.png", page(body, w, h), w, h)


# ---------- figure 2: functions discovered ----------
def discovery():
    w, h = 1000, 640
    H, T, B = 452, 14, 62
    names = [("random", "Random weights", MUTE, True), ("t13", "Temperature 1.3", ORANGE, False),
             ("tries64", "64 tries", AQUA, False), ("t08", "Temperature 0.8", BLUE, False)]
    ph = H - T - B
    Yp = lambda v: T + (205 - v) / 205 * ph
    ser = [([(i + 1, v) for i, v in enumerate(DISCOVERED[k])], c, d) for k, _, c, d in names]
    ends = {k: DISCOVERED[k][-1] for k, *_ in names}
    pos = dodge({k: Yp(v) + 6 for k, v in ends.items()}, gap=24)
    fmt = lambda v: f"{v:.1f}" if v != int(v) else str(int(v))
    labels = [(t, fmt(ends[k]), pos[k], True) for k, t, _, _ in names]

    def mark(X, Y):
        x, y = X(1), Y(112.0)
        return (f'<text x="{x+14:.1f}" y="{y+30:.1f}" font-size="19" fill="{INK}">112 after one round</text>'
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{MUTE}" stroke="{SURF}" stroke-width="2"/>'
                f'<line x1="{X(0.85):.1f}" x2="{X(10):.1f}" y1="{Y(205):.1f}" y2="{Y(205):.1f}" stroke="{MUTE}" stroke-width="1.5"/>'
                f'<text x="{X(10):.1f}" y="{Y(205)+24:.1f}" text-anchor="end" font-size="17" fill="{MUTE}">all 205 training functions</text>')
    svg = line_chart(ser, 0.85, 10, 0, 205, w - 40, H, [(t, str(t)) for t in (0, 50, 100, 150, 200)],
                     [(t, str(t)) for t in range(1, 11)], "Round", labels, mark, m=(66, 220, T, B))
    body = f"""<div style="padding:34px 20px 0 40px"><div class=h>Random weights in one round beat any loop in ten</div>
<div class=s>Distinct training functions discovered, by round. Random weights: mean of 20 simulated runs.</div>
{legend([(t, c) for _, t, c, _ in names])}{svg}<div class=foot style="margin-top:2px">{FOOT}</div></div>"""
    render("04-discovery.png", page(body, w, h), w, h)


# ---------- section 5: three tests ----------
def not_rsi():
    w, h = 1000, 470
    card = lambda t, q, why: (f'<div style="flex:1;border:1px solid {GRID};border-radius:10px;padding:20px 18px 18px;background:#fff">'
                              f'<div style="font:700 24px/1.2 {SERIF};white-space:nowrap">{t}</div><div style="font-size:19px;color:{SEC};line-height:1.35;margin-top:10px;min-height:52px">{q}</div>'
                              f'<div style="font-size:17px;color:{INK};border-top:1px solid {GRID};padding-top:10px;margin-top:10px;line-height:1.3">'
                              f'<b style="color:{ORANGE};font-size:19px">No.</b> {why}</div></div>')
    body = f"""<div style="padding:34px 40px 0"><div class=h>The loop fails all three tests</div>
<div class=s>Recursive self-improvement: a system that gets better at making itself better.</div>
<div style="display:flex;gap:16px;margin-top:26px">{card("Self-application", "Does it improve the process that improves it?", "The loop (how to explore, what to keep, how to train) is fixed code. The model only gets better at the task.")}
{card("Compounding", "Do gains feed further gains?", "The temperature 0.8 loop gained 11.7 points in round 1 and about 0.5 points per round at the end.")}
{card("Open-ended measure", "Can the target keep growing?", "There are 205 fixed training tasks. Once they are solved, there is nothing left to gain.")}</div>
<div style="text-align:center;font:400 22px/1.3 {SERIF};color:{SEC};margin-top:22px">The model gets better at the task, not at getting better.</div>
<div class=foot style="margin-top:14px">{FOOT}</div></div>"""
    render("05-not-rsi.png", page(body, w, h), w, h)


# ---------- table 3: self-chosen temperature ----------
def self_chosen():
    rows = [("Self-chosen temperature", "89", "32.2%", "42.4%"),
            ("Random choice", "92", "38.0%", "44.9%"),
            ("Fixed 1.4", "85", "41.5%", "41.5%")]
    w, h = 1000, 440
    body = f"""<div style="padding:34px 40px">
<div class=h>Choosing its own temperature did not beat chance</div>
<div class=s>Three hindsight runs of 10 rounds with the extra T = v line. A gap of three functions is within noise.</div>
{table(["", "Functions known<br>(of 205)", "First try", "Within<br>16 tries"], rows, {(1, 1), (2, 2), (1, 3)})}
<div class=foot style="margin-top:14px">Bold: best in the column. The extra line changes the prompt, so compare these runs only with each other.</div>
<div class=foot style="margin-top:6px">{FOOT}</div></div>"""
    render("06-self-chosen.png", page(body, w, h), w, h)


# ---------- figure 3: chosen temperature ----------
def temperature():
    w, h = 1000, 640
    H, T, B = 470, 14, 62
    pts = list(enumerate(SELF_T))

    def extra(X, Y):
        o = ""
        for i, v in pts[:-1]:
            o += f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4" fill="{BLUE}"/>'
        o += (f'<text x="{X(0.1):.1f}" y="{Y(0.8)+56:.1f}" font-size="19" fill="{INK}">Near 0.8 for rounds 0 to 3</text>'
              f'<text x="{X(10):.1f}" y="{Y(0.62):.1f}" text-anchor="end" font-size="19" fill="{INK}">Round 10: 1.383 on tasks solved before,</text>'
              f'<text x="{X(10):.1f}" y="{Y(0.62)+26:.1f}" text-anchor="end" font-size="19" fill="{INK}">1.378 on tasks never solved</text>')
        return o
    svg = line_chart([(pts, BLUE, False)], 0, 10, 0.4, 1.75, w - 40, H, [(t, f"{t:.1f}") for t in MENU],
                     [(t, str(t)) for t in range(0, 11)], "Round", [("Mean chosen", "1.38", None, True)], extra, m=(66, 200, 24, B))
    body = f"""<div style="padding:34px 20px 0 40px"><div class=h>It learned to sample hotter, but not per task</div>
<div class=s>Mean temperature the model chose, by round. Grid lines: the five values it could pick.</div>
<div style="margin-top:14px">{svg}</div><div class=foot style="margin-top:2px">{FOOT}</div></div>"""
    render("07-temperature.png", page(body, w, h), w, h)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for f in (header, results, first_try, discovery, not_rsi, self_chosen, temperature):
        f()
