#!/usr/bin/env python3
"""Render the X article graphics from the real logs (../results) with headless Chrome."""
import json, re, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT.parent / "results"
OUT = ROOT / "img"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
INK, SEC, MUTE, GRID, SURF = "#0b0b0b", "#52514e", "#85837d", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SANS = "-apple-system, 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif"
SERIF = "Georgia, 'Times New Roman', serif"
TPS = 65536


def curve(name):
    pts = {}
    for line in open(RES / "logs" / f"{name}.jsonl"):
        r = json.loads(line)
        if r.get("event") == "start":
            pts[0] = r["heldout_loss"]
        elif "step" in r and "heldout_loss" in r:
            pts[r["step"]] = r["heldout_loss"]
    return sorted(pts.items())


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


def line_chart(series, x0, x1, y0, y1, w, h, ticks_y, ticks_x, xlabel, labels, extra="", m=(70, 190, 20, 62)):
    L, R, T, B = m
    pw, ph = w - L - R, h - T - B
    X = lambda v: L + (v - x0) / (x1 - x0) * pw
    Y = lambda v: T + (y1 - v) / (y1 - y0) * ph
    o = [f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" font-family="{SANS}">']
    for t in ticks_y:
        o.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="{GRID}"/>')
        o.append(f'<text x="{L-12}" y="{Y(t)+6:.1f}" text-anchor="end" font-size="18" fill="{SEC}">{t:.2f}</text>')
    for t, lab in ticks_x:
        o.append(f'<text x="{X(t):.1f}" y="{T+ph+30}" text-anchor="middle" font-size="18" fill="{SEC}">{lab}</text>')
    o.append(f'<text x="{L+pw/2}" y="{T+ph+56}" text-anchor="middle" font-size="18" fill="{MUTE}">{xlabel}</text>')
    o.append(extra.replace("@X", "") if not callable(extra) else extra(X, Y))
    for pts, col in series:
        d = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in pts)
        o.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
    for (pts, col), (txt, val, ly) in zip(series, labels):
        a, b = pts[-1]
        o.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="5.5" fill="{col}" stroke="{SURF}" stroke-width="2"/>')
        yy = Y(b) + 6 if ly is None else ly
        o.append(f'<text x="{L+pw+14}" y="{yy:.1f}" font-size="19" font-weight="600" fill="{INK}">{val}</text>')
        o.append(f'<text x="{L+pw+70}" y="{yy:.1f}" font-size="19" fill="{SEC}">{txt}</text>')
    o.append("</svg>")
    return "".join(o)


def legend(items):
    return '<div style="display:flex;gap:26px;margin:14px 0 2px;font-size:19px;color:' + SEC + '">' + "".join(
        f'<span style="display:flex;align-items:center;gap:8px"><i style="width:22px;height:4px;border-radius:2px;background:{c};display:block"></i>{t}</span>'
        for t, c in items) + "</div>"


# ---------- header: the real head map of the grown model ----------
def header():
    w, h = 1500, 600
    q_map = list(range(7)) + [None] * 3 + list(range(7, 14)) + [None] * 3
    inserted = {4, 9, 13, 18}                      # grow.layer_plan(24, 28)
    cols = []
    for li in range(24):
        cols.append(True)
        if li in inserted:
            cols.append(False)
    assert len(cols) == 28
    cell, ox, oy = 24.5, 790, 55
    bg, fill, ring = "#0f1420", "#4f9bf0", "#46506a"
    s = [f'<svg width="{w}" height="{h}" style="position:absolute;left:0;top:0">']
    filled = 0
    for c, real in enumerate(cols):
        for r in range(20):
            x, y = ox + c * cell, oy + r * cell
            if real and q_map[r] is not None:
                s.append(f'<circle cx="{x}" cy="{y}" r="8" fill="{fill}"/>'); filled += 1
            else:
                s.append(f'<circle cx="{x}" cy="{y}" r="7" fill="none" stroke="{ring}" stroke-width="1.6"/>')
    assert filled == 24 * 14
    s.append("</svg>")
    t = f"""<div style="position:absolute;left:70px;top:0;height:{h}px;width:660px;display:flex;flex-direction:column;justify-content:center;color:#f4f3ef">
<div style="font:700 74px/1.06 {SERIF};letter-spacing:-1px">On Computable Growth</div>
<div style="font:400 31px/1.32 {SERIF};color:#c7cad4;margin-top:24px">Why doubling a model bought a rounding error</div>
<div style="font-size:19px;line-height:1.45;color:#8f96aa;margin-top:34px">Every dot is an attention head of the grown model.<br>Filled: from Qwen2.5-0.5B. Hollow: new and empty.</div></div>"""
    render("01-header.png", page("".join(s) + t, w, h, bg), w, h)


# ---------- results table ----------
def results():
    rows = [("Held-out loss, lower is better", 2.618, 2.584, 2.578, 3), ("HellaSwag", 52.1, 51.4, 51.1, 1), ("ARC-Easy", 58.4, 61.0, 61.4, 1),
            ("ARC-Challenge", 32.5, 32.7, 31.1, 1), ("PIQA", 70.3, 69.9, 70.4, 1), ("WinoGrande", 56.7, 56.7, 56.0, 1),
            ("WikiText perplexity, lower is better", 18.45, 20.37, 20.27, 2)]
    th = f"text-align:right;font-weight:600;padding:0 0 14px 22px;font-size:20px;color:{SEC};vertical-align:bottom;line-height:1.25"
    td = f"text-align:right;padding:15px 0 15px 22px;font-size:23px;font-variant-numeric:tabular-nums;border-top:1px solid {GRID}"
    tr = ""
    for n, a, b, c, d in rows:
        tr += (f'<tr><td style="padding:15px 0;font-size:21px;border-top:1px solid {GRID}">{n}</td><td style="{td};color:{SEC}">{a:.{d}f}</td>'
               f'<td style="{td}">{b:.{d}f}</td><td style="{td};font-weight:700">{c:.{d}f}</td><td style="{td};color:{SEC}">{c-b:+.{d}f}</td></tr>')
    w, h = 1000, 690
    body = f"""<div style="padding:34px 40px">
<div class=h>Same tokens, same benchmarks</div>
<div class=s>Twice the parameters moved held-out loss by 0.006 and no benchmark beyond noise.</div>
<table style="border-collapse:collapse;width:100%;margin-top:24px"><tr><th style="{th};text-align:left;padding-left:0"></th>
<th style="{th}">Source<br>0.5B</th><th style="{th}">Control<br>0.5B + 1B tokens</th><th style="{th}">Grown<br>1B + 1B tokens</th><th style="{th}">Grown<br>minus control</th></tr>{tr}</table>
<div class=foot style="margin-top:14px">One standard error of a benchmark difference is 0.7 to 2.0 points. One seed per model.</div></div>"""
    render("03-results.png", page(body, w, h), w, h)


# ---------- grown vs control ----------
def crossover():
    g, c = curve("heal-small-zero"), curve("control-small")
    tok = lambda pts: [(s * TPS / 1e6, v) for s, v in pts]
    gd, cd = dict(g), dict(c)
    cross = next(s for s in sorted(set(gd) & set(cd)) if s >= 5000 and gd[s] < cd[s])
    print("first step grown below control:", cross, gd[cross], cd[cross])
    w, h = 1000, 640
    def mark(X, Y):
        x = X(10000 * TPS / 1e6)
        return (f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{Y(2.695):.1f}" y2="{Y(2.60):.1f}" stroke="{MUTE}"/>'
                f'<text x="{x+10:.1f}" y="{Y(2.685):.1f}" font-size="19" fill="{INK}">Curves cross</text>'
                f'<text x="{x+10:.1f}" y="{Y(2.685)+24:.1f}" font-size="19" fill="{SEC}">near step 10,000</text>')
    gl, cl = g[-1][1], c[-1][1]
    svg = line_chart([(tok(g), BLUE), (tok(c), ORANGE)], 0, 1000, 2.56, 2.70, w - 40, 470,
                     [2.56, 2.58, 2.60, 2.62, 2.64, 2.66, 2.68, 2.70], [(0, "0"), (250, "250M"), (500, "500M"), (750, "750M"), (1000, "1B")],
                     "Training tokens", [("Grown 1B", f"{gl:.3f}", 366), ("Control", f"{cl:.3f}", 336)], mark, m=(66, 170, 14, 62))
    # dodge the two end labels, which sit 0.006 apart
    body = f"""<div style="padding:34px 20px 0 40px"><div class=h>The small model leads for two thirds of the run</div>
<div class=s>Held-out loss on fineweb-edu. Both start at 2.618.</div>{legend([("Grown 1B", BLUE), ("Control 0.5B, same tokens", ORANGE)])}{svg}</div>"""
    render("04-crossover.png", page(body, w, h), w, h)


# ---------- pilots ----------
def pilots():
    names = [("heal-small-zero", "Zero", BLUE), ("heal-small-llm", "LLM plan", ORANGE), ("heal-small-split", "Split", AQUA), ("heal-small-guided", "Guided", YELLOW)]
    ser, labs = [], []
    ys = {}
    for n, t, c in names:
        pts = [(s * TPS / 1e6, v) for s, v in curve(n) if s <= 2000]
        ser.append((pts, c)); ys[t] = pts[-1][1]
    w, h = 1000, 640
    H, T, B = 470, 14, 62
    Yp = lambda v: T + (2.70 - v) / (2.70 - 2.61) * (H - T - B)
    pos, last = {}, None
    for t in sorted(ys, key=lambda k: ys[k]):        # lowest loss is lowest on the chart
        pass
    order = sorted(ys, key=lambda k: -ys[k])          # top to bottom
    for t in order:
        y = Yp(ys[t]) + 6
        if last is not None and y - last < 26:
            y = last + 26
        pos[t] = y; last = y
    labels = [(t, f"{ys[t]:.3f}", pos[t]) for _, t, _ in names]
    svg = line_chart(ser, 0, 131.072, 2.61, 2.70, w - 40, H, [2.62, 2.64, 2.66, 2.68, 2.70],
                     [(0, "0"), (32.8, "33M"), (65.5, "66M"), (98.3, "98M"), (131.07, "131M")], "Training tokens", labels, "", m=(66, 170, T, B))
    body = f"""<div style="padding:34px 20px 0 40px"><div class=h>Four ways to fill the empty part</div>
<div class=s>Held-out loss, same schedule and data, one seed each. Zero and the LLM plan end level.</div>
{legend([(t, c) for _, t, c in names])}{svg}</div>"""
    render("05-pilots.png", page(body, w, h), w, h)


# ---------- three constructions ----------
def unit(x, y, color, reader, writer, name):
    """A neuron: reader lines in, circle, writer line out."""
    s = ""
    for dy in (-34, 0, 34):
        s += f'<line x1="{x-96}" y1="{y+dy}" x2="{x-20}" y2="{y}" stroke="{reader}" stroke-width="3"/>'
    if writer == "zero":
        s += f'<line x1="{x+20}" y1="{y}" x2="{x+96}" y2="{y}" stroke="{MUTE}" stroke-width="2.5" stroke-dasharray="3 7" stroke-linecap="round"/>'
    else:
        wd = {"full": 7, "half": 3.5}[writer]
        s += f'<line x1="{x+20}" y1="{y}" x2="{x+96}" y2="{y}" stroke="{INK}" stroke-width="{wd}" stroke-linecap="round"/>'
    edge = "#8f8d85" if color == "#ffffff" else SURF
    s += f'<circle cx="{x}" cy="{y}" r="20" fill="{color}" stroke="{edge}" stroke-width="2"/>'
    s += f'<text x="{x}" y="{y+50}" text-anchor="middle" font-size="17" fill="{SEC}">{name}</text>'
    return s


def constructions():
    w, h = 1000, 600
    cols = [("Zero", "Fresh unit", "Random reader, silent writer. Starts blind but free to learn anything.", [("#d9d8d2", "full", "existing", BLUE), None]),
            ("Split", "Exact copy", "Same reader, same writer, same gradient. Twin cosine 1.0000 after 131M tokens.", None),
            ("Reader copy", "LLM plan", "Same reader, silent writer. Fires from step 0, then gets its own gradients.", None)]
    px = [170, 500, 830]
    s = f'<svg width="{w}" height="340" xmlns="http://www.w3.org/2000/svg" font-family="{SANS}" style="display:block;margin-top:14px">'
    # panel 1
    s += unit(px[0], 80, BLUE, BLUE, "full", "existing")
    s += unit(px[0], 240, "#ffffff", "#b8b6ae", "zero", "new")
    # panel 2
    s += unit(px[1], 80, BLUE, BLUE, "half", "existing, output halved")
    s += unit(px[1], 240, BLUE, BLUE, "half", "new, identical twin")
    # panel 3
    s += unit(px[2], 80, BLUE, BLUE, "full", "existing, unchanged")
    s += unit(px[2], 240, ORANGE, BLUE, "zero", "new, shares the reader")
    s += "</svg>"
    txt = "".join(f'<div style="width:290px"><div style="font:700 25px/1.2 {SERIF}">{a}</div><div style="font-size:18px;color:{SEC};line-height:1.35;margin-top:6px">{c}</div></div>' for a, b, c, _ in cols)
    body = f"""<div style="padding:34px 40px 0"><div class=h>Three exact ways to add a neuron</div>
<div class=s>All three leave the model's output unchanged. Thick line: output weights. Dotted: zero.</div>{s}
<div style="display:flex;justify-content:space-between;margin-top:2px">{txt}</div></div>"""
    render("02-constructions.png", page(body, w, h), w, h)


# ---------- scale ----------
def scale():
    md = (RES / "results-small.md").read_text().splitlines()
    head = [c.strip() for c in md[0].strip("|").split("|")]
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in md[2:7]]
    means = {head[i]: sum(float(r[i]) for r in rows) / 5 for i in range(1, len(head))}
    print(means)
    names = {"source": "Source, Qwen2.5-0.5B", "source + same training (control)": "Control, 0.5B + 1B tokens", "grown + healed": "Grown 1B + 1B tokens",
             "pythia-1b": "Pythia-1B", "TinyLlama_v1.1": "TinyLlama-1.1B", "OLMo-2-0425-1B": "OLMo-2-1B", "Qwen2.5-1.5B": "Qwen2.5-1.5B", "SmolLM2-1.7B": "SmolLM2-1.7B"}
    items = sorted(((names[k], means[k], k) for k in names), key=lambda t: -t[1])
    ours = {"source", "source + same training (control)", "grown + healed"}
    w, h = 1000, 545
    L, top, row, maxw = 300, 12, 48, 570
    s = f'<svg width="{w-80}" height="{top+row*len(items)+6}" xmlns="http://www.w3.org/2000/svg" font-family="{SANS}">'
    for i, (n, v, k) in enumerate(items):
        y = top + i * row
        bw = v / 80 * maxw
        col = BLUE if k in ours else "#c9c7bf"
        s += f'<text x="{L-14}" y="{y+18}" text-anchor="end" font-size="20" fill="{INK if k in ours else SEC}" font-weight="{600 if k in ours else 400}">{n}</text>'
        s += f'<path d="M{L},{y} h{bw-4:.1f} a4,4 0 0 1 4,4 v16 a4,4 0 0 1 -4,4 h-{bw-4:.1f} z" fill="{col}"/>'
        s += f'<text x="{L+bw+10:.1f}" y="{y+18}" font-size="20" font-weight="600" fill="{INK}">{v:.1f}</text>'
    s += "</svg>"
    body = f"""<div style="padding:34px 40px 0"><div class=h>A billion tokens is not enough</div>
<div class=s>Mean accuracy on the five benchmarks, in percent. Blue: this experiment.</div><div style="margin-top:22px">{s}</div></div>"""
    render("06-scale.png", page(body, w, h), w, h)


# ---------- information sources ----------
def sources():
    w, h = 1000, 470
    card = lambda t, how, st: (f'<div style="flex:1;border:1px solid {GRID};border-radius:10px;padding:20px 20px 18px;background:#fff">'
                               f'<div style="font:700 25px/1.2 {SERIF}">{t}</div><div style="font-size:19px;color:{SEC};line-height:1.35;margin-top:10px;min-height:104px">{how}</div>'
                               f'<div style="font-size:17px;color:{INK};border-top:1px solid {GRID};padding-top:10px;margin-top:10px;line-height:1.3">{st}</div></div>')
    body = f"""<div style="padding:34px 40px 0"><div class=h>Growth adds no information</div>
<div class=s>Current weights + information = new weights. The information has to come from somewhere.</div>
<div style="display:flex;gap:16px;margin-top:26px">{card("Data", "Through a loss and gradient descent, in millions of small steps.", "Here: healing and the control. Worked, slowly.")}
{card("Another model", "Through its weights or outputs: a written init, or distillation.", "Here: the LLM plan. Level with empty.")}
{card("Verified compute", "Generate, check, train only on what passes.", "Here: not run.")}</div>
<div style="text-align:center;font:400 22px/1.3 {SERIF};color:{SEC};margin-top:24px">The growth step itself contributes none of these. That is why it is exact.</div></div>"""
    render("07-sources.png", page(body, w, h), w, h)


for f in (header, constructions, results, crossover, pilots, scale, sources):
    f()
