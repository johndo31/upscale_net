#!/usr/bin/env python3
"""Render the X article graphics for the neural graft experiment with headless Chrome.

Every number is read at build time from
  ../runs/robust_graft_results.json   (run 1: seeds, tuned baselines, gate, model-written circuits)
  ../runs/natural_graft_results.json  (run 2: addition in normal notation, Qwen uses the result)
and checked against the verified facts sheet with asserts. Same palette, fonts, sizes and rendering
as the weight-writer X article, so the articles read as one series.
"""
import json, subprocess, tempfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "img"
RUNS = ROOT.parent / "runs"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
INK, SEC, MUTE, GRID, SURF = "#0b0b0b", "#52514e", "#85837d", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
LIGHTBLUE = "#93bdf0"
SANS = "-apple-system, 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif"
SERIF = "Georgia, 'Times New Roman', serif"
MONO = "'SF Mono', Menlo, Consolas, monospace"

# ---------- data ----------
R1 = json.loads((RUNS / "robust_graft_results.json").read_text())
R2 = json.loads((RUNS / "natural_graft_results.json").read_text())
PA, PC, PD = R1["part_a"], R1["part_c"], R1["part_d"]
F, G1 = R2["results"]["f"], R2["results"]["g1"]

ARMS = [("base", "Base model", MUTE), ("lora", "LoRA", ORANGE),
        ("sgd_graft", "Trained circuit", YELLOW), ("compiled_graft_deadzone", "Written circuit", BLUE)]
TASKS = [("add_carry_64", "Addition", "carry chains, 64 digits", "cascade_L64"),
         ("real_parens_128", "Balanced brackets", "hard set, length 128", "hard_L128"),
         ("decimal_sorted_64", "Ascending digits", "length 64", "L64")]
D_NAMES = {  # part D task keys -> short plain labels
    "first_tok": "first token", "first_is_0": "first is 0", "first_is_1": "first is 1", "first_is_3": "first is 3",
    "last_is_0": "last is 0", "last_is_2": "last is 2", "has_2": "contains 2", "has_3": "contains 3",
    "has_1_or_2": "contains 1 or 2", "no_0": "no 0", "all_0": "all 0s", "majority_0": "majority of 0s",
    "more_2_than_1": "more 2s than 1s", "bigram_1_2": "1 then 2", "bigram_2_0": "2 then 0",
    "bigram_1_1": "1 then 1", "bigram_3_1": "3 then 1"}
LENS = ["id", "L16", "L32", "L64", "L128"]
LEN_LABELS = ["4–8", "16", "32", "64", "128"]


def pct(v):
    """Fraction -> percent text with one decimal, rounding halves up."""
    return str((Decimal(repr(v)) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def stat(task, arm, key):
    """(mean, min, max) as fractions for one part A cell; the base model is a single run."""
    v = PA[task]["summary"][arm][key]
    return (v, v, v) if isinstance(v, float) else (v["mean"], v["min"], v["max"])


def mean(xs):
    return sum(xs) / len(xs)


def seeds(prefix, field):
    """Run 2 cells named '<prefix>/sN': list of field(cell) over seeds."""
    cells = [v for k, v in G1.items() if k.startswith(prefix + "/s")] or [v for k, v in F.items() if k.startswith(prefix + "/s")]
    assert len(cells) == 3, prefix
    return [field(c) for c in cells]


def fact(prefix, L):
    vals = seeds(prefix, lambda c: c["held_out_by_family"]["fact"][L])
    return mean(vals), min(vals), max(vals)


def full_exact(prefix, L="L64"):
    vals = seeds(prefix, lambda c: c[L]["full_exact"])
    return mean(vals), min(vals), max(vals)


def dloss(v):
    """Loss change text: one decimal from 1 up, three below; anything that rounds to zero is 0.000."""
    if abs(v) < 0.0005:
        return "0.000"
    s = f"{v:+.1f}" if abs(v) >= 1 else f"{v:+.3f}"
    return s.replace("-", "−")


WRITTEN_N = [PA[t]["compiled_graft_deadzone"]["seed=0"]["n_trainable_params"] for t, *_ in TASKS]
GATE_N = {PC["damage"][f"compiled_graft_deadzone_gated|{t}"]["probe"]["n_params"] for t, *_ in TASKS}
EQ = {(t, arm, s): PC["summary"][f"{arm}|{t}"][s]["delta_nll_at_eq"]
      for t, *_ in TASKS for arm in ("lora", "compiled_graft_deadzone", "compiled_graft_deadzone_gated") for s in ("code", "maths")}
PROSE = {(t, arm): PC["summary"][f"{arm}|{t}"]["prose"]["delta_nll"]
         for t, *_ in TASKS for arm in ("compiled_graft_deadzone", "compiled_graft_deadzone_gated")}
NAT = {"base": (F["base"]["L64"]["full_exact"],) * 3, "lora": full_exact("lora_full"),
       "sgd_graft": full_exact("trained_circuit"), "compiled_graft_deadzone": full_exact("written_circuit")}
D_L64 = {t: (PD["summary"][t]["base"]["L64"], PD["summary"][t]["written"]["L64"]) for t in PD["summary"]}

# ---------- sanity checks against the verified facts sheet ----------
P = lambda t, a, k: pct(stat(t, a, k)[0])
assert WRITTEN_N == [15360, 3072, 3072] and GATE_N == {1537}
for t, *_ in TASKS:  # 5 seeds for circuits, 3 for LoRA, 512 cases per test length
    assert PA[t]["summary"]["lora"]["id"]["n_seeds"] == 3
    assert PA[t]["summary"]["sgd_graft"]["id"]["n_seeds"] == PA[t]["summary"]["compiled_graft_deadzone"]["id"]["n_seeds"] == 5
    assert all(c["n_cases"][k] == 512 for c in PA[t]["lora"].values() for k in c["n_cases"] if k not in ("id", "val", "hard_id"))
    assert PA[t]["lora"]["seed=0"]["n_cases"].get("hard_id", 1024) == 1024
    for k in PA[t]["summary"]["compiled_graft_deadzone"]:
        if k not in ("seeds", "lr", "hp_source"):
            assert PA[t]["summary"]["compiled_graft_deadzone"][k]["min"] == 1.0
assert [P("add_carry_64", a, "cascade_L64") for a in ("lora", "sgd_graft")] == ["52.3", "48.3"]
assert [P("add_carry_64", a, k) for k in ("cascade_L16", "cascade_L32") for a in ("lora", "sgd_graft")] == ["53.7", "48.5", "50.1", "49.1"]
assert [P("real_parens_128", a, "hard_L128") for a in ("lora", "sgd_graft")] == ["53.6", "50.1"]
assert [P("real_parens_128", a, "hard_L32") for a in ("lora", "sgd_graft")] == ["76.1", "51.1"]
assert P("real_parens_128", "lora", "hard_id") == "100.0"
assert [P("decimal_sorted_64", a, "L64") for a in ("lora", "sgd_graft")] == ["84.5", "62.9"]
assert [pct(NAT[a][0]) for a, *_ in ARMS] == ["0.0", "0.0", "0.0", "100.0"]
assert [dloss(EQ[t, "compiled_graft_deadzone", "code"]) for t, *_ in TASKS] == ["0.000", "+11.5", "+7.4"]
assert [dloss(EQ[t, "compiled_graft_deadzone", "maths"]) for t, *_ in TASKS] == ["+1.8", "+0.653", "+10.7"]
assert max(abs(EQ[t, "compiled_graft_deadzone_gated", s]) for t, *_ in TASKS for s in ("code", "maths")) == EQ["decimal_sorted_64", "compiled_graft_deadzone_gated", "code"]
assert dloss(EQ["decimal_sorted_64", "compiled_graft_deadzone_gated", "code"]) == "+0.024"
assert [dloss(EQ[t, "lora", "code"]) for t, *_ in TASKS] == ["+0.248", "+0.047", "+0.045"]
assert max(abs(v) for v in PROSE.values()) < 0.005
assert all(all(v == 1.0 for k, v in PC["damage"][f"compiled_graft_deadzone_gated|{t}"]["task_accuracy"].items() if k != "answer_nll")
           for t, *_ in TASKS)
assert [pct(G1["composed_base/chat_result_first"]["held_out_by_family"]["fact"][L]) for L in LENS] == ["56.7", "47.5", "50.8", "49.2", "49.9"]
assert [pct(G1["composed_oracle_text/chat_result_first"]["held_out_by_family"]["fact"][L]) for L in LENS] == ["98.6", "91.3", "95.8", "88.1", "97.9"]
assert [pct(fact("composed_graft_k9/chat_result_first", L)[0]) for L in LENS] == ["99.3", "99.9", "98.9", "99.8", "94.2"]
assert [pct(fact("composed_graft_k5/chat_result_first", L)[0]) for L in LENS] == ["94.7", "85.9", "75.7", "87.0", "72.3"]
assert [pct(fact("composed_graft_k19/chat_result_first", L)[0]) for L in LENS] == ["83.4", "70.6", "68.6", "69.7", "75.7"]
assert [pct(fact("composed_lora/chat_result_first", L)[0]) for L in LENS] == ["100.0", "87.1", "73.8", "61.1", "68.5"]
assert seeds("composed_graft_laststate/chat_result_first", lambda c: c["held_out"]) == [G1["composed_base/chat_result_first"]["held_out"]] * 3
assert len(D_L64) == 17 and set(D_L64) == set(D_NAMES)
assert sum(w == 1.0 for _, w in D_L64.values()) == 14
assert [pct(D_L64[t][1]) for t in ("first_tok", "first_is_0", "has_1_or_2")] == ["72.1", "79.5", "70.7"]
assert all(PD["summary"][t]["written"]["n_trainable_params"] == 6144 for t in D_L64)
assert all(PD["tasks"][t]["written"]["n_cases"]["L64"] == 512 for t in D_L64)
assert all(c[L]["n"] == 512 for k, c in F.items() if k.startswith(("written", "trained", "lora_full")) for L in ("id", "L64"))


def page(body, w, h, bg=SURF):
    return f"""<!doctype html><meta charset=utf-8><style>
html,body{{margin:0;padding:0;width:{w}px;height:{h}px;background:{bg};overflow:hidden;
font-family:{SANS};color:{INK};-webkit-font-smoothing:antialiased}}
.h{{font:700 34px/1.2 {SERIF};letter-spacing:-.3px}} .s{{font-size:21px;color:{SEC};line-height:1.35;margin-top:8px}}
.foot{{font-size:17px;color:{MUTE}}} .pt{{font:700 22px/1.2 {SERIF}}} .ps{{font-size:17px;color:{SEC};margin-top:4px}}
</style>{body}"""


def render(name, html, w, h):
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "p.html"; f.write_text(html)
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={td}/prof",
                        f"--screenshot={OUT / name}", f"--window-size={w},{h}", "--force-device-scale-factor=2",
                        f"file://{f}"], check=True, capture_output=True, timeout=90)
    print("rendered", name)


def legend(items, kind="line"):
    def key(c):
        if kind == "box" and isinstance(c, tuple):
            return f'<i style="width:18px;height:18px;border-radius:4px;background:{c[0]};border:2px solid {c[1]};display:block;box-sizing:border-box"></i>'
        if isinstance(c, str) and c.startswith("dash:"):
            return f'<i style="width:22px;height:0;border-top:3px dashed {c[5:]};display:block"></i>'
        if isinstance(c, str) and c.startswith("band:"):
            return f'<i style="width:22px;height:14px;border-radius:2px;background:{c[5:]};opacity:.25;display:block"></i>'
        return f'<i style="width:22px;height:4px;border-radius:2px;background:{c};display:block"></i>'
    return '<div style="display:flex;flex-wrap:wrap;gap:8px 26px;margin:14px 0 2px;font-size:19px;color:' + SEC + '">' + "".join(
        f'<span style="display:flex;align-items:center;gap:8px">{key(c)}{t}</span>' for t, c in items) + "</div>"


def dodge(items, gap=24):
    """items: {name: y in px}. Push labels apart, top to bottom, so none are closer than gap."""
    out, last = {}, None
    for k in sorted(items, key=items.get):
        y = items[k]
        if last is not None and y - last < gap:
            y = last + gap
        out[k] = y; last = y
    return out


def band_chart(series, xlabels, w, h, xlabel, chance=None, chance_label="", m=(62, 70, 36, 64), label_gap=22, label_dy=22):
    """series: list of (key, colour, [(mean, lo, hi) per x], dashed, width). Categorical x, 0-100% y.
    Shaded band = min to max over seeds; end labels dodged in the right margin."""
    L, Rm, T, B = m
    pw, ph = w - L - Rm, h - T - B
    n = len(xlabels)
    X = lambda i: L + 18 + i * (pw - 36) / (n - 1)
    Y = lambda v: T + (100 - v) / 100 * ph
    o = [f'<svg width="{w}" height="{h}" font-family="{SANS}">']
    for v in (0, 25, 50, 75, 100):
        o.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{GRID}"/>')
        o.append(f'<text x="{L-12}" y="{Y(v)+6:.1f}" text-anchor="end" font-size="18" fill="{SEC}">{v}%</text>')
    for i, lab in enumerate(xlabels):
        o.append(f'<text x="{X(i):.1f}" y="{T+ph+30}" text-anchor="middle" font-size="18" fill="{SEC}">{lab}</text>')
    o.append(f'<text x="{L+pw/2}" y="{T+ph+56}" text-anchor="middle" font-size="18" fill="{MUTE}">{xlabel}</text>')
    if chance is not None:
        o.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(chance):.1f}" y2="{Y(chance):.1f}" stroke="{MUTE}" stroke-width="2" stroke-dasharray="7 6"/>')
        if chance_label:
            o.append(f'<text x="{L+8}" y="{Y(chance)+label_dy:.1f}" font-size="16" fill="{MUTE}">{chance_label}</text>')
    for key, c, pts, dash, sw in series:
        if any(hi - lo > 1e-9 for _, lo, hi in pts):
            up = " ".join(f"{X(i):.1f},{Y(hi*100):.1f}" for i, (_, _, hi) in enumerate(pts))
            dn = " ".join(f"{X(i):.1f},{Y(lo*100):.1f}" for i, (_, lo, _) in reversed(list(enumerate(pts))))
            o.append(f'<polygon points="{up} {dn}" fill="{c}" fill-opacity=".16"/>')
    ends = {}
    for key, c, pts, dash, sw in series:
        d = " ".join(f"{X(i):.1f},{Y(mn*100):.1f}" for i, (mn, _, _) in enumerate(pts))
        da = ' stroke-dasharray="7 6"' if dash else ""
        o.append(f'<polyline points="{d}" fill="none" stroke="{c}" stroke-width="{sw}"{da} stroke-linejoin="round" stroke-linecap="round"/>')
        for i, (mn, _, _) in enumerate(pts):
            o.append(f'<circle cx="{X(i):.1f}" cy="{Y(mn*100):.1f}" r="4.5" fill="{c}" stroke="{SURF}" stroke-width="1.5"/>')
        ends[key] = Y(pts[-1][0] * 100) + 6
    pos = dodge(ends, gap=label_gap)
    for key, c, pts, _, _ in series:
        o.append(f'<text x="{L+pw+10:.1f}" y="{pos[key]:.1f}" font-size="18" font-weight="600" fill="{c}">{pct(pts[-1][0])}</text>')
    o.append("</svg>")
    return "".join(o)


# ---------- header: frozen model with a small circuit attached ----------
def header():
    w, h = 1500, 600
    bg, fill, ring, txt, faint = "#0f1420", "#4f9bf0", "#46506a", "#f4f3ef", "#8f96aa"
    s = [f'<svg width="{w}" height="{h}" style="position:absolute;left:0;top:0" font-family="{SANS}">']
    bx0, bx1, by0, by1 = 800, 1410, 96, 330
    s.append(f'<text x="{(bx0+bx1)/2}" y="{by0-22}" text-anchor="middle" font-size="18" fill="{faint}" letter-spacing="1.5">QWEN2.5-1.5B · ALL LAYERS FROZEN</text>')
    s.append(f'<rect x="{bx0}" y="{by0}" width="{bx1-bx0}" height="{by1-by0}" rx="18" fill="#161d2c" stroke="{ring}" stroke-width="2"/>')
    n = 30  # embeddings, 28 layers, final layer
    x0, x1 = bx0 + 34, bx1 - 34
    step = (x1 - x0) / (n - 1)
    xs = [x0 + i * step for i in range(n)]
    for i, x in enumerate(xs):
        edge = i in (0, n - 1)
        s.append(f'<rect x="{x-5:.1f}" y="{by0+26}" width="10" height="{by1-by0-52}" rx="4" '
                 f'fill="{"#9aa3b8" if edge else "#2a3346"}"/>')
    ex, fx = xs[0], xs[-1]
    cy = 448
    inp = [(1030, cy - 26), (1030, cy + 26)]
    hid = [(1110, cy - 44), (1110, cy), (1110, cy + 44)]
    out = (1190, cy)
    s.append(f'<path d="M{ex:.1f},{by1-26} V{cy} H{inp[0][0]-40}" fill="none" stroke="{fill}" stroke-width="2"/>')
    for p in inp:
        s.append(f'<line x1="{inp[0][0]-40}" y1="{cy}" x2="{p[0]}" y2="{p[1]}" stroke="{fill}" stroke-width="2"/>')
    s.append(f'<path d="M{out[0]},{cy} H{fx:.1f} V{by1-20}" fill="none" stroke="{fill}" stroke-width="2"/>'
             f'<path d="M{fx-7:.1f},{by1-12} L{fx:.1f},{by1-26} L{fx+7:.1f},{by1-12} z" fill="{fill}"/>')
    for a in inp:
        for b in hid:
            s.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="{fill}" stroke-opacity=".55" stroke-width="1.6"/>')
    for b in hid:
        s.append(f'<line x1="{b[0]}" y1="{b[1]}" x2="{out[0]}" y2="{out[1]}" stroke="{fill}" stroke-opacity=".55" stroke-width="1.6"/>')
    for x, y in inp:
        s.append(f'<circle cx="{x}" cy="{y}" r="11" fill="{bg}" stroke="{fill}" stroke-width="2.5"/>')
    for x, y in hid:
        s.append(f'<circle cx="{x}" cy="{y}" r="12" fill="{fill}"/>')
    s.append(f'<circle cx="{out[0]}" cy="{out[1]}" r="12" fill="{txt}"/>')
    s.append(f'<text x="{ex+14:.1f}" y="{by1+40}" font-size="17" fill="{faint}">token embeddings</text>')
    s.append(f'<text x="{fx-14:.1f}" y="{by1+40}" text-anchor="end" font-size="17" fill="{faint}">final layer</text>')
    s.append(f'<text x="{hid[0][0]}" y="{cy+88}" text-anchor="middle" font-size="18" fill="{txt}">written circuit</text>')
    s.append("</svg>")
    t = f"""<div style="position:absolute;left:70px;top:0;height:{h}px;width:660px;display:flex;flex-direction:column;justify-content:center;color:{txt}">
<div style="font:700 74px/1.06 {SERIF};letter-spacing:-1px">Neural Graft</div>
<div style="font:400 31px/1.32 {SERIF};color:#c7cad4;margin-top:24px">An exact skill, plugged into a frozen LLM, where fine-tuning fails</div>
<div style="font-size:19px;line-height:1.45;color:{faint};margin-top:34px">Qwen2.5-1.5B · trained on lengths 4–8 · exact up to 128</div>
<div style="font-size:16px;color:#5d6478;margin-top:18px">Niklas · drniklas.com</div></div>"""
    render("01-header.png", page("".join(s) + t, w, h, bg), w, h)


# ---------- wiring diagram ----------
FROZEN = ("#eeede8", MUTE)
WRITTEN = ("#e4eefb", BLUE)
TRAINED = ("#ddf3ea", AQUA)
PLAIN = ("#ffffff", GRID)


def wiring():
    w, h = 1000, 680
    lo, hi = min(WRITTEN_N), max(WRITTEN_N)
    my, mh = 222, 74            # main row centre and height
    sy, sh = 532, 120           # side row centre and height
    px, pr = 720, 18            # the (+)
    gx0, gx1, gy0, gy1 = 590, 960, 296, 410   # gate box
    qx, qw = 300, 236           # Qwen box
    l9 = qx + qw * 9 / 28       # x of layer 9 on the Qwen box
    main = [("Tokens", 40, 84, PLAIN), ("Token embeddings", 150, 124, FROZEN),
            ("Qwen2.5-1.5B,<br>all 28 layers frozen", qx, qw, FROZEN), ("Last hidden state", 566, 114, FROZEN),
            ("Output layer", 768, 96, FROZEN), ("Answer", 892, 68, PLAIN)]
    side = [("Read-in", "fitted from the embedding table, no labels, then clamped to 0/1", 100, 236, WRITTEN),
            ("Circuit", "weights written directly, frozen", 362, 200, WRITTEN),
            ("Write-out", f"trained ({lo:,} to {hi:,} numbers)", 590, 230, TRAINED)]
    o = []
    box = lambda x, y, bw, bh, f, st, inner, extra="": (
        f'<div style="position:absolute;left:{x}px;top:{y}px;width:{bw}px;height:{bh}px;box-sizing:border-box;'
        f'background:{f};border:2px solid {st};border-radius:10px;{extra}">{inner}</div>')
    for t, x, bw, (f, st) in main:
        o.append(box(x, my - mh / 2, bw, mh, f, st, t, "display:flex;align-items:center;justify-content:center;"
                     "text-align:center;font-size:17px;line-height:1.25;padding:0 8px"))
    for t, sub, x, bw, (f, st) in side:
        o.append(box(x, sy - sh / 2, bw, sh, f, st,
                     f'<div style="font-size:19px;font-weight:700">{t}</div>'
                     f'<div style="font-size:16px;line-height:1.3;color:{SEC};margin-top:6px">{sub}</div>', "padding:14px 14px 0"))
    gate_n = next(iter(GATE_N))
    o.append(box(gx0, gy0, gx1 - gx0, gy1 - gy0, *TRAINED,
                 f'<div style="font-size:19px;font-weight:700">Gate</div>'
                 f'<div style="font-size:16px;line-height:1.3;color:{SEC};margin-top:5px">read from Qwen\'s own hidden state, decides '
                 f'when the circuit may speak; {gate_n:,} trained numbers</div>', "padding:11px 14px 0"))
    svg = [f'<svg width="{w}" height="{h}" style="position:absolute;left:0;top:0" font-family="{SANS}"><defs>']
    for mid, col in (("a", SEC), ("b", BLUE), ("g", AQUA)):
        svg.append(f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
                   f'<path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
    svg.append("</defs>")
    arrow = lambda x1, y1, x2, y2, c=SEC, m="a": svg.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="2" marker-end="url(#{m})"/>')
    ends = [(x, x + bw) for _, x, bw, _ in main]
    for (a0, a1), (b0, b1) in zip(ends, ends[1:]):
        if a1 == 680:     # last hidden state -> (+) -> output layer
            arrow(a1 + 2, my, px - pr - 2, my); arrow(px + pr + 2, my, b0 - 2, my)
        else:
            arrow(a1 + 2, my, b0 - 2, my)
    ex = 150 + 124 / 2
    lh = 566 + 114 / 2
    arrow(ex, my + mh / 2 + 2, ex, sy - sh / 2 - 2, BLUE, "b")                 # embeddings -> read-in
    arrow(100 + 236 + 2, sy, 362 - 2, sy, BLUE, "b")                          # read-in -> circuit
    arrow(362 + 200 + 2, sy, 590 - 2, sy, BLUE, "b")                          # circuit -> write-out
    arrow(px, sy - sh / 2 - 2, px, gy1 + 2, AQUA, "g")                        # write-out -> gate
    arrow(px, gy0 - 2, px, my + pr + 2, AQUA, "g")                            # gate -> (+)
    arrow(lh, my + mh / 2 + 2, lh, gy0 - 2, SEC, "a")                         # last hidden state -> gate
    svg.append(f'<circle cx="{px}" cy="{my}" r="{pr}" fill="#fff" stroke="{INK}" stroke-width="2"/>'
               f'<path d="M{px-8},{my} H{px+8} M{px},{my-8} V{my+8}" stroke="{INK}" stroke-width="2.5"/>')
    # dashed alternative: write-out result injected at layer 9 instead
    yj = 438
    svg.append(f'<path d="M610,{sy-sh/2-2} V{yj} H{l9:.1f} V{my+mh/2+3}" fill="none" stroke="{AQUA}" stroke-width="2" '
               f'stroke-dasharray="7 5" marker-end="url(#g)"/>')
    for i, line in enumerate(("or injected at layer 9 of 28,", "so Qwen's upper layers", "use the result")):
        svg.append(f'<text x="{l9+12:.1f}" y="{318+i*20}" font-size="16" fill="{INK}">{line}</text>')
    svg.append("</svg>")
    body = f"""<div style="padding:34px 40px 0"><div class=h>The model stays frozen; a circuit is wired beside it</div>
<div class=s>The circuit reads the token embeddings. A trained gate decides when its answer is added.</div>
{legend([("Frozen model", FROZEN), ("Set without training, then frozen", WRITTEN), ("Trained", TRAINED), ("Alternative wiring", "dash:" + AQUA)], "box")}</div>
{"".join(o)}{"".join(svg)}
<div class=foot style="position:absolute;left:40px;top:{h-44}px">Qwen2.5-1.5B. Trained on lengths 4–8. Qwen's own weights never change.</div>"""
    render("02-wiring.png", page(body, w, h), w, h)


# ---------- results table ----------
def table(head, rows, bold):
    """head: column titles (first is the row label). rows: label + cells (html). bold: set of (row, col)."""
    th = f"text-align:right;font-weight:600;padding:0 0 14px 18px;font-size:19px;color:{SEC};vertical-align:bottom;line-height:1.25"
    td = f"text-align:right;padding:10px 0 10px 18px;font-size:23px;font-variant-numeric:tabular-nums;border-top:1px solid {GRID};vertical-align:top"
    out = f'<table style="border-collapse:collapse;width:100%;margin-top:24px"><tr><th style="{th};text-align:left;padding-left:0">{head[0]}</th>'
    out += "".join(f'<th style="{th}">{c}</th>' for c in head[1:]) + "</tr>"
    for r, row in enumerate(rows):
        out += f'<tr><td style="padding:10px 0;font-size:21px;border-top:1px solid {GRID};line-height:1.2;vertical-align:top">{row[0]}</td>'
        for c, v in enumerate(row[1:], 1):
            st = f";font-weight:700;color:{BLUE}" if (r, c) in bold else ""
            out += f'<td style="{td}{st}">{v}</td>'
        out += "</tr>"
    return out + "</table>"


def results():
    small = f"font-size:15px;color:{MUTE};font-weight:400;margin-top:2px"
    cells = []   # per row: list of (mean, lo, hi) per arm
    labels = []
    for t, name, sub, key in TASKS:
        cells.append([stat(t, a, key) for a, *_ in ARMS]); labels.append((name, sub))
    cells.append([NAT[a] for a, *_ in ARMS]); labels.append(("Addition in normal notation", "whole sum, 64 digits"))
    rows, bold = [], set()
    for r, ((name, sub), vals) in enumerate(zip(labels, cells)):
        row = [f'{name}<div style="font-size:16px;color:{MUTE};margin-top:2px">{sub}</div>']
        for a, (mn, lo, hi) in zip(ARMS, vals):
            txt = f"{pct(mn)}%"
            if a[0] in ("lora", "sgd_graft") and (hi - lo) * 100 > 2:
                txt += f'<div style="{small}">{pct(lo)}–{pct(hi)}</div>'
            row.append(txt)
        rows.append(row)
        best = max(v[0] for v in vals)
        bold |= {(r, c) for c, v in enumerate(vals, 1) if v[0] == best}
    w, h = 1000, 560
    body = f"""<div style="padding:34px 40px">
<div class=h>Only the written circuit is exact on every task</div>
<div class=s>Accuracy on the longest, hardest test. Mean over seeds; small type: range over seeds.</div>
{table([""] + [n for _, n, _ in ARMS], rows, bold)}
<div class=foot style="margin-top:16px;line-height:1.4">Blue: best in the row. Trained on lengths 4–8. 512 test cases per cell. Baselines tuned on in-range data.<br>5 seeds for circuits, 3 for LoRA (3 each for normal notation).</div></div>"""
    render("03-results.png", page(body, w, h), w, h)


# ---------- length: carry chains and hard brackets, with seed ranges ----------
def length():
    w, h = 1000, 720
    arms = ARMS[1:]
    left_keys = ["cascade_L16", "cascade_L32", "cascade_L64"]
    right_keys = ["hard_id", "hard_L32", "hard_L128"]
    mk = lambda t, keys: [(a, c, [stat(t, a, k) for k in keys], False, 3 if a.startswith("compiled") else 2.5) for a, _, c in arms]
    left = band_chart(mk("add_carry_64", left_keys), ["16", "32", "64"], 455, 470, "Digits", 50, "guessing the carry", label_dy=46)
    right = band_chart(mk("real_parens_128", right_keys), ["4–8", "32", "128"], 455, 470, "Length", 50, "")
    body = f"""<div style="padding:34px 30px 0 40px"><div class=h>Beyond training lengths, trained models guess</div>
<div class=s>The written circuit stays at 100% on every seed. Dashed: 50%, the level of guessing.</div>
{legend([(n, c) for _, n, c in arms] + [("range over seeds", "band:" + INK)])}
<div style="display:flex;gap:20px;margin-top:14px">
<div><div class=pt>Addition, carry chains</div><div class=ps>Does the top digit carry?</div>{left}</div>
<div><div class=pt>Balanced brackets, hard set</div><div class=ps>Counting brackets does not help</div>{right}</div></div>
<div class=foot style="margin-top:0">Trained on lengths 4–8. 512 test cases per point (1,024 in the training range). 5 seeds for circuits, 3 for LoRA.</div></div>"""
    render("04-length.png", page(body, w, h), w, h)


# ---------- gate: loss at "=" positions ----------
def gate():
    w, h = 1000, 740
    arms = [("lora", "LoRA", ORANGE), ("compiled_graft_deadzone", "Written circuit, no gate", LIGHTBLUE),
            ("compiled_graft_deadzone_gated", "Written circuit + gate", BLUE)]
    vmax = 12

    def panel(s):
        cw, ch = 450, 450
        L, pw = 10, 330
        X = lambda v: L + v / vmax * pw
        o = [f'<svg width="{cw}" height="{ch}" font-family="{SANS}">']
        top, grp, bar, gap = 30, 128, 22, 6
        bottom = top + grp * 3 - 18
        for v in (0, 4, 8, 12):
            o.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{top-14}" y2="{bottom}" stroke="{INK if v == 0 else GRID}" stroke-width="{1.5 if v == 0 else 1}"/>')
            o.append(f'<text x="{X(v):.1f}" y="{bottom+24}" text-anchor="middle" font-size="17" fill="{SEC}">{"+" if v else ""}{v}</text>')
        for i, (t, name, _, _) in enumerate(TASKS):
            y = top + i * grp
            o.append(f'<text x="{L+8}" y="{y+2}" font-size="18" fill="{INK}">{name}</text>')
            for j, (a, _, c) in enumerate(arms):
                d = EQ[t, a, s]
                yy = y + 14 + j * (bar + gap)
                o.append(f'<rect x="{L}" y="{yy}" width="{max(X(max(d, 0)) - L, 2):.1f}" height="{bar}" rx="2" fill="{c}"/>')
                o.append(f'<text x="{X(max(d, 0)) + 7:.1f}" y="{yy+17}" font-size="17" font-weight="600" fill="{INK}">{dloss(d)}</text>')
        o.append(f'<text x="{L+pw/2}" y="{bottom+50}" text-anchor="middle" font-size="17" fill="{MUTE}">Change in loss at “=” (higher is worse)</text>')
        o.append("</svg>")
        return "".join(o)
    n_eq = {s: PC["summary"]["base"][s]["n_eq"] for s in ("code", "maths")}
    gmax = max(EQ[t, "compiled_graft_deadzone_gated", s] for t, *_ in TASKS for s in ("code", "maths"))
    body = f"""<div style="padding:34px 30px 0 40px"><div class=h>Without a gate the circuit breaks text with “=”</div>
<div class=s>Change in loss at “=” positions vs the base model. With the gate: 0.000 (max {dloss(gmax)}).</div>
{legend([(n, c) for _, n, c in arms])}
<div style="display:flex;gap:40px;margin-top:16px">
<div><div class=pt>Python code</div><div class=ps>{n_eq["code"]:,} “=” positions</div>{panel("code")}</div>
<div><div class=pt>Maths lines</div><div class=ps>{n_eq["maths"]:,} “=” positions</div>{panel("maths")}</div></div>
<div class=foot style="margin-top:12px">Prose without “=” is unchanged by the circuit in both cases. Task accuracy with the gate stays 100%.</div></div>"""
    render("05-gate.png", page(body, w, h), w, h)


# ---------- Qwen uses the result ----------
def uses_result():
    w, h = 1000, 752
    base = G1["composed_base/chat_result_first"]["held_out_by_family"]["fact"]
    told = G1["composed_oracle_text/chat_result_first"]["held_out_by_family"]["fact"]
    single = lambda d: [(d[L],) * 3 for L in LENS]
    series = [("told", INK, single(told), True, 2),
              ("k9", BLUE, [fact("composed_graft_k9/chat_result_first", L) for L in LENS], False, 3),
              ("lora", ORANGE, [fact("composed_lora/chat_result_first", L) for L in LENS], False, 2.5),
              ("base", MUTE, single(base), False, 2.5)]
    left = band_chart(series, LEN_LABELS, 530, 470, "Length of the bracket string", 50, "", label_gap=22)
    # right: accuracy at length 64 by injection depth
    depth = [("Layer<br>5", fact("composed_graft_k5/chat_result_first", "L64")),
             ("Layer<br>9", fact("composed_graft_k9/chat_result_first", "L64")),
             ("Layer<br>19", fact("composed_graft_k19/chat_result_first", "L64")),
             ("Before<br>output", (mean(seeds("composed_graft_laststate/chat_result_first", lambda c: c["held_out"]["L64"])),) * 3)]
    bw_, bh = 376, 470
    L2, R2, T2, B2 = 62, 10, 36, 64
    pw2, ph2 = bw_ - L2 - R2, bh - T2 - B2
    Y2 = lambda v: T2 + (100 - v) / 100 * ph2
    o = [f'<svg width="{bw_}" height="{bh}" font-family="{SANS}">']
    for v in (0, 25, 50, 75, 100):
        o.append(f'<line x1="{L2}" x2="{L2+pw2}" y1="{Y2(v):.1f}" y2="{Y2(v):.1f}" stroke="{GRID}"/>')
        o.append(f'<text x="{L2-10}" y="{Y2(v)+6:.1f}" text-anchor="end" font-size="18" fill="{SEC}">{v}%</text>')
    slot = pw2 / 4
    for i, (lab, (mn, lo, hi)) in enumerate(depth):
        cx = L2 + slot * (i + 0.5)
        c = BLUE if i < 3 else MUTE
        o.append(f'<rect x="{cx-slot*0.3:.1f}" y="{Y2(mn*100):.1f}" width="{slot*0.6:.1f}" height="{Y2(0)-Y2(mn*100):.1f}" rx="3" fill="{c}"/>')
        if hi - lo > 1e-9:
            o.append(f'<path d="M{cx:.1f},{Y2(lo*100):.1f} V{Y2(hi*100):.1f} M{cx-6:.1f},{Y2(lo*100):.1f} H{cx+6:.1f} M{cx-6:.1f},{Y2(hi*100):.1f} H{cx+6:.1f}" stroke="{INK}" stroke-width="1.5"/>')
        o.append(f'<text x="{cx:.1f}" y="{Y2(hi*100)-9:.1f}" text-anchor="middle" font-size="18" font-weight="600" fill="{INK}">{pct(mn)}</text>')
        for j, line in enumerate(lab.split("<br>")):
            o.append(f'<text x="{cx:.1f}" y="{T2+ph2+28+j*21}" text-anchor="middle" font-size="17" fill="{SEC}">{line}</text>')
    o.append(f'<line x1="{L2}" x2="{L2+pw2}" y1="{Y2(50):.1f}" y2="{Y2(50):.1f}" stroke="{MUTE}" stroke-width="2" stroke-dasharray="7 6"/>')
    o.append("</svg>")
    right = "".join(o)
    body = f"""<div style="padding:34px 30px 0 40px"><div class=h>Qwen's frozen layers use the circuit's result</div>
<div class=s>Questions like “If the brackets balance, name the capital of France, else of Japan.”</div>
{legend([("Told the result in words", "dash:" + INK), ("Injected at layer 9", BLUE), ("LoRA", ORANGE), ("Base model", MUTE), ("range over seeds", "band:" + INK)])}
<div style="display:flex;gap:24px;margin-top:14px">
<div><div class=pt>Accuracy by length</div><div class=ps>Dashed grey: 50%, guessing</div>{left}</div>
<div><div class=pt>Where to inject</div><div class=ps>Length 64</div>{right}</div></div>
<div class=foot style="margin-top:0;line-height:1.4">5 unseen fact questions, 3 seeds. Facts in the test never appear in training.<br>Before output: the original wiring, where the result cannot reach Qwen's layers.</div></div>"""
    render("06-uses-result.png", page(body, w, h), w, h)


# ---------- circuits written by a model ----------
def model_written():
    order = sorted(D_L64, key=lambda t: (-D_L64[t][1], D_NAMES[t]))
    n100 = sum(D_L64[t][1] == 1.0 for t in order)
    row_h, top = 30, 40
    cw = 920
    L, Rm = 200, 70
    pw = cw - L - Rm
    X = lambda v: L + v / 100 * pw
    ch = top + row_h * len(order) + 56
    o = [f'<svg width="{cw}" height="{ch}" font-family="{SANS}">']
    bottom = top + row_h * len(order) - 8
    for v in (0, 25, 50, 75, 100):
        o.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{top-18}" y2="{bottom}" stroke="{GRID}"/>')
        o.append(f'<text x="{X(v):.1f}" y="{bottom+24}" text-anchor="middle" font-size="17" fill="{SEC}">{v}%</text>')
    o.append(f'<text x="{L+pw+14}" y="{top-14}" font-size="15" fill="{MUTE}">written</text>')
    for i, t in enumerate(order):
        b, wv = (v * 100 for v in D_L64[t])
        y = top + i * row_h
        o.append(f'<text x="{L-16}" y="{y+6}" text-anchor="end" font-size="18" fill="{INK}">{D_NAMES[t]}</text>')
        o.append(f'<line x1="{X(min(b, wv)):.1f}" x2="{X(max(b, wv)):.1f}" y1="{y}" y2="{y}" stroke="{GRID}" stroke-width="3"/>')
        o.append(f'<circle cx="{X(b):.1f}" cy="{y}" r="6.5" fill="{MUTE}"/>')
        o.append(f'<circle cx="{X(wv):.1f}" cy="{y}" r="7" fill="{BLUE if wv == 100 else ORANGE}" stroke="{SURF}" stroke-width="1.5"/>')
        o.append(f'<text x="{L+pw+14}" y="{y+6}" font-size="17" font-weight="600" fill="{INK}">{pct(wv/100)}</text>')
        if b == wv:
            o.append(f'<text x="{X(wv)+16:.1f}" y="{y+5}" font-size="15" fill="{MUTE}">same as base model</text>')
    o.append(f'<text x="{L+pw/2}" y="{bottom+50}" text-anchor="middle" font-size="17" fill="{MUTE}">Accuracy at length 64</text>')
    o.append("</svg>")
    w, h = 1000, ch + 230
    body = f"""<div style="padding:34px 40px 0"><div class=h>{n100} of {len(order)} model-written circuits graft at 100%</div>
<div class=s>Written by a fine-tuned model in an earlier experiment, wired into frozen Qwen2.5-1.5B.</div>
{legend([("Base model", MUTE), ("Written circuit, 100%", BLUE), ("Written circuit, below 100%", ORANGE)])}
<div style="margin-top:12px">{"".join(o)}</div>
<div class=foot style="margin-top:0">Synthetic tokens (digits 0 to 3). 6,144 trained numbers per circuit. 512 test cases per task.</div></div>"""
    render("07-model-written.png", page(body, w, h), w, h)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    keep = {"01-header.png", "02-wiring.png", "03-results.png", "04-length.png", "05-gate.png",
            "06-uses-result.png", "07-model-written.png"}
    for p in OUT.glob("*.png"):
        if p.name not in keep:
            p.unlink(); print("removed stale", p.name)
    for f in (header, wiring, results, length, gate, uses_result, model_written):
        f()
