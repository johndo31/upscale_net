#!/usr/bin/env python3
"""Render the X post charts for Experiment 06 (Domingos RNN vs. MicroGPT) via headless Chrome.

Every number is loaded directly from:
  ../runs/domingos_results.json       (65.5M tokens, batch 32, 4,000 steps)
  ../runs/domingos_250m_results.json  (250M tokens, batch 64, 7,630 steps)
and checked with asserts. Uses the same editorial palette, typography, and 2x DPI rendering
as 04-neural-graft/x-article/build_graphics.py.
"""
import json
import subprocess
import tempfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent / "runs"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Editorial palette (matched to 04-neural-graft/x-article/build_graphics.py)
INK, SEC, MUTE, GRID, SURF = "#0b0b0b", "#52514e", "#85837d", "#e6e5e0", "#fcfcfb"
BLUE, AQUA, ORANGE, AMBER = "#2a78d6", "#1baf7a", "#eb6834", "#d48a00"
SANS = "-apple-system, 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif"
SERIF = "Georgia, 'Times New Roman', serif"

R65 = json.loads((RUNS / "domingos_results.json").read_text())
R250 = json.loads((RUNS / "domingos_250m_results.json").read_text())

MODELS = [
    ("microgpt_rope", "MicroGPT-RoPE", BLUE, 3.2),
    ("microgpt", "MicroGPT (sinusoidal)", AQUA, 2.8),
    ("domingos_lstm", "Domingos-LSTM", ORANGE, 2.8),
    ("domingos_rnn", "Domingos-RNN", AMBER, 2.8),
]
BUCKETS = ["0-32", "32-64", "64-128", "128-256", "256-512"]
BUCKET_LABELS = ["0–32", "32–64", "64–128", "128–256", "256–512"]

# ---------- sanity checks against run JSONs ----------
finals_250 = {k: round(R250["results"][k]["final"]["val_ppl"], 1) for k, *_ in MODELS}
assert finals_250 == {
    "microgpt_rope": 58.2,
    "microgpt": 66.8,
    "domingos_lstm": 79.4,
    "domingos_rnn": 97.0,
}, finals_250

finals_65 = {k: round(R65["results"][k]["final"]["val_ppl"], 1) for k in ("microgpt", "domingos_lstm", "domingos_rnn")}
assert finals_65 == {"microgpt": 170.9, "domingos_lstm": 125.4, "domingos_rnn": 158.7}, finals_65

# Verify crossover between 62.5M and 93.8M checkpoints in 250M run
h_gpt = R250["results"]["microgpt"]["history"]
h_lstm = R250["results"]["domingos_lstm"]["history"]
assert h_lstm[2]["val_ppl"] < h_gpt[2]["val_ppl"]  # 62.5M: LSTM (127.6) < GPT (146.4)
assert h_gpt[3]["val_ppl"] < h_lstm[3]["val_ppl"]  # 93.8M: GPT (103.0) < LSTM (108.1)


def page(body: str, w: int, h: int, bg: str = SURF) -> str:
    return f"""<!doctype html><meta charset=utf-8><style>
html,body{{margin:0;padding:0;width:{w}px;height:{h}px;background:{bg};overflow:hidden;
font-family:{SANS};color:{INK};-webkit-font-smoothing:antialiased}}
.h{{font:700 32px/1.18 {SERIF};letter-spacing:-.3px}}
.s{{font-size:20px;color:{SEC};line-height:1.35;margin-top:8px}}
.foot{{font-size:16px;color:{MUTE};line-height:1.38}}
</style>{body}"""


def render(name: str, html: str, w: int, h: int) -> None:
    out_path = ROOT / name
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "p.html"
        f.write_text(html)
        subprocess.run(
            [
                CHROME,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                f"--user-data-dir={td}/prof",
                f"--screenshot={out_path}",
                f"--window-size={w},{h}",
                "--force-device-scale-factor=2",
                f"file://{f}",
            ],
            check=True,
            capture_output=True,
            timeout=90,
        )
    print(f"rendered {out_path}")


def legend(items) -> str:
    spans = []
    for label, color in items:
        spans.append(
            f'<span style="display:flex;align-items:center;gap:8px">'
            f'<i style="width:22px;height:4px;border-radius:2px;background:{color};display:block"></i>'
            f"{label}</span>"
        )
    return (
        f'<div style="display:flex;flex-wrap:wrap;gap:8px 26px;margin:14px 0 6px;font-size:18px;color:{SEC}">'
        + "".join(spans)
        + "</div>"
    )


def dodge(items: dict[str, float], gap: float = 24.0) -> dict[str, float]:
    out, last = {}, None
    for k in sorted(items, key=items.get):
        y = items[k]
        if last is not None and y - last < gap:
            y = last + gap
        out[k] = y
        last = y
    return out


def ppl_crossover() -> None:
    w, h = 1000, 660
    cw, ch = 920, 420
    L, Rm, T, B = 68, 92, 22, 62
    pw, ph = cw - L - Rm, ch - T - B
    xmax = 250.0
    ymin, ymax = 40.0, 280.0

    X = lambda m: L + (m / xmax) * pw
    Y = lambda v: T + (ymax - v) / (ymax - ymin) * ph

    # Linear interpolation for exact crossing of sinusoidal MicroGPT and Domingos-LSTM
    t1, t2 = h_gpt[2]["tokens"] / 1e6, h_gpt[3]["tokens"] / 1e6
    g1, g2 = h_gpt[2]["val_ppl"], h_gpt[3]["val_ppl"]
    l1, l2 = h_lstm[2]["val_ppl"], h_lstm[3]["val_ppl"]
    frac = (l1 - g1) / ((g2 - g1) - (l2 - l1))
    cross_x = t1 + frac * (t2 - t1)
    cross_y = g1 + frac * (g2 - g1)

    svg = [f'<svg width="{cw}" height="{ch}" font-family="{SANS}">']

    # Subtle shaded crossover window (62.5M to 93.8M)
    x_win0, x_win1 = X(t1), X(t2)
    svg.append(
        f'<rect x="{x_win0:.1f}" y="{T}" width="{x_win1 - x_win0:.1f}" height="{ph}" '
        f'fill="{GRID}" fill-opacity="0.55"/>'
    )

    # Horizontal gridlines
    for v in (50, 100, 150, 200, 250):
        svg.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{GRID}"/>')
        svg.append(
            f'<text x="{L-12}" y="{Y(v)+6:.1f}" text-anchor="end" font-size="18" fill="{SEC}">{v}</text>'
        )

    # Vertical gridlines & x-axis ticks
    for m in (0, 50, 100, 150, 200, 250):
        svg.append(f'<line x1="{X(m):.1f}" x2="{X(m):.1f}" y1="{T}" y2="{T+ph}" stroke="{GRID}" stroke-opacity="0.6"/>')
        lab = "0" if m == 0 else f"{m}M"
        svg.append(
            f'<text x="{X(m):.1f}" y="{T+ph+28}" text-anchor="middle" font-size="18" fill="{SEC}">{lab}</text>'
        )

    # Axis labels
    svg.append(
        f'<text x="{L+pw/2:.1f}" y="{T+ph+56}" text-anchor="middle" font-size="18" fill="{MUTE}">'
        f"Training tokens (FineWeb-Edu)</text>"
    )
    svg.append(
        f'<text transform="translate(18,{T+ph/2:.1f}) rotate(-90)" text-anchor="middle" '
        f'font-size="18" fill="{MUTE}">Validation PPL</text>'
    )

    # Plot curves (skip step 0 where PPL ~54,000)
    ends = {}
    for key, _, color, sw in MODELS:
        hist = [pt for pt in R250["results"][key]["history"] if pt["step"] > 0]
        pts = [(pt["tokens"] / 1e6, pt["val_ppl"]) for pt in hist]
        poly = " ".join(f"{X(mx):.1f},{Y(vy):.1f}" for mx, vy in pts)
        svg.append(
            f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="{sw}" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for mx, vy in pts:
            svg.append(
                f'<circle cx="{X(mx):.1f}" cy="{Y(vy):.1f}" r="4.2" fill="{color}" stroke="{SURF}" stroke-width="1.5"/>'
            )
        ends[key] = Y(pts[-1][1]) + 6

    # Crossover marker + callout
    cx_px, cy_px = X(cross_x), Y(cross_y)
    svg.append(
        f'<circle cx="{cx_px:.1f}" cy="{cy_px:.1f}" r="6.5" fill="none" stroke="{INK}" stroke-width="2"/>'
    )
    ann_x, ann_y = X(112), Y(210)
    svg.append(
        f'<line x1="{ann_x-6:.1f}" y1="{ann_y+12:.1f}" x2="{cx_px+7:.1f}" y2="{cy_px-7:.1f}" '
        f'stroke="{SEC}" stroke-width="1.5" stroke-dasharray="4 3"/>'
    )
    svg.append(
        f'<text x="{ann_x:.1f}" y="{ann_y:.1f}" font-size="17" font-weight="600" fill="{INK}">'
        f"Crossover (~63–94M tokens)</text>"
    )
    svg.append(
        f'<text x="{ann_x:.1f}" y="{ann_y+21:.1f}" font-size="16" fill="{SEC}">'
        f"Sinusoidal MicroGPT overtakes LSTM</text>"
    )

    # Right-margin final PPL labels
    pos = dodge(ends, gap=23)
    for key, _, color, _ in MODELS:
        val = R250["results"][key]["final"]["val_ppl"]
        svg.append(
            f'<text x="{L+pw+12:.1f}" y="{pos[key]:.1f}" font-size="19" font-weight="700" fill="{color}">'
            f"{val:.1f}</text>"
        )
    svg.append(
        f'<text x="{L+pw+12:.1f}" y="{T+14}" font-size="15" fill="{MUTE}">250M PPL</text>'
    )

    svg.append("</svg>")

    body = f"""<div style="padding:32px 40px 0">
<div class=h>The rotary GPT leads throughout; the sinusoidal GPT passes the LSTM by 94M tokens</div>
<div class=s>Held-out Common Crawl (FineWeb-Edu) validation perplexity, lower is better.</div>
{legend([(name, col) for _, name, col, _ in MODELS])}
<div style="margin-top:6px">{"".join(svg)}</div>
<div class=foot style="margin-top:6px">~30M total params (10.6M non-embedding), 6 layers, d=384, seq_len 512, batch 64, 1 seed. Init PPL (~54k at 0 tokens) is above the y-axis.</div>
</div>"""
    render("01-ppl-crossover.png", page(body, w, h), w, h)


def fmt2(v: float) -> str:
    return str(Decimal(repr(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def pos_loss() -> None:
    w, h = 1000, 660
    cw, ch = 920, 420
    L, Rm, T, B = 68, 112, 22, 62
    pw, ph = cw - L - Rm, ch - T - B
    ymin, ymax = 3.90, 4.85
    n = len(BUCKETS)

    X = lambda i: L + 36 + i * (pw - 72) / (n - 1)
    Y = lambda v: T + (ymax - v) / (ymax - ymin) * ph

    svg = [f'<svg width="{cw}" height="{ch}" font-family="{SANS}">']

    # Horizontal gridlines
    for v in (4.0, 4.2, 4.4, 4.6, 4.8):
        svg.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{GRID}"/>')
        svg.append(
            f'<text x="{L-12}" y="{Y(v)+6:.1f}" text-anchor="end" font-size="18" fill="{SEC}">{v:.1f}</text>'
        )

    # Subtle vertical divider after 32-64 where recurrent models flatten/rise
    x_div = (X(1) + X(2)) / 2
    svg.append(
        f'<line x1="{x_div:.1f}" x2="{x_div:.1f}" y1="{T}" y2="{T+ph}" '
        f'stroke="{MUTE}" stroke-width="1.5" stroke-dasharray="6 5" stroke-opacity="0.7"/>'
    )

    # X-axis ticks
    for i, lab in enumerate(BUCKET_LABELS):
        svg.append(
            f'<text x="{X(i):.1f}" y="{T+ph+28}" text-anchor="middle" font-size="18" fill="{SEC}">{lab}</text>'
        )
    svg.append(
        f'<text x="{L+pw/2:.1f}" y="{T+ph+56}" text-anchor="middle" font-size="18" fill="{MUTE}">'
        f"Token position in 512-token context</text>"
    )
    svg.append(
        f'<text transform="translate(18,{T+ph/2:.1f}) rotate(-90)" text-anchor="middle" '
        f'font-size="18" fill="{MUTE}">Validation loss (nats)</text>'
    )

    # Plot curves
    ends = {}
    for key, _, color, sw in MODELS:
        pl = R250["results"][key]["final"]["pos_loss"]
        ys = [pl[b] for b in BUCKETS]
        poly = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(ys))
        svg.append(
            f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="{sw}" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for i, v in enumerate(ys):
            svg.append(
                f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4.5" fill="{color}" stroke="{SURF}" stroke-width="1.5"/>'
            )
        ends[key] = Y(ys[-1]) + 6

    # Callout annotations
    svg.append(
        f'<text x="{X(2.9):.1f}" y="{Y(4.70):.1f}" text-anchor="middle" font-size="17" font-weight="600" fill="{ORANGE}">'
        f"RNN &amp; LSTM flatten or rise past pos 64</text>"
    )
    svg.append(
        f'<text x="{X(2.9):.1f}" y="{Y(3.94):.1f}" text-anchor="middle" font-size="17" font-weight="600" fill="{BLUE}">'
        f"Both GPTs improve monotonically across all 512 tokens</text>"
    )

    # Right-margin end labels (256-512 loss)
    pos = dodge(ends, gap=24)
    for key, _, color, _ in MODELS:
        val = R250["results"][key]["final"]["pos_loss"]["256-512"]
        svg.append(
            f'<text x="{L+pw+12:.1f}" y="{pos[key]:.1f}" font-size="19" font-weight="700" fill="{color}">'
            f"{fmt2(val)}</text>"
        )
    svg.append(
        f'<text x="{L+pw+12:.1f}" y="{T+14}" font-size="15" fill="{MUTE}">pos 256–512</text>'
    )

    svg.append("</svg>")

    body = f"""<div style="padding:32px 40px 0">
<div class=h>Transformers keep using longer context; RNNs stall past 64 tokens</div>
<div class=s>Per-position validation cross-entropy loss at 250M training tokens (lower is better).</div>
{legend([(name, col) for _, name, col, _ in MODELS])}
<div style="margin-top:6px">{"".join(svg)}</div>
<div class=foot style="margin-top:6px">Same 131K-token held-out FineWeb-Edu validation split (256 × 512 tokens). At pos 0–32, LSTM (4.63) is ahead of sinusoidal GPT (4.68); by pos 256–512, both GPTs pull well ahead.</div>
</div>"""
    render("02-pos-loss.png", page(body, w, h), w, h)


if __name__ == "__main__":
    ppl_crossover()
    pos_loss()
