"""
Build an offline HTML report for Test B (the separate hard-case test set).

Reads : results/test_b_scores.csv   (one row per message, scam scores per run)
Writes: results/test_b/results.html

Run: .venv/bin/python make_test_b_report.py
"""

import html
import json
from pathlib import Path

import pandas as pd

from text_utils import prepare

SRC = Path("results/test_b_scores.csv")
OUT = Path("results/test_b/results.html")
RUNS = [c for c in ["run2", "run3", "run4"] if c in pd.read_csv(SRC, nrows=1).columns]

df = pd.read_csv(SRC, keep_default_na=False)
df["has_link"] = df["text"].map(prepare).str.contains("<URL>", regex=False)
y = df["label"].eq("scam")


def score(pred, mask):
    yy, p = y[mask], pred[mask]
    tp, fp = int((p & yy).sum()), int((p & ~yy).sum())
    fn, tn = int((~p & yy).sum()), int((~p & ~yy).sum())
    n_scam, n_safe = tp + fn, fp + tn
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / n_scam if n_scam else 0
    return {"acc": (tp + tn) / max(1, len(yy)), "prec": prec, "rec": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0,
            "fa": fp / n_safe if n_safe else 0, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_scam": n_scam, "n_safe": n_safe}


groups = {"All": slice(None), "Easy": df["difficulty"].eq("easy"), "Hard": df["difficulty"].eq("hard"),
          "With link": df["has_link"], "No link": ~df["has_link"]}
stats = {r: {g: score(df[r] > 0.5, m) for g, m in groups.items()} for r in RUNS}
best = max(RUNS, key=lambda r: stats[r]["All"]["acc"])

rows = [{"id": r["id"], "text": r["text"], "label": r["label"], "difficulty": r["difficulty"],
         "country": r["country"], "category": r["category"], "why": r["why"], "link": bool(r["has_link"]),
         "p": {k: round(float(r[k]), 3) for k in RUNS}} for _, r in df.iterrows()]


def pct(x):
    return f"{x:.1%}"


# ---------- tiles ----------
tiles = ""
for r in RUNS:
    s = stats[r]["All"]
    tiles += f"""<div class="tile{' best' if r == best else ''}">
  <div class="tname">{r.upper()}{' · best' if r == best else ''}</div>
  <div class="big">{pct(s['acc'])}</div><div class="sub">accuracy</div>
  <div class="row"><span>Scams caught</span><b>{s['tp']}/{s['n_scam']} ({s['rec']:.0%})</b></div>
  <div class="row"><span>False alarms</span><b>{s['fp']}/{s['n_safe']} ({s['fa']:.0%})</b></div>
  <div class="row"><span>Precision</span><b>{pct(s['prec'])}</b></div>
  <div class="row"><span>F1</span><b>{pct(s['f1'])}</b></div>
</div>"""

# ---------- grouped bars (SVG) ----------
def bars(metric, title, good_high=True):
    W, H, pad_l, pad_b, pad_t = 560, 230, 40, 34, 14
    gw = (W - pad_l) / len(groups)
    bw = gw / (len(RUNS) + 1.2)
    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">']
    for t in [0, 0.25, 0.5, 0.75, 1]:
        yy = pad_t + (H - pad_t - pad_b) * (1 - t)
        svg.append(f'<line x1="{pad_l}" x2="{W}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid"/>'
                   f'<text x="{pad_l - 6}" y="{yy + 4:.1f}" class="axis" text-anchor="end">{t:.0%}</text>')
    for gi, g in enumerate(groups):
        x0 = pad_l + gi * gw + bw * 0.6
        for ri, r in enumerate(RUNS):
            v = stats[r][g][metric]
            h = (H - pad_t - pad_b) * v
            x = x0 + ri * bw
            svg.append(f'<rect x="{x:.1f}" y="{H - pad_b - h:.1f}" width="{bw - 3:.1f}" height="{h:.1f}" '
                       f'class="b{ri}" rx="2"><title>{r} · {g}: {v:.1%}</title></rect>')
        svg.append(f'<text x="{pad_l + gi * gw + gw / 2:.1f}" y="{H - 12}" class="axis" text-anchor="middle">{g}</text>')
    svg.append("</svg>")
    return f'<div class="card"><h3>{title}</h3>{"".join(svg)}</div>'

legend = "".join(f'<span class="lg"><i class="b{i}"></i>{r.upper()}</span>' for i, r in enumerate(RUNS))
charts = (bars("acc", "Accuracy") + bars("rec", "Scams caught (recall)")
          + bars("fa", "False alarms (lower is better)"))

# ---------- confusion grids ----------
conf = ""
for r in RUNS:
    s = stats[r]["All"]
    conf += f"""<div class="card conf"><h3>{r.upper()}</h3>
<table><tr><th></th><th>Said safe</th><th>Said scam</th></tr>
<tr><th>Really safe</th><td class="ok">{s['tn']}</td><td class="bad">{s['fp']}</td></tr>
<tr><th>Really scam</th><td class="bad">{s['fn']}</td><td class="ok">{s['tp']}</td></tr></table></div>"""

# ---------- per-group table ----------
gt = "<table class='gt'><tr><th>Group</th><th>Messages</th>" + "".join(
    f"<th>{r.upper()} acc</th><th>{r.upper()} caught</th><th>{r.upper()} false alarms</th>" for r in RUNS) + "</tr>"
for g, m in groups.items():
    n = len(df[m]) if not isinstance(m, slice) else len(df)
    gt += f"<tr><td>{g}</td><td>{n}</td>" + "".join(
        f"<td>{pct(stats[r][g]['acc'])}</td><td>{stats[r][g]['rec']:.0%}</td><td>{stats[r][g]['fa']:.0%}</td>"
        for r in RUNS) + "</tr>"
gt += "</table>"

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Test B Results</title>
<style>
:root {{ --bg:#f7f7f5; --card:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#e3e2dc;
  --c0:#a3a3a0; --c1:#5b8def; --c2:#1f9d74; --ok:#e3f4ec; --okt:#16704f; --bad:#fbe6e4; --badt:#a8322a; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#151514; --card:#1f1f1d; --ink:#ecebe6; --muted:#9a9993;
  --line:#33332f; --c0:#6f6f6b; --c1:#6e9cf2; --c2:#33b88c; --ok:#163428; --okt:#7fd8b3; --bad:#3b1d1a; --badt:#f19a92; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 -apple-system, system-ui, sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 60px; }}
h1 {{ margin:0 0 4px; font-size:26px; }} h2 {{ margin:34px 0 12px; font-size:19px; }} h3 {{ margin:0 0 8px; font-size:15px; }}
.lead {{ color:var(--muted); margin:0 0 20px; }}
.tiles, .charts, .confs {{ display:grid; gap:14px; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); }}
.tile, .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; }}
.tile.best {{ border-color:var(--c2); box-shadow:0 0 0 1px var(--c2) inset; }}
.tname {{ font-weight:600; color:var(--muted); font-size:13px; letter-spacing:.04em; }}
.big {{ font-size:34px; font-weight:700; margin-top:4px; }} .sub {{ color:var(--muted); margin:-4px 0 10px; font-size:13px; }}
.row {{ display:flex; justify-content:space-between; border-top:1px solid var(--line); padding:6px 0; font-size:14px; }}
svg {{ width:100%; height:auto; }} .grid {{ stroke:var(--line); }} .axis {{ fill:var(--muted); font-size:11px; }}
.b0 {{ fill:var(--c0); background:var(--c0); }} .b1 {{ fill:var(--c1); background:var(--c1); }} .b2 {{ fill:var(--c2); background:var(--c2); }}
.legend {{ display:flex; gap:16px; margin:0 0 10px; color:var(--muted); font-size:13px; }}
.lg i {{ display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:6px; vertical-align:-1px; }}
.conf table {{ border-collapse:collapse; width:100%; }} .conf th, .conf td {{ padding:8px; text-align:center; font-size:13px; }}
.conf td {{ font-size:20px; font-weight:700; border-radius:8px; }} .ok {{ background:var(--ok); color:var(--okt); }} .bad {{ background:var(--bad); color:var(--badt); }}
.scroll {{ overflow-x:auto; }}
.gt {{ border-collapse:collapse; width:100%; background:var(--card); font-size:13px; }}
.gt th, .gt td {{ border-bottom:1px solid var(--line); padding:7px 9px; text-align:right; white-space:nowrap; }}
.gt th:first-child, .gt td:first-child {{ text-align:left; }}
.filters {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:10px; }}
select, input {{ background:var(--card); color:var(--ink); border:1px solid var(--line); border-radius:8px; padding:6px 9px; font-size:14px; }}
input {{ flex:1; min-width:180px; }}
.mt {{ border-collapse:collapse; width:100%; background:var(--card); font-size:13px; }}
.mt th, .mt td {{ border-bottom:1px solid var(--line); padding:8px; vertical-align:top; text-align:left; }}
.mt td.msg {{ min-width:260px; white-space:pre-wrap; }} .why {{ color:var(--muted); font-size:12px; margin-top:4px; }}
.pill {{ display:inline-block; padding:1px 8px; border-radius:20px; font-size:12px; font-weight:600; }}
.scam {{ background:var(--bad); color:var(--badt); }} .safe {{ background:var(--ok); color:var(--okt); }}
.p {{ font-variant-numeric:tabular-nums; white-space:nowrap; }} .p.w {{ color:var(--badt); font-weight:700; }}
.count {{ color:var(--muted); font-size:13px; margin:6px 0; }}
ul.notes {{ margin:0; padding-left:20px; }} ul.notes li {{ margin:4px 0; }}
</style></head><body><main>
<h1>Test B Results</h1>
<p class="lead">400 new messages (199 scam, 201 safe, 18 countries) written by a different AI from a generic prompt.
Never used for training. Each model was scored once. A scam score above 50% counts as “scam”.</p>

<div class="tiles">{tiles}</div>

<h2>By group</h2>
<div class="legend">{legend}</div>
<div class="charts">{charts}</div>
<div class="scroll" style="margin-top:14px">{gt}</div>

<h2>Right and wrong answers</h2>
<div class="confs">{conf}</div>

<h2>What to know</h2>
<div class="card"><ul class="notes">
<li>Our own test set gave ~98–99%. Test B shows about <b>90%</b>, which is closer to real life.</li>
<li>Links are masked to &lt;URL&gt;, so the model cannot see fake-looking domains. Many missed scams depend only on the domain.</li>
<li>Test B has no long or multi-line messages, so it does not test that weakness.</li>
<li>Test B is now “seen”: don’t tune the model to its mistakes. A future check needs another fresh set.</li>
</ul></div>

<h2>Messages</h2>
<div class="filters">
  <select id="run">{''.join(f'<option value="{r}"{" selected" if r == best else ""}>{r.upper()}</option>' for r in RUNS)}</select>
  <select id="show"><option value="wrong">Wrong only</option><option value="all">All</option>
    <option value="fn">Missed scams</option><option value="fp">False alarms</option>
    <option value="changed">Run 2 vs this run differ</option></select>
  <select id="diff"><option value="">Easy + hard</option><option value="easy">Easy</option><option value="hard">Hard</option></select>
  <select id="lnk"><option value="">Link or not</option><option value="1">With link</option><option value="0">No link</option></select>
  <input id="q" placeholder="Search text…">
</div>
<div class="count" id="count"></div>
<div class="scroll"><table class="mt"><thead><tr><th>ID</th><th>Label</th><th>Message</th>
{''.join(f'<th>{r.upper()}</th>' for r in RUNS)}</tr></thead><tbody id="tb"></tbody></table></div>
</main>
<script>
const ROWS = {json.dumps(rows, ensure_ascii=False)};
const RUNS = {json.dumps(RUNS)};
const $ = id => document.getElementById(id);
const esc = s => s.replace(/[&<>"]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}}[c]));
function draw() {{
  const run = $("run").value, show = $("show").value, d = $("diff").value, l = $("lnk").value, q = $("q").value.toLowerCase();
  const out = ROWS.filter(r => {{
    const scam = r.label === "scam", said = r.p[run] > 0.5, wrong = said !== scam;
    if (show === "wrong" && !wrong) return false;
    if (show === "fn" && !(scam && !said)) return false;
    if (show === "fp" && !(!scam && said)) return false;
    if (show === "changed" && ((r.p.run2 > 0.5) === said)) return false;
    if (d && r.difficulty !== d) return false;
    if (l !== "" && String(+r.link) !== l) return false;
    if (q && !(r.text + " " + r.category + " " + r.country).toLowerCase().includes(q)) return false;
    return true;
  }});
  $("count").textContent = out.length + " messages";
  $("tb").innerHTML = out.map(r => `<tr><td>${{r.id}}<div class="why">${{esc(r.country)}} · ${{esc(r.difficulty)}}</div></td>
    <td><span class="pill ${{r.label}}">${{r.label}}</span></td>
    <td class="msg">${{esc(r.text)}}<div class="why">${{esc(r.category)}} — ${{esc(r.why)}}</div></td>
    ${{RUNS.map(k => `<td class="p ${{(r.p[k] > 0.5) !== (r.label === "scam") ? "w" : ""}}">${{Math.round(r.p[k]*100)}}%</td>`).join("")}}</tr>`).join("");
}}
["run","show","diff","lnk","q"].forEach(id => $(id).addEventListener("input", draw));
draw();
</script></body></html>"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(page, encoding="utf-8")
print(f"Saved -> {OUT}")
