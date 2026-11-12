"""
Build a local HTML results page for a training run (works offline, no libraries).

Reads : results/<run>/history.json, results/<run>/test_mistakes.csv
Writes: results/<run>/results.html

Run: .venv/bin/python make_report.py run2
"""

import html
import json
import sys
from pathlib import Path

import pandas as pd

run = sys.argv[1] if len(sys.argv) > 1 else "run2"
folder = Path("results") / run
history = json.loads((folder / "history.json").read_text())
mistakes = pd.read_csv(folder / "test_mistakes.csv", keep_default_na=False)

data = {
    "run": run,
    "settings": history["settings"],
    "stepsPerEpoch": history["steps_per_epoch"],
    "points": history["points"],
    "epochs": history["epochs"],
    "test": history["test"],
    "mistakes": mistakes.to_dict(orient="records"),
}

page = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scam Detector __RUN__ Results</title>
<style>
  :root { --bg:#f7f8fa; --panel:#fff; --fg:#1d2330; --muted:#5d6678; --line:#e3e6ec;
          --train:#2f6fd0; --val:#e07b24; --bad:#c8443c; --ok:#2e8b57; --grid:#eef0f4; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14171d; --panel:#1c2028; --fg:#e7eaf0; --muted:#9aa3b5; --line:#2c323d;
            --train:#6ea2f0; --val:#f0a05a; --bad:#ec6f66; --ok:#5cc58a; --grid:#262b35; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif; }
  main { max-width:1100px; margin:0 auto; padding:28px 16px 60px; display:grid; gap:22px; }
  h1 { margin:0; font-size:26px; font-weight:600; }
  h2 { margin:0 0 4px; font-size:17px; font-weight:600; }
  .sub, .note { color:var(--muted); font-size:13px; margin:4px 0 0; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:18px; min-width:0; }
  .tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
  .tile .v { font-size:28px; font-weight:600; font-variant-numeric:tabular-nums; }
  .tile .k { color:var(--muted); font-size:13px; }
  .tile.bad .v { color:var(--bad); }
  .row { display:grid; grid-template-columns:1.6fr 1fr; gap:22px; }
  @media (max-width:800px) { .row { grid-template-columns:1fr; } }
  .chart { position:relative; }
  svg { display:block; width:100%; height:auto; overflow:visible; }
  .axis text { fill:var(--muted); font-size:11px; }
  .tip { position:absolute; pointer-events:none; background:var(--panel); border:1px solid var(--line);
         box-shadow:0 4px 14px rgba(0,0,0,.15); border-radius:8px; padding:8px 10px; font-size:12px; display:none; white-space:nowrap; }
  .legend { display:flex; gap:16px; font-size:13px; color:var(--muted); margin:6px 0 4px; flex-wrap:wrap; }
  .legend i { display:inline-block; width:14px; height:3px; border-radius:2px; margin-right:6px; vertical-align:middle; }
  .cm { display:grid; grid-template-columns:auto 1fr 1fr; gap:6px; align-items:stretch; font-size:13px; }
  .cm .h { color:var(--muted); text-align:center; align-self:end; }
  .cm .r { color:var(--muted); align-self:center; text-align:right; padding-right:6px; }
  .cm .c { border-radius:8px; padding:18px 8px; text-align:center; font-size:24px; font-weight:600; font-variant-numeric:tabular-nums; }
  .cm .c small { display:block; font-size:12px; font-weight:400; color:var(--muted); }
  .cm .good { background:color-mix(in srgb, var(--ok) 18%, transparent); }
  .cm .wrong { background:color-mix(in srgb, var(--bad) 16%, transparent); }
  .filters { display:flex; gap:8px; flex-wrap:wrap; margin:8px 0 10px; }
  .filters button { font:inherit; font-size:13px; padding:5px 12px; border-radius:999px; border:1px solid var(--line);
                    background:transparent; color:var(--fg); cursor:pointer; }
  .filters button[aria-pressed=true] { background:var(--fg); color:var(--bg); border-color:var(--fg); }
  .tablewrap { overflow-x:auto; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
  th { color:var(--muted); font-weight:500; white-space:nowrap; }
  td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
  td.msg { min-width:260px; }
  .bar { height:6px; border-radius:3px; background:var(--grid); margin-top:4px; min-width:60px; }
  .bar span { display:block; height:100%; border-radius:3px; background:var(--bad); }
  .pill { font-size:12px; padding:1px 8px; border-radius:999px; border:1px solid var(--line); white-space:nowrap; }
  details summary { cursor:pointer; color:var(--muted); font-size:13px; }
</style>
</head>
<body>
<main>
  <header>
    <h1>Scam Detector &middot; <span id="runName"></span></h1>
    <p class="sub" id="lede"></p>
  </header>

  <section class="tiles" id="tiles"></section>

  <section class="row">
    <div class="card">
      <h2>Loss during training</h2>
      <p class="note">Lower is better. Validation loss uses real messages the model never learns from. Point at the chart to read values.</p>
      <div class="legend"><span><i style="background:var(--train)"></i>Train loss</span><span><i style="background:var(--val)"></i>Validation loss</span><span><i style="background:var(--muted);height:0;border-top:2px dashed var(--muted)"></i>Best model saved</span></div>
      <div class="chart" id="loss"><div class="tip"></div></div>
    </div>
    <div class="card">
      <h2>Right vs wrong on the test set</h2>
      <p class="note">Each box counts real test messages.</p>
      <div class="cm" id="cm" style="margin-top:14px"></div>
    </div>
  </section>

  <section class="card">
    <h2>Every test mistake</h2>
    <p class="note">Scam score is how sure the model was that the message is a scam. Above 50% it says scam.</p>
    <div class="filters" id="filters">
      <button data-f="all" aria-pressed="true">All</button>
      <button data-f="scam">Missed scams</button>
      <button data-f="safe">False alarms</button>
    </div>
    <div class="tablewrap"><table>
      <thead><tr><th>Message</th><th>Really</th><th>Model said</th><th>Source</th><th style="text-align:right">Scam score</th></tr></thead>
      <tbody id="rows"></tbody>
    </table></div>
  </section>

  <section class="card">
    <details><summary>Training settings and each epoch</summary>
      <div class="tablewrap" style="margin-top:10px"><table>
        <thead><tr><th>Epoch</th><th style="text-align:right">Train loss</th><th style="text-align:right">Validation loss</th><th style="text-align:right">Seconds</th><th>Best so far</th></tr></thead>
        <tbody id="epochs"></tbody>
      </table></div>
      <p class="note" id="settings"></p>
    </details>
  </section>
</main>

<script>
const D = __DATA__;
const pct = v => (v * 100).toFixed(1) + '%';
const el = (tag, attrs = {}, text) => { const e = document.createElement(tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (text != null) e.textContent = text; return e; };
const T = D.test;

document.getElementById('runName').textContent = D.run;
document.getElementById('lede').textContent =
  `Caught ${T.tp} of ${T.tp + T.fn} scams and wrongly flagged ${T.fp} of ${T.fp + T.tn} safe messages, on ${T.rows} real test messages.`;

// Score tiles
const tiles = [['Accuracy', T.accuracy, 'all answers right'], ['Precision', T.precision, 'when it says scam, how often right'],
  ['Recall', T.recall, 'share of scams caught'], ['F1', T.f1, 'balance of precision and recall'],
  ['False alarm rate', T.false_alarm_rate, 'safe messages flagged as scam', true]];
for (const [k, v, hint, bad] of tiles) {
  const t = el('div', { class: 'card tile' + (bad ? ' bad' : '') });
  t.append(el('div', { class: 'v' }, pct(v)), el('div', { class: 'k' }, k), el('div', { class: 'note' }, hint));
  document.getElementById('tiles').append(t);
}

// Confusion matrix
const cm = document.getElementById('cm');
const cell = (n, of, cls, label) => { const c = el('div', { class: 'c ' + cls }, String(n)); c.append(el('small', {}, label + ' · ' + pct(n / of))); return c; };
cm.append(el('div'), el('div', { class: 'h' }, 'Model said safe'), el('div', { class: 'h' }, 'Model said scam'));
cm.append(el('div', { class: 'r' }, 'Really safe'), cell(T.tn, T.tn + T.fp, 'good', 'correct'), cell(T.fp, T.tn + T.fp, 'wrong', 'false alarm'));
cm.append(el('div', { class: 'r' }, 'Really scam'), cell(T.fn, T.fn + T.tp, 'wrong', 'missed'), cell(T.tp, T.fn + T.tp, 'good', 'caught'));

// Loss chart (inline SVG)
function drawLoss() {
  const box = document.getElementById('loss'), tip = box.querySelector('.tip');
  box.querySelectorAll('svg').forEach(s => s.remove());
  const W = Math.max(320, box.clientWidth), H = 300, m = { t: 12, r: 14, b: 32, l: 44 };
  const pts = D.points.map(p => ({ x: p.step / D.stepsPerEpoch, step: p.step, train: p.train_loss, val: p.val_loss }));
  const maxE = Math.max(Math.ceil(pts[pts.length - 1].x), 1);
  const yMax = Math.ceil(Math.max(...pts.map(p => Math.max(p.train, p.val))) * 10) / 10;
  const X = v => m.l + (v / maxE) * (W - m.l - m.r), Y = v => H - m.b - (v / yMax) * (H - m.t - m.b);
  const NS = 'http://www.w3.org/2000/svg', s = document.createElementNS(NS, 'svg');
  s.setAttribute('viewBox', `0 0 ${W} ${H}`); s.setAttribute('role', 'img');
  s.setAttribute('aria-label', 'Line chart of train and validation loss across epochs');
  const add = (tag, a) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(a)) e.setAttribute(k, v); s.append(e); return e; };
  const g = add('g', { class: 'axis' });
  for (let i = 0; i <= 5; i++) {
    const v = yMax * i / 5, y = Y(v);
    add('line', { x1: m.l, x2: W - m.r, y1: y, y2: y, stroke: 'var(--grid)' });
    const t = document.createElementNS(NS, 'text'); t.setAttribute('x', m.l - 6); t.setAttribute('y', y + 4); t.setAttribute('text-anchor', 'end'); t.textContent = v.toFixed(2); g.append(t);
  }
  for (let e = 0; e <= maxE; e++) {
    const t = document.createElementNS(NS, 'text'); t.setAttribute('x', X(e)); t.setAttribute('y', H - 10); t.setAttribute('text-anchor', 'middle'); t.textContent = e === 0 ? 'Start' : 'Epoch ' + e; g.append(t);
  }
  const best = D.epochs.filter(e => e.best).pop();
  if (best) add('line', { x1: X(best.epoch), x2: X(best.epoch), y1: m.t, y2: H - m.b, stroke: 'var(--muted)', 'stroke-dasharray': '4 3' });
  for (const [key, color] of [['train', 'var(--train)'], ['val', 'var(--val)']]) {
    add('path', { d: pts.map((p, i) => (i ? 'L' : 'M') + X(p.x) + ',' + Y(p[key])).join(''), fill: 'none', stroke: color, 'stroke-width': 2 });
    pts.forEach(p => add('circle', { cx: X(p.x), cy: Y(p[key]), r: 3, fill: color }));
  }
  const cross = add('line', { y1: m.t, y2: H - m.b, stroke: 'var(--muted)', visibility: 'hidden' });
  const hit = add('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent' });
  hit.addEventListener('pointermove', ev => {
    const r = s.getBoundingClientRect(), xv = ((ev.clientX - r.left) * W / r.width - m.l) / (W - m.l - m.r) * maxE;
    const p = pts.reduce((a, b) => Math.abs(b.x - xv) < Math.abs(a.x - xv) ? b : a);
    cross.setAttribute('x1', X(p.x)); cross.setAttribute('x2', X(p.x)); cross.setAttribute('visibility', 'visible');
    tip.textContent = ''; tip.append(el('div', {}, `Step ${p.step} · epoch ${p.x.toFixed(2)}`), el('div', {}, `Train loss ${p.train.toFixed(4)}`), el('div', {}, `Validation loss ${p.val.toFixed(4)}`));
    tip.style.display = 'block';
    const px = X(p.x) * r.width / W; tip.style.top = '30px';
    tip.style.left = (px + 12 + tip.offsetWidth > box.clientWidth ? px - tip.offsetWidth - 12 : px + 12) + 'px';
  });
  hit.addEventListener('pointerleave', () => { tip.style.display = 'none'; cross.setAttribute('visibility', 'hidden'); });
  box.prepend(s);
}
drawLoss();
new ResizeObserver(drawLoss).observe(document.getElementById('loss'));

// Mistakes table with filter
function drawRows(f) {
  const tb = document.getElementById('rows'); tb.textContent = '';
  D.mistakes.filter(r => f === 'all' || r.label === f).forEach(r => {
    const tr = el('tr');
    tr.append(el('td', { class: 'msg' }, r.text), el('td', {}, r.label === 'scam' ? 'Scam' : 'Safe'));
    const said = el('td'); said.append(el('span', { class: 'pill' }, r.pred === 'scam' ? 'Scam' : 'Safe')); tr.append(said);
    tr.append(el('td', {}, r.source + (r.scam_type ? ' / ' + r.scam_type : '')));
    const sc = el('td', { class: 'num' }, pct(r.scam_prob)); const b = el('div', { class: 'bar' }); const fill = el('span'); fill.style.width = (r.scam_prob * 100) + '%'; b.append(fill); sc.append(b); tr.append(sc);
    tb.append(tr);
  });
}
document.querySelectorAll('#filters button').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('#filters button').forEach(x => x.setAttribute('aria-pressed', x === b));
  drawRows(b.dataset.f);
}));
drawRows('all');

// Epochs + settings
D.epochs.forEach(e => {
  const tr = el('tr');
  tr.append(el('td', {}, e.epoch), el('td', { class: 'num' }, e.train_loss.toFixed(4)), el('td', { class: 'num' }, e.val_loss.toFixed(4)), el('td', { class: 'num' }, e.seconds), el('td', {}, e.best ? '✓' : ''));
  document.getElementById('epochs').append(tr);
});
const S = D.settings;
document.getElementById('settings').textContent = `Base model ${S.base_model} · max length ${S.max_len} · batch ${S.batch_size} · up to ${S.epochs} epochs · learning rate ${S.lr} · patience ${S.patience} · seed ${S.seed}`;
</script>
</body>
</html>
"""

page = page.replace("__RUN__", html.escape(run)).replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
out = folder / "results.html"
out.write_text(page)
print(f"Saved -> {out}")
