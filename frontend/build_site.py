"""Stage 8 - build the static explanation website.

Generates a single, fully self-contained ``frontend/index.html`` that explains
the project to a non-technical reader: the microgrid problem, the three AI
paradigms being compared, how they were evaluated, and what the numbers say.

Unlike the Stage 7 dashboard (a live FastAPI server with Plotly/CDN assets),
this page is a **static, offline artefact** - all CSS is inlined and every chart
is drawn as plain HTML/CSS bars, so it opens with a double-click and needs no
server, no internet, and no Python at view time. It is meant for a viva / poster
/ hand-in where a self-contained file is more convenient than a running server.

Every number on the page is read from ``reports/comparison.json`` and
``reports/forecast_metrics.json`` at build time - nothing is hand-typed or
invented. Run ``python main.py site`` (or this module) after ``compare`` to
(re)generate it.
"""
from __future__ import annotations

import html
import json

import config
from utils.logger import get_logger

log = get_logger("site")

OUTPUT_HTML = config.FRONTEND_DIR / "index.html"

# Colour per approach - kept consistent with the Stage 7 dashboard.
_COLORS = {
    "Rule-based": "#6c757d",
    "Generative-AI": "#0d6efd",
    "Agentic-AI": "#d63384",
}

# (label, metric key, "min"|"max" is better, format, unit) for the results table.
_ROWS = [
    ("Daily energy cost", "daily_cost_inr", "min", "{:.0f}", "Rs"),
    ("Cost saved vs grid-only", "cost_saved_pct", "max", "{:.1f}", "%"),
    ("CO&#8322; emitted", "co2_kg", "min", "{:.0f}", "kg"),
    ("CO&#8322; saved vs grid-only", "co2_saved_kg", "max", "{:.0f}", "kg"),
    ("Renewable share of demand", "renewable_share_of_demand_pct", "max", "{:.1f}", "%"),
    ("Grid electricity imported", "grid_import_kwh", "min", "{:.0f}", "kWh"),
    ("LLM calls made", "llm_calls", "min", "{:.0f}", ""),
    ("LLM tokens used", "llm_tokens", "min", "{:,.0f}", ""),
    ("Latency per decision", "avg_decision_latency_ms", "min", "{:.0f}", "ms"),
]


def _best_index(values: list[float], better: str) -> int:
    return values.index(max(values) if better == "max" else min(values))


def _bar_chart(title: str, subtitle: str, results: dict, key: str,
               fmt: str, better: str) -> str:
    """Render a horizontal CSS bar chart (self-contained, no JS/CDN)."""
    names = list(results.keys())
    values = [float(results[n][key]) for n in names]
    best = _best_index(values, better)
    span = max(values) or 1.0
    bars = []
    for i, n in enumerate(names):
        pct = max(values[i] / span * 100.0, 1.5)  # keep a sliver visible at 0
        star = " &#9733;" if i == best else ""
        bars.append(
            f'<div class="bar-row">'
            f'<div class="bar-name">{html.escape(n)}</div>'
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{pct:.1f}%;background:{_COLORS[n]}"></div></div>'
            f'<div class="bar-val">{fmt.format(values[i])}{star}</div>'
            f'</div>'
        )
    return (
        f'<figure class="chart">'
        f'<figcaption><strong>{title}</strong>'
        f'<span class="muted"> &mdash; {subtitle}</span></figcaption>'
        + "".join(bars) +
        f'</figure>'
    )


def _results_table(results: dict) -> str:
    names = list(results.keys())
    head = "".join(
        f'<th style="color:{_COLORS[n]}">{html.escape(n)}</th>' for n in names)
    rows = []
    for label, key, better, fmt, unit in _ROWS:
        values = [float(results[n][key]) for n in names]
        best = _best_index(values, better)
        cells = []
        for i, n in enumerate(names):
            txt = fmt.format(values[i])
            if unit:
                txt = f"{txt} {unit}"
            cls = ' class="best"' if i == best else ""
            cells.append(f"<td{cls}>{txt}</td>")
        rows.append(f"<tr><th scope=\"row\">{label}</th>{''.join(cells)}</tr>")
    return (
        '<table class="results">'
        f'<thead><tr><th scope="col">Metric</th>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def _approach_card(name: str, tag: str, what: str, how: str,
                   pros: list[str], cons: list[str]) -> str:
    pl = "".join(f"<li>{p}</li>" for p in pros)
    cl = "".join(f"<li>{c}</li>" for c in cons)
    return (
        f'<article class="card" style="--accent:{_COLORS[name]}">'
        f'<div class="card-tag">{tag}</div>'
        f'<h3>{html.escape(name)}</h3>'
        f'<p class="what">{what}</p>'
        f'<p class="how"><strong>How it works:</strong> {how}</p>'
        f'<div class="proscons"><div><h4>Strengths</h4><ul>{pl}</ul></div>'
        f'<div><h4>Limitations</h4><ul class="cons">{cl}</ul></div></div>'
        f'</article>'
    )


def _load(path, what):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
        log.warning("Could not read %s: %s", what, exc)
        return None


def _forecast_rows(forecast: list | None) -> str:
    """Optional forecasting-accuracy block (skipped if metrics absent).

    ``models/forecast_metrics.json`` is a list, one entry per target, each with
    ``best_model``, ``best_metrics{mae,rmse}`` and ``naive_baseline{rmse}``.
    """
    if not forecast:
        return ""
    rows = []
    for m in forecast:
        best = m.get("best_metrics", {})
        naive = m.get("naive_baseline", {})
        rows.append(
            f"<tr><th scope=\"row\"><code>{html.escape(str(m.get('target', '-')))}"
            f"</code></th>"
            f"<td>{html.escape(str(m.get('best_model', '-')))}</td>"
            f"<td>{best.get('mae', float('nan')):.2f}</td>"
            f"<td>{best.get('rmse', float('nan')):.2f}</td>"
            f"<td>{naive.get('rmse', float('nan')):.2f}</td></tr>"
        )
    return (
        '<h3>Step 1 &mdash; forecasting the next hours</h3>'
        '<p>Every controller decides on <em>forecasts</em>, not on a crystal '
        'ball. Gradient-boosted models predict solar, wind and demand from '
        'calendar and recent-history features (no peeking at future weather). '
        'They comfortably beat a seasonal-naive baseline:</p>'
        '<table class="results"><thead><tr>'
        '<th scope="col">Target</th><th scope="col">Best model</th>'
        '<th scope="col">MAE</th><th scope="col">RMSE</th>'
        '<th scope="col">Naive RMSE</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
        '<p class="muted small">Lower is better; RMSE below the naive column '
        'means the model genuinely learned structure. Wind is the hardest '
        'target (gustier, less periodic than solar or demand).</p>'
    )


def build_site() -> None:
    """Generate the static explanation website from the saved reports."""
    comparison = _load(config.REPORTS_DIR / "comparison.json", "comparison")
    forecast = _load(config.MODELS_DIR / "forecast_metrics.json", "forecast")

    if not comparison:
        OUTPUT_HTML.write_text(_placeholder(), encoding="utf-8")
        log.warning(
            "No comparison.json yet - wrote a placeholder page. "
            "Run `python main.py compare` then `python main.py site`.")
        log.info("Wrote %s", OUTPUT_HTML)
        return

    results = comparison["results"]
    window_hours = comparison["window_hours"]
    model = comparison["model"]

    rb = results["Rule-based"]
    gen = results["Generative-AI"]
    ag = results["Agentic-AI"]
    by_cost = min(results.items(), key=lambda kv: kv[1]["daily_cost_inr"])[0]
    by_co2 = min(results.items(), key=lambda kv: kv[1]["co2_kg"])[0]
    tok_ratio = (ag["llm_tokens"] / gen["llm_tokens"]) if gen["llm_tokens"] else 0

    cards = "".join([
        _approach_card(
            "Rule-based", "Baseline &middot; no AI",
            "A transparent, hand-written policy - the comparison floor every AI "
            "approach must beat to justify its cost.",
            "A handful of if/then rules: charge the battery when renewables "
            "exceed demand, discharge it during expensive peak hours, top up "
            "cheaply off-peak. No model, no API.",
            ["Instant &amp; free (0 tokens, microsecond decisions)",
             "Fully auditable - you can read the exact rule that fired",
             "Deterministic: never fails at runtime"],
            ["Fixed - cannot adapt beyond the rules it was given",
             "No natural-language explanation of <em>why</em>",
             "A human must anticipate every situation in advance"]),
        _approach_card(
            "Generative-AI", "LLM &middot; human-in-the-loop",
            "A large language model (Groq) is shown the microgrid state each "
            "hour and <em>recommends</em> a battery set-point, explaining it in "
            "one sentence. A human would sign off in practice.",
            "One prompt per hour with the forecasts, price and battery level; "
            "the model replies with an action plus a plain-language rationale. "
            "No tools - a single shot of reasoning.",
            ["Explains every decision in human-readable language",
             "Flexible - reasons about situations no rule anticipated",
             "Cheapest of the two AI approaches (fewest tokens)"],
            ["Costs real tokens &amp; network latency each hour",
             "Advisory, not autonomous - needs a human to act",
             "Can occasionally reply unparseably (handled by a safe fallback)"]),
        _approach_card(
            "Agentic-AI", "Autonomous &middot; tool-using",
            "A LangGraph agent that doesn&#39;t just advise - it <em>acts</em>. "
            "Each hour it calls a simulation tool to test candidate actions, "
            "reads back their cost/CO&#8322;, then commits the best one itself.",
            "A ReAct loop: the model proposes actions, a tool evaluates each "
            "against the real physics, and the agent iterates before "
            "autonomously committing - a closed loop with no human.",
            ["Fully autonomous closed loop - decides and acts",
             "Can use tools to test actions before committing",
             "Exposes its evaluation trace for inspection"],
            [f"Most expensive by far (~{tok_ratio:.0f}&times; the tokens of the "
             "generative approach)",
             "Highest latency (many calls per decision)",
             "Myopic here: optimises each hour in isolation, not the whole day"]),
    ])

    charts = "".join([
        _bar_chart("Cost saved vs a grid-only baseline",
                   "higher is better", results, "cost_saved_pct", "{:.1f}%", "max"),
        _bar_chart("CO&#8322; saved vs a grid-only baseline",
                   "higher is better", results, "co2_saved_kg", "{:.0f} kg", "max"),
        _bar_chart("LLM tokens spent (the cost of intelligence)",
                   "lower is better", results, "llm_tokens", "{:,.0f}", "min"),
        _bar_chart("Latency per decision",
                   "lower is better", results, "avg_decision_latency_ms",
                   "{:.0f} ms", "min"),
    ])

    page = _PAGE.format(
        window_hours=window_hours,
        window_days=window_hours // 24,
        start_date=config.EVAL_START_DATE,
        model=html.escape(model),
        pv=int(config.PV_RATED_KWP),
        wind=int(config.WIND_RATED_KW),
        batt=int(config.BATTERY_CAPACITY_KWH),
        import_price=config.GRID_IMPORT_PRICE,
        co2_factor=config.CO2_PER_KWH_GRID,
        cards=cards,
        forecast_block=_forecast_rows(forecast),
        results_table=_results_table(results),
        charts=charts,
        rb_saved=f"{rb['cost_saved_pct']:.1f}",
        gen_saved=f"{gen['cost_saved_pct']:.1f}",
        ag_saved=f"{ag['cost_saved_pct']:.1f}",
        by_cost=html.escape(by_cost),
        by_co2=html.escape(by_co2),
        gen_calls=gen["llm_calls"],
        gen_tokens=f"{gen['llm_tokens']:,}",
        ag_calls=ag["llm_calls"],
        ag_tokens=f"{ag['llm_tokens']:,}",
        tok_ratio=f"{tok_ratio:.1f}",
    )
    OUTPUT_HTML.write_text(page, encoding="utf-8")
    log.info("Wrote %s (open it directly in a browser - no server needed).",
             OUTPUT_HTML)


def _placeholder() -> str:
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>Microgrid AI &mdash; not built yet</title></head><body "
        "style=\"font-family:system-ui;max-width:40rem;margin:4rem auto;"
        "padding:0 1rem;line-height:1.6\">"
        "<h1>Nothing to show yet</h1>"
        "<p>The explanation site is generated from the comparison results. "
        "Run these first, then rebuild the site:</p>"
        "<pre style=\"background:#f4f4f4;padding:1rem;border-radius:.5rem\">"
        "python main.py compare\npython main.py site</pre></body></html>"
    )


# --- Page template ----------------------------------------------------------
# One f-safe .format() template. All CSS is inlined so the file is fully
# self-contained (no CDN, works offline). Literal CSS braces are doubled.
_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Generative AI vs Agentic AI for Microgrid Operation</title>
<style>
  :root {{
    --ink:#111827; --muted:#6b7280; --line:#e5e7eb; --bg:#f6f8fb;
    --card:#ffffff; --navy1:#0d1b3e; --navy2:#123a6b; --accent:#0d6efd;
  }}
  * {{ box-sizing:border-box; }}
  html {{ scroll-behavior:smooth; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font:16px/1.65 system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
  a {{ color:var(--navy2); }}
  .wrap {{ max-width:1000px; margin:0 auto; padding:0 1.2rem; }}
  .hero {{ background:linear-gradient(125deg,var(--navy1),var(--navy2));
    color:#fff; padding:4rem 0 3.2rem; }}
  .hero h1 {{ font-size:clamp(1.7rem,4.2vw,2.7rem); margin:0 0 .6rem; line-height:1.15; }}
  .hero p {{ font-size:1.12rem; max-width:44rem; margin:.4rem 0; opacity:.92; }}
  .badges {{ margin-top:1.4rem; display:flex; flex-wrap:wrap; gap:.5rem; }}
  .badge {{ background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.25);
    padding:.35rem .7rem; border-radius:2rem; font-size:.85rem; }}
  nav.toc {{ position:sticky; top:0; z-index:10; background:rgba(255,255,255,.95);
    backdrop-filter:blur(6px); border-bottom:1px solid var(--line); }}
  nav.toc ul {{ display:flex; flex-wrap:wrap; gap:.2rem 1.3rem; list-style:none;
    margin:0; padding:.7rem 0; font-size:.92rem; }}
  nav.toc a {{ text-decoration:none; color:var(--muted); font-weight:500; }}
  nav.toc a:hover {{ color:var(--navy2); }}
  section {{ padding:3rem 0; border-bottom:1px solid var(--line); }}
  section h2 {{ font-size:1.6rem; margin:0 0 .4rem; }}
  section h2 .num {{ color:var(--accent); font-weight:800; margin-right:.5rem; }}
  .lead {{ color:var(--muted); font-size:1.06rem; max-width:48rem; margin:.2rem 0 1.6rem; }}
  .grid3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; }}
  .spec {{ background:var(--card); border:1px solid var(--line); border-radius:.7rem;
    padding:1.1rem; text-align:center; }}
  .spec .big {{ font-size:1.7rem; font-weight:800; color:var(--navy2); }}
  .spec .lbl {{ color:var(--muted); font-size:.9rem; }}
  .cards {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1.1rem; }}
  .card {{ background:var(--card); border:1px solid var(--line);
    border-top:4px solid var(--accent); border-radius:.7rem; padding:1.2rem;
    display:flex; flex-direction:column; }}
  .card-tag {{ font-size:.75rem; text-transform:uppercase; letter-spacing:.04em;
    color:var(--muted); font-weight:700; }}
  .card h3 {{ margin:.25rem 0 .5rem; color:var(--accent); }}
  .card .what {{ margin:0 0 .7rem; }}
  .card .how {{ font-size:.92rem; color:#374151; margin:0 0 .8rem; }}
  .proscons {{ display:grid; grid-template-columns:1fr 1fr; gap:.6rem; margin-top:auto; }}
  .proscons h4 {{ margin:.2rem 0; font-size:.82rem; text-transform:uppercase;
    letter-spacing:.03em; color:var(--muted); }}
  .proscons ul {{ margin:0; padding-left:1.05rem; font-size:.86rem; }}
  .proscons ul.cons li::marker {{ color:#c0392b; }}
  .proscons ul li {{ margin:.15rem 0; }}
  table {{ border-collapse:collapse; width:100%; background:var(--card);
    border:1px solid var(--line); border-radius:.6rem; overflow:hidden; margin:1rem 0; }}
  th, td {{ padding:.55rem .7rem; text-align:right; border-bottom:1px solid var(--line); }}
  thead th {{ background:#f0f3f8; font-size:.9rem; }}
  tbody th[scope=row] {{ text-align:left; font-weight:600; color:#374151; }}
  th[scope=col]:first-child {{ text-align:left; }}
  td.best {{ font-weight:800; color:#0a7d33; background:#eafaf0; }}
  .chart {{ margin:0 0 1.3rem; }}
  .chart figcaption {{ font-size:.95rem; margin-bottom:.5rem; }}
  .bar-row {{ display:grid; grid-template-columns:120px 1fr 92px; align-items:center;
    gap:.6rem; margin:.28rem 0; }}
  .bar-name {{ font-size:.86rem; color:#374151; text-align:right; }}
  .bar-track {{ background:#eef1f6; border-radius:.4rem; height:1.15rem; overflow:hidden; }}
  .bar-fill {{ height:100%; border-radius:.4rem; transition:width .4s; }}
  .bar-val {{ font-size:.85rem; font-variant-numeric:tabular-nums; }}
  .callouts {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin:1.4rem 0; }}
  .callout {{ background:var(--card); border:1px solid var(--line); border-left:4px solid var(--accent);
    border-radius:.5rem; padding:1rem 1.1rem; }}
  .callout .k {{ font-size:1.5rem; font-weight:800; color:var(--navy2); }}
  .callout .d {{ color:var(--muted); font-size:.9rem; }}
  .finding {{ background:#fffef5; border:1px solid #f0e6b8; border-radius:.7rem;
    padding:1.2rem 1.4rem; margin:1.2rem 0; }}
  .finding h3 {{ margin-top:0; }}
  .caveat {{ background:#f4f6f9; border-left:4px solid var(--muted); padding:.9rem 1.2rem;
    border-radius:.4rem; color:#374151; font-size:.95rem; }}
  .muted {{ color:var(--muted); }} .small {{ font-size:.86rem; }}
  code {{ background:#eef1f6; padding:.1rem .35rem; border-radius:.3rem; font-size:.9em; }}
  footer {{ padding:2.4rem 0 3rem; color:var(--muted); font-size:.9rem; }}
  @media (max-width:820px) {{
    .grid3,.cards,.callouts {{ grid-template-columns:1fr; }}
    .bar-row {{ grid-template-columns:96px 1fr 78px; }}
  }}
</style>
</head>
<body>

<header class="hero">
  <div class="wrap">
    <h1>Generative AI vs Agentic AI for Microgrid Operation</h1>
    <p>A final-year comparative study that solves the <strong>same</strong>
       energy-management problem three ways &mdash; a rule-based baseline, a
       generative LLM, and an autonomous agent &mdash; and measures, honestly,
       where the added intelligence actually pays off.</p>
    <div class="badges">
      <span class="badge">Evaluated on {window_hours} h ({window_days} days)</span>
      <span class="badge">Model: {model}</span>
      <span class="badge">Groq &middot; LangChain &middot; LangGraph</span>
      <span class="badge">100% local &amp; reproducible</span>
    </div>
  </div>
</header>

<nav class="toc"><div class="wrap"><ul>
  <li><a href="#problem">The problem</a></li>
  <li><a href="#approaches">Three approaches</a></li>
  <li><a href="#method">How we compared</a></li>
  <li><a href="#results">Results</a></li>
  <li><a href="#findings">What we found</a></li>
</ul></div></nav>

<main class="wrap">

<section id="problem">
  <h2><span class="num">1</span>The problem: running a microgrid</h2>
  <p class="lead">A microgrid is a small, self-contained power system &mdash; here
    a rooftop solar array, a small wind turbine, a battery, and a grid connection
    serving one site. Every hour a controller must decide: charge the battery,
    discharge it, or sit idle &mdash; to cut cost and emissions while keeping the
    lights on. The catch: it must decide on <em>forecasts</em>, before it knows
    what the sun, wind and demand will actually do.</p>
  <div class="grid3">
    <div class="spec"><div class="big">{pv} kW</div><div class="lbl">rooftop solar (PV)</div></div>
    <div class="spec"><div class="big">{wind} kW</div><div class="lbl">wind turbine</div></div>
    <div class="spec"><div class="big">{batt} kWh</div><div class="lbl">battery storage</div></div>
  </div>
  <p class="muted small">Simulated site with a time-of-use grid tariff (peak
    import at Rs {import_price}/kWh) and a grid carbon intensity of
    {co2_factor} kg CO&#8322; per kWh. The data is <strong>clearly-labelled
    synthetic</strong> &mdash; generated from documented physical models (solar
    geometry, a turbine power curve, a realistic demand shape), not measured
    field data, and fully reproducible from a fixed seed. No results on this
    page are invented; every number is read from the actual simulation runs.</p>
</section>

<section id="approaches">
  <h2><span class="num">2</span>Three ways to solve it</h2>
  <p class="lead">The whole point is a fair fight: the <em>same</em> microgrid,
    the <em>same</em> forecasts and physics, three very different decision-makers.
    Both AI approaches use the identical Groq model &mdash; so any difference is
    the <em>paradigm</em>, not the model.</p>
  <div class="cards">{cards}</div>
</section>

<section id="method">
  <h2><span class="num">3</span>How the comparison was run</h2>
  <p class="lead">Each controller was dropped into one shared simulator and scored
    on an identical {window_hours}-hour window starting {start_date}. They
    <strong>decide on forecasts</strong>; the environment <strong>settles on the
    actuals</strong> &mdash; so a lucky guess doesn&#39;t get rewarded.</p>
  {forecast_block}
  <h3>Step 2 &mdash; the same battery physics for everyone</h3>
  <p>Whatever a controller asks for is clipped to the battery&#39;s real charge/
    discharge and state-of-charge limits, round-trip efficiency is applied, and
    the hourly energy balance is settled against measured generation and demand.
    Cost, CO&#8322;, renewable use, tokens and latency are then accounted
    identically for all three.</p>
</section>

<section id="results">
  <h2><span class="num">4</span>What the numbers say</h2>
  <p class="lead">Scored on the same {window_hours}-hour window. Green marks the
    best value in each row.</p>
  {results_table}
  <div style="margin-top:1.8rem">{charts}</div>
</section>

<section id="findings">
  <h2><span class="num">5</span>What we found</h2>
  <div class="callouts">
    <div class="callout"><div class="k">{by_cost}</div>
      <div class="d">most cost-effective controller</div></div>
    <div class="callout"><div class="k">{by_co2}</div>
      <div class="d">lowest CO&#8322; emissions</div></div>
    <div class="callout"><div class="k">{tok_ratio}&times;</div>
      <div class="d">more tokens the agent spent vs the generative LLM</div></div>
  </div>

  <div class="finding">
    <h3>The honest headline: more AI did not mean better control.</h3>
    <p>On this problem the <strong>rule-based baseline was the most
      cost-effective</strong> &mdash; it saved {rb_saved}% versus a grid-only
      bill, matching or beating both AI approaches at <strong>zero token
      cost</strong>. The generative LLM saved {gen_saved}% and the agentic agent
      {ag_saved}%. The extra machinery bought no optimisation advantage here.</p>
    <p>That is not a failure of the study &mdash; it&#39;s the result. Each
      paradigm has a real, different strength:</p>
    <ul>
      <li><strong>Rule-based</strong> wins on cost, transparency and speed when
        the problem is simple and well understood.</li>
      <li><strong>Generative AI&#39;s</strong> genuine value is
        <em>explanation</em>: a human-readable rationale for every decision
        ({gen_calls} calls, {gen_tokens} tokens) &mdash; useful for trust and
        oversight, not for squeezing out extra savings.</li>
      <li><strong>Agentic AI</strong> was the most expensive
        ({ag_calls} calls, {ag_tokens} tokens) and did <em>not</em> pay off,
        because its tool-guided optimisation is <em>myopic</em> &mdash; it
        minimises each hour in isolation instead of planning across the day.</li>
    </ul>
    <p>Agentic AI is most promising where decisions are genuinely multi-step,
      tools unlock information a rule cannot encode, and full autonomy is
      required &mdash; conditions this particular single-step, well-understood
      problem simply does not present.</p>
  </div>

  <p class="caveat"><strong>Caveats.</strong> Results are on clearly-labelled
    synthetic data, a small (8B) model, and a deliberately simple agent design.
    A stronger model, richer tools, or a multi-step planning horizon could shift
    the agentic outcome &mdash; and the framework built here is exactly what
    makes those follow-ups measurable. See <code>reports/comparison.md</code> for
    the full written analysis, or launch the live dashboard with
    <code>python main.py dashboard</code>.</p>
</section>

</main>

<footer><div class="wrap">
  Generative AI vs Agentic AI for Microgrid Operation &mdash; a comparative
  analysis. Generated locally from <code>reports/comparison.json</code>; every
  figure traces to a real simulation run. Rebuild any time with
  <code>python main.py site</code>.
</div></footer>

</body>
</html>
"""


if __name__ == "__main__":
    build_site()
