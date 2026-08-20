"""Benchmark report rendering."""

import html as html_lib
from collections.abc import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to a markdown report with a summary analysis section."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality (heuristic) | Judge score | "
        "Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        judge = "" if item.judge_score is None else f"{item.judge_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {judge} "
            f"| {citation} | {failure} | {item.notes} |"
        )

    lines.append("")
    lines.append("## Analysis")
    lines.append("")
    lines.extend(_analysis_bullets(metrics))
    lines.append("")

    return "\n".join(lines) + "\n"


def _analysis_bullets(metrics: list[BenchmarkMetrics]) -> list[str]:
    if not metrics:
        return ["- No runs recorded."]

    by_name = {m.run_name: m for m in metrics}
    bullets: list[str] = []

    if len(metrics) > 1:
        fastest = min(metrics, key=lambda m: m.latency_seconds)
        bullets.append(
            f"- **Fastest run:** `{fastest.run_name}` at {fastest.latency_seconds:.2f}s."
        )

        scored = [m for m in metrics if m.quality_score is not None]
        if scored:
            best_quality = max(scored, key=lambda m: m.quality_score or 0)
            bullets.append(
                f"- **Highest heuristic quality score:** `{best_quality.run_name}` "
                f"({best_quality.quality_score:.1f}/10)."
            )

        judged = [m for m in metrics if m.judge_score is not None]
        if judged:
            best_judged = max(judged, key=lambda m: m.judge_score or 0)
            bullets.append(
                f"- **Highest LLM-judge score:** `{best_judged.run_name}` "
                f"({best_judged.judge_score:.1f}/10) — {best_judged.judge_rationale}"
            )

        costed = [m for m in metrics if m.estimated_cost_usd is not None]
        if costed:
            cheapest = min(costed, key=lambda m: m.estimated_cost_usd or 0)
            bullets.append(
                f"- **Cheapest run:** `{cheapest.run_name}` "
                f"(${cheapest.estimated_cost_usd:.4f})."
            )

        baseline = by_name.get("baseline")
        multi = by_name.get("multi-agent")
        if baseline and multi:
            latency_delta = multi.latency_seconds - baseline.latency_seconds
            bullets.append(
                f"- Multi-agent was {'slower' if latency_delta > 0 else 'faster'} than "
                f"baseline by {abs(latency_delta):.2f}s."
            )
            if baseline.judge_score is not None and multi.judge_score is not None:
                judge_delta = multi.judge_score - baseline.judge_score
                bullets.append(
                    f"- Multi-agent judge score was {abs(judge_delta):.1f} points "
                    f"{'higher' if judge_delta >= 0 else 'lower'} than baseline."
                )

    for item in metrics:
        if item.failure_rate and item.failure_rate > 0:
            reason = item.notes or "see trace"
            bullets.append(f"- `{item.run_name}` had a nonzero failure rate: {reason}.")

    return bullets or ["- Runs completed with no notable differences to flag."]


_RUN_COLORS = {"baseline": "#6366a8", "multi-agent": "#1a9c72"}
_FALLBACK_COLORS = ["#e08a3c", "#c65d7b", "#3b82b4", "#8b6cc7"]


def render_html_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics as a self-contained, theme-aware HTML dashboard."""

    rows_html = "\n".join(_metric_row_html(m) for m in metrics)
    bullets_html = "\n".join(
        f"<li>{html_lib.escape(b.lstrip('- '))}</li>" for b in _analysis_bullets(metrics)
    )
    charts_html = _charts_html(metrics)
    run_count = len(metrics)
    distinct_names = len({m.run_name for m in metrics})
    query_count = run_count // distinct_names if distinct_names else 0

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Benchmark Report</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Inter:wght@400;500;600&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #f6f7fb;
    --surface: #ffffff;
    --surface-2: #eef0f8;
    --border: #dfe2ee;
    --text: #1b1d2b;
    --text-muted: #62667a;
    --accent: #6366a8;
    --accent-2: #1a9c72;
    --accent-3: #e08a3c;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #0d0e14;
      --surface: #16171f;
      --surface-2: #1c1e29;
      --border: #2a2c3a;
      --text: #eceefb;
      --text-muted: #9598ad;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #0d0e14;
    --surface: #16171f;
    --surface-2: #1c1e29;
    --border: #2a2c3a;
    --text: #eceefb;
    --text-muted: #9598ad;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: "Inter", -apple-system, "Segoe UI", Roboto, sans-serif;
    padding: 3rem 1.5rem 5rem;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 1040px; margin: 0 auto; }}
  .masthead {{ margin-bottom: 2.5rem; }}
  .eyebrow {{
    font-family: "Space Mono", monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 0 0 0.6rem;
  }}
  h1 {{
    font-family: "Sora", sans-serif;
    font-weight: 700;
    font-size: clamp(1.6rem, 3vw, 2.15rem);
    margin: 0 0 0.5rem;
    text-wrap: balance;
    letter-spacing: -0.01em;
  }}
  .subtitle {{ color: var(--text-muted); margin: 0 0 1.5rem; max-width: 62ch; line-height: 1.55; }}
  .meta-row {{ display: flex; gap: 1.25rem; flex-wrap: wrap; }}
  .meta-chip {{
    font-family: "Space Mono", monospace;
    font-size: 0.76rem;
    color: var(--text-muted);
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.35rem 0.7rem;
  }}
  .meta-chip b {{ color: var(--text); font-weight: 700; }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem 1.65rem;
    margin-bottom: 1.5rem;
    overflow-x: auto;
  }}
  .card h2 {{
    font-family: "Sora", sans-serif;
    font-weight: 600;
    font-size: 1rem;
    margin: 0 0 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }}
  .card h2 .dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--accent); flex-shrink: 0; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.87rem; white-space: nowrap; }}
  th, td {{ text-align: left; padding: 0.6rem 0.85rem; border-bottom: 1px solid var(--border); }}
  td {{ font-variant-numeric: tabular-nums; }}
  th {{
    color: var(--text-muted); font-weight: 600; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.05em;
    font-family: "Space Mono", monospace;
  }}
  tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover {{ background: var(--surface-2); }}
  .run-pill {{
    display: inline-block; padding: 0.2rem 0.65rem; border-radius: 999px;
    font-size: 0.76rem; font-weight: 600; color: white; font-family: "Sora", sans-serif;
  }}
  ul.analysis {{ margin: 0; padding-left: 1.15rem; line-height: 1.75; }}
  ul.analysis li {{ color: var(--text); }}
  ul.analysis li::marker {{ color: var(--accent); }}
  .chart-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 1.75rem; }}
  .chart h3 {{
    font-size: 0.76rem; color: var(--text-muted); margin: 0 0 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em; font-family: "Space Mono", monospace;
  }}
  .bar-row {{ display: flex; align-items: center; gap: 0.65rem; margin-bottom: 0.55rem; }}
  .bar-label {{ width: 84px; font-size: 0.78rem; color: var(--text-muted); flex-shrink: 0; }}
  .bar-track {{ flex: 1; background: var(--surface-2); border-radius: 6px; height: 13px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 6px; }}
  .bar-value {{
    width: 58px; text-align: right; font-size: 0.78rem; color: var(--text-muted);
    flex-shrink: 0; font-variant-numeric: tabular-nums; font-family: "Space Mono", monospace;
  }}
  footer {{
    color: var(--text-muted); font-size: 0.76rem; margin-top: 2.5rem; text-align: center;
    font-family: "Space Mono", monospace;
  }}
</style>
</head>
<body>
<div class="wrap">
  <div class="masthead">
    <p class="eyebrow">Multi-Agent Research Lab</p>
    <h1>Benchmark Report</h1>
    <p class="subtitle">
      Single-agent baseline vs. multi-agent workflow (Supervisor → Researcher → Analyst →
      Writer → Critic), measured across latency, cost, citation coverage, and an
      LLM-as-judge quality score.
    </p>
    <div class="meta-row">
      <span class="meta-chip">runs: <b>{run_count}</b></span>
      <span class="meta-chip">queries: <b>{query_count}</b></span>
    </div>
  </div>

  <div class="card">
    <h2><span class="dot"></span>Comparison charts</h2>
    <div class="chart-row">
{charts_html}
    </div>
  </div>

  <div class="card">
    <h2><span class="dot" style="background:var(--accent-2)"></span>Raw metrics</h2>
    <table>
      <thead>
        <tr>
          <th>Run</th><th>Latency (s)</th><th>Cost (USD)</th><th>Quality (heuristic)</th>
          <th>Judge score</th><th>Citation cov.</th><th>Failure rate</th><th>Notes</th>
        </tr>
      </thead>
      <tbody>
{rows_html}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2><span class="dot" style="background:var(--accent-3)"></span>Analysis</h2>
    <ul class="analysis">
{bullets_html}
    </ul>
  </div>

  <footer>generated by multi_agent_research_lab.evaluation.report</footer>
</div>
</body>
</html>
"""


def _run_color(run_name: str, index: int) -> str:
    return _RUN_COLORS.get(run_name, _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)])


def _metric_row_html(item: BenchmarkMetrics) -> str:
    color = _run_color(item.run_name, 0)
    cost = "—" if item.estimated_cost_usd is None else f"${item.estimated_cost_usd:.4f}"
    quality = "—" if item.quality_score is None else f"{item.quality_score:.1f}"
    judge = "—" if item.judge_score is None else f"{item.judge_score:.1f}"
    citation = "—" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
    failure = "—" if item.failure_rate is None else f"{item.failure_rate:.0%}"
    name = html_lib.escape(item.run_name)
    notes = html_lib.escape(item.notes)
    return (
        "<tr>"
        f'<td><span class="run-pill" style="background:{color}">{name}</span></td>'
        f"<td>{item.latency_seconds:.2f}</td><td>{cost}</td><td>{quality}</td>"
        f"<td>{judge}</td><td>{citation}</td><td>{failure}</td><td>{notes}</td>"
        "</tr>"
    )


def _charts_html(metrics: list[BenchmarkMetrics]) -> str:
    if not metrics:
        return '<p style="color:var(--text-muted)">No runs recorded.</p>'

    specs: list[tuple[str, Callable[[BenchmarkMetrics], float | None], str]] = [
        ("Latency (s)", lambda m: m.latency_seconds, "{:.1f}s"),
        ("Cost (USD)", lambda m: m.estimated_cost_usd, "${:.4f}"),
        ("Judge score /10", lambda m: m.judge_score, "{:.1f}"),
        ("Citation coverage", lambda m: (m.citation_coverage or 0) * 100, "{:.0f}%"),
    ]
    charts = []
    for title, getter, fmt in specs:
        values: list[tuple[str, float]] = [
            (m.run_name, v) for m in metrics if (v := getter(m)) is not None
        ]
        if not values:
            continue
        max_value = max(v for _, v in values) or 1.0
        bar_rows = []
        for i, (name, value) in enumerate(values):
            width = max(2, value / max_value * 100)
            color = _run_color(name, i)
            bar_rows.append(
                f'<div class="bar-row">\n'
                f'  <span class="bar-label">{html_lib.escape(name)}</span>\n'
                f'  <span class="bar-track">'
                f'<span class="bar-fill" style="width:{width:.1f}%;background:{color}">'
                f"</span></span>\n"
                f'  <span class="bar-value">{fmt.format(value)}</span>\n'
                f"</div>"
            )
        bars = "\n".join(bar_rows)
        charts.append(f'<div class="chart"><h3>{html_lib.escape(title)}</h3>{bars}</div>')

    return "\n".join(charts)
