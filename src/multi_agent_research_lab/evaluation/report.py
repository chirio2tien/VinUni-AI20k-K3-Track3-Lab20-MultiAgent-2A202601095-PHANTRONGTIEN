"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to a markdown report with a summary analysis section."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
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
                f"- **Highest quality score:** `{best_quality.run_name}` "
                f"({best_quality.quality_score:.1f}/10)."
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
            if baseline.quality_score is not None and multi.quality_score is not None:
                quality_delta = multi.quality_score - baseline.quality_score
                bullets.append(
                    f"- Multi-agent quality score was {abs(quality_delta):.1f} points "
                    f"{'higher' if quality_delta >= 0 else 'lower'} than baseline."
                )

    for item in metrics:
        if item.failure_rate and item.failure_rate > 0:
            reason = item.notes or "see trace"
            bullets.append(f"- `{item.run_name}` had a nonzero failure rate: {reason}.")

    return bullets or ["- Runs completed with no notable differences to flag."]
