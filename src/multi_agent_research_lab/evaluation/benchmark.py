"""Benchmark harness for single-agent vs multi-agent."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_MIN_ANSWER_WORDS = 40


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run `runner` once and compute latency, cost, citation coverage, and failure rate.

    Quality is a cheap heuristic (0-10) meant as a stand-in for real peer review: it checks
    that a final answer exists, is reasonably long, and is (partially) grounded in cited
    sources. Replace/augment with `docs/peer_review_rubric.md` scores for real evaluation.
    """

    started = perf_counter()
    failed = False
    notes = ""
    try:
        state = runner(query)
    except Exception as exc:  # noqa: BLE001 - benchmark must capture failures, not crash
        latency = perf_counter() - started
        empty_state = ResearchState(request={"query": query})  # type: ignore[arg-type]
        metrics = BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=latency,
            failure_rate=1.0,
            notes=f"runner raised {type(exc).__name__}: {exc}",
        )
        return empty_state, metrics

    latency = perf_counter() - started

    if not state.final_answer:
        failed = True
        notes = "no final_answer produced"
    elif state.errors:
        notes = f"{len(state.errors)} non-fatal error(s) recorded"

    cost = sum(
        result.metadata.get("cost_usd") or 0.0
        for result in state.agent_results
        if result.metadata.get("cost_usd") is not None
    )
    cost = cost or None

    citation_coverage = _citation_coverage(state)
    quality = _quality_score(state)

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=cost,
        quality_score=quality,
        citation_coverage=citation_coverage,
        failure_rate=1.0 if failed else 0.0,
        notes=notes,
    )
    return state, metrics


def _citation_coverage(state: ResearchState) -> float | None:
    if not state.final_answer or not state.sources:
        return None
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", state.final_answer) if s.strip()]
    if not sentences:
        return None
    cited = sum(1 for s in sentences if _CITATION_PATTERN.search(s))
    return cited / len(sentences)


def _quality_score(state: ResearchState) -> float | None:
    if not state.final_answer:
        return None

    score = 0.0
    word_count = len(state.final_answer.split())
    score += min(4.0, 4.0 * word_count / _MIN_ANSWER_WORDS)

    coverage = _citation_coverage(state)
    if coverage is not None:
        score += 4.0 * coverage
    elif not state.sources:
        # No sources were ever gathered (e.g. single-agent baseline) - citations aren't
        # expected, so don't penalize on this axis.
        score += 2.0

    score += 2.0 if not state.errors else max(0.0, 2.0 - len(state.errors))

    return round(min(10.0, score), 1)
