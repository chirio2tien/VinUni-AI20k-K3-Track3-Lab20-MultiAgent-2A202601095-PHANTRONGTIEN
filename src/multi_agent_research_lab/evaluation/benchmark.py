"""Benchmark harness for single-agent vs multi-agent."""

import logging
import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, JudgeVerdict
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, LLMClientError

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_MIN_ANSWER_WORDS = 40

_JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluator scoring a research assistant's answer to a query. "
    "Score 0-10 on accuracy, completeness, clarity, and appropriate use of citations. "
    "Be strict: an unsupported claim or missing citation for a factual statement should "
    "lower the score. Give a short rationale and list concrete strengths/weaknesses."
)


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
    llm_client: LLMClient | None = None,
    use_judge: bool = True,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run `runner` once and compute latency, cost, citation coverage, and failure rate.

    `quality_score` is a cheap heuristic (0-10): answer length + citation coverage + error
    count. `judge_score` (when `use_judge=True`) is a second, LLM-as-judge opinion on the
    same answer — a closer stand-in for the peer review in `docs/peer_review_rubric.md`.
    Both are reported so a reader can see where they agree or diverge.
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

    citation_coverage = _citation_coverage(state)
    quality = _quality_score(state)

    judge_score: float | None = None
    judge_rationale = ""
    if use_judge and state.final_answer and not failed:
        verdict, judge_cost = _judge(query, state, llm_client or LLMClient())
        if verdict is not None:
            judge_score = verdict.score
            judge_rationale = verdict.rationale
            cost += judge_cost

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=cost or None,
        quality_score=quality,
        citation_coverage=citation_coverage,
        failure_rate=1.0 if failed else 0.0,
        judge_score=judge_score,
        judge_rationale=judge_rationale,
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


def _judge(
    query: str, state: ResearchState, llm_client: LLMClient
) -> tuple[JudgeVerdict | None, float]:
    """Ask an LLM to score the final answer.

    Returns (verdict, cost); verdict is None if the judge call itself failed.
    """

    sources_block = "\n".join(
        f"[{i}] {doc.title} — {doc.snippet}" for i, doc in enumerate(state.sources, start=1)
    )
    user_prompt = (
        f"Query: {query}\n\n"
        f"Answer:\n{state.final_answer}\n\n"
        f"Sources available to the answer (if any):\n{sources_block or 'none'}\n\n"
        "Score this answer."
    )
    try:
        verdict, response = llm_client.complete_structured(
            _JUDGE_SYSTEM_PROMPT, user_prompt, JudgeVerdict
        )
        return verdict, response.cost_usd or 0.0
    except LLMClientError as exc:
        logger.warning("LLM-as-judge scoring failed for run: %s", exc)
        return None, 0.0
