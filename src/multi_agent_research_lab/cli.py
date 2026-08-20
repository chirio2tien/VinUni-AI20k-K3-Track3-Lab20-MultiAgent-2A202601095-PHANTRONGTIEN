"""Command-line entrypoint for the lab starter."""

import uuid
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_html_report, render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def _run_baseline(query: str) -> ResearchState:
    request = _parse_query(query)
    state = ResearchState(request=request)

    llm = LLMClient()
    system_prompt = (
        "You are a single research assistant handling everything yourself: search your "
        "own knowledge, analyze it, and write a clear final answer for the given audience. "
        "Note where your knowledge may be incomplete or outdated."
    )
    response = llm.complete(system_prompt, request.query)
    state.final_answer = response.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
    )
    return state


def _run_multi_agent(query: str, thread_id: str | None = None) -> ResearchState:
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    return workflow.run(state, thread_id=thread_id)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline that answers the query directly with one LLM call."""

    _init()
    state = _run_baseline(query)
    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow: Supervisor -> Researcher -> Analyst -> Writer -> Critic.

    Streams progress node-by-node (checkpointed under a fresh thread id, so a crashed run
    could be resumed by re-invoking the workflow with the same thread id).
    """

    _init()
    thread_id = str(uuid.uuid4())
    workflow = MultiAgentWorkflow()
    initial_state = ResearchState(request=_parse_query(query))
    final_state = initial_state

    try:
        with console.status("Starting workflow...") as status:
            for step_state in workflow.run_streaming(initial_state, thread_id=thread_id):
                final_state = step_state
                last_route = step_state.route_history[-1] if step_state.route_history else "?"
                status.update(f"[{step_state.iteration}] last step: {last_route}")
                console.print(f"  -> {last_route} (iteration {step_state.iteration})")
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc

    console.print(final_state.model_dump_json(indent=2))


@app.command()
def benchmark(
    config_path: Annotated[
        str, typer.Option("--config", "-c", help="Path to a lab config YAML")
    ] = "configs/lab_default.yaml",
    output: Annotated[
        str, typer.Option("--output", "-o", help="Relative report path under reports/")
    ] = "benchmark_report.md",
) -> None:
    """Run baseline and multi-agent over the configured queries and write a report."""

    _init()
    queries = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["benchmark"]["queries"]

    metrics = []
    for idx, query in enumerate(queries, start=1):
        console.print(f"[{idx}/{len(queries)}] baseline: {query}")
        _, baseline_metrics = run_benchmark("baseline", query, _run_baseline)
        metrics.append(baseline_metrics)

        console.print(f"[{idx}/{len(queries)}] multi-agent: {query}")
        _, multi_metrics = run_benchmark("multi-agent", query, _run_multi_agent)
        metrics.append(multi_metrics)

    store = LocalArtifactStore()
    md_path = store.write_text(output, render_markdown_report(metrics))
    html_output = Path(output).with_suffix(".html").as_posix()
    html_path = store.write_text(html_output, render_html_report(metrics))
    console.print(
        Panel.fit(
            f"Reports written to {md_path} and {html_path}", title="Benchmark", style="green"
        )
    )


if __name__ == "__main__":
    app()
