"""Tracing hooks.

Emits a minimal local span (used to build `state.trace`) and, when `LANGSMITH_API_KEY` is
configured, also logs the span to LangSmith via its `Client.create_run` API so a real trace
UI is available without hard-binding the rest of the code to one provider.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context used across the workflow.

    Always records a local span dict (name, attributes, duration_seconds). When LangSmith
    credentials are present, also emits the span as a LangSmith run so it shows up in the
    LangSmith trace UI.
    """

    settings = get_settings()
    started_at = datetime.now(UTC)
    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    error: Exception | None = None
    try:
        yield span
    except Exception as exc:  # noqa: BLE001 - re-raised below after logging duration/trace
        error = exc
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        if settings.langsmith_api_key:
            _log_to_langsmith(settings.langsmith_project, name, span, started_at, error)


def _log_to_langsmith(
    project: str,
    name: str,
    span: dict[str, Any],
    started_at: datetime,
    error: Exception | None,
) -> None:
    try:
        from langsmith import Client

        client = Client()
        client.create_run(
            name=name,
            run_type="chain",
            project_name=project,
            inputs=span.get("attributes", {}),
            outputs={"duration_seconds": span["duration_seconds"]},
            start_time=started_at,
            end_time=datetime.now(UTC),
            error=str(error) if error else None,
        )
    except ImportError:
        logger.debug("langsmith not installed; skipping remote trace for %s", name)
    except Exception:  # noqa: BLE001 - tracing must never break the main workflow
        logger.warning("Failed to log span %s to LangSmith", name, exc_info=True)
