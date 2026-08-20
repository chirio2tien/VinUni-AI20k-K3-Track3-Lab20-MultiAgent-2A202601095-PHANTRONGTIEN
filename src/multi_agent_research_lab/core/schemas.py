"""Public schemas exchanged between CLI, agents, and evaluators."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AgentName(StrEnum):
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    CRITIC = "critic"


class ResearchQuery(BaseModel):
    query: str = Field(..., min_length=5)
    max_sources: int = Field(default=5, ge=1, le=20)
    audience: str = "technical learners"


class AgentResult(BaseModel):
    agent: AgentName
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceDocument(BaseModel):
    title: str
    url: str | None = None
    snippet: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkMetrics(BaseModel):
    run_name: str
    latency_seconds: float
    estimated_cost_usd: float | None = None
    quality_score: float | None = Field(default=None, ge=0, le=10)
    citation_coverage: float | None = Field(default=None, ge=0, le=1)
    failure_rate: float | None = Field(default=None, ge=0, le=1)
    judge_score: float | None = Field(default=None, ge=0, le=10)
    judge_rationale: str = ""
    notes: str = ""


class Claim(BaseModel):
    text: str
    supporting_source_indices: list[int] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class AnalysisResult(BaseModel):
    """Structured output for AnalystAgent."""

    key_claims: list[Claim] = Field(default_factory=list)
    conflicting_viewpoints: list[str] = Field(default_factory=list)
    weak_evidence_flags: list[str] = Field(default_factory=list)
    summary: str = ""


class CriticIssue(BaseModel):
    severity: str = Field(description="one of: low, medium, high")
    description: str


class CriticReview(BaseModel):
    """Structured output for CriticAgent."""

    issues: list[CriticIssue] = Field(default_factory=list)
    citation_coverage: float = Field(ge=0, le=1)
    verdict: str = Field(description="one of: pass, needs_revision")


class JudgeVerdict(BaseModel):
    """LLM-as-judge structured output used by the benchmark harness."""

    score: float = Field(ge=0, le=10)
    rationale: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
