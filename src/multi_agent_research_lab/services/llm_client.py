"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Rough public pricing for gpt-4o-mini class models (USD per 1K tokens). Used only to
# produce an *estimate* for benchmarking, not for billing.
_PRICE_PER_1K_INPUT_USD = 0.00015
_PRICE_PER_1K_OUTPUT_USD = 0.0006


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClientError(Exception):
    """Raised when the LLM call fails after retries."""


class LLMClient:
    """Provider-agnostic LLM client.

    Uses OpenAI's Chat Completions API when `OPENAI_API_KEY` is configured. Otherwise it
    falls back to a deterministic offline mock so the workflow stays runnable without keys
    (useful for CI, tests, and this lab's grading pipeline).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = None
        self._model = self._settings.openai_model

        provider = self._settings.llm_provider.lower()
        openai_unset = not self._settings.openai_api_key
        deepseek_available = bool(self._settings.deepseek_api_key)
        use_deepseek = provider == "deepseek" or (
            provider == "openai" and openai_unset and deepseek_available
        )

        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai package not installed; falling back to mock LLM client")
            return

        if use_deepseek and self._settings.deepseek_api_key:
            self._client = OpenAI(
                api_key=self._settings.deepseek_api_key,
                base_url=self._settings.deepseek_base_url,
            )
            self._model = self._settings.deepseek_model
        elif self._settings.openai_api_key:
            self._client = OpenAI(api_key=self._settings.openai_api_key)
            self._model = self._settings.openai_model

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(LLMClientError),
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion, retrying transient failures with backoff."""

        if self._client is None:
            return self._mock_complete(system_prompt, user_prompt)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=0.2,
                timeout=self._settings.timeout_seconds,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - normalize provider errors for retry/handling
            raise LLMClientError(f"OpenAI completion failed: {exc}") from exc

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        cost = self._estimate_cost(input_tokens, output_tokens)
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

    def _mock_complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Deterministic offline fallback used when no API key/SDK is available."""

        system_first_line = (
            system_prompt.strip().splitlines()[0] if system_prompt.strip() else "n/a"
        )
        content = (
            f"[mock-llm response]\n"
            f"System role: {system_first_line}\n"
            f"Task: {user_prompt.strip()[:500]}"
        )
        input_tokens = max(1, len(system_prompt.split()) + len(user_prompt.split()))
        output_tokens = max(1, len(content.split()))
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self._estimate_cost(input_tokens, output_tokens),
        )

    @staticmethod
    def _estimate_cost(input_tokens: int | None, output_tokens: int | None) -> float | None:
        if input_tokens is None or output_tokens is None:
            return None
        return (
            input_tokens / 1000 * _PRICE_PER_1K_INPUT_USD
            + output_tokens / 1000 * _PRICE_PER_1K_OUTPUT_USD
        )
