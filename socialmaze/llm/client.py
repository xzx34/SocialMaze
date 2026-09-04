"""One OpenAI-compatible chat client for every provider.

Design
------
Every provider used by SocialMaze (OpenAI, DeepSeek, DeepInfra, OpenRouter,
the OpenAI-compatible endpoints of Anthropic and Gemini, local vLLM or Ollama
servers) speaks the chat completions API, so a single client built on the
``openai`` SDK serves all of them. What differs between providers is captured
in :class:`ModelSpec` (base URL, key variable, whether ``temperature`` is
accepted, the name of the completion-cap parameter, extra request fields) and
nothing else. :func:`build_request_kwargs` turns a spec and a conversation
into the request; it is a pure function so the parameter mapping can be
tested without network access.

Failure handling is deliberately simple for a batch evaluator:

* transient failures (rate limits, connection problems, timeouts, 5xx) are
  retried with exponential backoff and jitter, see :func:`should_retry`;
* everything else (bad request, authentication, unexpected exceptions) is
  not retried;
* :meth:`ChatClient.chat` never raises. A final failure is returned as a
  :class:`ChatResult` whose ``error`` is set, so one bad call cannot abort a
  run over hundreds of scenarios. :mod:`socialmaze.hrd.evaluate` records the
  error per round and re-queries such records on the next run.

:func:`make_client` returns the offline :class:`socialmaze.llm.mock.MockClient`
for the ``mock`` provider and a :class:`ChatClient` for everything else.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, Sequence

import openai

DEFAULT_MAX_RETRIES = 5
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 30.0

#: Placeholder key for servers that do not check credentials (vLLM, Ollama).
EMPTY_API_KEY = "EMPTY"
LOCAL_PROVIDER = "local"
MOCK_PROVIDER = "mock"

#: ``finish_reason`` reported when the completion cap cut the reply.
TRUNCATED_FINISH_REASON = "length"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """Everything needed to talk to one model through the chat completions API.

    ``name`` is the registry key or command-line name, ``model`` the id sent
    to the provider. ``max_tokens_param`` is ``"max_tokens"`` for most models,
    ``"max_completion_tokens"`` for OpenAI o-series reasoning models and
    ``None`` to send no cap at all. ``extra_body`` holds provider-specific
    request fields that are passed through unchanged.
    """

    name: str
    provider: str
    model: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    supports_temperature: bool = True
    max_tokens_param: Optional[str] = "max_tokens"
    max_tokens: int = 4096
    extra_body: dict = field(default_factory=dict)
    timeout: float = 600.0
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChatResult:
    """The outcome of one chat call; ``error`` is set instead of raising."""

    text: str
    reasoning_text: Optional[str] = None
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_s: float = 0.0
    error: Optional[str] = None
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.error is None


class BaseClient:
    """Interface shared by :class:`ChatClient` and the mock client."""

    def chat(
        self,
        messages: Sequence[dict],
        temperature: float,
        max_tokens: Optional[int] = None,
        context: Optional[dict] = None,
    ) -> ChatResult:
        """Send ``messages`` and return a :class:`ChatResult`, never raising.

        ``context`` carries evaluation bookkeeping (ground truth, round
        index, scenario id); the real client ignores it, the mock uses it.
        """
        raise NotImplementedError


# --------------------------------------------------------------------------
# Pure helpers (testable without network)
# --------------------------------------------------------------------------


def api_key_for(spec: ModelSpec) -> str:
    """The API key for ``spec`` from the environment.

    Providers without a key variable (mock, local servers) get the
    placeholder ``EMPTY``; a local server whose variable is unset does too.
    Any other provider with an unset variable is a configuration error and
    is reported before a single request is made.
    """
    if spec.api_key_env is None:
        return EMPTY_API_KEY
    value = os.environ.get(spec.api_key_env)
    if value:
        return value
    if spec.provider == LOCAL_PROVIDER:
        return EMPTY_API_KEY
    raise ValueError(
        f"environment variable {spec.api_key_env} is not set (needed for model "
        f"{spec.name!r}); export it or add it to .env (see .env.example)"
    )


def build_openai_client(spec: ModelSpec) -> openai.OpenAI:
    """An ``openai.OpenAI`` client for ``spec`` with the SDK's own retries off."""
    return openai.OpenAI(
        api_key=api_key_for(spec),
        base_url=spec.base_url,
        timeout=spec.timeout,
        max_retries=0,
    )


def build_request_kwargs(
    spec: ModelSpec,
    messages: Sequence[dict],
    temperature: float,
    max_tokens: Optional[int] = None,
) -> dict:
    """Keyword arguments for ``chat.completions.create`` for one call.

    ``temperature`` is sent only if the model supports it, the completion cap
    (``max_tokens`` argument, else ``spec.max_tokens``) under the parameter
    name the model expects, and ``spec.extra_body`` is passed through.
    """
    kwargs: dict[str, Any] = {"model": spec.model, "messages": list(messages)}
    if spec.supports_temperature:
        kwargs["temperature"] = temperature
    if spec.max_tokens_param:
        kwargs[spec.max_tokens_param] = max_tokens if max_tokens is not None else spec.max_tokens
    if spec.extra_body:
        kwargs["extra_body"] = dict(spec.extra_body)
    return kwargs


def should_retry(exc: BaseException) -> bool:
    """Whether ``exc`` is transient: rate limit, connection or timeout, 5xx."""
    if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500
    return False


def backoff_delay(
    attempt: int,
    rng: random.Random,
    base: float = BACKOFF_BASE_S,
    cap: float = BACKOFF_CAP_S,
) -> float:
    """Seconds to wait after failed attempt number ``attempt`` (1-based).

    Exponential growth capped at ``cap``, multiplied by a jitter factor drawn
    uniformly from ``[0.5, 1.5)`` so that parallel workers do not retry in
    lockstep.
    """
    return min(cap, base * 2 ** (attempt - 1)) * (0.5 + rng.random())


def describe_error(exc: BaseException) -> str:
    """``"ExceptionType: message"`` as stored in result records."""
    return f"{type(exc).__name__}: {exc}"


def result_from_completion(completion: Any, latency_s: float, attempts: int) -> ChatResult:
    """Convert a chat completion object into a :class:`ChatResult`.

    ``reasoning_text`` is taken from ``message.reasoning_content`` (DeepSeek)
    or ``message.reasoning`` (OpenRouter) when present. An empty ``content``
    is an error unless the reply was cut by the completion cap, in which case
    it is a truncation and is scored as such by the evaluator.
    """
    if not completion.choices:
        return ChatResult(text="", error="empty response: no choices", latency_s=latency_s, attempts=attempts)
    choice = completion.choices[0]
    message = choice.message
    text = message.content or ""
    reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    usage = completion.usage
    result = ChatResult(
        text=text,
        reasoning_text=reasoning if isinstance(reasoning, str) and reasoning else None,
        finish_reason=choice.finish_reason,
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
        latency_s=latency_s,
        attempts=attempts,
    )
    if not text.strip() and result.finish_reason != TRUNCATED_FINISH_REASON:
        result.error = f"empty response (finish_reason={result.finish_reason})"
    return result


# --------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------


class ChatClient(BaseClient):
    """Chat completions with retries; see the module docstring.

    ``client`` (an object with ``chat.completions.create``) and ``sleep`` can
    be injected for tests; by default an ``openai.OpenAI`` client is built
    from the spec and retries wait with :func:`time.sleep`.
    """

    def __init__(
        self,
        spec: ModelSpec,
        max_retries: int = DEFAULT_MAX_RETRIES,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.spec = spec
        self.max_retries = max_retries
        self._client = client if client is not None else build_openai_client(spec)
        self._sleep = sleep
        self._rng = rng if rng is not None else random.Random()

    def chat(
        self,
        messages: Sequence[dict],
        temperature: float,
        max_tokens: Optional[int] = None,
        context: Optional[dict] = None,
    ) -> ChatResult:
        kwargs = build_request_kwargs(self.spec, messages, temperature, max_tokens)
        start = time.monotonic()
        attempt = 0
        while True:
            attempt += 1
            try:
                completion = self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - every failure becomes a result
                if attempt <= self.max_retries and should_retry(exc):
                    self._sleep(backoff_delay(attempt, self._rng))
                    continue
                return ChatResult(
                    text="",
                    error=describe_error(exc),
                    latency_s=time.monotonic() - start,
                    attempts=attempt,
                )
            return result_from_completion(completion, time.monotonic() - start, attempt)


def make_client(spec: ModelSpec, **kwargs: Any) -> BaseClient:
    """A :class:`MockClient` for the ``mock`` provider, else a :class:`ChatClient`."""
    if spec.provider == MOCK_PROVIDER:
        from .mock import MockClient

        return MockClient(spec.model)
    return ChatClient(spec, **kwargs)


__all__ = [
    "DEFAULT_MAX_RETRIES", "EMPTY_API_KEY", "LOCAL_PROVIDER", "MOCK_PROVIDER",
    "TRUNCATED_FINISH_REASON", "ModelSpec", "ChatResult", "BaseClient",
    "api_key_for", "build_openai_client", "build_request_kwargs", "should_retry",
    "backoff_delay", "describe_error", "result_from_completion", "ChatClient",
    "make_client",
]
