"""The model registry and the request/retry logic of the chat client, all offline."""

from types import SimpleNamespace

try:  # the openai SDK depends on httpx2 from 3.0 on and on httpx before
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx
import openai
import pytest
import yaml

from socialmaze.llm.client import (
    EMPTY_API_KEY,
    ChatClient,
    ChatResult,
    ModelSpec,
    api_key_for,
    backoff_delay,
    build_request_kwargs,
    make_client,
    result_from_completion,
    should_retry,
)
from socialmaze.llm.mock import BEHAVIOURS, MockClient
from socialmaze.llm.registry import DEFAULT_CONFIG_PATH, Registry, load_env

PAPER_MODELS = [
    "gpt-4o", "gpt-4o-mini", "o1", "o3-mini", "deepseek-v3", "deepseek-r1", "qwq-32b",
    "qwen-2.5-72b", "llama-3.3-70b", "llama-3.1-8b", "phi-4", "gemini-2.5-pro",
]
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


@pytest.fixture(scope="module")
def registry():
    return Registry.load()


# -- registry -------------------------------------------------------------


def test_default_yaml_loads_with_paper_models(registry):
    assert registry.path == DEFAULT_CONFIG_PATH
    names = registry.names()
    assert names[:12] == PAPER_MODELS
    assert {"mock", "mock-wrong", "mock-garbage", "mock-truncate", "mock-flaky"} <= set(names)
    for name in PAPER_MODELS:
        spec = registry.resolve(name)
        assert spec.name == name and spec.model and spec.provider in registry.providers


def test_resolve_exact_o1_mapping(registry):
    spec = registry.resolve("o1")
    assert spec.provider == "openai" and spec.model == "o1"
    assert spec.supports_temperature is False
    assert spec.max_tokens_param == "max_completion_tokens"
    assert spec.max_tokens == 8192
    assert spec.base_url is None and spec.api_key_env == "OPENAI_API_KEY"
    assert spec.timeout == 600.0


def test_resolve_defaults_and_long_cot_caps(registry):
    assert registry.resolve("gpt-4o").max_tokens == 4096
    assert registry.resolve("gpt-4o").supports_temperature is True
    for name in ("o1", "o3-mini", "deepseek-r1", "qwq-32b", "gemini-2.5-pro"):
        assert registry.resolve(name).max_tokens == 8192, name
    deepseek = registry.resolve("deepseek-r1")
    assert deepseek.base_url == "https://api.deepseek.com" and deepseek.api_key_env == "DEEPSEEK_API_KEY"


def test_resolve_provider_slash_model(registry):
    spec = registry.resolve("openrouter/anthropic/claude-sonnet-4.5")
    assert spec.provider == "openrouter"
    assert spec.model == "anthropic/claude-sonnet-4.5"
    assert spec.base_url == "https://openrouter.ai/api/v1"
    assert spec.api_key_env == "OPENROUTER_API_KEY"
    assert spec.max_tokens == 4096 and spec.max_tokens_param == "max_tokens"
    local = registry.resolve("local/my-model")
    assert local.base_url == "http://localhost:8000/v1" and local.api_key_env is None


def test_resolve_mock_variants(registry):
    assert registry.resolve("mock").provider == "mock"
    assert registry.resolve("mock").model == "oracle"
    assert registry.resolve("mock:wrong").model == "wrong"
    assert registry.resolve("mock-flaky").model == "flaky"
    for behaviour in BEHAVIOURS:
        assert registry.resolve(f"mock:{behaviour}").model == behaviour
    with pytest.raises(ValueError, match="mock behaviour"):
        registry.resolve("mock:bogus")


def test_resolve_unknown_names(registry):
    with pytest.raises(ValueError) as exc:
        registry.resolve("gpt-5-ultra")
    assert "gpt-4o" in str(exc.value) and "provider/model-id" in str(exc.value)
    with pytest.raises(ValueError):
        registry.resolve("noprovider/some-model")
    with pytest.raises(ValueError):
        registry.resolve("openai/")


def test_registry_validation(tmp_path):
    good = {"providers": {"p": {"base_url": "http://x", "api_key_env": "K"}},
            "defaults": {"max_tokens": 10}, "models": {"m": {"provider": "p", "model": "id", "timeout": 5}}}
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(good))
    reg = Registry.load(path)
    spec = reg.resolve("m")
    assert spec.max_tokens == 10 and spec.timeout == 5.0 and spec.base_url == "http://x"

    bad_key = {**good, "models": {"m": {"provider": "p", "model": "id", "max_token": 1}}}
    path.write_text(yaml.safe_dump(bad_key))
    with pytest.raises(ValueError, match="unknown key"):
        Registry.load(path)

    bad_provider = {**good, "models": {"m": {"provider": "nope", "model": "id"}}}
    path.write_text(yaml.safe_dump(bad_provider))
    with pytest.raises(ValueError, match="unknown provider"):
        Registry.load(path)

    path.write_text("")
    assert Registry.load(path).names() == []


def test_model_entry_can_override_provider_fields(tmp_path):
    cfg = {"providers": {"local": {"base_url": "http://localhost:8000/v1", "api_key_env": None}},
           "models": {"other": {"provider": "local", "model": "x", "base_url": "http://gpu:9000/v1",
                                "extra_body": {"top_k": 5}}}}
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(cfg))
    spec = Registry.load(path).resolve("other")
    assert spec.base_url == "http://gpu:9000/v1" and spec.extra_body == {"top_k": 5}


# -- request mapping ------------------------------------------------------


def test_build_request_kwargs_mapping(registry):
    o1 = build_request_kwargs(registry.resolve("o1"), MESSAGES, 0.7)
    assert o1 == {"model": "o1", "messages": MESSAGES, "max_completion_tokens": 8192}
    gpt = build_request_kwargs(registry.resolve("gpt-4o"), MESSAGES, 0.7)
    assert gpt == {"model": "gpt-4o", "messages": MESSAGES, "temperature": 0.7, "max_tokens": 4096}
    assert build_request_kwargs(registry.resolve("gpt-4o"), MESSAGES, 0.0, max_tokens=100)["max_tokens"] == 100
    no_cap = ModelSpec("x", "local", "x", max_tokens_param=None, extra_body={"top_k": 5})
    kwargs = build_request_kwargs(no_cap, MESSAGES, 0.5)
    assert "max_tokens" not in kwargs and "max_completion_tokens" not in kwargs
    assert kwargs["extra_body"] == {"top_k": 5}
    kwargs["messages"].append({"role": "user", "content": "x"})
    assert len(MESSAGES) == 2


# -- retry classification -------------------------------------------------


def _request():
    return httpx.Request("POST", "https://example.invalid/v1/chat/completions")


def _status_error(cls, status):
    return cls("failed", response=httpx.Response(status, request=_request()), body=None)


@pytest.mark.parametrize("exc,expected", [
    (_status_error(openai.RateLimitError, 429), True),
    (_status_error(openai.InternalServerError, 500), True),
    (_status_error(openai.APIStatusError, 502), True),
    (_status_error(openai.APIStatusError, 503), True),
    (openai.APIConnectionError(request=_request()), True),
    (openai.APITimeoutError(request=_request()), True),
    (_status_error(openai.AuthenticationError, 401), False),
    (_status_error(openai.BadRequestError, 400), False),
    (_status_error(openai.PermissionDeniedError, 403), False),
    (_status_error(openai.NotFoundError, 404), False),
    (_status_error(openai.APIStatusError, 422), False),
    (ValueError("not an api error"), False),
])
def test_should_retry(exc, expected):
    assert should_retry(exc) is expected


def test_backoff_delay_grows_and_is_capped():
    import random

    rng = random.Random(0)
    delays = [backoff_delay(a, rng, base=1.0, cap=30.0) for a in range(1, 10)]
    for attempt, delay in enumerate(delays, start=1):
        nominal = min(30.0, 2 ** (attempt - 1))
        assert 0.5 * nominal <= delay < 1.5 * nominal
    assert max(delays) < 45.0


# -- the client without network ------------------------------------------


class FakeCompletions:
    """``chat.completions.create`` that raises the queued exceptions first, then returns."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def fake_client(outcomes):
    completions = FakeCompletions(outcomes)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def completion(content="Final Criminal Is Player 4.\nMy Role Is Lunatic.", finish_reason="stop", reasoning=None):
    message = SimpleNamespace(content=content, role="assistant")
    if reasoning is not None:
        message.reasoning_content = reasoning
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )


def test_chat_client_retries_transient_errors_then_succeeds():
    spec = ModelSpec("m", "local", "m")
    client, completions = fake_client([
        _status_error(openai.RateLimitError, 429), openai.APITimeoutError(request=_request()), completion(),
    ])
    sleeps = []
    chat = ChatClient(spec, max_retries=5, client=client, sleep=sleeps.append)
    result = chat.chat(MESSAGES, 0.7)
    assert result.ok and result.attempts == 3 and len(sleeps) == 2
    assert result.text.endswith("My Role Is Lunatic.")
    assert result.prompt_tokens == 11 and result.completion_tokens == 7
    assert result.finish_reason == "stop" and result.reasoning_text is None
    assert len(completions.calls) == 3 and completions.calls[0]["model"] == "m"


def test_chat_client_does_not_retry_auth_errors():
    client, completions = fake_client([_status_error(openai.AuthenticationError, 401), completion()])
    sleeps = []
    result = ChatClient(ModelSpec("m", "local", "m"), client=client, sleep=sleeps.append).chat(MESSAGES, 0.7)
    assert not result.ok and result.attempts == 1 and sleeps == []
    assert result.error.startswith("AuthenticationError:") and result.text == ""
    assert len(completions.calls) == 1


def test_chat_client_gives_up_after_max_retries_without_raising():
    errors = [_status_error(openai.InternalServerError, 500) for _ in range(3)]
    client, _ = fake_client(errors)
    result = ChatClient(ModelSpec("m", "local", "m"), max_retries=2, client=client, sleep=lambda s: None).chat(MESSAGES, 0.7)
    assert result.error.startswith("InternalServerError:") and result.attempts == 3


def test_chat_client_unexpected_exception_becomes_error():
    client, _ = fake_client([RuntimeError("weird")])
    result = ChatClient(ModelSpec("m", "local", "m"), client=client, sleep=lambda s: None).chat(MESSAGES, 0.7)
    assert result.error == "RuntimeError: weird" and result.attempts == 1


def test_result_from_completion_edge_cases():
    empty = result_from_completion(completion(content=None), 0.1, 1)
    assert empty.error.startswith("empty response") and empty.text == ""
    cut = result_from_completion(completion(content="", finish_reason="length"), 0.1, 1)
    assert cut.ok and cut.finish_reason == "length" and cut.text == ""
    reasoned = result_from_completion(completion(reasoning="thinking..."), 0.2, 2)
    assert reasoned.reasoning_text == "thinking..." and reasoned.attempts == 2 and reasoned.latency_s == 0.2
    no_choices = result_from_completion(SimpleNamespace(choices=[], usage=None), 0.0, 1)
    assert no_choices.error == "empty response: no choices"


def test_result_from_real_sdk_types():
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    message = ChatCompletionMessage(role="assistant", content="hello", reasoning_content="why")
    cc = ChatCompletion(id="x", created=0, model="m", object="chat.completion",
                        choices=[Choice(index=0, finish_reason="stop", message=message)],
                        usage=CompletionUsage(prompt_tokens=3, completion_tokens=1, total_tokens=4))
    result = result_from_completion(cc, 0.0, 1)
    assert result.text == "hello" and result.reasoning_text == "why" and result.prompt_tokens == 3


def test_chat_result_ok():
    assert ChatResult(text="x").ok is True
    assert ChatResult(text="", error="e").ok is False


# -- keys, environment, factory --------------------------------------------


def test_api_key_for(monkeypatch, registry):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert api_key_for(registry.resolve("gpt-4o")) == "sk-test"
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        api_key_for(registry.resolve("gpt-4o"))
    assert api_key_for(registry.resolve("local/x")) == EMPTY_API_KEY
    assert api_key_for(registry.resolve("mock")) == EMPTY_API_KEY
    monkeypatch.delenv("LOCAL_KEY", raising=False)
    assert api_key_for(ModelSpec("l", "local", "x", api_key_env="LOCAL_KEY")) == EMPTY_API_KEY


def test_make_client(monkeypatch, registry):
    assert isinstance(make_client(registry.resolve("mock:wrong")), MockClient)
    assert make_client(registry.resolve("mock:wrong")).behaviour == "wrong"
    local = make_client(registry.resolve("local/x"))
    assert isinstance(local, ChatClient)
    assert str(local._client.base_url).startswith("http://localhost:8000/v1")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        make_client(registry.resolve("deepseek-v3"))


def test_load_env_reads_dotenv_from_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SOCIALMAZE_TEST_KEY", raising=False)
    assert load_env() == []
    (tmp_path / ".env").write_text("SOCIALMAZE_TEST_KEY=from-file\n")
    loaded = load_env()
    assert loaded and loaded[0] == (tmp_path / ".env").resolve()
    import os
    assert os.environ["SOCIALMAZE_TEST_KEY"] == "from-file"
    monkeypatch.setenv("SOCIALMAZE_TEST_KEY", "from-env")
    load_env()
    assert os.environ["SOCIALMAZE_TEST_KEY"] == "from-env"
