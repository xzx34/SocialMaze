"""Model registry: ``configs/models.yaml`` resolved into :class:`ModelSpec` objects.

The YAML file has three sections:

``providers``
    name -> ``{base_url, api_key_env}``. ``base_url: null`` means the SDK's
    default (api.openai.com); ``api_key_env: null`` means no key is needed.
``defaults``
    request settings shared by all models (``max_tokens``, ``timeout``,
    ``supports_temperature``, ``max_tokens_param``, ``extra_body``).
``models``
    name -> ``{provider, model, <overrides of the defaults>}``. A model entry
    may also override its provider's ``base_url`` or ``api_key_env``.

:meth:`Registry.resolve` accepts, in this order: a key of ``models``; a
``provider/model-id`` string split at the first slash, so any model of a
known provider can be named on the command line without editing the file;
``mock`` or ``mock:<behaviour>`` for the offline client. Unknown names raise
``ValueError`` listing what is available. :func:`load_env` reads API keys
from a ``.env`` file in the working directory or the repository root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import yaml

from .client import MOCK_PROVIDER, ModelSpec
from .mock import BEHAVIOURS, DEFAULT_BEHAVIOUR

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "models.yaml"
ENV_FILE = ".env"
MOCK_PREFIX = f"{MOCK_PROVIDER}:"

PROVIDER_KEYS = frozenset({"base_url", "api_key_env"})
DEFAULT_KEYS = frozenset({"max_tokens", "timeout", "supports_temperature", "max_tokens_param", "extra_body"})
MODEL_KEYS = frozenset({"provider", "model", "notes"} | PROVIDER_KEYS | DEFAULT_KEYS)

PathLike = Union[str, Path]


def _check_keys(section: str, name: str, entry: dict, allowed: frozenset) -> None:
    unknown = sorted(set(entry) - allowed)
    if unknown:
        raise ValueError(
            f"{section} {name!r}: unknown key(s) {', '.join(unknown)}; "
            f"allowed: {', '.join(sorted(allowed))}"
        )


@dataclass
class Registry:
    """The parsed registry; build it with :meth:`load`."""

    providers: dict[str, dict] = field(default_factory=dict)
    defaults: dict = field(default_factory=dict)
    models: dict[str, dict] = field(default_factory=dict)
    path: Optional[Path] = None

    @classmethod
    def load(cls, path: Optional[PathLike] = None) -> "Registry":
        """Parse ``path`` (default ``configs/models.yaml`` in the repository)."""
        path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        registry = cls(
            providers={k: dict(v or {}) for k, v in (raw.get("providers") or {}).items()},
            defaults=dict(raw.get("defaults") or {}),
            models={k: dict(v or {}) for k, v in (raw.get("models") or {}).items()},
            path=path,
        )
        registry.validate()
        return registry

    def validate(self) -> None:
        """Raise ``ValueError`` on unknown keys or references to unknown providers."""
        for name, entry in self.providers.items():
            _check_keys("provider", name, entry, PROVIDER_KEYS)
        _check_keys("section", "defaults", self.defaults, DEFAULT_KEYS)
        for name, entry in self.models.items():
            _check_keys("model", name, entry, MODEL_KEYS)
            if "provider" not in entry or "model" not in entry:
                raise ValueError(f"model {name!r} needs both 'provider' and 'model'")
            if entry["provider"] not in self.providers:
                raise ValueError(
                    f"model {name!r} uses unknown provider {entry['provider']!r}; "
                    f"known providers: {', '.join(self.providers)}"
                )

    def names(self) -> list[str]:
        """Model names in file order."""
        return list(self.models)

    def resolve(self, name: str) -> ModelSpec:
        """The :class:`ModelSpec` for a registry key, ``provider/model`` or ``mock[:behaviour]``."""
        if name in self.models:
            return self.spec(name, self.models[name])
        if "/" in name:
            provider, model = name.split("/", 1)
            if provider in self.providers and model:
                return self.spec(name, {"provider": provider, "model": model})
        if name == MOCK_PROVIDER or name.startswith(MOCK_PREFIX):
            behaviour = name[len(MOCK_PREFIX):] if name.startswith(MOCK_PREFIX) else DEFAULT_BEHAVIOUR
            if behaviour not in BEHAVIOURS:
                raise ValueError(
                    f"unknown mock behaviour {behaviour!r}; expected one of {', '.join(BEHAVIOURS)}"
                )
            return self.spec(name, {"provider": MOCK_PROVIDER, "model": behaviour})
        raise ValueError(
            f"unknown model {name!r}. Use a name from {self.path or 'the registry'} "
            f"({', '.join(self.names())}), 'provider/model-id' with one of the providers "
            f"({', '.join(self.providers)}), or 'mock'."
        )

    def spec(self, name: str, entry: dict) -> ModelSpec:
        """Build the spec for ``entry`` by layering defaults, provider and entry."""
        provider = entry["provider"]
        provider_entry = self.providers.get(provider)
        if provider_entry is None:
            if provider != MOCK_PROVIDER:
                raise ValueError(f"model {name!r} uses unknown provider {provider!r}")
            provider_entry = {}
        merged = dict(self.defaults)
        merged.update({k: v for k, v in entry.items() if k not in ("provider", "model")})
        return ModelSpec(
            name=name,
            provider=provider,
            model=str(entry["model"]),
            base_url=merged.get("base_url", provider_entry.get("base_url")),
            api_key_env=merged.get("api_key_env", provider_entry.get("api_key_env")),
            supports_temperature=bool(merged.get("supports_temperature", True)),
            max_tokens_param=merged.get("max_tokens_param", "max_tokens"),
            max_tokens=int(merged.get("max_tokens", 4096)),
            extra_body=dict(merged.get("extra_body") or {}),
            timeout=float(merged.get("timeout", 600.0)),
            notes=str(merged.get("notes") or ""),
        )


def load_env() -> list[Path]:
    """Load ``.env`` from the working directory and the repository root, if present.

    Existing environment variables win over the file. Returns the files that
    were read (possibly none); a missing file is not an error.
    """
    from dotenv import load_dotenv

    loaded: list[Path] = []
    for candidate in (Path.cwd() / ENV_FILE, REPO_ROOT / ENV_FILE):
        candidate = candidate.resolve()
        if candidate.is_file() and candidate not in loaded:
            load_dotenv(candidate, override=False)
            loaded.append(candidate)
    return loaded


__all__ = ["REPO_ROOT", "DEFAULT_CONFIG_PATH", "Registry", "load_env"]
