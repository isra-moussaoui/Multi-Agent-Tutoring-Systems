"""Per-role LLM configuration passed through LangGraph RunnableConfig."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RoleConfig:

    provider: str = "mistral"

    model: str = "mistral-large-latest"

    temperature: float = 0.2

    max_tokens: int = 3072

    # IMPORTANT:
    # No artificial delay by default.
    sleep_s: float = 0.0


@dataclass(frozen=True)
class GraphRoleConfigs:

    tutor: RoleConfig

    verifier: RoleConfig

    recovery: RoleConfig

    def to_configurable(self) -> dict[str, Any]:

        return {
            "tutor": asdict(self.tutor),
            "verifier": asdict(self.verifier),
            "recovery": asdict(self.recovery),
        }


def role_from_configurable(
    configurable: dict[str, Any] | None,
    role: str,
) -> RoleConfig:

    configurable = configurable or {}

    raw = configurable.get(role) or {}

    defaults = RoleConfig()

    return RoleConfig(

        provider=raw.get(
            "provider",
            defaults.provider,
        ),

        model=raw.get(
            "model",
            defaults.model,
        ),

        temperature=float(
            raw.get(
                "temperature",
                defaults.temperature,
            )
        ),

        max_tokens=int(
            raw.get(
                "max_tokens",
                defaults.max_tokens,
            )
        ),

        sleep_s=float(
            raw.get(
                "sleep_s",
                defaults.sleep_s,
            )
        ),
    )