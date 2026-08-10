from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from typing import Any
from uuid import UUID

from ..config import Settings
from ..protocol import ToolDescriptor
from ..storage import Storage, StoredArtifact

ProgressCallback = Callable[[float, str | None], None]


@dataclass(frozen=True, slots=True)
class LoraExecution:
    slot: str
    resource_id: str
    path: Path
    strength: float


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    variant_key: str
    family: str
    model_resource_id: str | None = None
    model_path: Path | None = None
    model_format: str | None = None
    loras: tuple[LoraExecution, ...] = ()
    optimizations: dict[str, Any] | None = None
    runtime_parameters: dict[str, Any] | None = None


@dataclass(slots=True)
class ToolContext:
    job_id: UUID
    settings: Settings
    storage: Storage
    cancel_event: Event
    progress: ProgressCallback
    execution: ExecutionPlan | None = None

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise ToolCancelled("Generation canceled")

    def resolve_asset(self, asset_id: UUID) -> Path:
        return self.storage.resolve_asset(asset_id)

    def with_execution(self, execution: ExecutionPlan) -> ToolContext:
        return replace(self, execution=execution)


class ToolCancelled(RuntimeError):
    pass


class Tool(ABC):
    @property
    @abstractmethod
    def descriptor(self) -> ToolDescriptor:
        raise NotImplementedError

    @abstractmethod
    def run(self, context: ToolContext, inputs: dict[str, Any]) -> list[StoredArtifact]:
        raise NotImplementedError

    def provenance(self) -> dict[str, Any]:
        return {}

    def execution_capabilities(self) -> set[str]:
        """Runtime features this tool can honor when wrapped by a data-defined variant."""

        return set()
