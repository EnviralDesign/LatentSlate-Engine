from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Callable
from uuid import UUID

from ..config import Settings
from ..protocol import ToolDescriptor
from ..storage import Storage, StoredArtifact


ProgressCallback = Callable[[float, str | None], None]


@dataclass(slots=True)
class ToolContext:
    job_id: UUID
    settings: Settings
    storage: Storage
    cancel_event: Event
    progress: ProgressCallback

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise ToolCancelled("Generation canceled")

    def resolve_asset(self, asset_id: UUID) -> Path:
        return self.storage.resolve_asset(asset_id)


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
