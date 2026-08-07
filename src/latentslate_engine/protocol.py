from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


PROTOCOL_VERSION = "1.0"


class WorkflowKind(StrEnum):
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    FIRST_FRAME_LAST_FRAME_VIDEO = "first_frame_last_frame_video"
    VIDEO_TO_VIDEO = "video_to_video"
    VIDEO_TO_BRIDGE = "video_to_bridge"
    TEXT_TO_AUDIO = "text_to_audio"
    AUDIO_TO_AUDIO = "audio_to_audio"
    CUSTOM = "custom"


class MediaType(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class InputType(StrEnum):
    TEXT = "text"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    CHOICE = "choice"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    RESOURCE = "resource"


class InputRole(StrEnum):
    PROMPT = "prompt"
    NEGATIVE_PROMPT = "negative_prompt"
    WIDTH = "width"
    HEIGHT = "height"
    SEED = "seed"
    DURATION_SECONDS = "duration_seconds"
    FPS = "fps"
    FRAME_COUNT = "frame_count"
    SOURCE_IMAGE = "source_image"
    START_IMAGE = "start_image"
    END_IMAGE = "end_image"
    MASK = "mask"
    SOURCE_VIDEO = "source_video"
    LEFT_VIDEO = "left_video"
    RIGHT_VIDEO = "right_video"


class BundleStatus(StrEnum):
    UNKNOWN = "unknown"
    INSTALLED = "installed"
    MISSING = "missing"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class ChoiceOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    label: str
    description: str | None = None


class InputUi(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group: str | None = None
    advanced: bool = False
    multiline: bool = False
    placeholder: str | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str | None = None


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    label: str
    type: InputType
    required: bool = False
    default: Any = None
    role: InputRole | None = None
    options: list[ChoiceOption] = Field(default_factory=list)
    resource_kind: str | None = None
    multiple: bool = False
    ui: InputUi | None = None

    @model_validator(mode="after")
    def validate_specialized_fields(self) -> "ToolInput":
        if self.type == InputType.CHOICE and not self.options:
            raise ValueError("choice inputs require options")
        if self.type != InputType.CHOICE and self.options:
            raise ValueError("options are only valid for choice inputs")
        if self.type == InputType.RESOURCE and not self.resource_kind:
            raise ValueError("resource inputs require resource_kind")
        return self


class ToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: MediaType
    primary_role: str = "primary"
    supports_multiple_artifacts: bool = False


class ToolRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundle_id: str
    required: bool = True


class ToolDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    key: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    schema_revision: int = Field(ge=1)
    schema_hash: str = ""
    name: str
    description: str | None = None
    workflow_kind: WorkflowKind
    output: ToolOutput
    inputs: list[ToolInput]
    requirements: list[ToolRequirement] = Field(default_factory=list)
    available: bool = True
    unavailable_reason: str | None = None

    def with_schema_hash(self) -> "ToolDescriptor":
        payload = self.model_dump(mode="json", exclude={"schema_hash", "available", "unavailable_reason"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return self.model_copy(update={"schema_hash": f"sha256:{hashlib.sha256(encoded).hexdigest()}"})


class BundleDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    name: str
    description: str | None = None
    source: Literal["huggingface"]
    repo_id: str
    revision: str | None = None
    status: BundleStatus = BundleStatus.UNKNOWN
    install_command: str | None = None


class CatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: str = PROTOCOL_VERSION
    engine_version: str
    tools: list[ToolDescriptor]
    bundles: list[BundleDescriptor]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    protocol_version: str = PROTOCOL_VERSION
    engine_version: str
    queued_jobs: int
    running_jobs: int


class AssetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    filename: str
    content_type: str | None = None
    size_bytes: int


class AssetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["asset"] = "asset"
    asset_id: UUID


class JobCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_id: UUID
    schema_revision: int = Field(ge=1)
    schema_hash: str
    inputs: dict[str, Any] = Field(default_factory=dict)


class ArtifactDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    role: str = "primary"
    media_type: MediaType
    filename: str
    content_type: str
    size_bytes: int
    download_url: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    tool_id: UUID
    status: JobStatus
    progress: float | None = Field(default=None, ge=0.0, le=1.0)
    message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    artifacts: list[ArtifactDescriptor] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    error: "ErrorBody | None" = None


class ErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: ErrorBody
