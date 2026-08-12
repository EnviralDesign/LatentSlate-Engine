from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..protocol import ToolDescriptor
from ..recipes import DeploymentPlan
from ..resources import (
    ArtifactPrecision,
    ArtifactQuantization,
    ResourceDescriptor,
    ResourceFormat,
    ResourceKind,
    ResourceSource,
)
from ..variants import VariantDefinition


class AuthoringSourceType(StrEnum):
    AUTO = "auto"
    HUGGINGFACE = "huggingface"
    CIVITAI = "civitai"
    HTTPS = "https"
    LOCAL = "local"


class ResourceInspectRequest(BaseModel):
    """Read-only source inspection request shared by CLI and HTTP clients."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    source_type: AuthoringSourceType = AuthoringSourceType.AUTO
    revision: str | None = None
    filename: str | None = None
    file_id: int | None = Field(default=None, ge=1)
    allow_patterns: list[str] = Field(default_factory=list)
    ignore_patterns: list[str] = Field(default_factory=list)
    token_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]*$")
    requires_auth: bool = False
    expected_size_bytes: int | None = Field(default=None, gt=0)
    expected_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-fA-F0-9]{64}$",
    )

    @model_validator(mode="after")
    def validate_source_options(self) -> ResourceInspectRequest:
        if self.filename and (self.allow_patterns or self.ignore_patterns):
            raise ValueError("an exact file cannot also declare snapshot patterns")
        if self.file_id is not None and self.source_type not in {
            AuthoringSourceType.AUTO,
            AuthoringSourceType.CIVITAI,
        }:
            raise ValueError("file_id is valid only for a CivitAI source")
        return self


class SafeTensorsFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tensor_count: int = Field(ge=0)
    tensor_keys: list[str] = Field(default_factory=list)
    dtypes: list[str] = Field(default_factory=list)
    shapes: dict[str, list[int]] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    schema_sha256: str
    truncated: bool = False


class ArtifactFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    format: ResourceFormat = ResourceFormat.UNKNOWN
    precision: ArtifactPrecision = ArtifactPrecision.UNKNOWN
    quantization: ArtifactQuantization = ArtifactQuantization.UNKNOWN
    safetensors: SafeTensorsFacts | None = None


class SourceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    filename: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResourceInspectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: AuthoringSourceType
    canonical_source: str
    facts: ArtifactFacts
    exact_source: ResourceSource | None = None
    candidates: list[SourceCandidate] = Field(default_factory=list)
    detected: dict[str, Any] = Field(default_factory=dict)
    recommended: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ResourceAddRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inspection: ResourceInspectRequest
    resource_id: str = Field(pattern=r"^[a-z][a-z0-9_.:/-]*$")
    kind: ResourceKind
    family: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    name: str | None = Field(default=None, min_length=1)
    relative_path: str | None = None
    format: ResourceFormat | None = None
    precision: ArtifactPrecision | None = None
    quantization: ArtifactQuantization | None = None
    base_model: str | None = None
    component: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    replace: bool = False


class CatalogActivation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    published: bool
    active_in_current_process: bool
    required_action: Literal["none", "restart_engine", "next_cli_invocation"]
    disk_revision: str


class ResourcePublicationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource: ResourceDescriptor
    declaration_path: str
    artifact_path: str
    inspection: ResourceInspectionResult
    activation: CatalogActivation


class ResourceCatalogValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    resource_id: str | None = None
    resource: ResourceDescriptor | None = None
    errors: list[str] = Field(default_factory=list)
    search_paths: list[str] = Field(default_factory=list)


class RecipeDraftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    definition: VariantDefinition
    replace: bool = False


class RecipeValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    definition: VariantDefinition
    toml: str
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    closure: DeploymentPlan | None = None


class RecipeDraftResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft_key: str
    draft_path: str
    validation: RecipeValidationResult


class RecipePublishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replace: bool = False


class RecipePublicationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipe_key: str
    recipe_path: str
    validation: RecipeValidationResult
    activation: CatalogActivation


class BaseToolAuthoringCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    descriptor: ToolDescriptor
    family: str
    runtime_available: bool
    runtime_unavailable_reason: str | None = None
    execution: dict[str, Any]
    model_resource_components: list[str] = Field(default_factory=list)


class AuthoringCapabilitiesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    recipe_schema: dict[str, Any]
    optimization_schema: dict[str, Any]
    resource_schema: dict[str, Any]
    base_tools: list[BaseToolAuthoringCapability]


class CatalogStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loaded_revision: str
    disk_revision: str
    stale: bool
    required_action: Literal["none", "restart_engine"]
