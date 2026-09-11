"""Portable catalog semantics shared by publication lineage and the HTTP API."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from .authoring import compile_document
from .klein9b.recipes import KLEIN9B_T2I_POLICY, KLEIN9B_TWO_IMAGE_EXPLICIT_POLICY
from .ltx23.recipes import LTX23_FLF_POLICY, LTX23_I2V_POLICY, LTX23_T2V_POLICY
from .wan2214b.recipes import (
    WAN2214B_FLF_POLICY,
    WAN2214B_I2V_POLICY,
    WAN2214B_T2V_POLICY,
)

T2V_ID = "46bdb57c-3b19-5397-8949-4e20ffe757c9"
I2V_ID = "5d6e2d6f-216c-5f35-a4ec-1565d6e56ee7"
FLF_ID = "1a8f9c0b-410e-56e4-90de-23bcb9d644ca"
KLEIN_T2I_ID = "e7dcbbde-d58f-4354-ad36-b684b5c236f3"
KLEIN_TWO_IMAGE_ID = "a7489e73-3bb9-4bb9-888f-fa592c8f4430"
WAN_T2V_ID = "34e57585-95a3-4bb6-b3de-fca5dd924ba6"
WAN_I2V_ID = "aac35e26-08e7-400b-bf9b-dc389809ddd5"
WAN_FLF_ID = "d0c202bf-7dd5-4df8-b116-f7633dc94cfe"


def _input(
    key: str,
    label: str,
    input_type: str,
    *,
    required: bool = True,
    default: Any = None,
    role: str | None = None,
    ui: dict[str, Any] | None = None,
) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "key": key,
        "label": label,
        "type": input_type,
        "required": required,
    }
    if default is not None:
        descriptor["default"] = default
    if role is not None:
        descriptor["role"] = role
    if ui is not None:
        descriptor["ui"] = ui
    return descriptor


def _video_policy_inputs(
    surface: tuple[dict[str, object], ...], image_labels: dict[str, str]
) -> list[dict[str, Any]]:
    """Present the six video caller surfaces under the HTTP contract."""
    labels = {
        "prompt": "Prompt",
        **image_labels,
        "width": "Width",
        "height": "Height",
        "duration_seconds": "Duration",
        "seed": "Seed",
    }
    hints = {
        "prompt": {"multiline": True, "placeholder": "Describe the shot"},
        "duration_seconds": {"unit": "seconds"},
    }
    published_constraints = {
        "width": ("min", "step"),
        "height": ("min", "step"),
        "duration_seconds": ("min", "max", "step"),
    }
    # HTTP requires these keys even though recipe resolution supplies defaults.
    required_on_wire = {"width", "height", "duration_seconds", "seed"}
    inputs = []
    for item in surface:
        key = item["key"]
        ui = dict(hints.get(key, {}))
        for constraint in published_constraints.get(key, ()):
            ui[constraint] = item["constraints"][constraint]
        inputs.append(
            _input(
                key,
                labels[key],
                item["type"],
                required=key in required_on_wire or item["required"],
                default=item.get("default"),
                role=item.get("role"),
                ui=ui or None,
            )
        )
    return inputs


def _tool_schema(
    tool_id: str,
    key: str,
    name: str,
    workflow_kind: str,
    alignment: int,
    *,
    inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    inputs = [
        {**item, "image_dimensions": "match_output_canvas"}
        if item["type"] == "image"
        else item
        for item in inputs
    ]
    return {
        "id": tool_id,
        "key": key,
        "schema_revision": 3 if any(item["type"] == "image" for item in inputs) else 2,
        "name": name,
        "description": "Generate LTX 2.3 video with synchronized audio.",
        "workflow_kind": workflow_kind,
        "output": {"type": "video"},
        "inputs": inputs,
        "canvas": {
            "alignment": alignment,
            "min_side": 64,
            "max_pixels": 942_080,
        },
    }


def _klein_policy_inputs(
    surface: tuple[dict[str, object], ...],
) -> list[dict[str, Any]]:
    """Present the two Klein products under the existing HTTP contract."""
    labels = {
        "prompt": "Prompt",
        "image_1": "Image 1",
        "image_2": "Image 2",
        "width": "Width",
        "height": "Height",
        "seed": "Seed",
    }
    inputs = []
    for item in surface:
        key = item["key"]
        ui = None
        if key == "prompt":
            ui = {"multiline": True, "placeholder": "Describe the image"}
        elif key in {"width", "height"}:
            ui = {name: item["constraints"][name] for name in ("min", "step")}
        inputs.append(
            _input(
                key,
                labels[key],
                item["type"],
                required=key in {"width", "height", "seed"} or item["required"],
                default=item.get("default"),
                role=item.get("role"),
                ui=ui,
            )
        )
    return inputs


def _klein_tool_schema(
    tool_id: str,
    key: str,
    name: str,
    workflow_kind: str,
    *,
    inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": tool_id,
        "key": key,
        "schema_revision": 1,
        "name": name,
        "description": "Generate an image with FLUX.2 Klein 9B distilled.",
        "workflow_kind": workflow_kind,
        "output": {"type": "image"},
        "inputs": inputs,
        "canvas": {
            "alignment": 16,
            "min_side": 256,
            "max_pixels": 1_048_576,
            "max_aspect": 4.0,
        },
    }


def _wan_tool_schema(
    tool_id: str,
    key: str,
    name: str,
    workflow_kind: str,
    *,
    inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": tool_id,
        "key": key,
        "schema_revision": 2,
        "name": name,
        "description": "Generate Wan 2.2 14B turbo video at a fixed 16 fps.",
        "workflow_kind": workflow_kind,
        "output": {"type": "video"},
        "inputs": inputs,
        "canvas": {
            "alignment": 16,
            "min_side": 480,
            "max_pixels": 921_600,
            "max_aspect": 16 / 9,
        },
    }


def _schema_hash(schema: dict[str, Any]) -> str:
    encoded = json.dumps(
        schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _tool_definitions() -> list[dict[str, Any]]:
    schemas = [
        _tool_schema(
            T2V_ID,
            "ltx23.text_to_video",
            "LTX 2.3 Text to Video",
            "text_to_video",
            64,
            inputs=_video_policy_inputs(LTX23_T2V_POLICY.surface(), {}),
        ),
        _tool_schema(
            I2V_ID,
            "ltx23.image_to_video",
            "LTX 2.3 Image to Video",
            "image_to_video",
            64,
            inputs=_video_policy_inputs(
                LTX23_I2V_POLICY.surface(), {"start_image": "Start Image"}
            ),
        ),
        _tool_schema(
            FLF_ID,
            "ltx23.first_last_frame_to_video",
            "LTX 2.3 First/Last Frame to Video",
            "first_frame_last_frame_video",
            32,
            inputs=_video_policy_inputs(
                LTX23_FLF_POLICY.surface(),
                {"start_image": "First Frame", "end_image": "Last Frame"},
            ),
        ),
        _klein_tool_schema(
            KLEIN_T2I_ID,
            "flux2_klein9b.text_to_image",
            "FLUX.2 Klein 9B Text to Image",
            "text_to_image",
            inputs=_klein_policy_inputs(KLEIN9B_T2I_POLICY.surface()),
        ),
        _klein_tool_schema(
            KLEIN_TWO_IMAGE_ID,
            "flux2_klein9b.two_image_to_image",
            "FLUX.2 Klein 9B Two-Image",
            "image_to_image",
            inputs=_klein_policy_inputs(KLEIN9B_TWO_IMAGE_EXPLICIT_POLICY.surface()),
        ),
        _wan_tool_schema(
            WAN_T2V_ID,
            "wan2214b_turbo.text_to_video",
            "Wan 2.2 14B Turbo Text to Video",
            "text_to_video",
            inputs=_video_policy_inputs(WAN2214B_T2V_POLICY.surface(), {}),
        ),
        _wan_tool_schema(
            WAN_I2V_ID,
            "wan2214b_turbo.image_to_video",
            "Wan 2.2 14B Turbo Image to Video",
            "image_to_video",
            inputs=_video_policy_inputs(
                WAN2214B_I2V_POLICY.surface(), {"start_image": "Start Image"}
            ),
        ),
        _wan_tool_schema(
            WAN_FLF_ID,
            "wan2214b_turbo.first_last_frame_to_video",
            "Wan 2.2 14B Turbo First/Last Frame to Video",
            "first_frame_last_frame_video",
            inputs=_video_policy_inputs(
                WAN2214B_FLF_POLICY.surface(),
                {"start_image": "First Frame", "end_image": "Last Frame"},
            ),
        ),
    ]
    tools = [{**schema, "schema_hash": _schema_hash(schema)} for schema in schemas]
    for tool in tools:
        if tool["id"] in {T2V_ID, I2V_ID, FLF_ID}:
            tool["timing"] = {
                "fps": {"mode": "fixed", "value": 30.0},
                "duration_seconds": {
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.5,
                    "output_frame_counts": [
                        {"duration_seconds": half_seconds / 2, "frame_count": frames}
                        for half_seconds, frames in enumerate(
                            (
                                25,
                                41,
                                57,
                                73,
                                89,
                                105,
                                121,
                                129,
                                145,
                                161,
                                177,
                                193,
                                209,
                                225,
                                241,
                                249,
                                265,
                                281,
                                297,
                            ),
                            start=2,
                        )
                    ],
                },
            }
        elif tool["id"] in {WAN_T2V_ID, WAN_I2V_ID, WAN_FLF_ID}:
            tool["timing"] = {
                "fps": {"mode": "fixed", "value": 16.0},
                "duration_seconds": {"min": 1.0, "max": 5.0, "step": 0.25},
            }
    return tools


TOOLS = _tool_definitions()
TOOLS_BY_ID = {tool["id"]: tool for tool in TOOLS}
TOOL_OPERATIONS = {
    T2V_ID: "t2v",
    I2V_ID: "i2v",
    FLF_ID: "flf",
    KLEIN_T2I_ID: "klein_t2i",
    KLEIN_TWO_IMAGE_ID: "klein_two_image",
    WAN_T2V_ID: "wan_t2v",
    WAN_I2V_ID: "wan_i2v",
    WAN_FLF_ID: "wan_flf",
}
RECIPE_TO_BUILTIN = {
    policy.capabilities.key: tool_id
    for policy, tool_id in (
        (LTX23_T2V_POLICY, T2V_ID),
        (LTX23_I2V_POLICY, I2V_ID),
        (LTX23_FLF_POLICY, FLF_ID),
        (KLEIN9B_T2I_POLICY, KLEIN_T2I_ID),
        (KLEIN9B_TWO_IMAGE_EXPLICIT_POLICY, KLEIN_TWO_IMAGE_ID),
        (WAN2214B_T2V_POLICY, WAN_T2V_ID),
        (WAN2214B_I2V_POLICY, WAN_I2V_ID),
        (WAN2214B_FLF_POLICY, WAN_FLF_ID),
    )
}


def user_request_schema(document: dict) -> dict:
    """Project the caller and media contract, excluding model/provenance metadata."""
    definition = compile_document(document)
    template = TOOLS_BY_ID[RECIPE_TO_BUILTIN[document["operation"]]]
    result = {
        key: deepcopy(template[key])
        for key in ("workflow_kind", "output", "canvas", "timing")
        if key in template
    }
    labels = {item["key"]: item["label"] for item in template["inputs"]}
    inputs = []
    for surface in definition.surface():
        item = deepcopy(surface)
        item["label"] = labels.get(item["key"], item["key"].replace("_", " ").title())
        if "constraints" in item:
            item["ui"] = item.pop("constraints")
        if item["type"] == "image" and document["operation"].startswith("ltx23."):
            item["image_dimensions"] = "match_output_canvas"
        inputs.append(item)
    result["inputs"] = inputs
    fields = {item.capability.key: item for item in definition.fields}
    for dimension in ("width", "height"):
        item = fields[dimension]
        if not item.exposed and item.value is not None:
            result["canvas"][f"fixed_{dimension}"] = item.value
    if any(item.get("image_dimensions") == "match_output_canvas" for item in inputs):
        for dimension in ("width", "height"):
            if f"fixed_{dimension}" not in result["canvas"] and not any(
                item.get("role") == dimension and not item.get("nullable", False)
                for item in inputs
            ):
                raise ValueError(
                    f"Media preparation requires a determinate {dimension}"
                )
    if "duration_seconds" in fields:
        duration = fields["duration_seconds"]
        timing = result["timing"]["duration_seconds"]
        if not duration.exposed:
            timing.update(
                mode="fixed",
                value=duration.value,
                min=duration.value,
                max=duration.value,
            )
        else:
            descriptor = next(
                item for item in inputs if item["key"] == "duration_seconds"
            )
            timing.update(
                {key: descriptor["ui"][key] for key in ("min", "max", "step")}
            )
        if "output_frame_counts" in timing:
            reachable = []
            for row in timing["output_frame_counts"]:
                value = row["duration_seconds"]
                if not duration.exposed and value != duration.value:
                    continue
                try:
                    duration.validate(value)
                except ValueError:
                    continue
                reachable.append(row)
            timing["output_frame_counts"] = reachable
    return result


def user_request_schema_hash(document: dict) -> str:
    """Hash exactly the client-significant projection used in the catalog."""
    return _schema_hash(user_request_schema(document))
