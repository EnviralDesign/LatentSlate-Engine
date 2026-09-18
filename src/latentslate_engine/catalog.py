"""Portable catalog semantics shared by publication lineage and the HTTP API."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from .authoring import compile_document
from .krea2.recipes import KREA2_T2I_POLICY
from .krea2.contracts import ALIGNMENT, MIN_SIDE, MAX_PIXELS
from .klein9b.recipes import KLEIN9B_T2I_POLICY, KLEIN9B_TWO_IMAGE_EXPLICIT_POLICY
from .ltx23.recipes import LTX23_FLF_POLICY, LTX23_I2V_POLICY, LTX23_T2V_POLICY
from .ltx25.recipes import POLICIES as LTX25_POLICIES
from .h3.recipes import POLICIES as H3_POLICIES
from .h3.contracts import ALIGNMENT as H3_ALIGNMENT, MIN_SIDE as H3_MIN_SIDE, FRAME_RATE as H3_FPS
from .qwen2511.recipes import QWEN2511_EDIT_POLICY
from .zimage.recipes import ZIMAGE_T2I_POLICY
from .zimage import contracts as zimage_contracts
from .ideogram4 import contracts as ideogram4_contracts
from .ideogram4.recipes import IDEOGRAM4_T2I_POLICY
from .sdxl import contracts as sdxl_contracts
from .sdxl.recipes import SDXL_T2I_POLICY
from .metaview.recipes import METAVIEW_POLICY
from .metaview import contracts as metaview_contracts
from .wan2214b.recipes import (
    WAN2214B_FLF_POLICY,
    WAN2214B_I2V_POLICY,
    WAN2214B_T2V_POLICY,
)

SDXL_T2I_ID = "a33f4d77-f475-517c-b7d8-208006f30eb2"
METAVIEW_ID = str(uuid5(NAMESPACE_URL, "latentslate:metaview:novel_view"))
H3_IDS = {operation: str(uuid5(NAMESPACE_URL, f"latentslate:h3:{operation}")) for operation in H3_POLICIES}
LTX25_IDS = {
    operation: str(uuid5(NAMESPACE_URL, f"latentslate:ltx25:{operation}"))
    for operation in LTX25_POLICIES
}
KREA2_T2I_ID = "fbdce87a-02cb-546e-98a3-4d268d35025b"
IDEOGRAM4_T2I_ID = "fa51168b-e904-51c9-bb0d-a61d367d9895"
ZIMAGE_T2I_ID = "8c7ab8cb-3670-5aed-a74a-dbf16e694cf9"
QWEN2511_EDIT_ID = "b89fecef-a923-5108-8100-c49b7f469cdc"
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
        "fps": "FPS",
        "prompt_enhancement": "Prompt enhancement",
        "turbo": "Turbo",
    }
    hints = {
        "prompt": {"multiline": True, "placeholder": "Describe the shot"},
        "duration_seconds": {"unit": "seconds"},
    }
    published_constraints = {
        "width": ("min", "step"),
        "height": ("min", "step"),
        "duration_seconds": ("min", "max", "step"),
        "fps": ("min", "max", "step"),
    }
    # HTTP requires these keys even though recipe resolution supplies defaults.
    required_on_wire = {"width", "height", "duration_seconds", "seed"}
    inputs = []
    for item in surface:
        key = item["key"]
        ui = dict(hints.get(key, {}))
        for constraint in published_constraints.get(key, ()):
            if constraint in item["constraints"]:
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
        (
            {**item, "image_dimensions": "match_output_canvas"}
            if item["type"] == "image"
            else item
        )
        for item in inputs
    ]
    return {
        "id": tool_id,
        "key": key,
        "schema_revision": 4,
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


def _image_policy_inputs(
    surface: tuple[dict[str, object], ...],
) -> list[dict[str, Any]]:
    """Present image products under the existing HTTP contract."""
    labels = {
        "negative_prompt": "Negative prompt",
        "steps": "Steps",
        "cfg": "CFG",
        "sampler": "Sampler",
        "scheduler": "Scheduler",
        "prompt_enhancement": "Prompt enhancement",
        "prompt": "Prompt",
        "image_1": "Image 1",
        "image_2": "Image 2",
        "image_3": "Image 3",
        "width": "Width",
        "height": "Height",
        "seed": "Seed",
    }
    inputs = []
    for item in surface:
        key = item["key"]
        ui = None
        if key in {"prompt", "negative_prompt"}:
            ui = {"multiline": True, "placeholder": "Describe the image"}
        elif key in {"width", "height"}:
            ui = {name: item["constraints"][name] for name in ("min", "step")}
        if key in {"steps", "cfg", "sampler", "scheduler"}:
            ui = dict(item["constraints"])
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
    for item in inputs:
        if item["type"] == "image":
            item["prompt_reference_token"] = "image {index}"
        if item["key"] in {"image_2", "image_3"}:
            item["nullable"] = True
    return {
        "id": tool_id,
        "key": key,
        "schema_revision": 3 if tool_id == KLEIN_TWO_IMAGE_ID else 1,
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


def _metaview_schema():
    schema = {
        "id": METAVIEW_ID, "key": "metaview.novel_view", "schema_revision": 1,
        "name": "Qwen MetaView Novel View", "description": "Render a source image from a target camera viewpoint.",
        "workflow_kind": "image_to_image", "output": {"type": "image"},
        "inputs": [
            _input(item["key"], item["key"].replace("_", " ").capitalize(), item["type"],
                   required=item["required"], default=item.get("default"),
                   role=item.get("role"), ui=item.get("constraints") or None)
            for item in METAVIEW_POLICY.surface()
        ],
        "canvas": {"alignment": metaview_contracts.ALIGNMENT, "min_side": metaview_contracts.MIN_SIDE,
                   "max_pixels": metaview_contracts.MAX_PIXELS},
    }
    for _item in schema["inputs"]:
        if _item["key"] == "radius":
            _item.update(description="Orbit radius; zero derives it from source depth.")
    return schema


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
            inputs=_image_policy_inputs(KLEIN9B_T2I_POLICY.surface()),
        ),
        _klein_tool_schema(
            KLEIN_TWO_IMAGE_ID,
            "flux2_klein9b.two_image_to_image",
            "FLUX.2 Klein 9B Image to Image",
            "image_to_image",
            inputs=_image_policy_inputs(KLEIN9B_TWO_IMAGE_EXPLICIT_POLICY.surface()),
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
        {
            "id": KREA2_T2I_ID,
            "key": "krea2_turbo.text_to_image",
            "schema_revision": 2,
            "name": "Krea 2 Turbo Text to Image",
            "description": "Generate an image with Krea 2 Turbo and optional prompt enhancement.",
            "workflow_kind": "text_to_image",
            "output": {"type": "image"},
            "inputs": _image_policy_inputs(KREA2_T2I_POLICY.surface()),
            "canvas": {
                "alignment": ALIGNMENT,
                "min_side": MIN_SIDE,
                "max_pixels": MAX_PIXELS,
                "max_aspect": 4.0,
            },
        },
    ]
    qwen_inputs = _image_policy_inputs(QWEN2511_EDIT_POLICY.surface())
    for item in qwen_inputs:
        if item["type"] == "image":
            item["prompt_reference_token"] = f"Picture {item['key'].rsplit('_', 1)[1]}"
        if item["key"] in {"image_2", "image_3"}:
            item["nullable"] = True
    schemas.append({
        "id": QWEN2511_EDIT_ID,
        "key": "qwen2511.edit",
        "schema_revision": 2,
        "name": "Qwen Image Edit 2511",
        "description": "Edit Image 1 using up to three ordered reference images; Image 1 determines the output canvas.",
        "workflow_kind": "image_to_image",
        "output": {"type": "image"},
        "inputs": qwen_inputs,
    })
    schemas.append({
        "id": ZIMAGE_T2I_ID,
        "key": "zimage_turbo.text_to_image",
        "schema_revision": 1,
        "name": "Z-Image Turbo Text to Image",
        "description": "Generate an image with Z-Image Turbo.",
        "workflow_kind": "text_to_image",
        "output": {"type": "image"},
        "inputs": _image_policy_inputs(ZIMAGE_T2I_POLICY.surface()),
        "canvas": {
            "alignment": zimage_contracts.ALIGNMENT,
            "min_side": zimage_contracts.MIN_SIDE,
            "max_pixels": zimage_contracts.MAX_PIXELS,
            "max_aspect": 4.0,
        },
    })
    schemas.append({
        "id": IDEOGRAM4_T2I_ID,
        "key": "ideogram4.text_to_image",
        "schema_revision": 1,
        "name": "Ideogram v4 Text to Image",
        "description": "Generate an image with Ideogram v4.",
        "workflow_kind": "text_to_image",
        "output": {"type": "image"},
        "inputs": _image_policy_inputs(IDEOGRAM4_T2I_POLICY.surface()),
        "canvas": {
            "alignment": ideogram4_contracts.ALIGNMENT,
            "min_side": ideogram4_contracts.MIN_SIDE,
            "max_pixels": ideogram4_contracts.MAX_PIXELS,
            "max_aspect": 4.0,
        },
    })
    schemas.append({
        "id": SDXL_T2I_ID,
        "key": "sdxl.text_to_image",
        "schema_revision": 1,
        "name": "SDXL Text to Image",
        "description": "Generate an image with SDXL.",
        "workflow_kind": "text_to_image",
        "output": {"type": "image"},
        "inputs": _image_policy_inputs(SDXL_T2I_POLICY.surface()),
        "canvas": {
            "alignment": sdxl_contracts.ALIGNMENT,
            "min_side": sdxl_contracts.MIN_SIDE,
            "max_pixels": sdxl_contracts.MAX_PIXELS,
            "max_aspect": 4.0,
        },
    })
    for operation, label, kind in (
        ("t2v", "Text to Video", "text_to_video"),
        ("i2v", "Image to Video", "image_to_video"),
        ("flf", "First/Last Frame", "first_frame_last_frame_video"),
    ):
        schemas.append({
            "id": LTX25_IDS[operation],
            "key": f"ltx25.{operation}",
            "schema_revision": 1,
            "name": f"LTX 2.5 {label}",
            "description": "Generate LTX 2.5 video with synchronized audio.",
            "workflow_kind": kind,
            "output": {"type": "video"},
            "inputs": _video_policy_inputs(
                LTX25_POLICIES[operation].surface(),
                {"start_image": "First Frame", "end_image": "Last Frame"},
            ),
            "canvas": {"alignment": 32 if operation == "flf" else 64, "min_side": 64},
            "timing": {
                "fps": {"mode": "input"},
                "duration_seconds": {
                    "min": 1.0, "max": 10.0, "step": 0.0,
                    "frame_step": 8, "frame_offset": 1,
                },
            },
        })
    for operation, label, kind in (
        ("t2v", "Text to Video", "text_to_video"),
        ("i2v", "Image to Video", "image_to_video"),
        ("r2v", "Reference to Video", "reference_to_video"),
    ):
        policy = H3_POLICIES[operation]
        surface = policy.surface()
        media_labels = {
            item["key"]: item["key"].replace("_", " ").title()
            for item in surface
            if item["type"] in {"image", "video", "audio"}
        }
        inputs = _video_policy_inputs(surface, media_labels)
        for item, field in zip(inputs, surface):
            if field.get("nullable"):
                item["nullable"] = True
            if operation == "i2v" and item["type"] == "image":
                item["image_dimensions"] = "match_output_canvas"
            if operation == "r2v":
                if item["type"] in {"image", "video", "audio"}:
                    reference_label = {"image": "Picture", "video": "Video", "audio": "Audio"}[
                        item["type"]
                    ]
                    item["prompt_reference_token"] = f"<{reference_label} {{index}}>"
                if item["key"] == "prompt":
                    item["description"] = (
                        "Write references manually as <Picture N>, <Video N> or <Audio N>. "
                        "Pictures and videos each count occupied slots from 1 in slot order. "
                        "Audio counts enabled video soundtracks first in video-slot order, "
                        "then occupied standalone audio slots. Clearing an earlier slot or "
                        "changing soundtrack inclusion can renumber later references. "
                        "Slot labels are not prompt numbers. Prompts are never rewritten."
                    )
                elif item["key"].startswith("reference_video_audio_"):
                    index = item["key"].rsplit("_", 1)[1]
                    item["label"] = f"Video slot {index} soundtrack"
                    item["paired_video_input"] = f"reference_video_{index}"
                    item["description"] = (
                        "Optional soundtrack paired with this video. Use its embedded "
                        "audio with the same sample and interval, or select a separate "
                        "audio source aligned to the video. This remains paired "
                        "conditioning, not a standalone audio reference."
                    )
                elif item["type"] in {"image", "video", "audio"}:
                    index = item["key"].rsplit("_", 1)[1]
                    item["label"] = f"{item['type'].title()} slot {index}"
                    item["description"] = (
                        "Optional reference for a new video. Slot order determines prompt "
                        "numbering among occupied references; see the prompt instructions."
                    )
        duration = policy.capabilities["duration_seconds"]
        schemas.append(
            {
                "id": H3_IDS[operation],
                "key": f"h3.{operation}",
                "schema_revision": 4 if operation == "r2v" else 2,
                "name": f"MiniMax H3 {label}",
                "description": "Generate MiniMax H3 video with synchronized audio.",
                "workflow_kind": kind,
                "output": {"type": "video"},
                "inputs": inputs,
                "canvas": {"alignment": H3_ALIGNMENT, "min_side": H3_MIN_SIDE},
                "timing": {
                    "fps": {"mode": "fixed", "value": H3_FPS},
                    "duration_seconds": {
                        "min": duration.minimum,
                        "max": duration.maximum,
                        "step": 0.0,
                        "frame_step": 17,
                        "frame_offset": 5,
                    },
                },
            }
        )
    schemas.append(_metaview_schema())
    tools = [{**schema, "schema_hash": _schema_hash(schema)} for schema in schemas]
    for tool in tools:
        if tool["id"] in {T2V_ID, I2V_ID, FLF_ID}:
            tool["timing"] = {
                "fps": {"mode": "fixed", "value": 30.0},
                "duration_seconds": {
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.0,
                    "frame_step": 8,
                    "frame_offset": 1,
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
    METAVIEW_ID: "metaview_novel_view",
    **{tool_id: f"ltx25_{operation}" for operation, tool_id in LTX25_IDS.items()},
    **{tool_id: f"h3_{operation}" for operation, tool_id in H3_IDS.items()},
    ZIMAGE_T2I_ID: "zimage_t2i",
    IDEOGRAM4_T2I_ID: "ideogram4_t2i",
    SDXL_T2I_ID: "sdxl_t2i",
    QWEN2511_EDIT_ID: "qwen2511_edit",
    KREA2_T2I_ID: "krea2_t2i",
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
        (METAVIEW_POLICY, METAVIEW_ID),
        (ZIMAGE_T2I_POLICY, ZIMAGE_T2I_ID),
        (IDEOGRAM4_T2I_POLICY, IDEOGRAM4_T2I_ID),
        (SDXL_T2I_POLICY, SDXL_T2I_ID),
        (QWEN2511_EDIT_POLICY, QWEN2511_EDIT_ID),
        (KREA2_T2I_POLICY, KREA2_T2I_ID),
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
RECIPE_TO_BUILTIN.update({
    policy.capabilities.key: LTX25_IDS[operation]
    for operation, policy in LTX25_POLICIES.items()
})
RECIPE_TO_BUILTIN.update({policy.capabilities.key: H3_IDS[operation] for operation, policy in H3_POLICIES.items()})


def user_request_schema(document: dict) -> dict:
    """Project the caller and media contract, excluding model/provenance metadata."""
    definition = compile_document(document, policy_only=True)
    template = TOOLS_BY_ID[RECIPE_TO_BUILTIN[document["operation"]]]
    result = {
        key: deepcopy(template[key])
        for key in ("workflow_kind", "output", "canvas", "timing")
        if key in template
    }
    presentation = {
        item["key"]: {
            key: deepcopy(item[key])
            for key in (
                "label", "description", "paired_video_input", "prompt_reference_token"
            )
            if key in item
        }
        for item in template["inputs"]
    }
    inputs = []
    for surface in definition.surface():
        item = deepcopy(surface)
        item.update(
            presentation.get(
                item["key"], {"label": item["key"].replace("_", " ").title()}
            )
        )
        if "constraints" in item:
            item["ui"] = item.pop("constraints")
        if item["type"] == "image" and (document["operation"].startswith("ltx23.") or document["operation"] == "h3.i2v"):
            item["image_dimensions"] = "match_output_canvas"
        inputs.append(item)
    result["inputs"] = inputs
    fields = {item.capability.key: item for item in definition.fields}
    for dimension in ("width", "height"):
        item = fields.get(dimension)
        if item is not None and not item.exposed and item.value is not None:
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
    if "fps" in fields:
        fps = fields["fps"]
        result["timing"]["fps"] = {"mode": "input"} if fps.exposed else {"mode": "fixed", "value": fps.value}
    if "duration_seconds" in fields:
        duration = fields["duration_seconds"]
        timing = result["timing"]["duration_seconds"]
        if not duration.exposed:
            timing.update(
                mode="fixed",
                value=duration.value,
                **({} if "frame_step" in timing else {"min": duration.value, "max": duration.value}),
            )
        else:
            descriptor = next(
                item for item in inputs if item["key"] == "duration_seconds"
            )
            timing.update(
                {key: descriptor["ui"][key] for key in ("min", "max", "step") if key in descriptor["ui"]}
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
