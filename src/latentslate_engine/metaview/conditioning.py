"""Single-image DA3 geometry and camera conditioning for MetaView.

Adapted from MetaView's inference path (Apache-2.0). DA3 supplies its own
model math and processors; no demo, export or Comfy runtime is imported.
"""

import gc
import numpy as np
import torch
from torch.nn import functional as F
from depth_anything_3.cfg import create_object, load_config
from depth_anything_3.registry import MODEL_REGISTRY
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.io.output_processor import OutputProcessor
from latentslate_engine.mapped_checkpoint import MappedCheckpoint


def target_camera(yaw, pitch, radius):
    yaw, pitch = np.radians(yaw), np.radians(pitch)
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    rotation = ry @ rx
    center = np.array([0.0, 0.0, radius])
    target = np.eye(4)
    target[:3, :3] = rotation
    target[:3, 3] = center - rotation @ center
    return torch.tensor(np.stack((target, np.eye(4))), dtype=torch.float32)


def _predict(path, preset, image, process_res, device, layers=()):
    config = load_config(MODEL_REGISTRY[preset])
    # Gaussian splatting is not exercised by novel-view conditioning.
    geometry_config = config.anyview if "anyview" in config else config
    geometry_config.pop("gs_head", None)
    geometry_config.pop("gs_adapter", None)
    model = create_object(config).eval().requires_grad_(False)
    source = MappedCheckpoint(path)
    state = {}
    tied = source._header.get("__metadata__", {})
    for name in model.state_dict():
        key = "model." + name
        state[name] = source.tensor(tied.get(key, key))
    model.load_state_dict(state, strict=True, assign=True)
    model.to(device)
    pixels, _, _ = InputProcessor()([image], process_res=process_res)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        output = model(pixels.to(device)[None].float(), export_feat_layers=list(layers))
    result = OutputProcessor()(output)
    del model, state, source, output, pixels
    gc.collect()
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def source_geometry(giant, depth_model, image, width, height, device):
    """Extract the source-dependent geometry once, independent of target pose."""
    image = image.resize((width, height))
    resolution = max(width, height) * 14 // 16
    layers = (19, 27, 33, 39)
    features = _predict(giant, "da3-giant", image, resolution, device, layers)
    intrinsics = features.intrinsics[0]
    k = torch.tensor([
        [intrinsics[0, 0] / (intrinsics[0, 2] * 2), 0, 0],
        [0, intrinsics[1, 1] / (intrinsics[1, 2] * 2), 0],
        [0, 0, 1],
    ], dtype=torch.float32)
    feat = torch.cat([torch.from_numpy(features.aux[f"feat_layer_{layer}"]) for layer in layers], dim=-1)[0].to(torch.bfloat16)
    if tuple(feat.shape[:2]) != (height // 16, width // 16):
        feat = F.interpolate(feat.permute(2, 0, 1)[None].float(), size=(height // 16, width // 16), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).to(torch.bfloat16)
    del features
    prediction = _predict(depth_model, "da3nested-giant-large", image, resolution, device)
    depth = F.interpolate(torch.tensor(prediction.depth)[None], size=(height, width), mode="bilinear", align_corners=False)[0]
    depth = torch.cat((torch.zeros_like(depth), depth), dim=0)
    return {"ks": torch.stack((k, k)), "feat3d": feat.contiguous(), "depth": depth}


def camera_conditioning(source, yaw, pitch, radius):
    """Zero or None selects the reference's center-depth orbit radius."""
    if radius is None or radius == 0:
        depth = source["depth"][1]
        radius = float(depth[depth.shape[0] // 2, depth.shape[1] // 2])
    return {**source, "viewmats": target_camera(yaw, pitch, radius)}, radius
