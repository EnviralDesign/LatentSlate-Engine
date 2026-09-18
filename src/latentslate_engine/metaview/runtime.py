"""Own one MetaView model identity, source conditioning and camera requests."""

from dataclasses import dataclass
import gc
import math
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from latentslate_engine.identity import FileContentIdentity
from latentslate_engine.progress import report_progress
from latentslate_engine.qwen2511.text import QwenTextEncoder
from latentslate_engine.qwen2511.vae import load_vae
from latentslate_engine.qwen2511.weights import Linear, RMSNorm, QwenWeights
from .conditioning import source_geometry, camera_conditioning
from .contracts import validate_request
from .model import MetaViewDiT
from .sampling import sample

TRIGGER = "镜头视角转到指定位置"


@dataclass(frozen=True)
class GenerationResult:
    output: Path
    width: int
    height: int
    elapsed_seconds: float
    timings: dict[str, float]
    models_reused: bool
    source_reused: bool
    radius: float


class MetaViewRuntime:
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.identity = self.source = None
        self.model = self.weights = self.vae = self.text_encoder = None

    def close(self):
        if self.weights is not None:
            self.weights.close()
        if self.text_encoder is not None:
            self.text_encoder.close()
        self.model = self.weights = self.vae = self.text_encoder = None
        self.identity = self.source = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate(self, identity, *, image, width, height, seed, yaw, pitch,
                 radius=None, output, progress=None):
        validate_request(width, height, seed, yaw, pitch, radius)
        started = time.perf_counter()
        source_key = (FileContentIdentity.from_path(image), width, height)
        reused = identity == self.identity and self.model is not None
        if identity != self.identity:
            self.close()
            self.identity = identity
        source_reused = self.source is not None and self.source[0] == source_key
        timings = {}
        previous_reduction = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
        try:
            stage = time.perf_counter()
            if not source_reused:
                # Source changes run geometry before staging text/DiT weights.
                if self.weights is not None:
                    self.weights.close()
                    self.weights = self.model = None
                    reused = False
                    gc.collect()
                    torch.cuda.empty_cache()
                with Image.open(image) as opened:
                    if getattr(opened, "n_frames", 1) != 1:
                        raise ValueError("MetaView requires a still source image")
                    source = ImageOps.exif_transpose(opened).convert("RGB")
                report_progress(progress, 0.02, "Scene geometry")
                geometry = source_geometry(identity.geometry_model.path, identity.depth_model.path,
                                           source, width, height, self.device)
                report_progress(progress, 0.12, "Image and text conditioning")
                if self.vae is None:
                    self.vae = load_vae(identity.vae.path, self.device)
                pixels = torch.from_numpy(np.array(source.resize((width, height))).astype(np.float32) / 255.0)[None]
                reference = self.vae.encode(pixels)
                if self.text_encoder is None:
                    self.text_encoder = QwenTextEncoder(identity.text_encoder.path, identity.tokenizer, self.device)
                pixels = torch.from_numpy(np.array(source).astype(np.float32) / 255.0)[None]
                scale = math.sqrt(1024 * 1024 / (source.width * source.height))
                visual = F.interpolate(pixels.movedim(-1, 1), size=(round(source.height * scale), round(source.width * scale)), mode="area").movedim(1, -1)
                try:
                    positive = self.text_encoder.encode(TRIGGER, (visual,), (1,), numbered=False)
                finally:
                    self.text_encoder.offload()
                    gc.collect()
                    torch.cuda.empty_cache()
                self.source = (source_key, geometry, reference, positive)
            _, source_geometry_value, reference, positive = self.source
            timings["conditioning"] = time.perf_counter() - stage
            geometry, actual_radius = camera_conditioning(source_geometry_value, yaw, pitch, radius)
            stage = time.perf_counter()
            if self.model is None:
                report_progress(progress, 0.2, "Loading novel-view model")
                self.model = MetaViewDiT(device="meta", dtype=torch.bfloat16,
                    operations=SimpleNamespace(Linear=Linear, RMSNorm=RMSNorm, LayerNorm=torch.nn.LayerNorm)).eval().requires_grad_(False)
                self.weights = QwenWeights(identity.diffusion.path, self.model, self.device, scaled_fp8=True)
            timings["model_load"] = time.perf_counter() - stage
            stage = time.perf_counter()
            latent = sample(self.model, positive, reference, geometry, seed, width, height, self.device,
                lambda step, total: report_progress(progress, 0.25 + 0.6 * step / total, "Sampling",
                    stage_progress=step / total, detail=f"Step {step} of {total}"))
            timings["sampling"] = time.perf_counter() - stage
            stage = time.perf_counter()
            report_progress(progress, 0.87, "VAE decode")
            decoded = self.vae.decode(latent.to(device=self.device, dtype=torch.bfloat16))
            pixels = decoded.float().add_(1).div_(2).clamp_(0, 1).movedim(1, -1).cpu()[0, 0]
            timings["decode"] = time.perf_counter() - stage
            output = Path(output).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((pixels.numpy() * 255).clip(0, 255).astype(np.uint8)).save(output, format="PNG")
            report_progress(progress, 1, "Artifact encoding", stage_progress=1)
            return GenerationResult(output, width, height, time.perf_counter() - started,
                                    timings, reused, source_reused, actual_radius)
        except BaseException:
            self.close()
            raise
        finally:
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous_reduction)
