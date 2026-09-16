"""Family-local SDXL state ownership and generation."""

import gc
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

from latentslate_engine.progress import ProgressCallback, report_progress

from .contracts import SDXLIdentity, validate_request
from .model import UNet
from .sampling import DiscreteEPS, sample
from .text import TextEncoder
from .vae import decode, load_decoder


@dataclass(frozen=True)
class GenerationResult:
    output: Path
    elapsed_seconds: float
    timings: dict[str, float]
    models_reused: bool
    conditioning_reused: bool


class SDXLRuntime:
    """Own one immutable artifact identity and its last prompt's conditioning."""

    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.identity = self.model = self.vae = self.conditioning = None
        self.model_sampling = None

    def close(self):
        self.identity = self.model = self.vae = self.conditioning = None
        self.model_sampling = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate(
        self,
        identity: SDXLIdentity,
        prompt: str,
        seed: int,
        output: Path,
        *,
        negative_prompt="",
        steps=25,
        cfg=7.0,
        sampler="dpmpp_2m",
        scheduler="karras",
        width=1024,
        height=1024,
        progress: ProgressCallback | None = None,
    ):
        validate_request(width, height, seed)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("SDXL prompt must be nonempty text")
        started = time.perf_counter()
        reused = self.identity == identity and self.model is not None
        if self.identity != identity:
            self.close()
            self.identity = identity
        conditioning_reused = self.conditioning is not None and self.conditioning[
            0
        ] == (prompt, negative_prompt)
        timings = {}
        previous_reduction = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
        try:
            if not conditioning_reused:
                report_progress(progress, 0.02, "Text conditioning")
                stage = time.perf_counter()
                encoder = TextEncoder(
                    identity.checkpoint.path, identity.tokenizer, self.device
                )
                try:
                    self.conditioning = (
                        (prompt, negative_prompt),
                        encoder.encode(prompt),
                        encoder.encode(negative_prompt),
                    )
                finally:
                    del encoder
                    gc.collect()
                    torch.cuda.empty_cache()
                timings["conditioning"] = time.perf_counter() - stage
            stage = time.perf_counter()
            if self.model is None:
                report_progress(progress, 0.2, "Loading SDXL")
                with torch.device("meta"):
                    self.model = UNet()
                with safe_open(identity.checkpoint.path, framework="pt") as source:
                    tensor_names = source.keys()
                    if any(
                        key in tensor_names
                        for key in (
                            "edm_mean",
                            "edm_std",
                            "edm_vpred.sigma_max",
                            "v_pred",
                            "ztsnr",
                        )
                    ):
                        raise ValueError(
                            "SDXL requires an ordinary epsilon checkpoint; EDM/v-prediction is not supported"
                        )
                    state = {
                        key.removeprefix("model.diffusion_model."): source.get_tensor(
                            key
                        ).to(device=self.device, dtype=torch.float16)
                        for key in tensor_names
                        if key.startswith("model.diffusion_model.")
                    }
                self.model.load_state_dict(state, assign=True)
                self.model.eval().requires_grad_(False)
                self.model_sampling = DiscreteEPS()
                del state
            timings["model_load"] = time.perf_counter() - stage
            stage = time.perf_counter()
            latent = sample(
                self.model,
                self.conditioning[1],
                self.conditioning[2],
                seed,
                width,
                height,
                self.device,
                steps=steps,
                cfg=cfg,
                sampler=sampler,
                scheduler=scheduler,
                model_sampling=self.model_sampling,
                progress=lambda step, total: report_progress(
                    progress,
                    0.25 + 0.6 * step / total,
                    "Sampling",
                    stage_progress=step / total,
                    detail=f"Step {step} of {total}",
                ),
            )
            timings["sampling"] = time.perf_counter() - stage
            report_progress(progress, 0.87, "VAE decode")
            stage = time.perf_counter()
            if self.vae is None:
                self.vae = load_decoder(
                    identity.checkpoint.path
                    if identity.vae is None
                    else identity.vae.path,
                    self.device,
                    embedded=identity.vae is None,
                )
            pixels = decode(self.vae, latent, self.device)[0]
            timings["decode"] = time.perf_counter() - stage
            output = Path(output).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            report_progress(progress, 0.97, "Artifact encoding")
            Image.fromarray((pixels.numpy() * 255).clip(0, 255).astype(np.uint8)).save(
                output, format="PNG"
            )
            report_progress(progress, 1.0, "Artifact encoding", stage_progress=1.0)
            return GenerationResult(
                output,
                time.perf_counter() - started,
                timings,
                reused,
                conditioning_reused,
            )
        except BaseException:
            self.close()
            raise
        finally:
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous_reduction)
