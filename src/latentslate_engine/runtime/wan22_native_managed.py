"""Managed wrapper for the identity-bound native Wan 2.2 14B I2V runtime."""

from __future__ import annotations

import asyncio
import gc
from typing import Any

from ..wan22_recipe import Wan22RuntimeRequest, revalidate_runtime_request
from .kit import cleanup_accelerator_memory


class ManagedNativeWanI2VRuntime:
    """Lazy, reusable owner of one exact native Wan component recipe.

    The wrapper is deliberately small. It does not discover resources, download
    weights, convert storage, or reinterpret a recipe. It receives one validated
    inventory-owned request and materializes ``NativeWanI2VRuntime`` only when a job
    actually executes.
    """

    def __init__(self, request: Wan22RuntimeRequest) -> None:
        self.request = request
        self._runtime: Any | None = None

    def generate(
        self,
        generation_request: Any,
        *,
        device: str,
        progress: Any = None,
        cancelled: Any = None,
    ) -> Any:
        if cancelled is not None and cancelled():
            raise asyncio.CancelledError
        if not revalidate_runtime_request(self.request):
            self.unload()
            raise RuntimeError("native Wan recipe changed after catalog validation")
        from .wan22_i2v_runtime import validate_wan_i2v_request

        # Reject bad 4k+1 frame counts, canvas sizes, stage policies, and guidance
        # before allocating the high/low transformers or UMT5/VAE components.
        validate_wan_i2v_request(generation_request)
        if cancelled is not None and cancelled():
            raise asyncio.CancelledError
        runtime = self._load()
        return runtime.generate(
            generation_request,
            device=device,
            progress=progress,
            cancelled=cancelled,
        )

    def status(self) -> dict[str, Any]:
        return {
            "family": "wan22",
            "runtime": "native_wan_i2v_14b",
            "recipe_fingerprint": self.request.fingerprint,
            "loaded": self._runtime is not None,
            "components": self.request.public_component_manifest(),
            "cache_support": {"prompt": False, "media": False},
            "cache": {},
        }

    def clear_cache(self) -> None:
        """Native component execution currently has no cross-job tensor cache."""

    def unload(self) -> None:
        runtime = self._runtime
        self._runtime = None
        if runtime is not None:
            del runtime
        gc.collect()
        cleanup_accelerator_memory()

    def _load(self) -> Any:
        if self._runtime is not None:
            return self._runtime
        if not revalidate_runtime_request(self.request):
            raise RuntimeError("native Wan recipe is no longer identity-valid")

        from .wan22_i2v_runtime import NativeWanI2VRuntime, WanI2VArtifactPaths

        components = self.request.components
        paths = WanI2VArtifactPaths(
            support=self.request.support_plan.root,
            transformer_high=self.request.identities["transformer_high_noise"].path,
            transformer_low=self.request.identities["transformer_low_noise"].path,
            text_encoder=self.request.identities["text_encoder"].path,
            vae=self.request.identities["vae"].path,
        )
        # Cross-check the private support plan against the public component manifest
        # before handing paths to the heavyweight runtime.
        support_component = components["pipeline_support"]
        if support_component.get("path") != str(paths.support):
            raise RuntimeError("native Wan support path does not match its recipe manifest")
        self._runtime = NativeWanI2VRuntime.load(
            paths,
            support_plan=self.request.support_plan,
            adapter_plans=self.request.adapter_plans,
        )
        return self._runtime
