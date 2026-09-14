"""Content provenance for the curated Qwen service operation."""

from latentslate_engine.identity import FileContentIdentity


def _content(path):
    identity = FileContentIdentity.from_path(path)
    return {"sha256": identity.sha256, "size": identity.size}


def model_provenance(identity):
    """Hash the concrete model composition once per worker model identity."""
    return {
        "diffusion": _content(identity.diffusion.path),
        "text_encoder": _content(identity.text_encoder.path),
        "vae": _content(identity.vae.path),
        "tokenizer": {
            item.path.name: _content(item.path) for item in identity.tokenizer_files
        },
        "adapters": [
            {"position": position, "strength": strength, **_content(artifact.path)}
            for position, (artifact, strength) in enumerate(identity.adapters)
        ],
    }


def request_provenance(models, request):
    """Preserve logical reference roles and the effective curated execution policy."""
    return {
        "operation": "qwen2511.edit",
        "models": models,
        "prompt": request["prompt"],
        "seed": request["seed"],
        "inputs": [
            {"slot": key, **_content(request[key])}
            for key in ("image_1", "image_2", "image_3")
            if request[key] is not None
        ],
        "settings": {
            "steps": request["steps"], "cfg": request["cfg"], "shift": request["shift"],
            "sampler": "euler", "scheduler": "simple", "denoise": 1.0,
            "cfg_norm_strength": 1.0, "reference_latents_method": "index_timestep_zero",
            "negative_prompt": "",
        },
    }
