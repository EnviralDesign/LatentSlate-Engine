"""Bounded validation for complete family-specific Diffusers repositories.

The generic resource catalog intentionally recognizes directories cheaply. Runtime
adapters use the stricter contracts in this module before advertising a selected
folder as executable. Validation reads JSON documents and SafeTensors headers only;
it never materializes tensor payloads or changes stored weights.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import probe_safetensors, revalidate_artifact

_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_SUPPORT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class WeightComponentContract:
    name: str
    class_name: str
    weight_stem: str
    allowed_dtypes: frozenset[str]
    required_dtypes: frozenset[str]
    schema_sha256: str
    config_sha256: str
    transformers_component: bool = False


@dataclass(frozen=True, slots=True)
class DiffusersRepositoryContract:
    family: str
    root_class: str
    components: tuple[tuple[str, str, str], ...]
    weights: tuple[WeightComponentContract, ...]
    required_files: tuple[str, ...]
    file_fingerprints: tuple[tuple[str, int, str], ...]
    json_fingerprints: tuple[tuple[str, str], ...]
    mirrored_model_indexes: tuple[str, ...] = ()


H3_REPOSITORY_CONTRACT = DiffusersRepositoryContract(
    family="H3",
    root_class="MiniMaxH3ModularPipeline",
    components=(
        ("text_encoder", "transformers", "Qwen3VLForConditionalGeneration"),
        ("tokenizer", "transformers", "Qwen2TokenizerFast"),
        ("processor", "transformers", "Qwen3VLProcessor"),
        ("vae", "diffusers", "AutoencoderKLMiniMaxH3"),
        ("audio_vae", "diffusers", "AutoencoderKLMiniMaxH3Audio"),
        ("transformer", "diffusers", "MiniMaxH3Transformer3DModel"),
        ("transformer_ref", "diffusers", "MiniMaxH3Transformer3DModel"),
        ("scheduler", "diffusers", "MiniMaxH3Scheduler"),
        ("audio_scheduler", "diffusers", "MiniMaxH3Scheduler"),
    ),
    weights=(
        WeightComponentContract(
            "text_encoder",
            "Qwen3VLForConditionalGeneration",
            "model",
            frozenset({"BF16"}),
            frozenset({"BF16"}),
            "ec406450fd9d2f584cd7fc5ad11c547efe7c8db3c19eb0d20f560145a1470cd8",
            "78ba49d993b301d94f3c8a7176c301fe3d0f2414691f5757e336306599813619",
            transformers_component=True,
        ),
        WeightComponentContract(
            "transformer",
            "MiniMaxH3Transformer3DModel",
            "diffusion_pytorch_model",
            frozenset({"BF16", "F32"}),
            frozenset({"BF16"}),
            "c8f0ffb0f59107155f255ccc8516378fc2f798a07b3676cc9d6102551f6aec64",
            "5ffe00008ad74b23d59a4518efcedb12c3ed609c8804e2dcf3c278524e89b8e7",
        ),
        WeightComponentContract(
            "vae",
            "AutoencoderKLMiniMaxH3",
            "diffusion_pytorch_model",
            frozenset({"F32"}),
            frozenset({"F32"}),
            "23394f5bb29a9c088b268833aea8ef5abfe72aa7022cfe71e426aad97d4626c0",
            "16e871d5b302192ab0c8b7b8200f07d8b51565443620eafa59a1de61f80d507b",
        ),
        WeightComponentContract(
            "audio_vae",
            "AutoencoderKLMiniMaxH3Audio",
            "diffusion_pytorch_model",
            frozenset({"F32"}),
            frozenset({"F32"}),
            "6ae3060204ad967ced7d0bbdaf53185a0eec926b210558e8c42b0eaefdccdf79",
            "c1719be088d576139317ea61b596ad3d4b44867fa0f835fb67239b11aa6ab8b4",
        ),
    ),
    required_files=(
        "tokenizer/merges.txt",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "processor/preprocessor_config.json",
        "processor/tokenizer.json",
        "processor/tokenizer_config.json",
        "processor/video_preprocessor_config.json",
        "scheduler/scheduler_config.json",
        "audio_scheduler/scheduler_config.json",
    ),
    file_fingerprints=(
        (
            "tokenizer/merges.txt",
            1_671_839,
            "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
        ),
        (
            "tokenizer/tokenizer.json",
            7_032_403,
            "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
        ),
        (
            "tokenizer/tokenizer_config.json",
            11_003,
            "a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68",
        ),
        (
            "tokenizer/vocab.json",
            2_776_833,
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        ),
        (
            "processor/preprocessor_config.json",
            390,
            "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
        ),
        (
            "processor/tokenizer.json",
            7_032_403,
            "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
        ),
        (
            "processor/tokenizer_config.json",
            11_003,
            "a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68",
        ),
        (
            "processor/video_preprocessor_config.json",
            385,
            "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
        ),
    ),
    json_fingerprints=(
        (
            "scheduler/scheduler_config.json",
            "92b3c23e43b00b389e26e79e6acb47241b68248d192d1e89911ada39746693e1",
        ),
        (
            "audio_scheduler/scheduler_config.json",
            "58f9103009c9cb396a70426a511a57b34f498dc589f5889e40c7a0843ae07745",
        ),
    ),
    mirrored_model_indexes=("modular_model_index.json",),
)


LTX23_REPOSITORY_CONTRACT = DiffusersRepositoryContract(
    family="LTX 2.3",
    root_class="LTX2Pipeline",
    components=(
        ("audio_vae", "diffusers", "AutoencoderKLLTX2Audio"),
        ("connectors", "ltx2", "LTX2TextConnectors"),
        ("scheduler", "diffusers", "FlowMatchEulerDiscreteScheduler"),
        ("text_encoder", "transformers", "Gemma3ForConditionalGeneration"),
        ("tokenizer", "transformers", "GemmaTokenizerFast"),
        ("transformer", "diffusers", "LTX2VideoTransformer3DModel"),
        ("vae", "diffusers", "AutoencoderKLLTX2Video"),
        ("vocoder", "ltx2", "LTX2VocoderWithBWE"),
    ),
    weights=(
        WeightComponentContract(
            "audio_vae",
            "AutoencoderKLLTX2Audio",
            "diffusion_pytorch_model",
            frozenset({"BF16"}),
            frozenset({"BF16"}),
            "1fa2e3369df02a7263e6946981b5ce9417d331df9b4306fcb10e2e9c9f9b5970",
            "cb6ca1c21cfb87b617ad2e30a00dc08e61b8fe515eb2fe89a5b4dc20393f1bd1",
        ),
        WeightComponentContract(
            "connectors",
            "LTX2TextConnectors",
            "diffusion_pytorch_model",
            frozenset({"BF16"}),
            frozenset({"BF16"}),
            "a0860a7095e09ac2a6160be11cba2395f0cc3371d548c96c670b4fb2e587a79b",
            "b9be95275cfdb1cb312cf1c4bb3f91ff6a3f7eb5a68ae76ab17551f73472ecf0",
        ),
        WeightComponentContract(
            "text_encoder",
            "Gemma3ForConditionalGeneration",
            "model",
            frozenset({"F32"}),
            frozenset({"F32"}),
            "ba18a30f8f42c0200d1d419392bd1cafec54e15b7983eeb4127a202891da180e",
            "a95cfb754f4378da49fa8624cb2e41f31d6b49fd6fd1f32d0ec18e951b992b02",
            transformers_component=True,
        ),
        WeightComponentContract(
            "transformer",
            "LTX2VideoTransformer3DModel",
            "diffusion_pytorch_model",
            frozenset({"BF16", "F32"}),
            frozenset({"BF16"}),
            "cfe82969a244a70257d44dda5b1b852403f63a38b18f416eebf59072ea76aada",
            "4f13af5cc250c259609da2fcb5fd9d897972df0b586510482953810b64a1846f",
        ),
        WeightComponentContract(
            "vae",
            "AutoencoderKLLTX2Video",
            "diffusion_pytorch_model",
            frozenset({"BF16"}),
            frozenset({"BF16"}),
            "a8fbeac784ba1d1c25c0d4de32244e0c8058d234eb85d0c5949aae31a8cdf271",
            "0c68149a0164c9377d1122872717788208e4a29ac5bfc300a0c8439d9d323231",
        ),
        WeightComponentContract(
            "vocoder",
            "LTX2VocoderWithBWE",
            "diffusion_pytorch_model",
            frozenset({"BF16"}),
            frozenset({"BF16"}),
            "f70da9b0a83a72cd1802afd340f9b12347e375e239558e632c469f3d4b2acde4",
            "71473646a878d591d9633432eba33343d95281c0c73ddba0c87075e4de198270",
        ),
    ),
    required_files=(
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer.model",
        "tokenizer/tokenizer_config.json",
        "scheduler/scheduler_config.json",
    ),
    file_fingerprints=(
        (
            "tokenizer/tokenizer.json",
            33_384_568,
            "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795",
        ),
        (
            "tokenizer/tokenizer.model",
            4_689_074,
            "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
        ),
        (
            "tokenizer/tokenizer_config.json",
            1_155_387,
            "983c80895e7b188911f5ccfbb4f780a4e73fb3131ed2080a3e847ef8623cfdca",
        ),
    ),
    json_fingerprints=(
        (
            "scheduler/scheduler_config.json",
            "78c19021524413bbaafa12625dcc93c86d0a05f71a91ce0995469be79f3a55f4",
        ),
    ),
)


KLEIN4B_REPOSITORY_CONTRACT = DiffusersRepositoryContract(
    family="FLUX.2 Klein 4B",
    root_class="Flux2KleinPipeline",
    components=(
        ("scheduler", "diffusers", "FlowMatchEulerDiscreteScheduler"),
        ("text_encoder", "transformers", "Qwen3ForCausalLM"),
        ("tokenizer", "transformers", "Qwen2TokenizerFast"),
        ("transformer", "diffusers", "Flux2Transformer2DModel"),
        ("vae", "diffusers", "AutoencoderKLFlux2"),
    ),
    # The stored-FP8 adapter injects its own validated transformer. Only the
    # dense components actually loaded from pipeline support are weight-bound.
    weights=(
        WeightComponentContract(
            "text_encoder",
            "Qwen3ForCausalLM",
            "model",
            frozenset({"BF16"}),
            frozenset({"BF16"}),
            "74ecea286e99bd123ab783df0b935ebfa4f25fe80d0b1c2af39c2388da7d8ad3",
            "d8175578997c0a74914aaa139509da36b525433d8fc14c222eb06b77b42fbd3d",
            transformers_component=True,
        ),
        WeightComponentContract(
            "vae",
            "AutoencoderKLFlux2",
            "diffusion_pytorch_model",
            frozenset({"BF16", "I64"}),
            frozenset({"BF16", "I64"}),
            "055df6465e46c1ea4425d900519fceb1c126bdfba166ebcef4b6f6827a48934a",
            "fc997d9f5ba71e0f309eb2b48fa5fa8b994400829a862faa2e6d240228498f5b",
        ),
    ),
    required_files=(
        "tokenizer/merges.txt",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "scheduler/scheduler_config.json",
    ),
    file_fingerprints=(
        (
            "tokenizer/merges.txt",
            1_671_853,
            "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        ),
        (
            "tokenizer/tokenizer.json",
            11_422_654,
            "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        ),
        (
            "tokenizer/tokenizer_config.json",
            5_404,
            "443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0",
        ),
        (
            "tokenizer/vocab.json",
            2_776_833,
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        ),
    ),
    json_fingerprints=(
        (
            "scheduler/scheduler_config.json",
            "eaf1d846ab01d8fbca6c2916fd87282dde4f02cedf66e9678d604a5eea7c002d",
        ),
    ),
)


def validate_diffusers_repository(
    path: Path,
    contract: DiffusersRepositoryContract,
) -> None:
    """Prove that ``path`` contains the complete stored layout for ``contract``."""

    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"{contract.family} requires a complete Diffusers directory")

    model_index = _read_json(root / "model_index.json")
    _validate_model_index(model_index, contract, source="model_index.json")
    for relative in contract.mirrored_model_indexes:
        mirrored = _read_json(root / relative)
        _validate_model_index(mirrored, contract, source=relative)

    for relative in contract.required_files:
        required = _contained_file(root, relative)
        if required.stat().st_size <= 0:
            raise ValueError(f"{contract.family} support file is empty: {relative}")

    for relative, expected_size, expected_sha256 in contract.file_fingerprints:
        _validate_support_file(root, contract.family, relative, expected_size, expected_sha256)

    for relative, expected_fingerprint in contract.json_fingerprints:
        actual_fingerprint = _semantic_fingerprint(_read_json(root / relative))
        if actual_fingerprint != expected_fingerprint:
            raise ValueError(
                f"{contract.family} behavior config differs from the supported contract: {relative}"
            )

    weight_names = {component.name for component in contract.weights}
    for component in contract.weights:
        _validate_weight_component(root, contract.family, component)

    for component_name, _library, class_name in contract.components:
        if component_name in weight_names:
            continue
        if component_name.endswith("scheduler") or component_name == "scheduler":
            config = _read_json(root / component_name / "scheduler_config.json")
            _require_class(config, class_name, component_name, transformers=False)


def _validate_model_index(
    document: Any,
    contract: DiffusersRepositoryContract,
    *,
    source: str,
) -> None:
    if not isinstance(document, dict) or document.get("_class_name") != contract.root_class:
        raise ValueError(
            f"{contract.family} {source} must declare _class_name={contract.root_class!r}"
        )
    for name, library, class_name in contract.components:
        value = document.get(name)
        if (
            not isinstance(value, list)
            or len(value) < 2
            or value[0] != library
            or value[1] != class_name
        ):
            raise ValueError(f"{contract.family} {source} has an incompatible {name!r} component")


def _validate_weight_component(
    root: Path,
    family: str,
    contract: WeightComponentContract,
) -> None:
    component_root = root / contract.name
    config = _read_json(component_root / "config.json")
    _require_class(
        config,
        contract.class_name,
        contract.name,
        transformers=contract.transformers_component,
    )
    if _semantic_fingerprint(config) != contract.config_sha256:
        raise ValueError(
            f"{family} component {contract.name!r} config differs from the supported contract"
        )

    single = component_root / f"{contract.weight_stem}.safetensors"
    index = component_root / f"{contract.weight_stem}.safetensors.index.json"
    layouts = int(single.is_file()) + int(index.is_file())
    if layouts != 1:
        raise ValueError(
            f"{family} component {contract.name!r} must contain exactly one "
            "SafeTensors weight layout"
        )

    if single.is_file():
        shard_paths = (single,)
        indexed_keys: dict[str, set[str]] | None = None
    else:
        index_document = _read_json(index)
        weight_map = index_document.get("weight_map") if isinstance(index_document, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{family} component {contract.name!r} has an invalid shard index")
        indexed_keys = {}
        for tensor_name, shard_name in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name:
                raise ValueError(f"{family} component {contract.name!r} has an invalid tensor name")
            if (
                not isinstance(shard_name, str)
                or Path(shard_name).name != shard_name
                or not shard_name.endswith(".safetensors")
            ):
                raise ValueError(f"{family} component {contract.name!r} has an unsafe shard path")
            indexed_keys.setdefault(shard_name, set()).add(tensor_name)
        shard_paths = tuple(_contained_file(component_root, name) for name in sorted(indexed_keys))

    observed_dtypes: set[str] = set()
    observed_schema: dict[str, tuple[str, tuple[int, ...]]] = {}
    from safetensors import safe_open

    for shard in shard_paths:
        probe = probe_safetensors(shard)
        observed_dtypes.update(probe.tensor_dtypes)
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
            if indexed_keys is not None and actual_keys != indexed_keys[shard.name]:
                raise ValueError(
                    f"{family} component {contract.name!r} shard index does not exactly "
                    f"describe {shard.name!r}"
                )
            for key in sorted(actual_keys):
                if key in observed_schema:
                    raise ValueError(
                        f"{family} component {contract.name!r} duplicates tensor {key!r}"
                    )
                view = handle.get_slice(key)
                observed_schema[key] = (view.get_dtype(), tuple(view.get_shape()))
        if not revalidate_artifact(probe.identity):
            raise ValueError(
                f"{family} component {contract.name!r} changed during header validation"
            )
    if not contract.required_dtypes.issubset(observed_dtypes) or not observed_dtypes.issubset(
        contract.allowed_dtypes
    ):
        raise ValueError(
            f"{family} component {contract.name!r} has stored dtypes "
            f"{sorted(observed_dtypes)!r}; expected only "
            f"{sorted(contract.allowed_dtypes)!r} with "
            f"{sorted(contract.required_dtypes)!r} present"
        )
    if _schema_fingerprint(observed_schema) != contract.schema_sha256:
        raise ValueError(
            f"{family} component {contract.name!r} tensor schema is incomplete or incompatible"
        )


def _require_class(
    document: Any,
    expected: str,
    component: str,
    *,
    transformers: bool,
) -> None:
    if not isinstance(document, dict):
        raise TypeError(f"Component {component!r} config must be a JSON object")
    if transformers:
        architectures = document.get("architectures")
        valid = isinstance(architectures, list) and architectures == [expected]
    else:
        valid = document.get("_class_name") == expected
    if not valid:
        raise ValueError(f"Component {component!r} config does not declare {expected!r}")


def _read_json(path: Path) -> Any:
    file_path = Path(path)
    try:
        size = file_path.stat().st_size
    except OSError as exc:
        raise ValueError(f"Required repository file is missing: {file_path}") from exc
    if size <= 0 or size > _MAX_JSON_BYTES:
        raise ValueError(f"Repository JSON file has an invalid size: {file_path}")
    try:
        with file_path.open("rb") as handle:
            raw = handle.read(_MAX_JSON_BYTES + 1)
        if len(raw) != size:
            raise ValueError(f"Repository JSON file changed while being read: {file_path}")
        return json.loads(raw, object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Repository JSON file is invalid: {file_path}") from exc


def _contained_file(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe repository-relative path: {relative!r}")
    try:
        path = (root / relative_path).resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Required repository file is missing: {relative}") from exc
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Required repository file is missing: {relative}")
    return path


def _validate_support_file(
    root: Path,
    family: str,
    relative: str,
    expected_size: int,
    expected_sha256: str,
) -> None:
    path = _contained_file(root, relative)
    if expected_size <= 0 or expected_size > _MAX_SUPPORT_BYTES:
        raise ValueError(f"Invalid support contract size for {relative!r}")
    try:
        with path.open("rb") as handle:
            raw = handle.read(_MAX_SUPPORT_BYTES + 1)
    except OSError as exc:
        raise ValueError(f"{family} support file cannot be read: {relative}") from exc
    if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{family} support file differs from the supported contract: {relative}")


def _schema_fingerprint(schema: dict[str, tuple[str, tuple[int, ...]]]) -> str:
    payload = [(key, dtype, shape) for key, (dtype, shape) in sorted(schema.items())]
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _semantic_fingerprint(document: Any) -> str:
    ignored = {"_diffusers_version", "transformers_version", "_name_or_path"}

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items() if key not in ignored}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return hashlib.sha256(_canonical_json(normalize(document))).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result
