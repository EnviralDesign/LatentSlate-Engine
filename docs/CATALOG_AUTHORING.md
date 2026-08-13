# Custom catalog authoring

LatentSlate Engine supports a local, inspectable authoring workflow for custom
resources and runnable recipes. The implementation deliberately separates catalog
metadata from artifact acquisition and recipe installation:

- **inspect** reads defensible source facts without publishing anything;
- **add** publishes one resource declaration and imports local files;
- **fetch** materializes one already-declared remote resource;
- **install** materializes the fixed closure of one or more recipes or a deployment
  profile.

The command-line interface and authenticated authoring API use the same typed request,
validation, serialization, closure-planning, publication, and lifecycle services.

Catalog work must follow the
[Comfy authority and Engine execution policy](./COMFY_ENGINE_POLICY.md). Official
Comfy workflows are source evidence; they do not authorize recipes that launch
or depend on ComfyUI. Optimized recipes bind Engine-native implementations and
record actual Comfy Kitchen/native dispatch where claimed.

## Catalog locations

User-owned authoring output lives only below the configured Engine home:

```text
<ENGINE_HOME>/resource_declarations   published resource TOMLs
<ENGINE_HOME>/recipes                 published runnable recipe TOMLs
<ENGINE_HOME>/profiles                deployment profiles
<ENGINE_HOME>/drafts/recipes          editable recipe drafts
<ENGINE_HOME>/models                  imported/fetched model artifacts
<ENGINE_HOME>/loras                   imported/fetched LoRA artifacts
<ENGINE_HOME>/temp/catalog-authoring  transactional staging
```

Package-owned built-ins under `src/latentslate_engine/builtin_*` are read-only. Private
recipe roots configured through `LATENTSLATE_RECIPE_PATHS` remain discoverable but are
not mutated by the authoring service.

## Resource Editor

The Engine-hosted Resource Editor is a local browser client for resource declarations,
not recipes. Start the Engine in one terminal and open it from another:

```powershell
.\scripts\engine.ps1 serve
# separate terminal
.\scripts\engine.ps1 author
```

`author` verifies that Engine is already reachable and opens
`http://127.0.0.1:8765/authoring/`. `--url` may select another loopback HTTP(S)
Engine origin (`localhost`, `127.0.0.1`, or `[::1]`); it is normalized to
`/authoring/`. Redirects are refused, so it never opens or probes a remote target.

The page shell is public so it can load locally, but every `/v1` request keeps normal
bearer authentication. The token is entered into the browser for that session only.
The editor lists resources by family; built-ins are read-only, while local declarations
can be created, updated, or deleted. Deletion fails without changing disk state when a
published recipe or draft still references the resource. Once unreferenced, the editor
can remove only the declaration or remove both the declaration and installed artifact;
keeping the artifact may make it reappear as a read-only discovered resource. The normal
creation flow is inspect, review the generated declaration, validate/publish, then
optionally fetch the declared artifact. LoRA declarations require `base_model`.

Browser inspection/publication supports Hugging Face and CivitAI sources only. Local
file imports and direct HTTPS declarations remain trusted local CLI operations. Recipe
authoring remains TOML/CLI-only. Browser/API publication writes the catalog on disk but
requires an Engine restart before the running job registry uses the change.

## Resource workflow

### Read-only inspection

```powershell
.\scripts\engine.ps1 resources inspect `
  hf://black-forest-labs/FLUX.2-klein-4B/transformer/diffusion_pytorch_model.safetensors

.\scripts\engine.ps1 resources inspect `
  civitai://version/123456 `
  --file-id 789012 `
  --json

.\scripts\engine.ps1 resources inspect C:\models\custom.safetensors
```

Inspection keeps three categories separate:

- `facts`: exact or directly observed properties such as file name, bytes, SHA-256,
  file/container format, SafeTensors metadata, tensor keys, shapes, dtypes, and schema
  fingerprint;
- `detected`: bounded inferences derived from those facts;
- `recommended`: editable defaults such as a display name, candidate family, or
  component role.

Recommendations are never treated as proof that an unsupported architecture is
runnable. Recipe publication still has to resolve through a supported Engine family,
operation, runtime adapter, and typed recipe path.

### Add a declaration

Hugging Face exact file:

```powershell
.\scripts\engine.ps1 resources add `
  hf://example-org/example-model/weights/model.safetensors `
  --id model:custom:example-transformer `
  --kind model `
  --family klein4b `
  --component transformer `
  --name "Example transformer"
```

Filtered Hugging Face snapshot:

```powershell
.\scripts\engine.ps1 resources add hf://example-org/example-model `
  --allow-pattern model_index.json `
  --allow-pattern scheduler/* `
  --allow-pattern tokenizer/* `
  --allow-pattern transformer/config.json `
  --id model:custom:example-support `
  --kind model `
  --family klein4b `
  --component pipeline_support `
  --format directory `
  --name "Example pipeline support"
```

CivitAI exact file:

```powershell
.\scripts\engine.ps1 resources add civitai://version/123456 `
  --file-id 789012 `
  --requires-auth `
  --id lora:custom:example `
  --kind lora `
  --family klein4b `
  --name "Example LoRA"
```

Use `--requires-auth` for CivitAI files whose delivery policy requires a signed-in
download even when their metadata page is public. The declaration then records
`CIVITAI_TOKEN` as a required secret, preflight fails before network access when it
is absent, and the token is stripped from cross-origin delivery redirects.

Direct HTTPS file:

```powershell
.\scripts\engine.ps1 resources add https://models.example.com/model.safetensors `
  --size-bytes 123456789 `
  --sha256 <64-hex-digest> `
  --id model:custom:https-example `
  --kind model `
  --family custom `
  --name "HTTPS example"
```

Local file import:

```powershell
.\scripts\engine.ps1 resources add C:\models\model.safetensors `
  --id model:custom:local-example `
  --kind model `
  --family custom `
  --name "Local example"
```

Hugging Face revisions are resolved to an immutable commit. CivitAI model-version
pages are resolved to an exact file ID and publication is refused when selection is
ambiguous. Direct HTTPS declarations require an exact size and SHA-256 and remain
unmaterialized until `resources fetch` or a recipe/deployment install needs them.
Local files are copied into Engine-owned storage during `resources add`, hashed, and
published truthfully as manual imports because no remote reacquisition identity
exists.

Direct HTTPS is a first-class exact source rather than a disguised CivitAI or manual
source. Its stable URL and SHA-256 are serialized in the declaration; redirects remain
HTTPS-only and credentials are never attached. Private/local literal IP destinations
are rejected. The authenticated server does not accept arbitrary direct-URL authoring,
so the browser/API surface cannot be used as a general request proxy; trusted local
CLI authoring is required.

### Validate and fetch

```powershell
.\scripts\engine.ps1 resources validate
.\scripts\engine.ps1 resources validate model:custom:example-transformer --json
.\scripts\engine.ps1 resources fetch model:custom:example-transformer
```

`resources fetch` is a thin single-resource entry point over the existing deployment
installer. It reuses exact-source selection, secret lookup, redirect controls,
resumable staging, capacity checks, size/hash verification, no-clobber publication,
and final catalog rediscovery. It does not implement a second downloader.

## Recipe workflow

Recipe TOML remains the advanced escape hatch and the initial deterministic authoring
input. The Engine publishes a capabilities description so CLI/API/UI clients can build
family- and operation-aware forms without maintaining a second schema.

```powershell
.\scripts\engine.ps1 recipes capabilities
.\scripts\engine.ps1 recipes capabilities --json > authoring-capabilities.json

.\scripts\engine.ps1 recipes validate --file .\my-recipe.toml
.\scripts\engine.ps1 recipes create .\my-recipe.toml
.\scripts\engine.ps1 recipes publish my-model.text-to-image.custom
```

`recipes create` parses the file through the existing `VariantDefinition` models,
compiles it against the selected curated base adapter, validates resource roles and
runtime combinations, previews stable generated TOML, builds the deduplicated resource
closure, and saves an editable draft under `<ENGINE_HOME>/drafts/recipes`. Drafts may
contain errors so a person or agent can iterate. `recipes publish` refuses invalid,
unsafe, duplicate, or unsupported recipes and atomically moves the validated content
into the local recipe catalog.

A minimal fixed-model recipe can use the existing typed shape:

```toml
[runnable_recipe]
schema_version = 1
key = "example.text-to-image.custom"
schema_revision = 1
name = "Example text to image"
enabled = true
family = "klein4b"
base_tool = "flux2_klein4b.text_to_image"

[runnable_recipe.model]
resource = "model:custom:example"

[runnable_recipe.optimizations]
offload = "staged"
quantization = "fp8"
```

A fixed LoRA is another exact resource in the same closure. The runtime validates
the selected family/format combination before publication:

```toml
[[runnable_recipe.loras]]
slot = "style"
resource = "lora:klein9b:example-style"
strength = 0.8
```

For a user-selectable slot, set `exposed = true`, provide `allowed` resource IDs or
tags, and optionally expose its strength with `strength_exposed = true`. Fixed LoRAs
are preferable for reproducible studies because the recipe identity and deployment
lock include the exact adapter resource and strength. Klein stored FP8/NVFP4 recipes
apply compatible LoRAs as an additive Comfy-style branch beside the native Kitchen
matmul; the base quantized weight is never dequantized or silently replaced.

Generated TOML is intentionally readable, deterministic, free of credentials, and
suitable for source control. Hand edits remain supported; rerun `recipes validate
--file` and `recipes create --replace` before publishing a replacement.

## Authoring API

All authoring routes preserve normal Engine bearer-token authentication:

```text
GET  /v1/authoring/capabilities
GET  /v1/authoring/status
GET  /v1/authoring/resources
GET  /v1/authoring/resources/{resource_id}
POST /v1/authoring/resources/inspect
POST /v1/authoring/resources/suggest-id
POST /v1/authoring/resources/preview
POST /v1/authoring/resources
PUT  /v1/authoring/resources/{resource_id}
DELETE /v1/authoring/resources/{resource_id}
GET  /v1/authoring/resources/validate
POST /v1/authoring/resources/fetch?resource_id=...
POST /v1/authoring/recipes/validate
POST /v1/authoring/recipes/drafts
POST /v1/authoring/recipes/drafts/{recipe_key}/publish
```

The server endpoints intentionally reject local filesystem sources and generic direct
HTTPS authoring. This prevents a browser client from turning Engine into an arbitrary
filesystem reader or general-purpose server-side request proxy. Local imports and
direct-URL declarations are available through the trusted local CLI. Once an exact
resource has been deliberately published in the local catalog, the authenticated
single-resource fetch endpoint may materialize it through the normal installer.
Hugging Face and CivitAI metadata lookup and declaration publication remain available
through the API.

## Transaction and lifecycle behavior

Publication stages content below the Engine home, validates the generated TOML by
round-trip parsing, checks duplicate IDs/keys and artifact-path ownership, publishes
with atomic no-clobber filesystem operations, rediscovers the catalog, and rolls back
new artifacts and metadata if post-publication validation fails.

One-shot CLI commands build a fresh registry every invocation. A running Engine keeps
one concrete registry shared by catalog routes and the job manager, so silently
replacing only part of it would be unsafe. This tranche therefore uses an explicit
restart-required lifecycle:

- CLI publication reports `next_cli_invocation`;
- API publication reports `restart_engine`;
- `GET /v1/authoring/status` compares the startup catalog revision with the current
  on-disk revision and reports `stale=true` until Engine restarts.

A coordinated zero-active-jobs registry swap can be added later. Until then, the API
never claims that newly published recipes are active in the current process.

## Security and reproducibility boundaries

- Engine authentication applies to every authoring endpoint.
- The normal server default remains loopback-local.
- Tokens are read by environment-variable name and are never serialized into TOML,
  API responses, logs, or locks.
- Direct URLs require HTTPS, an exact size and SHA-256, reject
  userinfo/query strings/fragments and private/local literal IPs, and can be authored
  only through the trusted local CLI.
- Publication is confined to Engine-owned catalog/artifact roots.
- Existing targets are never silently overwritten; replacement must be explicit and
  cannot move an existing resource to a new path.
- Deletion is local-declaration-only, refuses resources referenced by any published
  recipe or draft, and removes artifact bytes only after an explicit request.
- Archive extraction and arbitrary directory ingestion are not implemented.
- Routine tests use tiny synthetic SafeTensors files and mocked metadata/transfers;
  they do not download production model weights.

Detection is a convenience, not a compatibility oracle. Unknown resources may be
registered and retained, but runnable recipes must still pass a supported family and
operation adapter. Runtime errors remain part of the intended expert author-test-edit
loop when semantics cannot be proven statically.
