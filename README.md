# LatentSlate Engine

LatentSlate Engine is the first-party, opinionated generation backend for
[LatentSlate](https://github.com/EnviralDesign/LatentSlate). It exposes a small
catalog of creative tools instead of a general node graph.

The first wedge intentionally stays simple:

- a versioned tool catalog that LatentSlate can render as native controls;
- HTTP upload/download transport that works unchanged on localhost, a LAN host,
  or a remote GPU instance;
- asynchronous jobs with progress, cancellation requests, and downloadable artifacts;
- a model-bundle registry with a CLI download seam;
- two MiniMax-H3 tools backed by the upstream Diffusers modular pipeline:
  text-to-video-and-audio and first/last-frame video-and-audio.

There are no custom H3 kernels or WanGP-derived runtime changes in this initial
implementation. The model runtime is deliberately isolated so optimization can
happen later without changing the client protocol or tool schemas.

## Install

The API and development tools:

```bash
uv sync --extra dev
```

Include the H3 runtime dependencies:

```bash
uv sync --extra dev --extra h3
```

Download the canonical H3 bundle into the Hugging Face cache:

```bash
uv run latentslate-engine bundles install h3-basic
```

Run the server:

```bash
uv run latentslate-engine serve --host 127.0.0.1 --port 8765
```

For a LAN, Vast.ai, or other remote deployment, bind to `0.0.0.0`, set a bearer
token, and expose the port through a secure tunnel or HTTPS reverse proxy:

```bash
export LATENTSLATE_ENGINE_TOKEN='replace-me'
uv run latentslate-engine serve --host 0.0.0.0 --port 8765
```

LatentSlate sends media as multipart HTTP uploads and downloads generated
artifacts over HTTP. It never assumes that the desktop app and engine share a
filesystem.

## Important environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `LATENTSLATE_ENGINE_HOME` | platform user data directory | Uploaded assets, job state, and artifacts |
| `LATENTSLATE_ENGINE_TOKEN` | unset | Optional bearer token required by every `/v1` route |
| `LATENTSLATE_H3_MODEL` | `MiniMaxAI/MiniMax-H3` | H3 Hugging Face repository |
| `LATENTSLATE_H3_PROFILE` | `consumer_int8` | `consumer_int8` or `bf16_auto_offload` |
| `LATENTSLATE_H3_DEVICE` | `cuda` | Torch device used by H3 |

The `consumer_int8` profile follows the upstream Diffusers low-VRAM loading
recipe. It is a correctness-first starting point, not the final optimized
LatentSlate runtime. MiniMax-H3 is large; 64 GB system RAM may still be tight
until lower-RAM checkpoints and loaders are added.

## Protocol

The initial endpoints are:

```text
GET    /v1/health
GET    /v1/catalog
GET    /v1/bundles
POST   /v1/assets
POST   /v1/jobs
GET    /v1/jobs/{job_id}
DELETE /v1/jobs/{job_id}
GET    /v1/jobs/{job_id}/artifacts/{artifact_id}
```

Tool IDs and input keys are stable machine identities. Labels and descriptions
may evolve. Every job carries the tool's schema revision and hash; stale clients
receive a structured `schema_mismatch` error instead of an implicit migration.

## Development

```bash
uv run pytest
uv run ruff check .
```

The tests use an in-process lightweight tool and do not download or execute H3.
