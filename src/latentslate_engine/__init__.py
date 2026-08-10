"""LatentSlate Engine package."""

from dotenv import find_dotenv, load_dotenv

from .model_store import configure_library_cache_environment, repository_root

# Load a repository-local .env before submodules import huggingface_hub or read
# Engine settings. Real process/container environment variables retain priority.
_repository_dotenv = repository_root() / ".env"
_dotenv_path = str(_repository_dotenv) if _repository_dotenv.is_file() else find_dotenv(usecwd=True)
if _dotenv_path:
    load_dotenv(_dotenv_path, override=False)

# Third-party model libraries must never choose a user-global cache for Engine
# downloads. Force their internal caches beneath LATENTSLATE_ENGINE_HOME before
# any Engine submodule imports Hugging Face, Diffusers, Transformers, or Torch.
configure_library_cache_environment()

__version__ = "0.1.0"
