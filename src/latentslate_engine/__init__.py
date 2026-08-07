"""LatentSlate Engine package."""

from dotenv import find_dotenv, load_dotenv

# Load a repository-local .env before submodules import huggingface_hub or read
# Engine settings. Real process/container environment variables retain priority.
_dotenv_path = find_dotenv(usecwd=True)
if _dotenv_path:
    load_dotenv(_dotenv_path, override=False)

__version__ = "0.1.0"
