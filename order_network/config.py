import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

LLM_BASE_URL = os.getenv("LITELLM_API", "http://localhost:4000")
LLM_API_KEY = os.getenv("LITELLM_KEY") or "none"
LLM_MODEL = os.getenv("LITELLM_DEFAULT_MODEL", "gpt-4o-mini")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
# "openai" = any OpenAI-compatible endpoint (LiteLLM, Ollama, ...); "google" = native Gemini API.
# Default: google when LITELLM_API points to the Gemini API, openai otherwise.
LLM_PROVIDER = os.getenv("LLM_PROVIDER") or ("google" if "generativelanguage.googleapis.com" in LLM_BASE_URL else "openai")

# "local" = this project's registry service; "litellm" = LiteLLM agent gateway (REGISTRY_URL = proxy URL).
REGISTRY_BACKEND = os.getenv("REGISTRY_BACKEND", "local").lower()
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8000").rstrip("/")
REGISTRY_TOKEN = os.getenv("REGISTRY_TOKEN") or (LLM_API_KEY if REGISTRY_BACKEND == "litellm" else None)
REGISTRY_TTL_SECONDS = float(os.getenv("REGISTRY_TTL_SECONDS", "30"))
HEARTBEAT_SECONDS = float(os.getenv("HEARTBEAT_SECONDS", "10"))

A2A_TIMEOUT_SECONDS = float(os.getenv("A2A_TIMEOUT_SECONDS", "300"))

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))

DEFAULT_PORTS = {
    "registry": 8000,
    "orchestrator": 8100,
    "order-intake": 8101,
    "inventory": 8102,
    "production": 8103,
}
