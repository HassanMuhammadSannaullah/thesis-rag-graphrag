"""
Central configuration for the thesis project.
Loads .env and provides all paths/parameters in one place.
"""
import os
from pathlib import Path

from dotenv import load_dotenv


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() == "true"

# Locate project root (where .env lives)
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Provider and credentials
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "local_openai").lower()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "local-dev-key")
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8001/v1")

# Model names
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
LOCAL_GENERATION_MODEL = os.getenv(
    "LOCAL_GENERATION_MODEL",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
LOCAL_GRAPHRAG_INDEX_MODEL = os.getenv(
    "LOCAL_GRAPHRAG_INDEX_MODEL",
    LOCAL_GENERATION_MODEL,
)
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "intfloat/e5-base-v2",
)
EVALUATION_FRAMEWORK = os.getenv("EVALUATION_FRAMEWORK", "classic").strip().lower()
if EVALUATION_FRAMEWORK not in {"ragas", "classic"}:
    EVALUATION_FRAMEWORK = "classic"
RAGAS_EVAL_BACKEND = os.getenv("RAGAS_EVAL_BACKEND", MODEL_BACKEND).lower()
RAGAS_EVAL_GENERATION_MODEL = os.getenv(
    "RAGAS_EVAL_GENERATION_MODEL",
    LOCAL_GENERATION_MODEL if RAGAS_EVAL_BACKEND == "local_openai" else GENERATION_MODEL,
)
RAGAS_EVAL_EMBEDDING_MODEL = os.getenv(
    "RAGAS_EVAL_EMBEDDING_MODEL",
    LOCAL_EMBEDDING_MODEL if RAGAS_EVAL_BACKEND == "local_openai" else EMBEDDING_MODEL,
)

# Local server/runtime settings
LOCAL_SERVER_HOST = os.getenv("LOCAL_SERVER_HOST", "127.0.0.1")
LOCAL_SERVER_PORT = int(os.getenv("LOCAL_SERVER_PORT", "8001"))
LOCAL_MAX_NEW_TOKENS = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "2048"))
LOCAL_MODEL_DEVICE = os.getenv("LOCAL_MODEL_DEVICE", "auto")
LOCAL_MODEL_DEVICE_MAP = os.getenv("LOCAL_MODEL_DEVICE_MAP", "none")
LOCAL_MODEL_TRUST_REMOTE_CODE = _env_flag("LOCAL_MODEL_TRUST_REMOTE_CODE")
LOCAL_MODEL_STORAGE_BUDGET_GB = int(os.getenv("LOCAL_MODEL_STORAGE_BUDGET_GB", "150"))
LOCAL_MODEL_VRAM_GB = int(os.getenv("LOCAL_MODEL_VRAM_GB", "24"))
LOCAL_MODEL_RAM_GB = int(os.getenv("LOCAL_MODEL_RAM_GB", "32"))

# Rate-limit / quota settings
MAX_RPM = int(os.getenv("MAX_RPM", "10"))
MAX_RPD = int(os.getenv("MAX_RPD", "500"))
EMBEDDING_RPM = int(os.getenv("EMBEDDING_RPM", "100"))
SLEEP_BETWEEN_REQ = float(os.getenv("SLEEP_BETWEEN_REQUESTS", "4.0"))
SLEEP_BETWEEN_EMB = float(os.getenv("SLEEP_BETWEEN_EMBEDDINGS", "0.5"))

# Sample sizes
DEV_SAMPLE_SIZE = int(os.getenv("DEV_SAMPLE_SIZE", "5"))
EVAL_SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", "10"))
MAX_LINKED_PASSAGES = int(os.getenv("MAX_LINKED_PASSAGES", "3"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
FAIR_BASELINE_MAX_CONTEXT_CHARS = int(os.getenv("FAIR_BASELINE_MAX_CONTEXT_CHARS", "16000"))
FAIR_BASELINE_MAX_ANSWER_TOKENS = int(os.getenv("FAIR_BASELINE_MAX_ANSWER_TOKENS", "128"))

# Pure GraphRAG local-search tuning used in thesis experiments.
# These stay within GraphRAG's own retrieval path; they do not introduce a RAG fallback.
GRAPHRAG_LOCAL_REPORT_COMMUNITY_LEVEL = int(os.getenv("GRAPHRAG_LOCAL_REPORT_COMMUNITY_LEVEL", "2"))
GRAPHRAG_LOCAL_COMMUNITY_PROP = float(os.getenv("GRAPHRAG_LOCAL_COMMUNITY_PROP", "0.05"))
GRAPHRAG_LOCAL_TEXT_UNIT_PROP = float(os.getenv("GRAPHRAG_LOCAL_TEXT_UNIT_PROP", "0.92"))
GRAPHRAG_LOCAL_TOP_K_MAPPED_ENTITIES = int(os.getenv("GRAPHRAG_LOCAL_TOP_K_MAPPED_ENTITIES", "8"))
GRAPHRAG_LOCAL_TOP_K_RELATIONSHIPS = int(os.getenv("GRAPHRAG_LOCAL_TOP_K_RELATIONSHIPS", "8"))
GRAPHRAG_LOCAL_MAX_CONTEXT_TOKENS = int(os.getenv("GRAPHRAG_LOCAL_MAX_CONTEXT_TOKENS", "3000"))
GRAPHRAG_LOCAL_INCLUDE_RELATIONSHIP_WEIGHT = _env_flag(
    "GRAPHRAG_LOCAL_INCLUDE_RELATIONSHIP_WEIGHT",
    "true",
)

# Directory paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "hybridqa"
HYBRIDQA_DIR = DATA_DIR / "hybridqa"
ORIGINAL_DIR = HYBRIDQA_DIR / "original"
SAMPLES_DIR = HYBRIDQA_DIR / "samples"
PARSED_DIR = HYBRIDQA_DIR / "parsed"
COMPLIANCE_DIR = DATA_DIR / "compliance"

CACHE_DIR = PROJECT_ROOT / "cache"
EMB_CACHE_DIR = CACHE_DIR / "embeddings"
INDEX_CACHE_DIR = CACHE_DIR / "indexes"
GEN_CACHE_DIR = CACHE_DIR / "generations"

RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
METRICS_DIR = RESULTS_DIR / "metrics"
OUTPUTS_DIR = RESULTS_DIR / "outputs"
LOCAL_MODELS_DIR = PROJECT_ROOT / "local_models"

# Ensure cache dirs exist
for _d in (
    EMB_CACHE_DIR,
    INDEX_CACHE_DIR,
    GEN_CACHE_DIR,
    LOGS_DIR,
    METRICS_DIR,
    OUTPUTS_DIR,
    LOCAL_MODELS_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)
