"""Central configuration. Every module imports paths and models from here."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
INDEX_ROOT = ROOT / "chroma_db"
EVAL_DIR = ROOT / "eval"

# Embeddings run locally, no API key needed.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Verify this string against the current Gemini model list before demoing.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# The two chunking strategies under comparison.
STRATEGIES = ("recursive", "section")
DEFAULT_STRATEGY = os.getenv("DEFAULT_STRATEGY", "section")

TOP_K = int(os.getenv("TOP_K", "5"))

# Cosine distance: 0.0 = identical, 2.0 = opposite. Lower is better.
# Anything above this is treated as "not relevant" and triggers a refusal.
# Calibrate with: python eval/run_eval.py --calibrate
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.75"))

REFUSAL_MESSAGE = (
    "I couldn't find that in the indexed documents. Try rephrasing, "
    "or it may simply not be covered."
)


@lru_cache(maxsize=1)
def get_embeddings():
    """One embedding model per process — loading it is slow, so cache it."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def index_path(strategy: str) -> str:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}. Choose from {STRATEGIES}.")
    return str(INDEX_ROOT / strategy)