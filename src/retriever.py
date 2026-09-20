"""Retrieval plus the relevance gate.

The gate is the important part. "Say I don't know" is not a prompting problem —
a prompt is a suggestion the model can ignore. Here nothing reaches the LLM
unless at least one chunk is closer than DISTANCE_THRESHOLD, so a refusal is a
guarantee rather than a hope.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from .config import DEFAULT_STRATEGY, DISTANCE_THRESHOLD, TOP_K
from .vectorstore import get_vectorstore


@dataclass
class Hit:
    document: Document
    distance: float

    @property
    def source(self) -> str:
        return self.document.metadata.get("source", "unknown")

    @property
    def pages(self) -> str:
        return str(self.document.metadata.get("pages", "?"))

    @property
    def section(self) -> str:
        return self.document.metadata.get("section", "") or ""

    def covers(self, pages: List[int]) -> bool:
        """True if this chunk overlaps any of the given 1-based page numbers."""
        start = int(self.document.metadata.get("page_start", 0))
        end = int(self.document.metadata.get("page_end", start))
        return any(start <= p <= end for p in pages)

    def as_dict(self, rank: int) -> dict:
        return {
            "rank": rank,
            "source": self.source,
            "pages": self.pages,
            "section": self.section,
            "distance": round(self.distance, 4),
            "text": self.document.page_content,
        }


def search(
    question: str,
    strategy: str = DEFAULT_STRATEGY,
    k: int = TOP_K,
) -> List[Hit]:
    """Raw top-k, ungated. Lower distance = more similar."""
    store = get_vectorstore(strategy)
    pairs: List[Tuple[Document, float]] = store.similarity_search_with_score(question, k=k)
    return [Hit(document=d, distance=float(s)) for d, s in pairs]


def retrieve(
    question: str,
    strategy: str = DEFAULT_STRATEGY,
    k: int = TOP_K,
    threshold: Optional[float] = None,
) -> Tuple[List[Hit], List[Hit]]:
    """Return (kept, all). An empty `kept` means the system must refuse."""
    limit = DISTANCE_THRESHOLD if threshold is None else threshold
    hits = search(question, strategy=strategy, k=k)
    kept = [h for h in hits if h.distance <= limit]
    return kept, hits


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "How many days of PTO do employees receive?"
    kept, hits = retrieve(q)
    print(f"\nQuestion: {q}\nKept {len(kept)} of {len(hits)} (threshold {DISTANCE_THRESHOLD})\n")
    for i, hit in enumerate(hits, start=1):
        flag = "KEPT" if hit in kept else "drop"
        print(f"[{flag}] {i}. {hit.source} p.{hit.pages}  distance={hit.distance:.4f}")
        if hit.section:
            print(f"        section: {hit.section}")
        print(f"        {hit.document.page_content[:200].strip()}...\n")