"""Answer generation with citations.

Every passage handed to the model is numbered and labelled with its source and
page. The model is required to cite those numbers inline, and the same passages
are returned to the caller — so the answer the user reads and the evidence they
can inspect are always the same set.
"""

import os
from typing import List, Optional

from .config import (
    DEFAULT_STRATEGY,
    DISTANCE_THRESHOLD,
    GEMINI_MODEL,
    REFUSAL_MESSAGE,
    TOP_K,
)
from .retriever import Hit, retrieve

PROMPT_TEMPLATE = """You answer questions about a set of company documents.

Rules:
1. Use ONLY the passages below. Never use outside knowledge.
2. Cite the passage number in square brackets after each claim, like [1] or [2][3].
3. If the passages do not contain the answer, reply with exactly:
   {refusal}
4. Do not guess, and do not soften a missing answer into a plausible one.

Passages:
{context}

Question: {question}

Answer:"""


def format_context(hits: List[Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        header = f"[{i}] {hit.source} — page {hit.pages}"
        if hit.section:
            header += f" — section: {hit.section}"
        blocks.append(f"{header}\n{hit.document.page_content.strip()}")
    return "\n\n---\n\n".join(blocks)


def _extract_text(content) -> str:
    """Newer Gemini models return content as a list of blocks, not a plain
    string — e.g. [{"type": "text", "text": "..."}]. Normalise both shapes."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return "" if content is None else str(content)


def _get_llm():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not set. Copy .env.example to .env.")
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)


def answer(
    question: str,
    strategy: str = DEFAULT_STRATEGY,
    k: int = TOP_K,
    threshold: Optional[float] = None,
) -> dict:
    """Return {answer, sources, refused, strategy, threshold, considered}."""
    limit = DISTANCE_THRESHOLD if threshold is None else threshold
    kept, all_hits = retrieve(question, strategy=strategy, k=k, threshold=limit)

    base = {
        "question": question,
        "strategy": strategy,
        "threshold": limit,
        "considered": [h.as_dict(i) for i, h in enumerate(all_hits, start=1)],
    }

    # Gate: nothing close enough, so nothing goes to the model.
    if not kept:
        return {**base, "answer": REFUSAL_MESSAGE, "sources": [], "refused": True}

    prompt = PROMPT_TEMPLATE.format(
        refusal=REFUSAL_MESSAGE,
        context=format_context(kept),
        question=question,
    )
    response = _get_llm().invoke(prompt)
    text = _extract_text(response.content).strip()

    return {
        **base,
        "answer": text,
        "sources": [h.as_dict(i) for i, h in enumerate(kept, start=1)],
        "refused": text.startswith(REFUSAL_MESSAGE[:40]),
    }


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or input("\nAsk a question: ")
    result = answer(question)

    print("\n" + "=" * 60)
    print(result["answer"])
    print("=" * 60)
    if result["sources"]:
        print("\nSources")
        for source in result["sources"]:
            label = f"  [{source['rank']}] {source['source']} p.{source['pages']}"
            if source["section"]:
                label += f" — {source['section']}"
            print(f"{label}  (distance {source['distance']})")
    else:
        best = result["considered"][0]["distance"] if result["considered"] else None
        print(f"\nNo passage passed the threshold {result['threshold']} "
              f"(closest was {best}).")