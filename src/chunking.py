"""Loading and chunking.

Two strategies are implemented so they can be compared on identical questions:

  recursive  Fixed-size character windows with overlap. The standard baseline.
             Fast and simple, but it happily cuts a policy in half.

  section    Detects the document's own headings, keeps each section whole,
             and only sub-splits sections that are too long to embed well.
             Hypothesis: a retrieved chunk that is a complete policy answers
             the question better than an arbitrary 1000-character window.

Both produce langchain Documents with the same metadata contract:
    source, page_start, page_end, pages, section, strategy, chunk_id
"""

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import DATA_DIR

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _furniture_key(line: str) -> str:
    """Normalise a line so 'Page 12' and 'Page 13' count as the same line."""
    return re.sub(r"\d+", "#", line.strip().lower())


def strip_page_furniture(pages: List[Document], min_share: float = 0.5) -> List[Document]:
    """Remove running headers/footers (e.g. 'Company - Handbook', 'Page 12').

    A line that appears on at least half of a document's pages (ignoring digits)
    is page furniture, not content. Left in, it pollutes every chunk and, worse,
    makes a section look like it runs onto the next page.
    """
    by_source: Dict[str, List[Document]] = defaultdict(list)
    for page in pages:
        by_source[page.metadata.get("source", "unknown")].append(page)

    for docs in by_source.values():
        if len(docs) < 4:
            continue
        counts: Counter = Counter()
        for doc in docs:
            counts.update({_furniture_key(l) for l in doc.page_content.splitlines() if l.strip()})
        furniture = {key for key, n in counts.items() if n >= min_share * len(docs)}
        for doc in docs:
            doc.page_content = "\n".join(
                l for l in doc.page_content.splitlines()
                if not l.strip() or _furniture_key(l) not in furniture
            )
    return pages


# A table-of-contents line: a section number (or appendix label) then a short title.
_TOC_LINE = re.compile(r"^\s*(\d+(\.\d+)*[.)]?|[A-Z]\.\d+|Appendix\s+[A-Z]\b:?)\s+\S")


def is_toc_page(text: str) -> bool:
    """True for table-of-contents pages. They match every question about a topic
    (they list every heading) but answer none, so they only add retrieval noise."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    if re.match(r"^(table of )?contents$", lines[0], re.I):
        return True
    if len(lines) < 6:
        return False
    toc = sum(
        1 for l in lines
        if _TOC_LINE.match(l) and len(l) <= 90 and not l.endswith((".", ",", ";"))
    )
    return toc / len(lines) >= 0.75


def count_pages(data_dir: Path = DATA_DIR) -> int:
    """Total PDF pages in the corpus (what the '50 pages' requirement measures)."""
    from pypdf import PdfReader

    return sum(len(PdfReader(str(p)).pages) for p in sorted(Path(data_dir).glob("*.pdf")))


def load_documents(data_dir: Path = DATA_DIR) -> List[Document]:
    """Load every PDF in data_dir, one Document per page, ready to chunk.

    - Page numbers are 1-based, matching a PDF viewer (and the eval's gold pages).
    - Running headers/footers are stripped.
    - Table-of-contents pages are skipped. They keep their page numbers, so
      citations for real content are unaffected.
    """
    pdfs = sorted(Path(data_dir).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {data_dir}. Add your corpus there.")

    pages: List[Document] = []
    for pdf in pdfs:
        for doc in PyPDFLoader(str(pdf)).load():
            doc.metadata["source"] = pdf.name
            doc.metadata["page"] = int(doc.metadata.get("page", 0)) + 1
            pages.append(doc)

    total = len(pages)
    strip_page_furniture(pages)
    content = [p for p in pages if not is_toc_page(p.page_content)]

    print(f"Loaded {total} pages from {len(pdfs)} PDF(s): "
          f"{', '.join(p.name for p in pdfs)} "
          f"({total - len(content)} table-of-contents page(s) not indexed)")
    return content


# --------------------------------------------------------------------------
# Strategy A — fixed-size recursive
# --------------------------------------------------------------------------


def chunk_recursive(
    pages: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    for i, chunk in enumerate(chunks):
        page = int(chunk.metadata.get("page", 1))
        chunk.metadata = {
            "source": chunk.metadata.get("source", "unknown"),
            "page_start": page,
            "page_end": page,
            "pages": str(page),
            "section": "",
            "strategy": "recursive",
            "chunk_id": f"recursive-{i}",
        }
    return chunks


# --------------------------------------------------------------------------
# Strategy B — section aware
# --------------------------------------------------------------------------

# A line is treated as a heading if it matches any of these and is short.
_HEADING_PATTERNS = [
    re.compile(r"^\s*\d+(\.\d+)*[.)]?\s+\S.{2,70}$"),          # 3.2 Paid Time Off
    re.compile(r"^\s*[A-Z]\.\d+\s+\S.{2,70}$"),                 # B.1 United States
    re.compile(r"^\s*(SECTION|ARTICLE|CHAPTER|APPENDIX|PART)\b.{0,60}$", re.I),
    re.compile(r"^[A-Z][A-Z0-9 &/,'\-()]{4,60}$"),              # PAID TIME OFF
]
_TITLE_CASE = re.compile(r"^([A-Z][\w'\-]*)(\s+(?:[A-Z][\w'\-]*|of|and|the|to|for|in))*$")


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not (3 <= len(stripped) <= 80):
        return False
    if stripped.endswith((".", ",", ";", ":")) and not stripped.endswith("..."):
        # Trailing punctuation usually means a sentence, not a heading.
        if not stripped.endswith(":"):
            return False
    if any(p.match(stripped) for p in _HEADING_PATTERNS):
        return True
    # Short title-case line with few words, no terminal period.
    words = stripped.rstrip(":").split()
    if 1 < len(words) <= 8 and _TITLE_CASE.match(stripped.rstrip(":")):
        return True
    return False


def _pages_to_lines(pages: List[Document]):
    """Flatten pages into (line, page_number) pairs, grouped per source file."""
    per_source: Dict[str, List[tuple]] = {}
    for page in pages:
        source = page.metadata.get("source", "unknown")
        page_no = int(page.metadata.get("page", 1))
        for line in page.page_content.splitlines():
            per_source.setdefault(source, []).append((line, page_no))
    return per_source


def chunk_by_section(
    pages: List[Document],
    max_chars: int = 1200,
    overlap: int = 150,
    min_chars: int = 80,
) -> List[Document]:
    """Group lines under their heading; sub-split sections over max_chars."""
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Document] = []
    counter = 0

    for source, lines in _pages_to_lines(pages).items():
        sections: List[dict] = []
        current = {"title": "(front matter)", "lines": [], "start": lines[0][1] if lines else 1, "end": 1}

        for line, page_no in lines:
            if _is_heading(line) and current["lines"]:
                sections.append(current)
                current = {"title": line.strip().rstrip(":"), "lines": [], "start": page_no, "end": page_no}
            elif _is_heading(line):
                current["title"] = line.strip().rstrip(":")
                current["start"] = page_no
            else:
                current["lines"].append(line)
                current["end"] = page_no
        if current["lines"]:
            sections.append(current)

        # Merge sections that are too small to stand alone (stray headings,
        # page furniture) into the previous one.
        merged: List[dict] = []
        for section in sections:
            body = "\n".join(section["lines"]).strip()
            if merged and len(body) < min_chars:
                merged[-1]["lines"].extend([section["title"]] + section["lines"])
                merged[-1]["end"] = section["end"]
            else:
                merged.append(section)

        for section in merged:
            body = "\n".join(section["lines"]).strip()
            if not body:
                continue
            title = section["title"]
            # Split the body only, so the heading can never end up as a chunk of
            # its own; then repeat the heading on every piece to keep context.
            if len(title) + 2 + len(body) > max_chars:
                pieces = sub_splitter.split_text(body)
            else:
                pieces = [body]
            for piece in pieces:
                piece = f"{title}\n\n{piece}" if title else piece
                chunks.append(
                    Document(
                        page_content=piece,
                        metadata={
                            "source": source,
                            "page_start": int(section["start"]),
                            "page_end": int(section["end"]),
                            "pages": (
                                str(section["start"])
                                if section["start"] == section["end"]
                                else f"{section['start']}-{section['end']}"
                            ),
                            "section": section["title"][:120],
                            "strategy": "section",
                            "chunk_id": f"section-{counter}",
                        },
                    )
                )
                counter += 1

    return chunks


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

CHUNKERS: Dict[str, Callable[[List[Document]], List[Document]]] = {
    "recursive": chunk_recursive,
    "section": chunk_by_section,
}


def chunk(strategy: str, pages: List[Document]) -> List[Document]:
    if strategy not in CHUNKERS:
        raise ValueError(f"Unknown strategy {strategy!r}. Choose from {list(CHUNKERS)}.")
    return CHUNKERS[strategy](pages)


def describe(chunks: List[Document]) -> str:
    lengths = [len(c.page_content) for c in chunks] or [0]
    return (
        f"{len(chunks)} chunks | "
        f"chars min {min(lengths)} / mean {sum(lengths) // len(lengths)} / max {max(lengths)}"
    )


if __name__ == "__main__":
    pages = load_documents()
    for name in CHUNKERS:
        print(f"{name:>10}: {describe(chunk(name, pages))}")