# Document Q&A with Citations

A Retrieval-Augmented Generation (RAG) system that answers questions about a PDF corpus, **shows the exact passages each answer came from**, and **says "I don't know" when the documents don't cover the question**.

The bundled corpus is the *YEAB Technologies Employee Handbook* (51 pages, `Data/`). Any PDFs dropped into `Data/` work.

## What it does

| Requirement | How it is met |
|---|---|
| Two chunking strategies, compared with evidence | `recursive` (fixed windows) and `section` (heading-aware) are both built, and `eval/run_eval.py` compares them on 32 labelled questions. See [Which chunking strategy won](#which-chunking-strategy-won). |
| Every answer shows the passages it used | The model must cite `[1]`, `[2]`... and the API/UI return those same numbered passages with file, page, section and distance. Clicking a citation in the UI jumps to its passage. |
| Says "I don't know" when nothing is relevant | Two layers. A **distance gate** blocks the LLM entirely when no chunk is close enough. If related-but-irrelevant chunks get through, the **prompt** forces an exact refusal message. `--e2e` tests both. |
| Corpus of at least 50 pages | 51 pages. `build_indexes.py` prints the page count and warns below 50. |

## How it works

```
                     INDEXING (once)
Data/*.pdf -> chunking.py -> vectorstore.py -> chroma_db/{recursive,section}
              - strip running headers ("Page 12")
              - skip table-of-contents pages
              - chunk two ways

                     QUERYING (each question)
question -> retriever.py -> any chunk within threshold? --no--> refuse (no LLM call)
            (top-k, cosine)            | yes
                                       v
                        qa.py -> Gemini (cite [n], refuse if not covered)
                                       ^
                           api.py  <-  frontend/index.html
```

## Project structure

```
.
├── Data/                     # source PDFs
├── src/
│   ├── config.py             # paths, models, top-k, distance threshold
│   ├── chunking.py           # PDF cleaning + both chunking strategies
│   ├── vectorstore.py        # build / load the Chroma indexes
│   ├── retriever.py          # top-k search + relevance gate
│   ├── qa.py                 # prompt, Gemini call, citations
│   └── api.py                # FastAPI: /ask, /health, serves the UI
├── scripts/build_indexes.py  # builds both indexes
├── frontend/index.html       # chat UI
├── eval/
│   ├── questions.json        # 24 answerable + 8 unanswerable, all labelled
│   ├── run_eval.py           # comparison, calibration, end-to-end test
│   ├── results.md            # generated: chunking comparison + verdict
│   ├── calibration.md        # generated: threshold sweep
│   └── e2e.md                # generated: full-pipeline test
├── .env.example
└── requirements.txt
```

## Setup

Python 3.10 or newer.

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # then put your Gemini key in .env
```

Get a free key at [Google AI Studio](https://aistudio.google.com/apikey). Only `GOOGLE_API_KEY` is required. `.env` is git-ignored.

## Run

From the project root:

```bash
python scripts/build_indexes.py     # rebuild whenever Data/ or chunking.py changes
uvicorn src.api:app --reload        # open http://127.0.0.1:8000
```

Command line alternatives:

```bash
python src/qa.py "How many days of paid time off do employees get?"
python -m src.retriever "What is the dress code?"     # retrieval only, no LLM
```

Open the UI from `http://127.0.0.1:8000`. If you serve `frontend/index.html` some other way (for example VS Code Live Server), it still finds the API on port 8000.

### API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, strategies, threshold, k |
| POST | `/ask` | `{"question": "...", "strategy": "section", "k": 5, "threshold": 0.7}` |
| GET | `/` | web UI |

`/ask` returns `answer`, `refused`, `sources` (the passages the model saw, numbered to match the `[n]` citations) and `considered` (everything retrieved, including passages the gate dropped).

## Evaluation

```bash
python eval/run_eval.py                 # compare strategies    -> eval/results.md
python eval/run_eval.py --calibrate     # choose the threshold  -> eval/calibration.md
python eval/run_eval.py --e2e           # full pipeline, Gemini -> eval/e2e.md
python eval/run_eval.py --chunks-only   # chunk integrity only, needs no embeddings
```

### The question set

`eval/questions.json` has 24 answerable and 8 unanswerable questions, every one checked against the actual PDF text.

- **gold_pages**: the pages that contain the answer (1-based, as in a PDF viewer).
- **alt_pages**: other pages that also answer it (for example the quick-reference table). They count as correct for Hit@k and MRR but not for recall.
- **answer_keys**: short phrases that must all appear for an answer to be complete.
- **kind** (unanswerable only): `off_topic` (nothing related in the corpus) or `in_domain_missing` (related topic, answer not in the handbook: dress code, gross misconduct, 2019 revenue, relocation leave).

Two answerable questions (q21, q22) are deliberately about policies that **straddle a page break**. Without them the comparison would not test whether a chunker cuts a policy in half.

### Metrics

| Metric | Meaning |
|---|---|
| **Fact@k** (headline) | Some retrieved chunk contains every answer key: the model was given the whole answer. |
| Fact@1 | Same, using only the top chunk. |
| Hit@k / MRR | A retrieved chunk lands on a correct page / how high the first one ranks. |
| Page recall | Share of gold pages covered by the top-k. |
| Intact | The answer sits whole inside one chunk of the index. Needs no embeddings. |
| Gate answered / refused | Whether the distance gate lets answerable questions through and blocks unanswerable ones. |

## Which chunking strategy won

**Strategies**

- `recursive`: cut each page into 1000-character windows with 200 overlap. The standard baseline.
- `section`: detect the handbook's own headings (`4.1 Paid Time Off`, `B.1 United States`), keep each section whole across page breaks, sub-split only sections over 1200 characters, and repeat the heading on every piece.

Both strategies get identical input: running headers and table-of-contents pages are removed for both.

**Evidence that needs no embeddings** (run `python eval/run_eval.py --chunks-only`):

| Strategy | Chunks | Mean chunk size | Answers kept whole in one chunk |
|---|---|---|---|
| recursive | 95 | 778 chars | 22 / 24 |
| section | 122 | 536 chars | **24 / 24** |

`recursive` cuts two answers in half: the postmortem rule (q21: "Every Sev1 and Sev2" ends page 16 while "...requires a blameless postmortem published within 5 business days" starts page 17) and the incident-reporting policy (q22: the one-hour reporting rule is on page 14 and the 30-minute acknowledgement target on page 15). It splits at page boundaries, so it cannot avoid this. `section` keeps each policy whole because it groups by heading, not by page or length.

**Retrieval evidence** (Fact@k, Fact@1, Hit@k, MRR, page recall, with an automatic verdict listing which questions each strategy alone answered completely) is generated into `eval/results.md` by `python eval/run_eval.py`.

**Why section chunking is expected to win, and its cost:** a handbook is organised as self-contained policies. A retrieved section is a complete policy with its heading, so the LLM receives the full answer and the citation names a real section (`4.1 Paid Time Off`), not an arbitrary window. The cost is more, smaller chunks (122 vs 95) and a dependence on the document having detectable headings; on a document without them, `section` degrades to `recursive`-like behaviour.

## Refusal behaviour

1. **Distance gate** (`DISTANCE_THRESHOLD`, cosine distance, lower = more similar). If no chunk is within the threshold, the LLM is never called and the refusal message is returned. This is a guarantee, not a hope.
2. **Prompt rule.** Related but irrelevant chunks can pass the gate ("What counts as gross misconduct?" is close to Code of Conduct text). The prompt requires the exact refusal message when the passages do not contain the answer.

A distance gate alone cannot catch `in_domain_missing` questions, which is why the eval reports them separately and `--e2e` runs the full pipeline (retrieval, gate and Gemini) to measure real refusal and citation behaviour.

**Tuning:** run `python eval/run_eval.py --calibrate`, then set `DISTANCE_THRESHOLD` in `.env`. Too high lets junk through, too low refuses valid questions.

## Design notes

- **Running headers removed.** A line repeated on at least half of a PDF's pages (ignoring digits) is treated as page furniture. Left in, "Page 13" text made sections appear to run onto the next page, which would have inflated page-based metrics for the `section` strategy.
- **Table-of-contents pages skipped.** A TOC lists every heading, so it matches every topic and answers nothing. Real content keeps its original page numbers, so citations are unaffected.
- **Local embeddings** (`all-MiniLM-L6-v2`): no embedding API key or cost.
- **One index per strategy** in separate Chroma collections, both forced to cosine distance so scores are comparable.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No index for 'section'` | Run `python scripts/build_indexes.py`. |
| Results look stale after editing `chunking.py` or `Data/` | Rebuild the indexes. |
| `GOOGLE_API_KEY is not set` | Add it to `.env` in the project root and restart uvicorn. |
| Gemini "model not found" | Change `GEMINI_MODEL` in `.env`. |
| `ModuleNotFoundError: src` | Run commands from the project root. |
| UI shows 404/405 errors | Open `http://127.0.0.1:8000`, and make sure uvicorn is running. |
| `--e2e` hits rate limits | Raise `--delay` (default 6.5 s between Gemini calls). |

## Tech stack

Python, LangChain, ChromaDB, Sentence-Transformers, Google Gemini, PyPDF, FastAPI, Uvicorn.