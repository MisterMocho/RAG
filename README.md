```markdown
*This project has been created as part of the 42 curriculum by luida-cu.*

# RAG against the machine: Will you answer my questions?

## Description
This project implements a complete Retrieval-Augmented Generation (RAG) pipeline designed to answer questions based on a given codebase (specifically the vLLM repository). It bridges the gap between static LLM knowledge and dynamic, external documentation by indexing raw files, retrieving the most relevant context via semantic search, and generating accurate, source-grounded answers using local LLM inference.

## System Architecture
The system is divided into modular, independently testable components:
* **Indexer (`indexer.py`):** Parses raw `.py` and `.md` files, applies smart chunking, and builds a BM25 matrix.
* **Searcher (`searcher.py`):** Tokenizes user queries and retrieves top-k relevant snippets. Implements **Lazy Loading** and **Result Caching** (RAM + Disk) for extreme performance.
* **Answerer (`answerer.py`):** Loads the local LLM on the most optimal hardware (CUDA/CPU), constructs context-aware prompts, and generates clean responses stripped of internal thinking tags.
* **Evaluator (`evaluator.py`):** Calculates Recall@k metrics by assessing the exact character overlap (min. 5%) between retrieved sources and ground truth data.
* **CLI (`cli.py` & `__main__.py`):** A robust command-line interface built with Python Fire to orchestrate the entire pipeline.

## Chunking Strategy
Document segmentation is handled by the `langchain_text_splitters` library. 
* **Python & Markdown Files:** Processed using `RecursiveCharacterTextSplitter` tailored to the specific language format.
* **Parameters:** Configured with a default chunk size of **2000 characters** and an overlap of 200 characters to prevent loss of context across chunk boundaries, fully respecting the maximum hard limit.

## Retrieval Method
The retrieval system relies on **BM25**, a highly optimized probabilistic information retrieval model. 
1. The query is tokenized and scored against the pre-computed BM25 index.
2. The top-K document indices are extracted.
3. Metadata (file path, start/end character indices) is fetched from a serialized `.pkl` chunk list.

**Bonus:** A sophisticated caching layer was implemented. It generates an MD5 hash of the query and parameters, storing results in disk and memory. Subsequent identical queries bypass the BM25 computation entirely.

## Performance Analysis
The system successfully meets and exceeds the baseline requirements established by the subject (validated via Moulinette):
* **Recall@5 (Code):** 54% (Subject requirement: >=50%)
* **Recall@5 (Docs):** 80% (Subject requirement: >=80%)
* **Throughput:** A dataset of 100 queries is searched in approximately ~1 second on cold start. With the implemented caching bonus, subsequent warm searches execute in <0.9 seconds (the physical limit of Python/CLI startup overhead).

## Design Decisions
* **Pydantic Validation:** All data flows through strict Pydantic models (`models.py`) to ensure data integrity and schema compliance (enforcing `question_str` mapping).
* **Lazy Loading:** The heavy BM25 index is only loaded into memory if a cache miss occurs, dramatically saving RAM and reducing latency.
* **Regex Cleaning:** Given the model's tendency to output its internal chain of thought, regex was utilized to parse and strip `<think>` tags, guaranteeing pristine JSON outputs.

## Challenges Faced
* **Semantic Chunking (AST) vs. Evaluation Truth:** An initial attempt at using Python's AST for intelligent semantic chunking resulted in a massive drop in Recall. The ground-truth dataset was built expecting sequential text chunks, meaning the removal of global variables and imports by the AST blinded the retriever. We reverted to character-based chunking with overlaps.
* **Query Expansion vs. Query Drift:** Implementing Pseudo-Relevance Feedback (PRF) for dynamic query expansion introduced query drift, where common code terms retrieved in the first pass skewed the final search. Result caching proved to be a much safer and highly effective optimization.

## Instructions

**1. Installation**
```bash
make install

```

*(To completely reinstall and clear environment, run `make re`)*

**2. Clean & Maintenance**

```bash
make lint         # Runs flake8 and mypy type checking
make clean        # Removes python cache files
make clear-cache  # Deletes search cache
make fclean       # Deep cleans everything including BM25 indexes and generated JSONs

```

## Example Usage

**1. Index the database**

```bash
make index
```

**2. Search both datasets (with Lazy Loading & Caching bonus)**

```bash
make search-all
```

**3. Generate LLM answers for datasets**

```bash
make answer-all
```

**4. Run full evaluation metrics (Recall@5)**

```bash
make evaluate-all
```

**5. Answer a single query with LLM**

```bash
uv run python -m src answer "How to configure OpenAI server?" --k 5
```

## Resources & AI Usage

* **Transformers & HuggingFace:** Core documentation for loading and managing the local model.
* **BM25s & Langchain:** Reference materials for text splitting and probabilistic retrieval.
* **AI Usage:** Generative AI was used critically and responsibly as a sounding board during development. It assisted in generating the regex patterns to clean the LLM output, brainstorming potential chunking strategies (such as AST parsing, which was later discarded), mapping variables to bypass evaluation edge-cases, and optimizing the caching mechanism for the CLI. All AI-generated logic was rigorously tested, reviewed, and deeply understood before integration.