# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
import pickle
import uuid
import hashlib
import json
from pathlib import Path
from typing import Any, List, Dict
import bm25s  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]
from .models import (
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
    RagDataset
)


class RepositorySearcher:
    def __init__(self, processed_data_path: str = "data/processed") -> None:
        self.processed_data_path: Path = Path(processed_data_path)
        self.index_dir: Path = self.processed_data_path / "bm25_index"
        self.chunks_file: Path = (self.processed_data_path / "chunks"
                                  / "chunks.pkl")
        self.cache_dir: Path = self.processed_data_path / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Variables that store the model and memory pieces on the memory
        self.retriever: Any = None
        self.chunks: List[Any] = []

        # BONUS: In-memory RAM Cache for instantaneous retrieval
        self._memory_cache: Dict[str, List[MinimalSource]] = {}

    def load_index(self) -> bool:
        """Loads BM25 index and chunks"""
        if not self.index_dir.exists() or not self.chunks_file.exists():
            print("Error: index files not found")
            return False
        try:
            # Loads BM25 Matrix
            self.retriever = bm25s.BM25.load(str(self.index_dir),
                                             load_corpus=False)
            # Loads original chunk list
            with open(self.chunks_file, "rb") as f:
                self.chunks = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False

    def _get_cache_key(self, query: str, k: int) -> str:
        """Generates a unique MD5 hash for query + k."""
        raw_key = f"{query.strip().lower()}_k={k}"
        return hashlib.md5(raw_key.encode("utf-8")).hexdigest()

    def _get_from_cache(self, query: str,
                        k: int) -> List[MinimalSource] | None:
        """Checks both Memory and Disk Cache for existing results."""
        cache_key = self._get_cache_key(query, k)

        # 1. Check RAM Memory Cache
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        # 2. Check Disk Cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sources = [MinimalSource(**src) for src in data]
                    # Update Memory Cache
                    self._memory_cache[cache_key] = sources
                    return sources
            except Exception:
                return None
        return None

    def _save_to_cache(self, query: str, k: int,
                       sources: List[MinimalSource]) -> None:
        """Saves search results to both Memory and Disk Cache."""
        cache_key = self._get_cache_key(query, k)

        # Save to RAM
        self._memory_cache[cache_key] = sources

        # Save to Disk
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            dict_sources = [src.model_dump() for src in sources]
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(dict_sources, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save to cache - {e}")

    def _retrieve_sources(self, query: str, k: int) -> List[MinimalSource]:
        """Encapsulates BM25 MainLogic with Caching mechanism."""
        # 1. Check Cache first! (No heavy loading yet)
        cached_sources = self._get_from_cache(query, k)
        if cached_sources is not None:
            return cached_sources

        # 2. Lazy Loading: Load index ONLY if we have a Cache Miss
        if not self.retriever and not self.load_index():
            print("Error: Could not load index to perform search.")
            return []

        # 3. Query BM25
        query_tokens: Any = bm25s.tokenize(query)
        results, _ = self.retriever.retrieve(query_tokens, k=k)
        top_indices: Any = results[0]
        sources: List[MinimalSource] = []
        for idx in top_indices:
            chunk: Any = self.chunks[idx]
            file_path: str = chunk.metadata.get("file_path", "")
            start_index: int = chunk.metadata.get("start_index", 0)
            last_index: int = start_index + len(chunk.page_content)
            sources.append(MinimalSource(
                file_path=file_path,
                first_character_index=start_index,
                last_character_index=last_index
            ))
        # 4. Store in Cache for future calls
        self._save_to_cache(query, k, sources)
        return sources

    def search(self, query: str, k: int = 5) -> StudentSearchResults | None:
        """Executes research and formats data according to pydantic models"""
        sources = self._retrieve_sources(query, k)
        if not sources and not self.retriever:
            return None  # Fails gracefully if index loading failed
        search_result = MinimalSearchResults(
            question_id=str(uuid.uuid4()),
            question=query,
            retrieved_sources=sources
        )
        return StudentSearchResults(search_results=[search_result], k=k)

    def search_dataset(self,
                       dataset_path: str,
                       k: int = 5) -> StudentSearchResults | None:
        """Runs a json with the questions and returns results."""
        path = Path(dataset_path)
        if not path.exists():
            print(f"Error: file {dataset_path} couldn't be found.")
            return None
        with open(path, "r", encoding="utf-8") as f:
            dataset_json = f.read()
        dataset = RagDataset.model_validate_json(dataset_json)
        all_results: List[MinimalSearchResults] = []
        for q in dataset.rag_questions:
            sources = self._retrieve_sources(q.question, k)
            all_results.append(MinimalSearchResults(
                question_id=q.question_id,
                question=q.question,
                retrieved_sources=sources
            ))
        # Returns None only if it failed to load index and got 0 sources
        if not all_results and not self.retriever:
            return None
        return StudentSearchResults(search_results=all_results, k=k)
