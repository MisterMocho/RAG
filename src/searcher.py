# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
import pickle
import uuid
from pathlib import Path
from typing import Any, List
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
        # Variables that store the model and memory pieces on the memory
        self.retriever: Any = None
        self.chunks: List[Any] = []

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

    def _retrieve_sources(self, query: str, k: int) -> List[MinimalSource]:
        """Encapsulates BM25 MainLogic and parses sources"""
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
        return sources

    def search(self, query: str, k: int = 5) -> StudentSearchResults | None:
        """Executes research and formats data according to pydantic models"""
        if not self.retriever and not self.load_index():
            return None
        sources = self._retrieve_sources(query, k)
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
        if not self.retriever and not self.load_index():
            return None
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
        return StudentSearchResults(search_results=all_results, k=k)
