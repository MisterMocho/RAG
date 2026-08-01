import fire  # pyright: ignore[reportMissingTypeStubs]
from .indexer import RepositoryIndexer


class RAGCLI:
    """Comand Line interface for the RAG system."""

    def index(self, max_chunk_size: int = 2000):
        """Builds indexing from the raw repository."""
        print(f"Beggining indexing with max_chunk_size:{max_chunk_size}...")
        indexer = RepositoryIndexer()
        indexer.build_index(max_chunk_size)

    def search(self, query: str, k: int):
        """Returns top-k sources in a single query."""
        print(f"Searching for: '{query}' (top {k})...")

    def search_dataset(self, dataset_path: str, k: int, save_directory: str):
        """Runs research for a complete dataset and stores it in a JSON."""
        print(f"Processing dataset: {dataset_path}...")

    def answer(self, query: str, k: int):
        """Generates an answer to a single query."""
        print(f"Generating answer to: '{query}'...")

    def answer_dataset(
            self, student_search_results_path: str, save_directory: str):
        """Generates answers for all queries in a researched dataset."""
        print(f"Generating answers for dataset: {student_search_results_path}")

    def evaluate(
            self,
            student_search_results_path: str,
            dataset_path: str,
            k: int = 10,
            max_context_length: int = 2000):
        """Evaluates research quality locally (recall@k)"""
        print("Calculating evaluation metrics...")


def main():
    fire.Fire(RAGCLI)  # pyright: ignore[reportUnknownMemberType]
