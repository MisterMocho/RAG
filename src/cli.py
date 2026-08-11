from .indexer import RepositoryIndexer


class RAGCLI:
    """Comand Line interface for the RAG system."""

    def index(self, max_chunk_size: int = 800) -> None:
        """Builds indexing from the raw repository."""
        print(f"Beggining indexing with max_chunk_size:{max_chunk_size}...")
        indexer = RepositoryIndexer()
        indexer.build_index(max_chunk_size)

    def search(self, query: str, k: int = 5) -> None:
        """Returns top-k sources in a single query."""
        print(f"Searching for: '{query}' (top {k})...")
        from .searcher import RepositorySearcher
        searcher = RepositorySearcher()
        results = searcher.search(query, k)
        if results:
            # Converts the model into a JSON
            print(results.model_dump_json(indent=2))

    def search_dataset(self,
                       dataset_path: str,
                       save_directory: str,
                       k: int = 5) -> None:
        """Runs research for a complete dataset and stores it in a JSON."""
        print(f"Processing dataset: {dataset_path}...")
        from pathlib import Path
        from .searcher import RepositorySearcher
        searcher = RepositorySearcher()
        results = searcher.search_dataset(dataset_path, k)
        if results:
            input_path = Path(dataset_path)
            # Makes sure destination folder exists
            out_dir = Path(save_directory)
            out_dir.mkdir(parents=True, exist_ok=True)
            # Final name file
            out_file = out_dir / input_path.name
            # Stores Json on the drive
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(results.model_dump_json(indent=2))
            print(f"Success! Results stored in: {out_file}")

    def answer(self, query: str, k: int = 5) -> None:
        """Generates an answer to a single query."""
        print(f"Generating answer to: '{query}'...")
        from .searcher import RepositorySearcher
        from .answerer import RepositoryAnswerer
        # Researches sources
        searcher = RepositorySearcher()
        search_results = searcher.search(query, k)
        if search_results and search_results.search_results:
            # 2. Passes the result to the LLM
            answerer = RepositoryAnswerer()
            final_answer = answerer.answer(search_results.search_results[0])
            print("\n--- Answer ---")
            print(final_answer.answer)
            print("--------------\n")
        else:
            print("No results found to answer the query.")

    def answer_dataset(
            self, student_search_results_path: str,
            save_directory: str) -> None:
        """Generates answers for all queries in a researched dataset."""
        from pathlib import Path
        from .answerer import RepositoryAnswerer
        from .models import (
            StudentSearchResults,
            StudentSearchResultsAndAnswer,
            MinimalAnswer,
        )
        print(f"Generating answers for dataset: {student_search_results_path}")
        input_path = Path(student_search_results_path)
        if not input_path.exists():
            print(f"Error: Could not find {student_search_results_path}")
            return
        # Loads research results
        with open(input_path, "r", encoding="utf-8") as f:
            search_data = StudentSearchResults.model_validate_json(f.read())
        # Initializes the model once for the whole lot
        answerer = RepositoryAnswerer()
        answered_results: list[MinimalAnswer] = []
        total = len(search_data.search_results)
        print(f"Loaded {total} questions. Starting generation...")
        # Generates the answer for each question
        for i, search_result in enumerate(search_data.search_results, 1):
            print(f"Processing question {i}/{total}...")
            answer = answerer.answer(search_result)
            answered_results.append(answer)
        # Builds final model
        final_output = StudentSearchResultsAndAnswer(
            search_results=answered_results,
            k=search_data.k
        )
        # Stores it on the drive
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / input_path.name
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(final_output.model_dump_json(indent=2))
        print(f"Success! Answers saved to {out_file}")

    def evaluate(
            self,
            student_search_results_path: str,
            dataset_path: str,
            k: int = 10,
            max_context_length: int = 800) -> None:
        """Evaluates research quality locally (recall@k)"""
        from .evaluator import SystemEvaluator
        print("Calculating evaluation metrics...")
        evaluator = SystemEvaluator()
        evaluator.evaluate(student_search_results_path, dataset_path, k)
