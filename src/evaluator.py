from pathlib import Path
from typing import Dict
from .models import (
    StudentSearchResults,
    RagDataset,
    AnsweredQuestion,
    MinimalSource,
)


class SystemEvaluator:
    def _calculate_overlap_percentage(self, expected: MinimalSource,
                                      retrieved: MinimalSource) -> float:
        """
        Calculates the percentage of overlap between the expected source
        and the retrieved chunk.
        """
        # If not the same file, overlap is automatically 0
        if expected.file_path != retrieved.file_path:
            return 0.0
        # Calculates interseption limits
        overlap_start = max(expected.first_character_index,
                            retrieved.first_character_index)
        overlap_end = min(expected.last_character_index,
                          retrieved.last_character_index)
        overlap_length = max(0, overlap_end - overlap_start)
        expected_length = (expected.last_character_index -
                           expected.first_character_index)
        # Prevents division by 0
        if expected_length == 0:
            return 0.0
        return overlap_length / expected_length

    def evaluate(self, student_path: str,
                 dataset_path: str, max_k: int = 10) -> None:
        """
        Calculates Recall@k matching the 5% overlap.
        """
        student_file = Path(student_path)
        dataset_file = Path(dataset_path)
        if not student_file.exists() or not dataset_file.exists():
            print("Error: Could not find the result or dataset files.")
            return
        # Loads Json files to pydantic models
        with open(student_file, "r", encoding="utf-8") as f:
            student_data = StudentSearchResults.model_validate_json(f.read())
        with open(dataset_file, "r", encoding="utf-8") as f:
            true_data = RagDataset.model_validate_json(f.read())
        # Creates a dictionary to find the correct sources to each question
        true_dict: Dict[str, AnsweredQuestion] = {}
        for q in true_data.rag_questions:
            if isinstance(q, AnsweredQuestion) and q.sources:
                true_dict[q.question_id] = q
        # Calculates Recall para K = 1, 3, 5 e 10 at once
        k_levels = [1, 3, 5, 10]
        results_sum = {k: 0.0 for k in k_levels}
        total_evaluated = 0
        for student_res in student_data.search_results:
            q_id = student_res.question_id
            # Ignores the question if not in original dataset
            if q_id not in true_dict:
                continue
            expected_sources = true_dict[q_id].sources
            total_expected = len(expected_sources)
            total_evaluated += 1
            for k in k_levels:
                # If K > number of sources user returned cuts by its limit
                limit_k = min(k, len(student_res.retrieved_sources))
                retrieved_k = student_res.retrieved_sources[:limit_k]
                found_count = 0
                # For each source, verifies if its in foreseen top-k
                for expected in expected_sources:
                    for retrieved in retrieved_k:
                        overlap = self._calculate_overlap_percentage(expected,
                                                                     retrieved)
                        # At least 5% Overlap
                        if overlap >= 0.05:
                            found_count += 1
                            break  # if source already found moves on to next
                results_sum[k] += found_count / total_expected
        print("Evaluation Results")
        print(f"Questions evaluated: {total_evaluated}")
        for k in k_levels:
            if k <= max_k:
                avg_recall = results_sum[k] / total_evaluated \
                    if total_evaluated > 0 else 0.0
                print(f"Recall@{k}: {avg_recall:.3f}")
