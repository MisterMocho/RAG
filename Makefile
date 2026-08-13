export UV_CACHE_DIR := $(PWD)/.uv_cache
NAME = RAG

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

# Builds BM25 index
index:
	uv run python -m src index --max_chunk_size 2000

# Researches both datasets code and docs using and BM25 + Cache
search-all:
	uv run python -m src search_dataset data/datasets/UnansweredQuestions/dataset_code_public.json data/output/search_results --k 5
	uv run python -m src search_dataset data/datasets/UnansweredQuestions/dataset_docs_public.json data/output/search_results --k 5

# Makes LLM generate answers for both datasets
answer-all:
	uv run python -m src answer_dataset data/output/search_results/dataset_code_public.json data/output/search_results_and_answer
	uv run python -m src answer_dataset data/output/search_results/dataset_docs_public.json data/output/search_results_and_answer

# Evaluates Recall@5 metrics for both datasets
evaluate-all:
	uv run python -m src evaluate --student_search_results_path data/output/search_results/dataset_code_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json --k 5
	uv run python -m src evaluate --student_search_results_path data/output/search_results/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json --k 5

clean:
	rm -rf __pycache__ .mypy_cache src/__pycache__

clear-cache:
	rm -rf data/processed/cache

fclean: clean clear-cache
	rm -rf data/processed/bm25_index
	rm -rf data/processed/chunks
	rm -rf data/output/search_results/*
	rm -rf data/output/search_results_and_answer/*

re: fclean install

lint:
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean clear-cache fclean re lint index search-all answer-all evaluate-all
