import os
import pickle
import bm25s  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]
from pathlib import Path
from typing import Any, List
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


class RepositoryIndexer:
    def __init__(self,
                 raw_data_path: str = "data/raw/vllm-0.10.1",
                 processed_data_path: str = "data/processed") -> None:
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        # Makes sure output folder exists
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

    def _get_splitter(self, file_extension: str, max_chunk_size: int) -> Any:
        """Returns the correct splitter based on file extension."""
        if file_extension == '.py':
            return (RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=max_chunk_size,
                chunk_overlap=200,  # Small overlap to not lose context
                add_start_index=True  # Stores first char index
            ))
        else:
            # Fallsback to MD and Normal Text
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=max_chunk_size,
                chunk_overlap=200,
                add_start_index=True
            )

    def process_files(self, max_chunk_size: int = 2000) -> List[Any]:
        """Reads the files, applies chunking and readies index."""
        all_chunks: List[Any] = []

        # Runs all the files in the folder Raw
        for root, _, files in os.walk(self.raw_data_path):
            for file in files:
                if not file.endswith(('.py', '.md')):
                    continue  # Ignores files that are not code or documents

                file_path = Path(root) / file

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    splitter: Any = self._get_splitter(file_path.suffix,
                                                       max_chunk_size)
                    chunks: List[Any] = splitter.create_documents(
                        texts=[content],
                        metadatas=[{"file_path": str(file_path)}]
                    )
                    all_chunks.extend(chunks)

                except Exception as e:
                    print(f"Erro ao ler o ficheiro {file_path}: {e}")

        return all_chunks

    def build_index(self, max_chunk_size: int = 2000) -> None:
        """Main Function that processes and generates BM25 index."""
        print("Reading and splitting files...")
        chunks: List[Any] = self.process_files(max_chunk_size)
        if not chunks:
            print(
                "Warning: No chunk has been generated.\n"
                "Please check if the folder data/raw has the correct files"
                )
        print(f"There was {len(chunks)} chunks generated.")
        print("Building BM25 index...")
        # Extracting text of each chunk to the BM25 Analyzer
        corpus: List[str] = [chunk.page_content for chunk in chunks]
        # Tokenizing text
        corpus_tokens: Any = bm25s.tokenize(corpus)  # pyright: ignore
        # Creating and training BM25 model
        retriever: Any = bm25s.BM25()
        retriever.index(corpus_tokens)
        # Stores index and chunks on the Drive
        print("Storing indexes...")
        retriever.save(str(self.processed_data_path / "bm25_index"))
        with open(self.processed_data_path / "chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
        print("Indexing concluded! Files stored in data/processed/")
