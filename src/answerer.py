# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
import torch
from pathlib import Path
from typing import List, Any
# pyright: ignore[reportMissingImports]
from transformers import AutoModelForCausalLM, AutoTokenizer
from .models import MinimalSearchResults, MinimalSource, MinimalAnswer


class RepositoryAnswerer:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        print(f"Loading LLM model: {model_name}...")
        # Detects if there is a GPU, apple silicon or CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto" if self.device != "mps" else None
        )
        if self.device == "mps":
            self.model.to("mps")
        print("Model loaded successfully!")

    def _extract_context(self, sources: List[MinimalSource]) -> str:
        """Reads the physical text from the retrieved sources."""
        context_blocks: List[str] = []

        for source in sources:
            file_path = Path(source.file_path)
            if not file_path.exists():
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Corta o texto exato
                snippet = content[source.first_character_index:
                                  source.last_character_index]
                context_blocks.append(f"--- From {file_path.name} ---\n"
                                      f"{snippet}\n")
            except Exception as e:
                print(f"Warning: Could not read {file_path} - {e}")

        return "\n".join(context_blocks)

    def answer(self, search_result: MinimalSearchResults) -> MinimalAnswer:
        """Generates an answer for a single query using context."""
        context_text = self._extract_context(search_result.retrieved_sources)

        # Builds the prompt
        prompt = (
            "You are a helpful coding assistant."
            "Answer the question based ONLY on the provided context.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {search_result.question}\n\n"
            "Answer:"
        )

        # Converts text to tokens
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generates the answer with a max of 200 tokens
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decodes tokens back to text
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens,
                                               skip_special_tokens=True)

        return MinimalAnswer(
            question_id=search_result.question_id,
            question=search_result.question,
            retrieved_sources=search_result.retrieved_sources,
            answer=generated_text.strip()
        )
