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
        messages = [
            {
                "role": "system",
                "content": "You are a concise and helpful coding assistant."
                "Answer the user's question based ONLY on provided context."
                "Do not repeat yourself."
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: "
                f"{search_result.question}"
            }
        ]
        # Applies the template of native chat to the tokenizer
        text_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # Converts text to tokens
        inputs = self.tokenizer(text_prompt,
                                return_tensors="pt").to(self.device)
        # Generates the answer with new strict rules
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            repetition_penalty=1.15,  # Punishes word repetition
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id  # Forces model to stop
        )
        # Decodes tokens back to text
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens,
                                               skip_special_tokens=True)
        import re
        # Removes everything between <think> and </think>
        clean_text = re.sub(r'<think>.*?</think>',
                            '', generated_text, flags=re.DOTALL)
        # If it got cut in between thinking we get rid of the rest
        clean_text = re.sub(r'<think>.*',
                            '', clean_text, flags=re.DOTALL).strip()
        # If cleanup fails because it spent all tokens thinking we warn
        if not clean_text:
            clean_text = "Error: The model ran out of tokens while thinking."
        return MinimalAnswer(
            question_id=search_result.question_id,
            question=search_result.question,
            retrieved_sources=search_result.retrieved_sources,
            answer=clean_text
        )
