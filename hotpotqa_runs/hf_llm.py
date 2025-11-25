import torch
from typing import Optional


class AnyHFLLM:
    """
    Adapter to mimic the callable LLM interface used in `hotpotqa_runs.agents.AnyOpenAILLM`.
    Works with Unsloth's FastLanguageModel + tokenizer or HF-style tokenizers that support
    `tokenizer(prompt, return_tensors="pt", ...)` and `tokenizer.decode(...)`.

    Usage:
        hf = AnyHFLLM(model, tokenizer, temperature=0.0, max_tokens=128)
        out = hf(prompt_text)
    """
    def __init__(self, model, tokenizer, temperature: float = 0.0, max_tokens: int = 250, device: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, prompt: str) -> str:
        # Encode prompt (assumes prompt is final text ready for tokenization)
        enc = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                temperature=float(self.temperature),
                pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
                return_dict_in_generate=True,
            )

        # decode only newly generated tokens
        new_tokens = out.sequences[0, enc.input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
