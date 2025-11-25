"""Adapter for UnsloTh FastLanguageModel so it can be used as a callable LLM in the repo.

This file implements `UnslothLLM` which exposes a simple `__call__(prompt)->str` API
compatible with the rest of the codebase.

Notes:
- UnsloTh (FastLanguageModel) must be available in your Colab environment.
- This adapter tries a few generation entrypoints (model.generate, model.chat)
  and produces a plain text string as output.
"""
from typing import Optional
import os


class UnslothLLM:
    def __init__(self,
                 model_name: str,
                 token: Optional[str] = None,
                 load_in_4bit: bool = True,
                 max_seq_length: int = 8192,
                 chat_template: str = "llama-3.1",
                 ) -> None:
        try:
            # unsloth provides FastLanguageModel
            from unsloth import FastLanguageModel
            from transformers import TextStreamer
            from unsloth.chat_templates import get_chat_template
        except Exception as e:
            raise RuntimeError("unsloth and transformers must be installed to use UnslothLLM (pip install unsloth transformers).") from e

        # If a token was supplied, expose it for huggingface downloads
        if token:
            os.environ['HUGGINGFACE_HUB_TOKEN'] = token

        # Load model/tokenizer using the FastLanguageModel helper
        # This returns (model, tokenizer) as per unsloth examples
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                token=token,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load unsloth model {model_name}: {e}") from e

        # Patch tokenizer for the chat template if available
        try:
            tokenizer = get_chat_template(
                tokenizer,
                chat_template=chat_template,
                mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
            )
        except Exception:
            # fallback: continue with original tokenizer
            pass

        # Optimize model for inference when available
        try:
            FastLanguageModel.for_inference(model)
        except Exception:
            # not fatal
            pass

        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, prompt: str) -> str:
        # Try a few generation call styles depending on what the model exposes.
        # 1) If model has `generate` like HF models, use it.
        try:
            if hasattr(self.model, 'generate'):
                inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=self.max_seq_length)
                # Move tensors to model device if possible
                try:
                    import torch
                    device = next(self.model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                except Exception:
                    pass
                out_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
                out = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
                # strip prompt if present
                if out.startswith(prompt):
                    return out[len(prompt):].strip()
                return out.strip()
        except Exception:
            # fallthrough to other methods
            pass

        # 2) If model has chat or stream style API, attempt simple calls
        try:
            # some FastLanguageModel instances implement a `chat` or `generate_text` method
            if hasattr(self.model, 'chat'):
                out = self.model.chat(prompt)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                return str(out).strip()
            if hasattr(self.model, 'generate_text'):
                out = self.model.generate_text(prompt)
                return str(out).strip()
        except Exception:
            pass

        # 3) As a last resort, try to call model on the tokenizer directly (some wrappers offer call)
        try:
            out = self.model(prompt)
            return str(out).strip()
        except Exception as e:
            raise RuntimeError(f"UnsloTh model couldn't generate text with any known API: {e}") from e
