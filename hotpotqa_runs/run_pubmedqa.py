"""Run PubMedQA dataset through existing agents and LLM adapters.

Usage examples:
  # use HF Inference API (requires HF token in HF_API_TOKEN)
  python run_pubmedqa.py --use-inference --model qiaojin/PubMedQA --split validation --limit 50 --out results_pubmed.csv

  # use local transformers adapter (requires model available locally or on HF hub)
  python run_pubmedqa.py --use-transformers --model google/flan-t5-small --split validation --limit 20 --out results_pubmed.csv

Notes:
- The script heuristically finds the question/context/answer fields in the dataset.
- For 8B Llama-style models you must run with GPU + required deps (bitsandbytes) and set `--use-transformers`.
"""

import argparse
import csv
import os
import sys
from typing import Tuple, Optional

# Ensure package imports work when running as module or script.
# Insert the repository root (parent of this file) into sys.path so
# imports like `hotpotqa_runs.*` resolve when the script is executed directly.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Note: import datasets lazily inside `run()` to avoid hard dependency at import-time

try:
    # Prefer repo-provided adapters if present
    from hf_inference_llm import HFInferenceLLM
except Exception:
    HFInferenceLLM = None

try:
    from hf_transformers_llm import HFTransformersLLM
except Exception:
    HFTransformersLLM = None

try:
    from agents import ReactAgent, CoTAgent, ReactReflectAgent, EM, ReflexionStrategy
except Exception:
    # fallback relative import
    from hotpotqa_runs.agents import ReactAgent, CoTAgent, ReactReflectAgent, EM, ReflexionStrategy


def map_reflexion_str(s: Optional[str]):
    if s is None:
        return ReflexionStrategy.REFLEXION
    s = s.lower()
    if s in ('none', 'base'):
        return ReflexionStrategy.NONE
    if s in ('last_attempt', 'last'):
        return ReflexionStrategy.LAST_ATTEMPT
    if s in ('reflexion', 'reflect'):
        return ReflexionStrategy.REFLEXION
    if s in ('last_attempt_and_reflexion', 'last_and_reflexion'):
        return ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION
    # default
    return ReflexionStrategy.REFLEXION


COMMON_QUESTION_FIELDS = ["question", "query", "prompt"]
COMMON_CONTEXT_FIELDS = ["context", "abstract", "passage", "article", "text", "background"]
COMMON_ANSWER_FIELDS = ["answer", "label", "answers", "final_answer", "target"]


def detect_fields(example: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (question_field, context_field, answer_field) or (None,...)"""
    keys = list(example.keys())
    q = next((k for k in keys if k.lower() in COMMON_QUESTION_FIELDS), None)
    c = next((k for k in keys if k.lower() in COMMON_CONTEXT_FIELDS), None)
    a = next((k for k in keys if k.lower() in COMMON_ANSWER_FIELDS), None)
    # fallback heuristics
    if q is None:
        # try any key that contains 'question'
        q = next((k for k in keys if 'question' in k.lower()), None)
    if c is None:
        c = next((k for k in keys if 'abstract' in k.lower() or 'context' in k.lower()), None)
    if a is None:
        a = next((k for k in keys if 'label' in k.lower() or 'answer' in k.lower() or 'target' in k.lower()), None)
    return q, c, a


def build_llm(args):
    """Instantiate either HF Inference or local Transformers adapter depending on args."""
    # shared token used for HF authentication when needed
    token = args.hf_token or os.environ.get('HF_API_TOKEN') or os.environ.get('HUGGINGFACE_API_TOKEN')

    # UnsloTh path (Colab-ready prequantized models)
    if getattr(args, 'use_unsloth', False):
        try:
            from unsloth_llm import UnslothLLM
        except Exception:
            from hotpotqa_runs.unsloth_llm import UnslothLLM
        return UnslothLLM(model_name=args.model, token=token, load_in_4bit=getattr(args, 'load_in_4bit', True), max_seq_length=getattr(args, 'max_seq_length', 8192))

    if args.use_transformers:
        # Build a local transformers-backed LLM inline so we can set trust_remote_code
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError("Local transformers path requires 'transformers' and 'torch' installed.") from e

        model_id = args.model

        # If user provided an HF token, set HUGGINGFACE_HUB_TOKEN for downloads
        token = args.hf_token or os.environ.get('HF_API_TOKEN') or os.environ.get('HUGGINGFACE_API_TOKEN')
        if token:
            os.environ['HUGGINGFACE_HUB_TOKEN'] = token

        bnb_cfg = None
        if getattr(args, 'load_in_4bit', False):
            try:
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception:
                bnb_cfg = None

        # Load tokenizer and model (trust_remote_code=True for models that require it)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True, use_auth_token=token)
        except OSError as e:
            raise RuntimeError(
                f"Could not access model '{model_id}'. This repository may be private or gated.\n"
                "Make sure you have accepted the model license on Hugging Face, then either: \n"
                "  1) run `huggingface-cli login` and authenticate, or\n"
                "  2) set your token in the env var `HF_API_TOKEN` (or pass --hf-token) before running this script.\n"
                "Example: export HF_API_TOKEN=hf_...\n"
            ) from e

        load_kwargs = {"device_map": "auto"}
        if bnb_cfg is not None:
            load_kwargs["quantization_config"] = bnb_cfg
        # prefer float16 on CUDA devices
        try:
            use_cuda = torch.cuda.is_available()
        except Exception:
            use_cuda = False
        if use_cuda:
            load_kwargs["torch_dtype"] = torch.float16

        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, use_auth_token=token, **load_kwargs)
        except OSError as e:
            raise RuntimeError(
                f"Could not download/load model '{model_id}'. Ensure your token has access and you accepted the model terms on Hugging Face.\n"
                "Either run `huggingface-cli login` or set `HF_API_TOKEN` env var with a token that has access.\n"
            ) from e

        class LocalTransformersLLM:
            def __init__(self, model, tokenizer):
                self.model = model
                self.tokenizer = tokenizer

            def __call__(self, prompt: str) -> str:
                inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True)
                # move inputs to model device if possible
                try:
                    device = next(self.model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                except Exception:
                    pass
                gen = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
                out = self.tokenizer.decode(gen[0], skip_special_tokens=True)
                # remove prompt prefix if returned
                if out.startswith(prompt):
                    return out[len(prompt):].strip()
                return out.strip()

        return LocalTransformersLLM(model, tokenizer)
    else:
        # HF Inference
        if HFInferenceLLM is None:
            raise RuntimeError("HF Inference adapter not available. Make sure hotpotqa_runs/hf_inference_llm.py exists.")
        token = args.hf_token or os.environ.get('HF_API_TOKEN') or os.environ.get('HUGGINGFACE_API_TOKEN')
        if not token:
            raise RuntimeError('HF API token required for inference. Provide via --hf-token or set HF_API_TOKEN env var')
        # HFInferenceLLM expects `model_id` and `api_token` keywords
        return HFInferenceLLM(model_id=args.model, api_token=token)


def extract_text(field_value) -> str:
    if field_value is None:
        return ''
    # If the dataset field is a list (e.g., contexts) join them into a single text
    if isinstance(field_value, list):
        return '\n\n'.join(map(lambda x: x if isinstance(x, str) else str(x), field_value))
    # If the field is a dict containing nested 'contexts' or text pieces, try to extract them
    if isinstance(field_value, dict):
        # common shape: {'contexts': [ ... ], 'some_meta': ...}
        if 'contexts' in field_value and isinstance(field_value['contexts'], list):
            return '\n\n'.join(map(lambda x: x if isinstance(x, str) else str(x), field_value['contexts']))
        # fallback to string representation
        return str(field_value)
    return str(field_value)


def run(args, external_llm=None):
    print(f"Loading dataset {args.dataset} split={args.split}...")
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("The 'datasets' library is required to load HF datasets. Install it with `pip install datasets`.") from e

    # If the dataset requires a config (name), pass it through. Example: PubMedQA requires one of
    # ['pqa_artificial', 'pqa_labeled', 'pqa_unlabeled'] as the config name.
    try:
        if args.dataset_config:
            ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
        else:
            ds = load_dataset(args.dataset, split=args.split)
    except ValueError as e:
        # Many HF datasets require specifying a config name; surface the error and provide guidance.
        raise RuntimeError(
            f"Failed to load dataset '{args.dataset}': {e}.\n"
            "If the dataset has multiple configs (e.g. PubMedQA), pass one with --dataset-config,\n"
            "for example: --dataset-config pqa_labeled\n"
        ) from e
    if len(ds) == 0:
        print('Empty split. Exiting.')
        return

    sample = ds[0]
    q_field, c_field, a_field = detect_fields(sample)
    print('Detected fields ->', q_field, c_field, a_field)
    if q_field is None:
        print('Could not detect a question field. Dataset feature keys:', list(sample.keys()))
        return

    # instantiate llm (or use a pre-initialized one when provided)
    if external_llm is not None:
        # Wrap external LLM in a safe callable that accepts either a prompt string
        # or token kwargs (e.g., input_ids) and normalizes outputs to string.
        def make_safe_llm(l):
            def _call(prompt: Optional[str] = None, **kwargs):
                try:
                    # If caller passed token tensors, forward them directly
                    if kwargs:
                        out = l(**kwargs)
                    else:
                        out = l(prompt)
                except TypeError as e:
                    # Some models raise TypeError when receiving unsupported kwargs
                    # Try common alternative call patterns
                    try:
                        if prompt is not None and hasattr(l, 'chat'):
                            out = l.chat(prompt)
                        elif prompt is not None and hasattr(l, 'generate_text'):
                            out = l.generate_text(prompt)
                        else:
                            # try tokenizing if tokenizer available
                            tok = getattr(l, 'tokenizer', None)
                            if tok is not None and prompt is not None:
                                inputs = tok(prompt, return_tensors='pt', truncation=True, max_length=getattr(args, 'max_seq_length', 8192))
                                try:
                                    import torch
                                    device = next(l.parameters()).device
                                    inputs = {k: v.to(device) for k, v in inputs.items()}
                                except Exception:
                                    pass
                                out = l(**inputs)
                            else:
                                raise
                    except Exception:
                        raise
                # normalize output shapes
                if isinstance(out, (list, tuple)):
                    out = out[0]
                if isinstance(out, dict):
                    out = out.get('generated_text') or out.get('text') or next(iter(out.values()), None)
                return '' if out is None else str(out)
            return _call

        llm = make_safe_llm(external_llm)
        print('Using externally provided LLM instance (wrapped)')
    else:
        llm = build_llm(args)
        print('LLM instantiated:', llm)

    # Try to configure a Wikipedia docstore for ReactAgent if LangChain is available.
    # If unavailable, docstore remains None and agents will fallback to their existing behavior.
    docstore = None
    try:
        from langchain import Wikipedia
        # Pass the Wikipedia instance into ReactAgent; ReactAgent will wrap it
        # with DocstoreExplorer if LangChain is available. Avoid double-wrapping
        # by creating the raw Wikipedia object here.
        docstore = Wikipedia()
        try:
            print('Configured Wikipedia source for ReactAgent (will be wrapped).', 'docstore repr:', repr(docstore))
        except Exception:
            print('Configured Wikipedia source for ReactAgent (will be wrapped).')
    except Exception:
        docstore = None

    def coerce_yes_no_maybe(pred_text: str, scratchpad: str) -> str:
        """Map a model output to 'yes'/'no'/'maybe' using heuristics and scratchpad search.

        Priority: explicit Finish[...] in scratchpad, then keyword heuristics on pred_text,
        then fallback to 'maybe'.
        """
        if pred_text is None:
            return 'maybe'
        s = pred_text.strip().lower()
        # 1) Check scratchpad for explicit Finish[...] occurrences
        import re
        m = re.search(r'finish\[([^\]]+)\]', scratchpad, flags=re.IGNORECASE)
        if m:
            val = m.group(1).strip().lower()
            if val in ('yes', 'y', 'true', '1'):
                return 'yes'
            if val in ('no', 'n', 'false', '0'):
                return 'no'
            if 'maybe' in val or 'possibly' in val or 'could' in val or 'likely' in val:
                return 'maybe'
            # If scratchpad contains other freeform text, try to detect affirmation/negation
            if any(tok in val for tok in ('yes','no','maybe','likely','possibly','uncertain','not')):
                if 'no' in val or 'not' in val or 'none' in val or 'absent' in val:
                    return 'no'
                if 'yes' in val or 'present' in val or 'found' in val:
                    return 'yes'

        # 2) Heuristic on pred_text
        if any(tok in s for tok in (' yes ', ' yes', 'yes.', 'yes\n', ' yes\'')) or s in ('yes','y','true'):
            return 'yes'
        if any(tok in s for tok in (' no ', ' no', 'no.', 'no\n')) or s in ('no','n','false'):
            return 'no'
        if any(tok in s for tok in ('maybe','possibly','could','likely','uncertain','unsure','unclear')):
            return 'maybe'

        # 3) As a last resort, look for polarity words
        yes_words = ('affirmative','positive','present','found','detected')
        no_words = ('absent','negative','not detected','none','no evidence')
        if any(w in s for w in yes_words):
            return 'yes'
        if any(w in s for w in no_words):
            return 'no'

        return 'maybe'

    # Choose agent type: use ReactAgent as default
    if args.agent == 'react':
        AgentClass = ReactAgent
    elif args.agent == 'cot':
        AgentClass = CoTAgent
    elif args.agent == 'react_reflect':
        AgentClass = ReactReflectAgent
    else:
        AgentClass = ReactAgent

    out_rows = []
    total = 0
    correct = 0

    for i, ex in enumerate(ds):
        if args.limit and i >= args.limit:
            break
        total += 1
        question = extract_text(ex.get(q_field))
        context = extract_text(ex.get(c_field)) if c_field else ''
        true_answer = extract_text(ex.get(a_field)) if a_field else ''

        # Prepare agent. ReactAgent: (question, key, ...) ; CoTAgent: (question, context, key, ...)
        if AgentClass is ReactAgent:
            agent = ReactAgent(question=question, key=true_answer, react_llm=llm, max_steps=args.max_steps, docstore=docstore, force_finish_format=getattr(args, 'force_finish_format', False))
        elif AgentClass is CoTAgent:
            agent = CoTAgent(question=question, context=context, key=true_answer, action_llm=llm, self_reflect_llm=llm, force_finish_format=getattr(args, 'force_finish_format', False))
        else:  # ReactReflectAgent
            agent = ReactReflectAgent(question=question, key=true_answer, react_llm=llm, reflect_llm=llm, max_steps=args.max_steps, docstore=docstore, force_finish_format=getattr(args, 'force_finish_format', False))

        try:
            # Dispatch run with optional reflexion strategy when supported
            # Debug: report whether the agent has a docstore configured
            try:
                has_doc = getattr(agent, 'docstore', None) is not None
                print(f'Agent docstore configured: {has_doc}')
            except Exception:
                pass
            if isinstance(agent, CoTAgent):
                # map string to ReflexionStrategy
                strat = map_reflexion_str(args.reflexion_strategy)
                agent.run(reflexion_strategy=strat)
            elif isinstance(agent, ReactReflectAgent):
                strat = map_reflexion_str(args.reflexion_strategy)
                agent.run(reset=True, reflect_strategy=strat)
            else:
                agent.run()
        except Exception as e:
            print(f"Error while running agent on example {i}: {e}")

        pred = getattr(agent, 'answer', '')
        # Some agents may leave answer empty; try to extract Finish[...] from scratchpad
        if not pred:
            # look for Finish[...] pattern in scratchpad
            import re
            s = getattr(agent, 'scratchpad', '')
            m = re.search(r'Finish\[(.*?)\]', s)
            if m:
                pred = m.group(1)
            else:
                # fallback: pick last Observation or last line
                lines = s.split('\n')
                if len(lines) > 0:
                    pred = lines[-1].strip()

        is_correct = False
        try:
            is_correct = EM(pred, true_answer)
        except Exception:
            is_correct = (pred.strip().lower() == true_answer.strip().lower())

        if is_correct:
            correct += 1

        out_rows.append({
            'index': i,
            'question': question,
            'context': context[:1000],
            'true_answer': true_answer,
            'predicted_answer': pred,
            'correct': is_correct,
        })

        if total % 10 == 0:
            print(f"Processed {total} examples. Acc={correct}/{total}={correct/total:.3f}")

    # write CSV
    out_path = args.out or f'results_{args.dataset.replace("/","_")}_{args.split}.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['index','question','context','true_answer','predicted_answer','correct'])
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f"Done. Processed={total}. Correct={correct}. Results saved to {out_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='qiaojin/PubMedQA', help='HF dataset id')
    p.add_argument('--split', default='validation', help='dataset split to run (train/validation/test)')
    p.add_argument('--limit', type=int, default=100, help='limit number of examples (0 means all)')
    p.add_argument('--model', default='google/flan-t5-small', help='model id for inference or transformers')
    p.add_argument('--use-transformers', action='store_true', help='Load model locally with transformers adapter')
    p.add_argument('--use-inference', dest='use_transformers', action='store_false', help='Use HF Inference API client (default)')
    p.add_argument('--hf-token', default=None, help='Hugging Face API token (if using inference)')
    p.add_argument('--out', default=None, help='CSV output path')
    p.add_argument('--agent', choices=['react','cot','react_reflect'], default='react', help='Agent style to run')
    p.add_argument('--dataset-config', default=None, help='Optional dataset config/name (for multi-config datasets like PubMedQA)')
    p.add_argument('--reflexion-strategy', choices=['none','last_attempt','reflexion','last_attempt_and_reflexion'], default='reflexion', help='Reflexion strategy to apply (when agent supports it)')
    p.add_argument('--max-steps', type=int, default=6, help='max steps for ReactAgent')
    # Optional explicit field mapping (overrides auto-detection)
    p.add_argument('--question-field', default=None, help='Explicit dataset field to use as the question')
    p.add_argument('--context-field', default=None, help='Explicit dataset field to use as the context/abstract')
    p.add_argument('--answer-field', default=None, help='Explicit dataset field to use as the gold answer/label')
    # Local transformers options
    p.add_argument('--device', default=None, help='Device to load model on (cuda/mps/cpu). If not set, adapter auto-detects')
    p.add_argument('--load-in-4bit', action='store_true', help='Load model in 4-bit using bitsandbytes (requires CUDA + bitsandbytes)')
    # UnsloTh / long-context options
    p.add_argument('--use-unsloth', action='store_true', help='Use UnsloTh prequantized models (Colab-friendly)')
    p.add_argument('--max-seq-length', type=int, default=8192, help='Max sequence length / context window for UnsloTh or long-context models')
    p.add_argument('--force-finish-format', action='store_true', help='Ask agents to output exactly one Finish[...] action with yes/no/maybe when finishing')
    args = p.parse_args()

    # normalize limit
    if args.limit == 0:
        args.limit = None

    # Fill use_transformers default: if flag not passed, default to False (inference) unless user passes --use-transformers
    # argparse configured so default is True when --use-transformers present, else False

    # Build boolean correctly
    args.use_transformers = bool(args.use_transformers)

    run(args)
