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
from typing import Optional

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
    q_field = getattr(args, 'question_field', 'question')
    c_field = getattr(args, 'context_field', 'context')
    a_field = getattr(args, 'answer_field', 'final_decision')
    long_field = getattr(args, 'long_answer_field', None)

    def _ensure_field(name: str):
        if name not in sample:
            raise RuntimeError(f"Field '{name}' not found in dataset sample. Available keys: {list(sample.keys())}")

    _ensure_field(q_field)
    _ensure_field(c_field)
    _ensure_field(a_field)
    if long_field:
        _ensure_field(long_field)

    fields_msg = f'Using fields -> question={q_field}, context={c_field}, answer={a_field}'
    if long_field:
        fields_msg += f', long={long_field}'
    print(fields_msg)

    # instantiate llm (or use a pre-initialized one when provided)
    # We expose two names: `llm_callable` (what agents should call) and
    # `llm_raw` (the underlying model/tokenizer object when available) so
    # we can compute log-probabilities / confidences when supported.
    if external_llm is not None:
        # Wrap external LLM in a safe proxy object that is callable and
        # attempts to expose `.model` and `.tokenizer` attributes when available.
        class _ProxyLLM:
            def __init__(self, raw):
                self._raw = raw
                # expose model/tokenizer if present on the raw adapter
                self.model = getattr(raw, 'model', None)
                self.tokenizer = getattr(raw, 'tokenizer', None)

            def __call__(self, prompt: Optional[str] = None, **kwargs):
                l = self._raw
                try:
                    if kwargs:
                        out = l(**kwargs)
                    else:
                        out = l(prompt)
                except TypeError:
                    # Try alternate call patterns used by some adapters
                    if prompt is not None and hasattr(l, 'chat'):
                        out = l.chat(prompt)
                    elif prompt is not None and hasattr(l, 'generate_text'):
                        out = l.generate_text(prompt)
                    else:
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

                # normalize output shapes
                if isinstance(out, (list, tuple)):
                    out = out[0]
                if isinstance(out, dict):
                    out = out.get('generated_text') or out.get('text') or next(iter(out.values()), None)
                return '' if out is None else str(out)

        llm_callable = _ProxyLLM(external_llm)
        # Prefer using the proxy as raw for scoring if it exposes model/tokenizer
        llm_raw = llm_callable if (getattr(llm_callable, 'model', None) is not None and getattr(llm_callable, 'tokenizer', None) is not None) else external_llm
        print('Using externally provided LLM instance (wrapped)')
    else:
        llm_raw = build_llm(args)
        llm_callable = llm_raw
        print('LLM instantiated:', llm_raw)

    if getattr(llm_raw, 'model', None) is None or getattr(llm_raw, 'tokenizer', None) is None:
        print('Notice: LLM instance does not expose model/tokenizer attributes. Confidence scoring and Brier metrics will be unavailable.')

    # Helper: compute relative probabilities for the discrete choices using
    # an underlying transformers-style model+tokenizer when available. Returns
    # a dict mapping choice->prob or None when scoring is not supported.
    def _score_choices_via_transformers(raw_llm, prompt: str, choices):
        model = getattr(raw_llm, 'model', None)
        tokenizer = getattr(raw_llm, 'tokenizer', None)
        if model is None or tokenizer is None:
            if getattr(args, 'print_logit_debug', False):
                print('Logit scoring skipped: LLM missing model/tokenizer attrs.')
            return None
        import math
        import torch
        import torch.nn.functional as F

        device = next(model.parameters()).device
        max_len = getattr(args, 'max_seq_length', 8192)

        def _logprob_via_generate(choice: str):
            choice_ids = tokenizer(choice, add_special_tokens=False, return_tensors='pt')['input_ids'][0].to(device)
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_len)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            target_len = choice_ids.shape[0]
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=target_len,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                    use_cache=True,
                )
            scores = out.scores  # list length target_len
            log_lik = 0.0
            for idx, tok_id in enumerate(choice_ids):
                logits = scores[idx][0]
                logp = F.log_softmax(logits, dim=-1)[tok_id].item()
                log_lik += logp
            return log_lik

        logls = []
        for choice in choices:
            logprob = None
            try:
                logprob = _logprob_via_generate(choice)
            except Exception as e_gen:
                if getattr(args, 'print_logit_debug', False):
                    print('Generate logprob failed:', type(e_gen).__name__, e_gen)
                return None
            logls.append(logprob)
        m = max(logls)
        exps = [math.exp(l - m) for l in logls]
        s = sum(exps)
        probs = [e / s for e in exps]
        return dict(zip(choices, probs))

    debug_enabled = getattr(args, 'print_debug', True)

    # Try to configure a Wikipedia docstore for ReactAgent if LangChain is available.
    # If unavailable, docstore remains None and agents will fallback to their existing behavior.
    docstore = None
    try:
        from langchain import Wikipedia
        # Pass the Wikipedia instance into ReactAgent; ReactAgent will wrap it
        # with DocstoreExplorer if LangChain is available. Avoid double-wrapping
        # by creating the raw Wikipedia object here.
        docstore = Wikipedia()
        if debug_enabled:
            try:
                print('Configured Wikipedia source for ReactAgent (will be wrapped).', 'docstore repr:', repr(docstore))
            except Exception:
                print('Configured Wikipedia source for ReactAgent (will be wrapped).')
    except Exception:
        docstore = None

    # Simple fallback docstore that exposes .search() and .lookup() using the
    # example's `context` text. This ensures ReactAgent can at least Search/Lookup
    # within the provided context when LangChain/Wikipedia is not available.
    class SimpleDocstore:
        def __init__(self, docs: dict):
            # docs: id -> text
            self.docs = docs
            self._last_doc_id = None

        def search(self, query: str) -> str:
            # naive search: return the first doc containing the query or the
            # full text of the single context if nothing matches.
            q = (query or '').lower()
            for doc_id, text in self.docs.items():
                if q in doc_id.lower() or q in text.lower():
                    self._last_doc_id = doc_id
                    return text
            # fallback: return concatenation of all docs
            if len(self.docs) > 0:
                # pick first doc
                did = next(iter(self.docs.keys()))
                self._last_doc_id = did
                return self.docs[did]
            return 'No documents available.'

        def lookup(self, term: str) -> str:
            if self._last_doc_id is None:
                raise ValueError('No last page searched.')
            text = self.docs.get(self._last_doc_id, '')
            for sent in text.split('.'):
                if term.lower() in sent.lower():
                    return sent.strip()
            return 'Term not found in last page.'

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

    limit = getattr(args, 'limit', None)
    if limit == 0:
        limit = None
    target_total = len(ds) if limit is None else min(len(ds), limit)

    out_rows = []
    total = 0
    correct = 0
    gold_labels: list[str] = []
    pred_labels: list[str] = []
    prob_records = []
    rouge_records = []
    readability_scores = []

    try:
        from rouge_score import rouge_scorer
        rouge_scorer_fn = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
    except Exception:
        rouge_scorer_fn = None
        print('rouge_score package not available; skipping rationale ROUGE metrics.')

    try:
        import textstat
    except Exception:
        textstat = None
        print('textstat not available; skipping readability metrics.')

    def _canon_label(val: str) -> str:
        return (val or '').strip().lower()

    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        total += 1
        question = extract_text(ex.get(q_field))
        context = extract_text(ex.get(c_field)) if c_field else ''
        true_answer = extract_text(ex.get(a_field)) if a_field else ''
        gold_label = _canon_label(true_answer)
        long_answer_text = extract_text(ex.get(long_field)) if long_field else ''

        # Prepare agent. ReactAgent: (question, key, ...) ; CoTAgent: (question, context, key, ...)
        # If a global LangChain Wikipedia docstore is not configured, create a
        # per-example SimpleDocstore using the example `context` so Search/Lookup
        # still return meaningful text.
        doc_for_agent = docstore if docstore is not None else SimpleDocstore({'context': context})
        if AgentClass is ReactAgent:
            agent = ReactAgent(question=question, key=true_answer, react_llm=llm_callable, max_steps=args.max_steps, docstore=doc_for_agent, force_finish_format=getattr(args, 'force_finish_format', False))
        elif AgentClass is CoTAgent:
            agent = CoTAgent(question=question, context=context, key=true_answer, action_llm=llm_callable, self_reflect_llm=llm_callable, force_finish_format=getattr(args, 'force_finish_format', False))
        else:  # ReactReflectAgent
            agent = ReactReflectAgent(question=question, key=true_answer, react_llm=llm_callable, reflect_llm=llm_callable, max_steps=args.max_steps, docstore=doc_for_agent, force_finish_format=getattr(args, 'force_finish_format', False))

        # Defensive: clear the agent's few-shot examples when running open-domain
        # tasks so the prompt only contains the current sample. For PubMedQA we
        # want to keep biomedical few-shots, so skip clearing when the dataset
        # name indicates PubMedQA or when the caller explicitly requests to keep
        # the examples (keep_fewshot_examples=True).
        should_clear_examples = True
        try:
            dataset_name = (getattr(args, 'dataset', '') or '').lower()
            if 'pubmedqa' in dataset_name:
                should_clear_examples = False
        except Exception:
            should_clear_examples = True
        if getattr(args, 'keep_fewshot_examples', False):
            should_clear_examples = False
        if should_clear_examples:
            try:
                if hasattr(agent, 'react_examples'):
                    agent.react_examples = ''
                if hasattr(agent, 'cot_examples'):
                    agent.cot_examples = ''
                if hasattr(agent, 'reflect_examples'):
                    agent.reflect_examples = ''
            except Exception:
                pass

        # Force-attach a SimpleDocstore fallback so agent.docstore is never None.
        # This guarantees Search/Lookup actions have a minimal implementation
        # (search/lookup over the example `context`) even when LangChain/Wikipedia
        # are unavailable or agent wrapping failed.
        try:
            fallback_ds = SimpleDocstore({'context': context})
            if getattr(agent, 'docstore', None) is None:
                try:
                    agent.docstore = fallback_ds
                    if debug_enabled:
                        print(f'Notice: Attached SimpleDocstore fallback to agent for example {i}')
                except Exception:
                    # best-effort: if the agent doesn't allow setting `.docstore`,
                    # try storing on a private attribute used by our debug prints
                    try:
                        setattr(agent, '_simple_docstore_fallback', fallback_ds)
                        if debug_enabled:
                            print(f'Notice: Stored SimpleDocstore fallback on agent._simple_docstore_fallback for example {i}')
                    except Exception:
                        if debug_enabled:
                            print('Warning: Could not attach SimpleDocstore fallback to agent')
            else:
                # Agent already had a docstore (e.g., Wikipedia). Print its type.
                if debug_enabled:
                    try:
                        print('Agent already has a docstore of type:', type(agent.docstore))
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            # Dispatch run with optional reflexion strategy when supported
            # Debug: report whether the agent has a docstore configured
            if debug_enabled:
                try:
                    has_doc = getattr(agent, 'docstore', None) is not None
                    print(f'Agent docstore configured: {has_doc}')
                except Exception:
                    pass
                # Additional debug: print the inputs provided to the agent/LLM
                try:
                    print('\n==== DEBUG: Inputs passed to agent for example {} ===='.format(i))
                    print('Question:', question)
                    if context is not None:
                        ctx_snip = (context[:1000] + '...') if len(context) > 1000 else context
                        print('Context (truncated 1000 chars):', ctx_snip)
                    else:
                        print('Context: <None>')
                    print('True/Label Answer:', true_answer)
                    try:
                        print('doc_for_agent repr:', repr(doc_for_agent))
                    except Exception:
                        print('doc_for_agent repr: <unprintable>')
                    # If the agent exposes a prompt builder, try to print the initial prompt
                    try:
                        if hasattr(agent, '_build_agent_prompt'):
                            built = agent._build_agent_prompt()
                            print('\n--- Built agent prompt (initial) ---')
                            print(built)
                            print('--- End prompt ---\n')
                    except Exception as e:
                        print('Could not build/print agent prompt:', type(e), e)
                except Exception:
                    import traceback
                    traceback.print_exc()
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

        # After the first run, try to evaluate the model's confidence on the
        # discrete choices (yes/no/maybe) using logits when the underlying
        # transformers model+tokenizer is available. If confidence is low,
        # attempt reflexion (if supported) and rerun to improve confidence.
        pred = getattr(agent, 'answer', '')
        prob_dict_for_example = None
        try:
            max_attempts = getattr(args, 'max_reflect_attempts', 2)
            attempts = 0
            # previous confidence baseline
            prev_conf = -1.0
            while attempts < max_attempts:
                # build a scoring prompt from the agent if possible
                scoring_prompt = None
                try:
                    if hasattr(agent, '_build_agent_prompt'):
                        scoring_prompt = agent._build_agent_prompt()
                except Exception:
                    scoring_prompt = None

                confs = None
                if scoring_prompt is not None:
                    confs = _score_choices_via_transformers(llm_raw, scoring_prompt, ['yes', 'no', 'maybe'])
                if confs is None:
                    # scoring not available; break out
                    break
                prob_dict_for_example = confs

                # coerce current prediction to canonical label
                cur_label = coerce_yes_no_maybe(pred, getattr(agent, 'scratchpad', ''))
                cur_conf = float(confs.get(cur_label, 0.0))
                if getattr(args, 'print_logit_debug', False):
                    print(f'Confidence for prediction \"{cur_label}\" = {cur_conf:.4f} (choices: {confs})')

                # If confidence is acceptable, stop retrying
                threshold = getattr(args, 'confidence_threshold', 0.6)
                if cur_conf >= threshold or cur_conf > prev_conf:
                    break

                # otherwise attempt one reflexion and rerun
                if hasattr(agent, 'reflect'):
                    try:
                        strat = map_reflexion_str(args.reflexion_strategy)
                        agent.reflect(strat)
                        # rerun without resetting to preserve context/scratchpad
                        agent.run(reset=False)
                        pred = getattr(agent, 'answer', '')
                    except Exception as e:
                        print('Reflection attempt failed:', e)
                        break
                else:
                    break

                prev_conf = cur_conf
                attempts += 1
        except Exception:
            pass
        prob_records.append((prob_dict_for_example, gold_label))
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

        # Canonicalize prediction to yes/no/maybe so downstream evaluation and
        # logging are consistent even if the model emitted extra prose.
        scratchpad = getattr(agent, 'scratchpad', '')
        rationale_text = ''
        if 'Reason:' in scratchpad:
            rationale_text = scratchpad.rsplit('Reason:', 1)[1].strip()
            rationale_text = rationale_text.split('Answer:', 1)[0].strip()
            rationale_text = rationale_text.split('\n', 1)[0].strip()
        else:
            rationale_text = f'Reason: {pred}.'

        pred = coerce_yes_no_maybe(pred, scratchpad)
        try:
            agent.answer = pred
        except Exception:
            pass
        pred_labels.append(pred)
        gold_labels.append(gold_label)

        is_correct = False
        try:
            is_correct = EM(pred, true_answer)
        except Exception:
            is_correct = (pred.strip().lower() == true_answer.strip().lower())

        if is_correct:
            correct += 1

        if rouge_scorer_fn and rationale_text and long_answer_text:
            try:
                score = rouge_scorer_fn.score(long_answer_text, rationale_text)
                rouge_records.append(score)
            except Exception:
                pass
        if textstat and rationale_text:
            try:
                readability_scores.append(textstat.flesch_reading_ease(rationale_text))
            except Exception:
                pass

        out_rows.append({
            'index': i,
            'question': question,
            'context': context[:1000],
            'true_answer': true_answer,
            'long_answer': long_answer_text,
            'predicted_answer': pred,
            'scratchpad': scratchpad,
            'reason_text': rationale_text,
            'correct': is_correct,
        })

        if total % 10 == 0 or total == target_total:
            pct = (total / target_total) * 100 if target_total else 0.0
            acc = correct / total if total else 0.0
            print(f"Progress: {total}/{target_total} ({pct:.1f}%)  Acc={acc:.3f}")

    # write CSV
    out_path = args.out or f'results_{args.dataset.replace("/","_")}_{args.split}.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['index','question','context','true_answer','long_answer','predicted_answer','scratchpad','reason_text','correct'])
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f"Done. Processed={total}. Correct={correct}. Results saved to {out_path}")

    # Aggregate accuracy / F1 / classification metrics if sklearn is available
    try:
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        if gold_labels and pred_labels:
            acc = accuracy_score(gold_labels, pred_labels)
            f1 = f1_score(gold_labels, pred_labels, average='macro')
            print(f'Accuracy: {acc:.4f}')
            print(f'Macro-F1: {f1:.4f}')
            print(classification_report(gold_labels, pred_labels, digits=4))
    except ImportError:
        print('sklearn not installed; skipping accuracy/F1 metrics.')

    # Compute mean Brier score when probability records are available
    valid_probs = [(prob, gold) for prob, gold in prob_records if prob is not None]
    if valid_probs:
        def _brier(prob_dict, gold):
            classes = ('yes', 'no', 'maybe')
            return sum((prob_dict.get(cls, 0.0) - (1.0 if gold == cls else 0.0))**2 for cls in classes)

        mean_brier = sum(_brier(prob, gold) for prob, gold in valid_probs) / len(valid_probs)
        print(f'Mean Brier (0-2 scale over {len(valid_probs)} examples): {mean_brier:.6f}')
    else:
        print('No probability data available to compute Brier score (model/tokenizer pair missing).')

    if rouge_records:
        r1 = sum(s['rouge1'].fmeasure for s in rouge_records) / len(rouge_records)
        rL = sum(s['rougeL'].fmeasure for s in rouge_records) / len(rouge_records)
        print(f'Avg ROUGE-1 F1 (rationale vs long_answer): {r1:.4f}')
        print(f'Avg ROUGE-L F1 (rationale vs long_answer): {rL:.4f}')
    else:
        print('ROUGE metrics not computed for rationales.')

    if readability_scores:
        avg_read = sum(readability_scores) / len(readability_scores)
        print(f'Avg Flesch Reading Ease (rationales): {avg_read:.2f}')
    else:
        print('Readability metrics not computed for rationales.')


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
    p.add_argument('--question-field', default='question', help='Dataset field to use as the question')
    p.add_argument('--context-field', default='context', help='Dataset field to use as the context/abstract')
    p.add_argument('--answer-field', default='final_decision', help='Dataset field to use as the gold answer/label')
    p.add_argument('--long-answer-field', default='long_answer', help='Dataset field containing the explanatory long answer/rationale (optional)')
    # Local transformers options
    p.add_argument('--device', default=None, help='Device to load model on (cuda/mps/cpu). If not set, adapter auto-detects')
    p.add_argument('--load-in-4bit', action='store_true', help='Load model in 4-bit using bitsandbytes (requires CUDA + bitsandbytes)')
    # UnsloTh / long-context options
    p.add_argument('--use-unsloth', action='store_true', help='Use UnsloTh prequantized models (Colab-friendly)')
    p.add_argument('--max-seq-length', type=int, default=8192, help='Max sequence length / context window for UnsloTh or long-context models')
    p.add_argument('--force-finish-format', action='store_true', help='Ask agents to output exactly one Finish[...] action with yes/no/maybe when finishing')
    p.add_argument('--confidence-threshold', type=float, default=0.6, help='Confidence threshold (0..1) to accept yes/no/maybe without further reflexion')
    p.add_argument('--max-reflect-attempts', type=int, default=2, help='Maximum number of reflexion+retry attempts when confidence is low')
    p.add_argument('--print-debug', dest='print_debug', action='store_true', help='Print verbose debug info (default)')
    p.add_argument('--no-print-debug', dest='print_debug', action='store_false', help='Disable verbose debug info')
    p.add_argument('--print-logit-debug', action='store_true', help='Print yes/no/maybe probability scores when evaluating confidence')
    p.set_defaults(print_debug=True, print_logit_debug=False)
    p.add_argument('--keep-fewshot-examples', action='store_true', help='Preserve builtin few-shot examples (otherwise cleared unless dataset contains PubMedQA)')
    args = p.parse_args()

    # normalize limit
    if args.limit == 0:
        args.limit = None

    # Fill use_transformers default: if flag not passed, default to False (inference) unless user passes --use-transformers
    # argparse configured so default is True when --use-transformers present, else False

    # Build boolean correctly
    args.use_transformers = bool(args.use_transformers)

    run(args)
