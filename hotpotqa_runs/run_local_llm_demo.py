import os
"""Demo: run CoTAgent using the Hugging Face Inference API.

Set environment variables:
  - HF_API_TOKEN: your Hugging Face inference API token
  - HF_MODEL_ID: e.g. meta-llama/Llama-3.1-8b-instruct

Then run from the repository root or from `hotpotqa_runs/`:
  python run_local_llm_demo.py
"""

try:
    # prefer package import when run from repo root
    from hotpotqa_runs.hf_inference_llm import HFInferenceLLM
except Exception:
    from hf_inference_llm import HFInferenceLLM

# Local transformers adapter is optional and loaded only when requested.
USE_TRANSFORMERS = os.environ.get('USE_TRANSFORMERS', '') in ['1', 'true', 'True']
if USE_TRANSFORMERS:
    try:
        from hotpotqa_runs.hf_transformers_llm import HFTransformersLLM
    except Exception:
        try:
            from hf_transformers_llm import HFTransformersLLM
        except Exception:
            HFTransformersLLM = None

try:
    from hotpotqa_runs.agents import CoTAgent, ReflexionStrategy
except Exception:
    from agents import CoTAgent, ReflexionStrategy


def main():
    # If user wants to use a local transformers model, set USE_TRANSFORMERS=1 and HF_LOCAL_MODEL
    if USE_TRANSFORMERS:
        if HFTransformersLLM is None:
            print('HFTransformersLLM adapter not available. Ensure `hotpotqa_runs/hf_transformers_llm.py` exists and dependencies are installed.')
            return
        local_model_id = os.environ.get('HF_LOCAL_MODEL', os.environ.get('HF_MODEL_ID'))
        if not local_model_id:
            print('Set HF_LOCAL_MODEL or HF_MODEL_ID to the local model id you want to load.')
            return

        try:
            # Pass `device=None` when HF_DEVICE is not set so adapter auto-detects
            device_env = os.environ.get('HF_DEVICE')
            llm = HFTransformersLLM(model_id=local_model_id, device=device_env, load_in_4bit=os.environ.get('HF_LOAD_4BIT','1') in ['1','true','True'])
        except Exception as e:
            print('Failed to construct HFTransformersLLM:', e)
            return

    else:
        hf_token = os.environ.get('HF_API_TOKEN')
        model_id = os.environ.get('HF_MODEL_ID', 'meta-llama/Llama-3.1-8b-instruct')
        if not hf_token:
            print('HF_API_TOKEN not set. Please export HF_API_TOKEN and HF_MODEL_ID to use the Hugging Face Inference API or set USE_TRANSFORMERS=1 to load models locally.')
            return

        llm = HFInferenceLLM(model_id=model_id, api_token=hf_token, temperature=0.0, max_new_tokens=256)

    agent = CoTAgent(
        question="Is aspirin recommended for reducing fever?",
        context="Abstract: ...",
        key="yes",
        self_reflect_llm=llm,
        action_llm=llm,
    )

    # Run one trial (run will call LLM when needed). This demo does not provide real context.
    try:
        agent.run(reflexion_strategy=ReflexionStrategy.REFLEXION)
    except Exception as e:
        print('Agent run failed:', e)
    print('Pred:', agent.answer)


if __name__ == '__main__':
    main()