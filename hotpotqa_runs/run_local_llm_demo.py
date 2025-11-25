import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from hf_llm import AnyHFLLM
from agents import CoTAgent, ReflexionStrategy

# load model (same as notebook)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length = 8192,
    load_in_4bit = True,
)
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
FastLanguageModel.for_inference(model)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

hf_llm = AnyHFLLM(model=model, tokenizer=tokenizer, temperature=0.0, max_tokens=128, device=DEVICE)

agent = CoTAgent(
    question="Is aspirin recommended for reducing fever?",
    context="Abstract: ...",  # put a small abstract or context
    key="yes",
    self_reflect_llm=hf_llm,
    action_llm=hf_llm,
)

agent.run(reflexion_strategy=ReflexionStrategy.REFLEXION)
print('Pred:', agent.answer)