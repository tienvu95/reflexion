try:
    from langchain.prompts import PromptTemplate
except Exception:
    try:
        from hotpotqa_runs.prompt_shim import PromptTemplate
    except Exception:
        from prompt_shim import PromptTemplate

COT_INSTRUCTION = """Answer a PubMedQA biomedical question by having a Thought, then Finish with your answer. First decide the label (yes, no, or maybe). After emitting the label, write a short justification prefixed with "Reason:". Thought can reason about the PubMed abstract context and the question. Finish[answer] returns the answer and finishes the task. Always rely on the provided context and restrict the final answer to yes, no, or maybe.
Here are some examples:
{examples}
(END OF EXAMPLES)
{reflections}
Relevant PubMed Context: {context} 
Question: {question}{scratchpad}"""

COT_AGENT_REFLECT_INSTRUCTION = """Answer a PubMedQA biomedical question by having a Thought, then Finish with your answer. First decide the label (yes, no, or maybe) and then provide a short justification prefixed with "Reason:". Thought can reason about the PubMed abstract context and the question. Finish[answer] returns the answer and finishes the task. Always rely on the provided context and restrict the final answer to yes, no, or maybe.
Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Relevant PubMed Context: {context}
Question: {question}{scratchpad}"""

COT_REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self reflection. You will be given a previous PubMedQA reasoning trial in which you read PubMed context and answered a yes/no/maybe question. You were unsuccessful either because you produced the wrong label with Finish[<answer>] or phrased the answer incorrectly. In a few sentences, diagnose a possible reason for failure and devise a new, concise, high level plan grounded in the PubMed evidence that mitigates the same failure. Use complete sentences and remember to restate when the `Reason:` justification should be used.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous trial:
Relevant PubMed Context: {context}
Question: {question}{scratchpad}

Reflection:"""

cot_agent_prompt = PromptTemplate(
                        input_variables=["examples", "reflections", "context", "question", "scratchpad"],
                        template = COT_INSTRUCTION,
                        )

cot_reflect_agent_prompt = PromptTemplate(
                        input_variables=["examples", "reflections", "context", "question", "scratchpad"],
                        template = COT_AGENT_REFLECT_INSTRUCTION,
                        )

cot_reflect_prompt = PromptTemplate(
                        input_variables=["examples", "context", "question", "scratchpad"],
                        template = COT_REFLECT_INSTRUCTION,
                        )

COT_SIMPLE_INSTRUCTION = """Answer a PubMedQA biomedical question by having a Thought, then Finish with your answer. First decide the label (yes, no, or maybe). After writing the label, provide a brief justification starting with "Reason:". Thought can reason about the PubMed abstract context and the question. Finish[answer] returns the answer and finishes the task. Always ground your reasoning in the provided context and respond with yes, no, or maybe.
Here are some examples:
{examples}
(END OF EXAMPLES)
{reflections}
Relevant PubMed Context: {context}
Question: {question}{scratchpad}"""

COT_SIMPLE_AGENT_REFLECT_INSTRUCTION = """Answer a PubMedQA biomedical question by having a Thought, then Finish with your answer. First decide the label (yes, no, or maybe), and after the label provide a short justification starting with "Reason:". Thought can reason about the PubMed abstract context and the question. Finish[answer] returns the answer and finishes the task. Always ground your reasoning in the provided context and respond with yes, no, or maybe.
Here are some examples:
{examples}
(END OF EXAMPLES)
Relevant PubMed Context: {context}
{reflections}

Question: {question}{scratchpad}"""

COT_SIMPLE_REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self reflection. You will be given a previous PubMedQA reasoning trial with a biomedical abstract and a yes/no/maybe question. You were unsuccessful either because you produced the wrong label with Finish[<answer>] or phrased the answer incorrectly. In a few sentences, diagnose the failure and propose a concise plan that explains how to better use the PubMed context to arrive at the correct yes/no/maybe answer, including when to present the `Reason:` justification.
Here are some examples:
{examples}
(END OF EXAMPLES)
Relevant PubMed Context: {context}
Previous trial:
Question: {question}{scratchpad}

Reflection:"""

cot_simple_agent_prompt = PromptTemplate(
                        input_variables=["examples", "question", "reflections", "context", "scratchpad"],
                        template = COT_SIMPLE_INSTRUCTION,
                        )

cot_simple_reflect_agent_prompt = PromptTemplate(
                        input_variables=["examples", "context", "reflections", "question", "scratchpad"],
                        template = COT_SIMPLE_AGENT_REFLECT_INSTRUCTION,
                        )

cot_simple_reflect_prompt = PromptTemplate(
                        input_variables=["examples", "question", "context", "scratchpad"],
                        template = COT_SIMPLE_REFLECT_INSTRUCTION,
                        )


REACT_INSTRUCTION = """Answer a PubMedQA biomedical question with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the provided biomedical docstore (PubMed context or Wikipedia fallback) and returns the first matching passage. If not found, it will return similar entries to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task. When you finish, the answer must be yes, no, or maybe followed by a short justification prefixed with "Reason:".
You may take as many steps as necessary, but always base your reasoning on the retrieved biomedical evidence and finish with yes, no, or maybe, then supply the Reason line.
Here are some examples:
{examples}
(END OF EXAMPLES)
Question: {question}{scratchpad}"""

# Stronger instruction variant that enforces Action formatting strictly
REACT_INSTRUCTION_STRICT = """Answer a PubMedQA biomedical question with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types:
(1) Search[entity], which searches the provided biomedical docstore (PubMed context or Wikipedia fallback) and returns the first matching passage. If not found, it will return similar entries to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task. When you choose Finish, output yes, no, or maybe and immediately add a "Reason:" line that briefly cites the supporting evidence.

IMPORTANT: When you output an Action, OUTPUT EXACTLY one line that begins with `Action:` followed by one of the three action forms above (for example: `Action: Search[Barack Obama]` or `Action: Finish[yes]`). Do not include extra commentary on the same line. If you want to reason, put it under `Thought:` lines only.
Only finish with the labels yes, no, or maybe.

Here are some examples:
{examples}
(END OF EXAMPLES)

Question: {question}{scratchpad}"""

REACT_REFLECT_INSTRUCTION = """Answer a PubMedQA biomedical question with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the provided biomedical docstore (PubMed context or Wikipedia fallback) and returns the first matching passage. If not found, it will return similar entries to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task, then you must add a concise "Reason:" line grounded in the retrieved evidence.
You may take as many steps as necessary, but always base your reasoning on the retrieved biomedical evidence and finish with yes, no, or maybe followed by a Reason line.
Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{scratchpad}"""

REFLECTION_HEADER = 'You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'
REFLECTION_AFTER_LAST_TRIAL_HEADER = 'The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'
LAST_TRIAL_HEADER = 'You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\n'

REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self reflection. You will be given a previous PubMedQA reasoning trial in which you had access to a biomedical docstore (PubMed context or Wikipedia fallback) and a yes/no/maybe question. You were unsuccessful either because you produced the wrong label with Finish[<answer>] or exhausted your reasoning steps. In a few sentences, diagnose a possible reason for failure and devise a new, concise, high level plan grounded in the biomedical evidence that mitigates the same failure. Use complete sentences.  
Here are some examples:
{examples}

Previous trial:
Question: {question}{scratchpad}

Reflection:"""

react_agent_prompt = PromptTemplate(
                        input_variables=["examples", "question", "scratchpad"],
                        template = REACT_INSTRUCTION_STRICT,
                        )

react_reflect_agent_prompt = PromptTemplate(
                        input_variables=["examples", "reflections", "question", "scratchpad"],
                        template = REACT_REFLECT_INSTRUCTION,
                        )

reflect_prompt = PromptTemplate(
                        input_variables=["examples", "question", "scratchpad"],
                        template = REFLECT_INSTRUCTION,
                        )
