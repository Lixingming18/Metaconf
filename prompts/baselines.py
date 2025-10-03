## Direct Prompting
DIRECT_PROMPTING_PROMPT = """
Answer the following question and provide a confidence score (0-10) indicating your certainty
in the response. A score closer to 10 indicates higher confidence in the answer’s accuracy.
The output should have the format: ”Answer: ⟨answer⟩, Confidence: ⟨confidence(0-10)⟩”
Question: {QUESTION}
Answer:
"""

## Chain of Thought
CHAIN_OF_THOUGHT_PROMPT = """
Analyze the following question step-by-step, then provide your answer with a confidence score
(0-10) indicating the likelihood your answer is correct. A score closer to 10 indicates higher
confidence in the answer’s accuracy.
The output should have the format: ”Reasoning: ⟨step-by-step analysis⟩, Answer: ⟨answer⟩,
Confidence: ⟨0-10⟩”
Respond only in the specified format with no additional text.
Question: {QUESTION}
Answer:
"""

## Self-Refine
SELF_REFINE_PROMPT_STEP_1 = """
Provide your answer for the following question. Question: {QUESTION}
Answer:
"""

SELF_REFINE_PROMPT_STEP_2 = """
You will be given a question and your own previous answer to it. Your task is to first critically
evaluate your previous answer. Then, based on the critique, provide a refined answer and a
confidence score (0-10) for it.
Your response should follow the format: Self-Critique: ⟨Your critique of the initial answer⟩,
Refined Answer: ⟨Your new, improved answer⟩, Confidence: ⟨Your confidence score (0-10)⟩
Original Question: {QUESTION}
Initial Answer: {INITIAL_ANSWER}
Answer:
"""
