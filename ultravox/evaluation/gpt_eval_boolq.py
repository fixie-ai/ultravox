from ultravox.evaluation import eval_types
from ultravox.evaluation import gpt_eval

BOOLQ_SYSTEM_PROMPT = f"""
You are an expert evaluator of AI systems.
Given a question with a known true/false answer, you will be rating the correctness of an AI model's answer to that same question.
Based on the supplied question, answer, and expected (correct) answer, you will rate the model's answer as either correct or incorrect.
Award 1 point if the model's answer matches the correct answer, and 0 points if the model's answer does not match, or cannot be converted to a true/false verdict.
Model answers of the form "True", "Yes", "Yeah", etc., should be considered to match a True answer.
Model answers of the form "False", "No", "Incorrect", etc., should be considered to match a False answer.
Only use the supplied correct answer to make your decision; DO NOT use your own knowledge to determine correctness.
Your response MUST start with either 0 or 1, followed by a space, and then a brief explanation for why you awarded that score.
"""
BOOLQ_USER_PROMPT = """
Using the supplied correct answer as ground truth, evaluate the model's answer to the question below:
Question: {{ question }}
Model answer: {{ generated_answer }}
Correct answer: {{ expected_answer }}
"""


def evaluate_answer_boolq(sample: eval_types.Sample) -> eval_types.InstructResult:
    return gpt_eval.gpt_evaluator.evaluate_binary_with_reason(
        BOOLQ_SYSTEM_PROMPT, BOOLQ_USER_PROMPT, sample
    )
