from ultravox.evaluation import eval_types
from ultravox.evaluation import gpt_eval

INSTRUCT_SYSTEM_PROMPT = f"""
You are an expert evaluator of AI systems.
Given a question with a specified instruction, you will be rating the correctness of an AI model's ability to follow that instruction.
Based on the supplied answer, and exemplary (correct) answer, you will rate the model's answer as either correct or incorrect.
Award 1 point if the model followed the instruction, and 0 points if it did not.
For example, given a question with an instruction of "Write a sentence about pickleball",
- if the model responds "Pickleball is a tennis-like game played with a wiffle ball.", you should award 1 point.
- if the model responds "Pickleball is a type of fruit", you should award 0 points.
- if the model responds with something off-topic or nonsensical, you should award 0 points.
Your response MUST start with either 0 or 1, followed by a space, and then an explanation for why you awarded that score.
"""
INSTRUCT_USER_PROMPT = """
Using the supplied correct answer as an example, evaluate the model's ability to follow the instructions in the question below:
Question: {{ question }}
Model answer: {{ generated_answer }}
Correct answer: {{ expected_answer }}
"""


def evaluate_answer_instruct(sample: eval_types.Sample) -> eval_types.InstructResult:
    return gpt_eval.gpt_evaluator.evaluate_binary_with_reason(
        INSTRUCT_SYSTEM_PROMPT, INSTRUCT_USER_PROMPT, sample
    )
