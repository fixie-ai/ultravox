from ultravox.evaluation import eval_types
from ultravox.evaluation import gpt_eval

AUDIOBENCH_SYSTEM_PROMPT = """
You are a helpful assistant.
"""

AUDIOBENCH_USER_PROMPT = """\
[Reference Answer]
{{ expected_answer }}

[Model Answer]
{{ generated_answer}}

[Question]
{{ transcript }}

[Task]
Rate the model's answer based on its alignment with the reference answer, focusing on accuracy and relevance to the reference provided. Please be critical on the details. If the model response is something like 'cannot decide', please rate as 0.
Criteria: Assess if the model's response mirrors the reference in terms of content, accuracy, and relevance.
Score0: The answer is refusing to give concrete results, providing something like 'cannot decide'.
Score0: The answer is completely misaligned, providing incorrect or irrelevant information compared to the reference.
Score1: The answer shows minimal alignment, often misunderstanding or providing irrelevant details unrelated to the reference.
Score2: The answer recognizes the topic but diverges significantly from the reference in accuracy or relevance.
Score3: The answer aligns with the reference generally but lacks detail or precise accuracy in some aspects.
Score4: The answer is mostly accurate and relevant, closely following the reference but could be clearer or more detailed.
Score5: The answer is highly accurate, detailed, and matches the reference answer perfectly, capturing its essence and detail.

Your response should be formatted as follows:
Explanation: (Provide a concise explanation of your rating, comparing the reference answer with the model's response. "The reference answer is [XXX], while the model's answer is [YYY]. I think ...")
Rating: (int)
"""


AUDIOBENCH_USER_PROMPT_BINARY = """\
[Reference Answer]
{{ expected_answer }}

[Model Answer]
{{ generated_answer}}

[Question]
{{ transcript }}

[Task]
Rate the model's answer based on its alignment with the reference answer, focusing on accuracy and relevance to the reference provided. Please be critical on the details.
Criteria: Assess if the model's response mirrors the reference in terms of content, accuracy, and relevance. Please give a score of 0 or 1. 
Score0: The answer is refusing to give concrete results, providing something like 'cannot decide'.
Score0: The answer is wrong, providing incorrect or irrelevant information compared to the reference. 
Score1: The answer is correct, capturing or covering the meaning from the reference.

Your response should be formatted as follows:
Explanation: (Provide a concise explanation of your rating, comparing the reference answer with the model's response. "The reference answer is [XXX], while the model's answer is [YYY]. I think ...")
Rating: (int)
"""


def evaluate_answer_audiobench(sample: eval_types.Sample) -> eval_types.InstructResult:
    return gpt_eval.gpt_evaluator.evaluate_score_scalar(
        AUDIOBENCH_SYSTEM_PROMPT, AUDIOBENCH_USER_PROMPT, sample
    )


def evaluate_answer_audiobench_binary(
    sample: eval_types.Sample,
) -> eval_types.InstructResult:
    return gpt_eval.gpt_evaluator.evaluate_binary_with_reason(
        AUDIOBENCH_SYSTEM_PROMPT,
        AUDIOBENCH_USER_PROMPT_BINARY,
        sample,
        rating_at_end=True,
    )
