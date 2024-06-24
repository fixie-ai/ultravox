import re

from ultravox.evaluation import eval_types


def match_last_word(sample: eval_types.Sample) -> eval_types.ExactMatchResult:
    last_words = re.findall(r"\b\w+\b(?=\W*$)", sample.generated_answer.lower())
    expected_tf = re.findall(r"\b\w+\b(?=\W*$)", sample.expected_answer.lower())[-1]

    if not last_words:
        return eval_types.ExactMatchResult(score=0, reason="No last word found")

    last_word: str = last_words[-1]
    if last_word in ["yes", "true"]:
        last_word = "true"
    elif last_word in ["no", "false"]:
        last_word = "false"
    else:
        return eval_types.ExactMatchResult(score=0, reason="Last word not true/false")

    return eval_types.ExactMatchResult(
        score=last_word == expected_tf, reason="exact_match check"
    )
