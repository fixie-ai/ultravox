import re

import sacrebleu

from ultravox.evaluation import eval_types
from typing import List



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


def bleu(sample: eval_types.Sample) -> eval_types.BleuResult:
    """
    Compute BLEU score for a single sample.

    Note: BLEU is supposed to be computed on a corpus level, not on a single sample.
    As such, reported values here might not be easily comparable to other metrics.
    """
    score = sacrebleu.sentence_bleu(
        hypothesis=sample.generated_answer,
        references=[sample.expected_answer],
    ).score
    return eval_types.BleuResult(score=score)

def corpus_bleu(samples: List[eval_types.Sample], **kwargs) -> eval_types.BleuResult:
    """
    Compute BLEU score for a list of samples.
    """
    references = [[sample.expected_answer] for sample in samples]
    hypotheses = [sample.generated_answer for sample in samples]
    score = sacrebleu.corpus_bleu(
        hypotheses=hypotheses, references=references, **kwargs).score
    return eval_types.BleuResult(score=score)
