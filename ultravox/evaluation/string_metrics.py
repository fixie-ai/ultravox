import re
from typing import Any, Dict, List

import jiwer
import sacrebleu

from ultravox.evaluation import eval_types


def wer(samples: List[eval_types.Sample], args: Dict[str, Any]) -> eval_types.WerResult:
    """
    Computes the WER (Word Error Rate) across multiple samples by summing
    all errors and dividing by the total reference words (the standard
    'global' WER).
    """
    transforms = jiwer.Compose(
        [
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveEmptyStrings(),
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )

    references = [sample.expected_answer for sample in samples]
    hypotheses = [sample.generated_answer for sample in samples]

    # jiwer.wer will aggregate errors over the entire collection
    score: float = jiwer.wer(
        references,
        hypotheses,
        truth_transform=transforms,
        hypothesis_transform=transforms,
        **args,
    )

    # Scale by 100 to be comparable to other metrics (e.g. BLEU)
    return eval_types.WerResult(score=score * 100)


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


def bleu(
    samples: List[eval_types.Sample], args: Dict[str, Any]
) -> eval_types.BleuResult:
    """
    Compute corpus BLEU score for a list of samples.
    """
    references = [[sample.expected_answer for sample in samples]]
    hypotheses = [sample.generated_answer for sample in samples]
    score = sacrebleu.corpus_bleu(
        hypotheses=hypotheses, references=references, **args
    ).score
    return eval_types.BleuResult(score=score)
