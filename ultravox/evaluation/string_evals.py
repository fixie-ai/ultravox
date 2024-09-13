import re
from typing import List

import jiwer
import sacrebleu

from ultravox.evaluation import eval_types

def wer(sample: eval_types.Sample):
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

    score = jiwer.wer(
        [sample.reference],
        [sample.hypothesis],
        truth_transform=transforms,
        hypothesis_transform=transforms,
    )

    return eval_types.WerResult(score=min(1.0, score))


def match_last_word(sample: eval_types.Sample) -> eval_types.ExactMatchResult:
    last_words = re.findall(r"\b\w+\b(?=\W*$)", sample.hypothesis.lower())
    expected_tf = re.findall(r"\b\w+\b(?=\W*$)", sample.reference.lower())[-1]

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


def bleu(samples: List[eval_types.Sample], **kwargs) -> eval_types.BleuResult:
    """
    Compute corpus BLEU score for a list of samples.
    """
    references = [[sample.reference for sample in samples]]
    hypotheses = [sample.hypothesis for sample in samples]
    score = sacrebleu.corpus_bleu(
        hypotheses=hypotheses, references=references, **kwargs
    ).score
    return eval_types.BleuResult(score=score)
