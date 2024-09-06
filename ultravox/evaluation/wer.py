# compute WER comparing a set of references with a set of hypotheses

import jiwer

from ultravox.evaluation import eval_types


def compute_wer(references, hypotheses):
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

    wer = jiwer.wer(
        references,
        hypotheses,
        truth_transform=transforms,
        hypothesis_transform=transforms,
    )

    return wer


def evaluate_answer_asr(sample: eval_types.Sample):
    wer_rate = compute_wer([sample.reference], [sample.hypothesis])
    return eval_types.WerResult(score=min(1.0, wer_rate))
