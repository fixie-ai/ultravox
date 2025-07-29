import argparse
import json
import re
from typing import Any, Dict, List

import evaluate
import sacrebleu
import whisper_normalizer.basic as whisper_basic
import whisper_normalizer.english as whisper_english

from ultravox.evaluation import eval_types

# Arabic diacritic marks
arabic_diacritics = re.compile(r"[\u064B-\u065F\u0670]")


def remove_diacritics(text):
    return arabic_diacritics.sub("", text)


def wer(samples: List[eval_types.Sample], args: Dict[str, Any]) -> eval_types.WerResult:
    """Compute WER or CER using Whisper's text normalization."""
    lang_id = args.get("lang_id", "<undefined>").lower()  # Ensure case-insensitive

    # Initialize the appropriate text normalizer
    if lang_id == "en":
        normalizer = whisper_english.EnglishTextNormalizer()
    else:
        normalizer = whisper_basic.BasicTextNormalizer()

    references = [sample.expected_answer for sample in samples]
    hypotheses = [sample.generated_answer for sample in samples]

    if lang_id == "ar":
        references = [remove_diacritics(ref) for ref in references]
        hypotheses = [remove_diacritics(hyp) for hyp in hypotheses]

    # Normalize both reference and hypothesis
    references = [normalizer(ref) for ref in references]
    hypotheses = [normalizer(hyp) for hyp in hypotheses]

    # Languages where we compute CER (space-separated characters)
    if lang_id in ["zh", "ja", "th", "lo", "my"]:
        # Convert to space-separated characters for CER
        references = [" ".join(list(ref)) for ref in references]
        hypotheses = [" ".join(list(hyp)) for hyp in hypotheses]

    # Cap the length of the hypothesis to some multiple of the reference length
    cap_hypothesis_len = args.get("cap_hypothesis_len", None)
    if cap_hypothesis_len is not None:
        hypotheses = [
            hyp[: int(len(ref) * cap_hypothesis_len)]
            for hyp, ref in zip(hypotheses, references)
        ]

    # Handle empty strings
    references = [e if e.strip() else "<silence>" for e in references]
    hypotheses = [s if s.strip() else "<silence>" for s in hypotheses]

    # Compute WER using space-separated words
    wer_metric = evaluate.load("wer")
    wer_score = wer_metric.compute(predictions=hypotheses, references=references)
    return eval_types.WerResult(score=wer_score * 100)


def match_last_word(sample: eval_types.Sample) -> eval_types.ExactMatchResult:
    # Handle the case where generated_answer might be a boolean
    generated_answer_str = (
        str(sample.generated_answer).lower()
        if isinstance(sample.generated_answer, bool)
        else sample.generated_answer.lower()
    )
    last_words = re.findall(r"\b\w+\b(?=\W*$)", generated_answer_str)

    # Handle the case where expected_answer might be a boolean
    expected_answer_str = (
        str(sample.expected_answer).lower()
        if isinstance(sample.expected_answer, bool)
        else sample.expected_answer.lower()
    )
    expected_tf = re.findall(r"\b\w+\b(?=\W*$)", expected_answer_str)[-1]

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


def partial_match(sample: eval_types.Sample) -> eval_types.ExactMatchResult:
    """Compute partial match score where expected answer should be part of generated answer.

    Returns a score of 1 if the expected answer is contained within the generated answer
    (case-insensitive), and 0 otherwise.
    """
    generated = sample.generated_answer.lower().strip()
    expected = sample.expected_answer.lower().strip()

    return eval_types.ExactMatchResult(
        score=int(expected in generated), reason="partial_match check"
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate JSON files using WER and BLEU."
    )
    parser.add_argument("input_file", type=str, help="Path to the input JSON file.")
    parser.add_argument(
        "--metric",
        type=str,
        choices=["wer", "bleu"],
        required=True,
        help="Metric to compute.",
    )
    parser.add_argument(
        "--lang_id", type=str, default="en", help="Language ID (e.g., en, zh, ja)."
    )
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = [eval_types.Sample(**sample) for sample in data]

    if args.metric == "wer":
        result = wer(samples, {"lang_id": args.lang_id})
    else:
        result = bleu(samples, {"tokenize": args.lang_id})

    print(f"{args.metric.upper()} Score: {result.score}")


if __name__ == "__main__":
    main()
