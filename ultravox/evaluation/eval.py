from ultravox.evaluation import eval_types
from ultravox.evaluation import gpt_eval
from ultravox.evaluation import string_based
from ultravox.evaluation import wer


def evaluate_answer(sample: eval_types.Sample, metric: str) -> eval_types.Result:
    if metric == "asr":
        return wer.evaluate_answer_asr(sample)
    elif metric == "boolq":
        return gpt_eval.evaluate_answer_boolq(sample)
    elif metric == "instruct":
        return gpt_eval.evaluate_answer_instruct(sample)
    elif metric == "exact_match_last_word":
        return string_based.match_last_word(sample)
    else:
        raise ValueError(f"Unknown metric: {metric}")
