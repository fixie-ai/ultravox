from typing import List

from ultravox.data import types
from ultravox.evaluation import eval_types
from ultravox.evaluation import gpt_eval_boolq
from ultravox.evaluation import gpt_eval_conv
from ultravox.evaluation import gpt_eval_instruct
from ultravox.evaluation import string_metrics

METRIC_REGISTRY = {
    "asr": string_metrics.wer,
    "boolq": gpt_eval_boolq.evaluate_answer_boolq,
    "instruct": gpt_eval_instruct.evaluate_answer_instruct,
    "conversation": gpt_eval_conv.evaluate_conversation_response,
    "exact_match_last_word": string_metrics.match_last_word,
}

CORPUS_METRIC_REGISTRY = {"bleu": string_metrics.bleu}


def evaluate_answer(sample: eval_types.Sample, metric: str) -> eval_types.Result:
    if metric in METRIC_REGISTRY:
        return METRIC_REGISTRY[metric](sample)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def evaluate_answers(
    samples: List[eval_types.Sample], metric_config: types.EvalConfig
) -> eval_types.Result:
    if metric_config.metric in CORPUS_METRIC_REGISTRY:
        return CORPUS_METRIC_REGISTRY[metric_config.metric](
            samples, **metric_config.args
        )
    else:
        raise ValueError(f"Unknown metric: {metric_config.metric}")
