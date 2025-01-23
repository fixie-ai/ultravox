from typing import Any, Callable, Dict, List

from ultravox.data import types
from ultravox.evaluation import eval_types
from ultravox.evaluation import gpt_eval_boolq
from ultravox.evaluation import gpt_eval_conv
from ultravox.evaluation import gpt_eval_instruct
from ultravox.evaluation import string_metrics

METRIC_REGISTRY: Dict[str, Callable[[eval_types.Sample], eval_types.Result]] = {
    "boolq": gpt_eval_boolq.evaluate_answer_boolq,
    "instruct": gpt_eval_instruct.evaluate_answer_instruct,
    "exact_match_last_word": string_metrics.match_last_word,
    "conversation": gpt_eval_conv.evaluate_conversation_response,
}

CORPUS_METRIC_REGISTRY: Dict[
    str, Callable[[List[eval_types.Sample], Dict[str, Any]], eval_types.Result]
] = {
    "bleu": string_metrics.bleu,
    "wer": string_metrics.wer,
}


def evaluate_answer(sample: eval_types.Sample, metric: str) -> eval_types.Result:
    if metric in METRIC_REGISTRY:
        return METRIC_REGISTRY[metric](sample)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def evaluate_answers(
    samples: List[eval_types.Sample], metric_config: types.EvalConfig
) -> eval_types.Result:
    if metric_config.metric in CORPUS_METRIC_REGISTRY:
        metric_func = CORPUS_METRIC_REGISTRY[metric_config.metric]
        return metric_func(samples, metric_config.args)
    else:
        raise ValueError(f"Unknown metric: {metric_config.metric}")
