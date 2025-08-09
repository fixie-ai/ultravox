from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Dict, List

from ultravox.data import types
from ultravox.evaluation import eval_types
from ultravox.evaluation import gpt_eval_audiobench
from ultravox.evaluation import gpt_eval_bigbench
from ultravox.evaluation import gpt_eval_boolq
from ultravox.evaluation import gpt_eval_conv
from ultravox.evaluation import gpt_eval_instruct
from ultravox.evaluation import gpt_eval_voicebench
from ultravox.evaluation import ifeval_voicebench
from ultravox.evaluation import string_metrics

METRIC_REGISTRY: Dict[str, Callable[[eval_types.Sample], eval_types.Result]] = {
    "boolq": gpt_eval_boolq.evaluate_answer_boolq,
    "instruct": gpt_eval_instruct.evaluate_answer_instruct,
    "conversation": gpt_eval_conv.evaluate_conversation_response,
    "bigbench": gpt_eval_bigbench.evaluate_answer_bigbench,
    "audiobench_binary": gpt_eval_audiobench.evaluate_answer_audiobench_binary,
    "audiobench_scalar": gpt_eval_audiobench.evaluate_answer_audiobench,
    "exact_match_last_word": string_metrics.match_last_word,
    "partial_match": string_metrics.partial_match,
    "voicebench_yes_no": gpt_eval_voicebench.evaluate_yes_no_voicebench,
    "voicebench_scalar": gpt_eval_voicebench.evaluate_answer_voicebench,
    "voicebench_mcq": gpt_eval_voicebench.evaluate_mcq_voicebench,
    "voicebench_bbh": gpt_eval_voicebench.evaluate_bbh_voicebench,
    "voicebench_harm": gpt_eval_voicebench.evaluate_harm_voicebench,
    "voicebench_ifeval": ifeval_voicebench.IFEvaluator.instruction_following_evaluate,
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
    elif metric_config.metric in METRIC_REGISTRY:
        metric_fn = METRIC_REGISTRY[metric_config.metric]
        partial_metric_fn = partial(metric_fn, **metric_config.args)
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(partial_metric_fn, samples))

        total_score = sum(result.score for result in results)
        return eval_types.MeanResult(score=total_score / len(samples))
    else:
        raise ValueError(f"Unknown metric: {metric_config.metric}")
