from typing import List
from ultravox.evaluation import eval_types
from ultravox.evaluation import gpt_eval_boolq
from ultravox.evaluation import gpt_eval_conv
from ultravox.evaluation import gpt_eval_instruct
from ultravox.evaluation import string_based
from ultravox.evaluation import wer

METRIC_REGISTRY = {
    "asr": wer.evaluate_answer_asr,
    "boolq": gpt_eval_boolq.evaluate_answer_boolq,
    "instruct": gpt_eval_instruct.evaluate_answer_instruct,
    "conversation": gpt_eval_conv.evaluate_conversation_response,
    "exact_match_last_word": string_based.match_last_word,
    "bleu": string_based.bleu,
    "corpus_bleu": string_based.corpus_bleu,
}


def evaluate_answer(sample: eval_types.Sample, metric: str) -> eval_types.Result:
    if metric in METRIC_REGISTRY:
        return METRIC_REGISTRY[metric](sample)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def evaluate_answers(samples: List[eval_types.Sample], metric_config: eval_types.EvalConfig) -> eval_types.Result:
    if metric_config.metric in METRIC_REGISTRY:
        return METRIC_REGISTRY[metric_config.metric](samples, **metric_config.args)
    else:
        raise ValueError(f"Unknown metric: {metric_config.metric}")