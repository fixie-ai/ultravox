import concurrent.futures
import dataclasses
import functools
import json
import os
from typing import List, Optional

import numpy as np
from torch.utils import data

from ultravox.data import datasets
from ultravox.evaluation import eval
from ultravox.evaluation import eval_types
from ultravox.inference import infer
from ultravox.training import ddp_utils


def dataset_infer(
    inference: infer.LocalInference,
    ds: data.IterableDataset,
    world_size: int = 1,
    local_rank: int = 0,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> List[eval_types.Sample]:
    eval_samples = []

    for sample in ddp_utils.sharded_iterator(ds, world_size, local_rank):
        # Store the original question and answer for JSON output.
        question_text = sample.audio_transcript or sample.messages[-2]["content"]
        expected_answer = sample.messages[-1]["content"]
        # Drop any assistant response from the sample.
        sample.messages = sample.messages[:-1]
        history = sample.messages[:-2]

        output = inference.infer(
            sample, max_tokens=max_new_tokens, temperature=temperature
        )
        eval_sample = eval_types.Sample(
            question=question_text,
            hypothesis=output.text,
            reference=expected_answer,
            history=history,
        )
        eval_samples.append(eval_sample)

    # Gather all the samples from all the processes.
    return ddp_utils.all_gather_list(eval_samples)


@dataclasses.dataclass
class EvalScenario:
    name: str
    dataset: str
    metric: str
    include_audio: bool = True
    include_context: bool = True
    new_tokens: Optional[int] = None


EVAL_SCENARIOS = [
    # automatic speech recognition scenarios
    EvalScenario("boolq__wer", "boolq_in", "asr"),
    # automatic speech translation scenarios
    EvalScenario("covost2_en_de__bleu", "covost2:en_de", "bleu"),
    EvalScenario("covost2_en_zh-CN__bleu", "covost2:en_zh-CN", "bleu"),
    EvalScenario("covost2_es_en__bleu", "covost2:es_en", "bleu"),
    EvalScenario(
        "covost2_en_de__bleu__text_only", "covost2:en_de", "bleu", include_audio=False
    ),
    EvalScenario(
        "covost2_en_zh-CN__bleu__text_only",
        "covost2:en_zh-CN",
        "bleu",
        include_audio=False,
    ),
    EvalScenario(
        "covost2_es_en__bleu__text_only", "covost2:es_en", "bleu", include_audio=False
    ),
    # SQA scenarios
    EvalScenario("anyinstruct__instruct_follow", "anyinstruct", "instruct"),
    EvalScenario(
        "boolq__binary", "boolq_extended", "exact_match_last_word", new_tokens=128
    ),
    EvalScenario(
        "anyinstruct__instruct_follow__text_only",
        "anyinstruct",
        "instruct",
        include_audio=False,
    ),
    EvalScenario(
        "boolq__binary__text_only",
        "boolq_extended",
        "exact_match_last_word",
        new_tokens=128,
        include_audio=False,
    ),
    # Conversation dialogue scenarios
    EvalScenario("soda__sensible_generation", "soda", "conversation", new_tokens=64),
    EvalScenario(
        "soda__sensible_generation__text_only",
        "soda",
        "conversation",
        new_tokens=64,
        include_audio=False,
    ),
]


def evaluate(
    inference: infer.LocalInference,
    data_dir: Optional[str] = None,
    num_samples: int = 200,
    num_procs: int = 8,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    log_dir: Optional[str] = None,
):
    metrics = {}

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if log_dir:
        log_dir = os.path.join(log_dir, "evals")
        os.makedirs(log_dir, exist_ok=True)

    for task in EVAL_SCENARIOS:
        ds_args = datasets.VoiceDatasetArgs(
            data_dir=data_dir,
            split=datasets.DatasetSplit.VALIDATION,
            include_audio=task.include_audio,
            include_context=task.include_context,
        )

        ds = datasets.Range(datasets.create_dataset(task.dataset, ds_args), num_samples)

        output_samples = dataset_infer(
            inference,
            ds=ds,
            max_new_tokens=task.new_tokens or max_new_tokens,
            temperature=temperature,
            world_size=world_size,
            local_rank=local_rank,
        )

        if local_rank != 0:
            # Only the master process should evaluate the samples.
            continue

        eval_per_sample = functools.partial(eval.evaluate_answer, metric=task.metric)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_procs) as executor:
            possibly_non_scores = [
                x.score for x in executor.map(eval_per_sample, output_samples)
            ]

        if None in possibly_non_scores:
            print(f"Failed to evaluate {task.metric} for {task.dataset}")
            continue

        scores = [x for x in possibly_non_scores if x is not None]

        average = np.mean(scores)
        std = np.std(scores) / np.sqrt(len(scores))
        metrics[f"eval_{task.name}"] = average
        metrics[f"eval_{task.name}_std"] = std

        has_audio_str = "with" if task.include_audio else "without"
        agg_score_str = f"Aggregate {task.metric} score for {task.dataset} ({has_audio_str} audio): {average:.2f} ± {std:.2f}"
        print(agg_score_str)

        if log_dir:
            eval_details = {
                "score": average,
                "confidence_interval": std,
                "task_info": dataclasses.asdict(task),
                "samples": [
                    {**dataclasses.asdict(sample), "score": score}
                    for sample, score in zip(output_samples, scores)
                ],
            }
            with open(os.path.join(log_dir, f"{task.name}.json"), "w") as f:
                json.dump(eval_details, f, indent=1)

    return metrics
