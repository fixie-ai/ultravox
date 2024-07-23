import concurrent.futures
import dataclasses
import functools
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
            generated_answer=output.text,
            expected_answer=expected_answer,
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
    EvalScenario("covost2_en_fa__bleu", "covost2:en_fa", "bleu"),
    EvalScenario(
        "covost2_en_fa__bleu__text_only", "covost2:en_fa", "bleu", include_audio=False
    ),
    # EvalScenario("anyinstruct__instruct_follow", "anyinstruct", "instruct"),
    # EvalScenario(
    #     "boolq__binary", "boolq_extended", "exact_match_last_word", new_tokens=128
    # ),
    # EvalScenario("boolq__wer", "boolq_in", "asr"),
    # EvalScenario("soda__sensible_generation", "soda", "conversation", new_tokens=64),
    # # Text-only scenarios: tests for catastrophic forgetting.
    # EvalScenario(
    #     "anyinstruct__instruct_follow__text_only",
    #     "anyinstruct",
    #     "instruct",
    #     include_audio=False,
    # ),
    # EvalScenario(
    #     "boolq__binary__text_only",
    #     "boolq_extended",
    #     "exact_match_last_word",
    #     new_tokens=128,
    #     include_audio=False,
    # ),
    # EvalScenario(
    #     "soda__sensible_generation__text_only",
    #     "soda",
    #     "conversation",
    #     new_tokens=64,
    #     include_audio=False,
    # ),
]


def evaluate(
    inference: infer.LocalInference,
    data_dir: Optional[str] = None,
    num_samples: int = 200,
    num_procs: int = 8,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    verbose: bool = False,
):
    metrics = {}

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

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

        if verbose:
            print(f"Eval for {task.dataset}:")
            for sample, score in zip(output_samples, scores):
                print("-" * 20)
                print(f"Q: {sample.question}")
                print(f"A: {sample.generated_answer}")
                print(f"X: {sample.expected_answer} [score: {score:.2f}]")

        average = np.mean(scores)
        std = np.std(scores) / np.sqrt(len(scores))
        metrics[f"eval_{task.name}"] = average
        metrics[f"eval_{task.name}_std"] = std

        print(
            f"Aggregate {task.metric} score for {task.dataset}: {average:.2f} ± {std:.2f}"
        )

    return metrics
