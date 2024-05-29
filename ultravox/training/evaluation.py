import concurrent.futures
import functools
from typing import List, Optional

import numpy as np
from torch.utils import data

from ultravox.data import datasets
from ultravox.evaluation import eval
from ultravox.evaluation import eval_types
from ultravox.inference import base


def dataset_infer(
    inference: base.VoiceInference,
    ds: data.IterableDataset,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> List[eval_types.Sample]:
    eval_samples = []
    # TODO for multiprocessing: ds -> split_batches or sharded reader
    for sample in ds:
        # Store the original question and answer for JSON output.
        question_text = sample.audio_transcript or sample.messages[0]["content"]
        expected_answer = sample.messages[1]["content"]
        # Drop any assistant response from the sample.
        sample.messages = sample.messages[:1]

        output = inference.infer(
            sample, max_tokens=max_new_tokens, temperature=temperature
        )
        eval_sample = eval_types.Sample(
            question=question_text,
            generated_answer=output.text,
            expected_answer=expected_answer,
        )
        eval_samples.append(eval_sample)

    # TODO for multiprocess: gather eval_samples

    return eval_samples


def get_metric_name(ds_name: str, metric: str) -> str:
    if ds_name == "boolq_in" and metric == "asr":
        return "boolq__wer"
    if ds_name == "boolq" and metric == "boolq":
        return "boolq__correctness"
    if metric == "instruct":
        return f"{ds_name}__instruct_follow"
    return f"{ds_name}__{metric}"


def evaluate(
    inference: base.VoiceInference,
    data_dir: Optional[str] = None,
    num_samples: int = 200,
    num_procs: int = 8,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    verbose: bool = False,
):
    metrics = {}

    ds_args = datasets.VoiceDatasetArgs(
        data_dir=data_dir, split=datasets.DatasetSplit.VALIDATION
    )

    for ds_name, metric in [
        ("boolq_in", "asr"),
        ("boolq", "boolq"),
        ("anyinstruct", "instruct"),
    ]:
        ds = datasets.Range(datasets.create_dataset(ds_name, ds_args), num_samples)

        output_samples = dataset_infer(
            inference, ds=ds, max_new_tokens=max_new_tokens, temperature=temperature
        )

        eval_per_sample = functools.partial(eval.evaluate_answer, metric=metric)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_procs) as executor:
            possibly_non_scores = [
                x.score for x in executor.map(eval_per_sample, output_samples)
            ]

        if None in possibly_non_scores:
            print(f"Failed to evaluate {metric} for {ds_name}")
            continue

        scores = [x for x in possibly_non_scores if x is not None]

        if verbose:
            print(f"Eval for {ds_name}:")
            for sample, score in zip(output_samples, scores):
                print("-" * 20)
                print(f"Q: {sample.question}")
                print(f"A: {sample.generated_answer}")
                print(f"X: {sample.expected_answer} [score: {score:.2f}]")

        average = np.mean(scores)
        std = np.std(scores)
        metric_name = get_metric_name(ds_name, metric)
        metrics[f"eval_{metric_name}"] = average
        metrics[f"eval_{metric_name}_std"] = std / np.sqrt(len(scores))

        print(f"Aggregate {metric} score for {ds_name}: {average:.2f} ± {std:.2f}")

    return metrics
