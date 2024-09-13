<<<<<<< HEAD
import json
import logging
import os
from typing import List, Optional

from tqdm import tqdm

from ultravox.data import datasets
from ultravox.evaluation import eval_helpers
=======
from typing import List

>>>>>>> d4ef647 (Update)
from ultravox.evaluation import eval_types
<<<<<<< HEAD
from ultravox.inference import infer
from ultravox.training import ddp_utils
=======
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
}
>>>>>>> ca4e476 (Update)


<<<<<<< HEAD
def dataset_infer(
    inference: infer.LocalInference,
    dataset: datasets.SizedIterableDataset,
    world_size: int = 1,
    local_rank: int = 0,
    batch_size: int = 1,
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
) -> List[eval_types.Sample]:
    results: List[eval_types.Sample] = []
    if local_rank == 0:
        total_batches = len(dataset) // (batch_size * world_size)
        progress_bar = tqdm(total=total_batches)
    for batch_input in ddp_utils.sharded_batch_iterator(
        dataset, batch_size, world_size, local_rank
    ):
        batch_indices = [idx for idx, _ in batch_input]
        batch_samples = [sample for _, sample in batch_input]
        batch_references = []
        for sample in batch_samples:
            assistant_message = sample.messages.pop()
            if assistant_message["role"] != "assistant":
                raise ValueError(
                    f"Expected assistant message but got: role={assistant_message['role']}, content={assistant_message['content']}"
                )
            batch_references.append(assistant_message["content"])
        batch_output = inference.infer_batch(
            batch_samples,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for index, sample, output, reference in zip(
            batch_indices, batch_samples, batch_output, batch_references
        ):
            results.append(
                eval_types.Sample(
                    index=index,
                    question=sample.messages[-1]["content"],
                    reference=reference,
                    hypothesis=output.text,
                )
            )

        if local_rank == 0:
            progress_bar.update(1)
    if local_rank == 0:
        progress_bar.close()
    return results


def run_infer(
    inference: infer.LocalInference,
    dataset_args: datasets.VoiceDatasetArgs,
    dataset_configs: List[datasets.DatasetConfig],
    world_size: int,
    local_rank: int,
    output_dir: Optional[str] = None,
):
    metrics = {}
    output_files = []
    for dataset_config in dataset_configs:
        logging.info(f"Evaluating dataset: {dataset_config.alias}")
        dataset = datasets.create_dataset(dataset_args, dataset_config)
        results = dataset_infer(
            inference,
            dataset,
            world_size=world_size,
            local_rank=local_rank,
            batch_size=dataset_args.batch_size,
            max_tokens=dataset_args.max_tokens,
            temperature=dataset_args.temperature,
        )
        results = ddp_utils.all_gather_list(results)

        # compute metrics, if specified, and save results only on the first process
        if local_rank == 0:
            # ensure same order
            results.sort(key=lambda x: x.index)
            dataset_alias = dataset_config.alias
            if dataset_config.eval_config:
                eval_result: eval_types.Result = eval_helpers.evaluate_answers(
                    results, dataset_config.eval_config
                )
                print(
                    f"Dataset: {dataset_alias}, Metric: {dataset_config.eval_config.metric}, Score: {eval_result.score:.2f}"
                )

                metrics[f"eval_{dataset_alias}-{dataset_config.eval_config.metric}"] = (
                    eval_result.score
                )

            if output_dir:
                output_file = os.path.join(output_dir, f"{dataset_alias}.json")
                with open(output_file, "w") as f:
                    results_json = [result.to_dict() for result in results]
                    json.dump(results_json, f, ensure_ascii=False, indent=2)
                print(f"Results saved to {output_file}")
                output_files.append(output_file)
    return metrics, output_files
=======
def evaluate_answer(sample: eval_types.Sample, metric: str) -> eval_types.Result:
    if metric in METRIC_REGISTRY:
        return METRIC_REGISTRY[metric](sample)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def evaluate_answers(
    samples: List[eval_types.Sample], metric_config: eval_types.EvalConfig
) -> eval_types.Result:
    if metric_config.metric in METRIC_REGISTRY:
        return METRIC_REGISTRY[metric_config.metric](samples, **metric_config.args)
    else:
        raise ValueError(f"Unknown metric: {metric_config.metric}")
>>>>>>> d4ef647 (Update)
