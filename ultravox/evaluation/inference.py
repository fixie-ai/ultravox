import json
import os
from typing import List, Optional

import numpy as np
from torch.utils import data

from ultravox.data import datasets
from ultravox.data import dataset_config
from ultravox.evaluation import eval
from ultravox.evaluation import eval_types
from ultravox.inference import infer
from ultravox.training import ddp_utils

def dataset_infer(
    inference: infer.LocalInference,
    dataset: datasets.VoiceDataset,
    world_size: int = 1,
    local_rank: int = 0,
    batch_size: Optional[int] = 1,
    max_tokens: Optional[int] = None,
    temperature: float = 0.0
) -> List[eval_types.Sample]:
    results = []
    for batch_input in ddp_utils.sharded_batch_iterator(dataset, batch_size, world_size, local_rank):
        batch_indices = [idx for idx, _ in batch_input]
        batch_samples = [sample for _, sample in batch_input]
        batch_references = []
        for sample in batch_samples:
            assistant_message = sample.messages.pop()
            if assistant_message['role'] != 'assistant':
                raise ValueError(f"Expected assistant message but got: role={assistant_message['role']}, content={assistant_message['content']}")
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
                    question=sample.messages[-1]['content'],
                    reference=reference,
                    hypothesis=output.text,
                )
            )
    return results

def run_infer(inference: infer.LocalInference, dataset_configs: List[dataset_config.DatasetConfig], world_size: int, local_rank: int, batch_size: int=1, max_tokens: int=512, temperature: float=0.0, output_dir: Optional[str] = None):
    dataset_args = datasets.VoiceDatasetArgs()
    metrics = {}
    for ds_cfg in dataset_configs:
        dataset = datasets.GenericVoiceDataset(dataset_args, ds_cfg)
        results = dataset_infer(inference, dataset, world_size=world_size, local_rank=local_rank,batch_size=batch_size, max_tokens=max_tokens, temperature=temperature)
        results = ddp_utils.all_gather_list(results)
        
        # compute metrics, if specified, and save results only on the first process
        if local_rank == 0:
            # ensure same order 
            results.sort(key=lambda x: x.index)
            dataset_alias = ds_cfg.alias
            if ds_cfg.eval_config:
                eval_result: eval_types.Result = eval.evaluate_answers(results, ds_cfg.eval_config)
                print(f"Dataset: {dataset_alias}, Metric: {ds_cfg.eval_config.metric}, Score: {eval_result.score:.2f}")
                
                metrics[f"eval/{dataset_alias}-{ds_cfg.eval_config.metric}"] = eval_result.score

            if output_dir:
                output_file = os.path.join(output_dir, f"{dataset_alias}.json")
                with open(output_file, "w") as f:   
                    results_json = [result.to_dict() for result in results]
                    json.dump(results_json, f, ensure_ascii=False,indent=2)
                print(f"Results saved to {output_file}")
    
    return metrics