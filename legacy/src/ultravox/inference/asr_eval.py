from dataclasses import dataclass

import datasets
import pyrallis
from train import data

from .speechlm_inference import SpeechLMInference
from .speechlm_inference import SpeechLMInferenceConfig


@dataclass
class InferenceConfig:
    model_path: str = "runs/tinyllama-hubertL-LS-10Hz-bs1"
    dataset_streaming: bool = False
    dataset_path: str = None
    num_samples: int = 200
    batch_size: int = 1
    num_beams: int = 4


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=InferenceConfig)
    infer = SpeechLMInference(SpeechLMInferenceConfig(path=cfg.model_path))

    if cfg.dataset_path:
        ds = datasets.load_dataset("audiofolder", data_dir=cfg.dataset_path)["train"]
        infer.evaluate(ds, batch_size=cfg.batch_size, num_beams=cfg.num_beams)
    else:
        dataset_types = [
            data.DatasetType.LIBRISPEECH,
            data.DatasetType.COMMON_VOICE,
        ]
        for ds_type in dataset_types:
            ds = data.get_dataset_split(
                dataset_name=ds_type,
                train=False,
                streaming=cfg.dataset_streaming,
                shuffle=True,
                max_num_samples=cfg.num_samples,
                max_duration_in_seconds=30,
            )
            print(f"{ds.info.dataset_name} validation set ({cfg.num_samples} samples):")
            infer.evaluate(ds, batch_size=cfg.batch_size, num_beams=cfg.num_beams)
