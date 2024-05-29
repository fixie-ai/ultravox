import glob
from dataclasses import dataclass

import evaluate
import pyrallis
import torch

from .speechlm_inference import SpeechLMInference
from .speechlm_inference import SpeechLMInferenceConfig


@dataclass
class InferenceConfig:
    model: SpeechLMInferenceConfig
    data_path: str = "../../../epd_data"
    threshold: float = None
    prompt: str = (
        "Transcribe speech to text and indicate whether user is done talking or they might continue with [END] or [...]: {audio}"
    )


# @torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False)
@torch.inference_mode
def main():
    cfg = pyrallis.parse(config_class=InferenceConfig)
    infer = SpeechLMInference(cfg.model)

    metrics = [
        evaluate.load("accuracy"),
        evaluate.load("confusion_matrix"),
    ]

    classes = {0: "false", 1: "true"}

    for gt, dirname in classes.items():
        for audio_path in glob.glob(f"{cfg.data_path}/{dirname}/*.wav"):
            inputs = infer.prep_audio(audio_path, cfg.prompt)
            audio_logits = infer.get_audio_logits(inputs)

            if cfg.threshold is None:
                audio_preds = audio_logits.argmax(dim=-1).cpu()
                is_eou = audio_preds == infer.data_prep_fn.eou_token_id
            else:
                eou_logit = audio_logits[..., infer.data_prep_fn.eou_token_id]
                mid_logit = audio_logits[..., infer.data_prep_fn.mid_token_id]
                is_eou = torch.sigmoid(eou_logit - mid_logit)
                # is_eou = audio_logits.softmax(-1)[..., infer.data_prep_fn.eou_token_id]
                is_eou = is_eou > cfg.threshold
                # print(is_eou[-1])
            epd_pred = is_eou[-1].int().item()
            for metric in metrics:
                metric.add(predictions=epd_pred, references=gt)

    print(metrics[0].compute())
    # normalize (str): Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population.
    print("Confusion matrix:")
    print(metrics[1].compute(normalize="true")["confusion_matrix"])


if __name__ == "__main__":
    main()
