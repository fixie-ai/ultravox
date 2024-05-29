import datetime
import glob
import json
import os
from dataclasses import dataclass
from dataclasses import field

import datasets
import evaluate
import jiwer
import safetensors.torch
import torch
import torch.utils.data
import torchaudio
import transformers
from train import data
from train.models import multimodal as multimodal_models


@dataclass
class SpeechLMInferenceConfig:
    path: str
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float32
    add_audio_tag_ratio: float = 1.0
    freezing_config: multimodal_models.FreezingConfig = field(
        default_factory=multimodal_models.FreezingConfig
    )


class SpeechLMInference:
    def __init__(self, config: SpeechLMInferenceConfig):
        self.config = config

        model_config = multimodal_models.SpeechLMConfig.from_pretrained(config.path)
        model_config.torch_dtype = config.dtype
        self.model = multimodal_models.SpeechLM(model_config)
        self.model.apply_lora_configs(config.freezing_config)
        processor = multimodal_models.SpeechLMProcessor.from_config(self.model.config)
        self.processor = processor

        for path in glob.glob(os.path.join(str(config.path), "model*.safetensors")):
            state_dict = safetensors.torch.load_file(path)
            mismatch = self.model.load_state_dict(state_dict, strict=False)
            if mismatch.unexpected_keys:
                raise ValueError(
                    f"Unexpected keys in state dict: {mismatch.unexpected_keys}"
                )

        self.model = self.model.to(config.device)

        self.tokenizer = processor.tokenizer
        audio_proc = processor.audio_processor
        total_audio_stride = processor.total_audio_stride

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.model.llm.config.use_cache = True
        processor.tokenizer.pad_token_id = 0
        processor.tokenizer.padding_side = "left"

        self.data_prep_fn = data.AudioTextTokenizer(
            audio_proc,
            self.tokenizer,
            total_audio_stride,
            cfg=data.AudioTextTokenizerConfig(
                inference_mode=True,
                add_audio_tag_ratio=config.add_audio_tag_ratio,
            ),
        )

        self.data_collator = data.DataCollatorForSeq2SeqWithAudio(
            self.tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
            audio_dtype=self.config.dtype,
        )

    def get_audio_logits(self, batch_of_1):
        out = self.model.forward(**batch_of_1)
        start = batch_of_1["audio_token_start_idx"].item()
        alen = batch_of_1["audio_token_len"].item()
        audio_logits = out.logits[0, start : start + alen]
        return audio_logits

    def prep_audio(self, audio_path, prompt):
        target_sr = 16_000
        array, sr = torchaudio.load(audio_path, normalize=True)
        array = torchaudio.functional.resample(array, sr, target_sr).sum(0)
        audio = {"array": array, "sampling_rate": target_sr}
        audio_only_input = self.data_prep_fn({"audio": audio, "prompt": prompt})
        audio_only_input.pop("prompt", None)
        audio_only_input.pop("labels")
        audio_only_input.pop("audio", None)
        audio_only_input.pop("text", "")
        audio_only_inputs = self.data_collator([audio_only_input])
        audio_only_inputs = {
            k: v.to(device=self.config.device) for k, v in audio_only_inputs.items()
        }
        return audio_only_inputs

    # @torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False)
    # @torch.inference_mode
    # def batch_transcribe(self, sample_audios, num_beams=4, temp=0.7):
    #     # prep and collate, but only the audio, not text/prompt, etc.
    #     audio_only_inputs = self.data_collator(
    #         [self.data_prep_fn({"audio": s}) for s in sample_audios]
    #     )

    #     audio_only_inputs = {
    #         k: v.to(device=self.config.device) for k, v in audio_only_inputs.items()
    #     }

    #     ## pass 1: get the transcript
    #     tokens = self.model.generate(
    #         **audio_only_inputs,
    #         max_new_tokens=100,
    #         num_beams=num_beams,
    #         do_sample=True,
    #         temperature=temp,
    #         top_k=10,
    #         top_p=0.95,
    #     )

    #     texts = [self.tokenizer.decode(t) for t in tokens.tolist()]

    #     return texts

    @torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False)
    @torch.inference_mode
    def evaluate(self, ds: datasets.Dataset, batch_size=1, num_beams=4):
        ds = ds.map(self.data_prep_fn)

        trainer = transformers.Seq2SeqTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=transformers.Seq2SeqTrainingArguments(
                output_dir="runs/eval",
                predict_with_generate=True,
                report_to=[],
                dataloader_num_workers=0,
                per_device_eval_batch_size=batch_size,
            ),
            eval_dataset=ds,
            data_collator=self.data_collator,
            compute_metrics=ComputeMetrics(self.tokenizer),
        )

        # sample = next(iter(ds))
        # collator = trainer._get_collator_with_removed_columns(self.data_collator)
        # batch = collator([sample])
        # batch = {k: v.to(device=self.config.device) for k, v in batch.items()}
        # out = self.model(**batch)

        print(
            trainer.evaluate(
                max_new_tokens=64,
                num_beams=num_beams,
                top_k=10,
                top_p=0.95,
                do_sample=True,
            )
        )


@dataclass
class ComputeMetrics:
    tokenizer: transformers.LlamaTokenizer
    preamble: str = "Transcript:"

    def _remove_prefix_suffix(self, text: str):
        text = text.split(self.preamble, maxsplit=1)[-1]
        text = text.rsplit(self.tokenizer.eos_token, maxsplit=1)[0]
        return text.strip()

    def __post_init__(self):
        self.cer_metric = evaluate.load("cer")
        self.wer_metric = evaluate.load("wer")

    def __call__(self, eval_preds: transformers.EvalPrediction):
        targets = eval_preds.label_ids
        logits = eval_preds.predictions

        ref_strs = []
        pred_strs = []

        for logit, target in zip(logits, targets):
            ref_ids = target[target > 0]
            pred_ids = logit[logit > 0]
            # CER complains if the string is empty
            ref_strs.append(self._remove_prefix_suffix(self.tokenizer.decode(ref_ids)))
            pred_strs.append(
                self._remove_prefix_suffix(self.tokenizer.decode(pred_ids))
            )

        base_transforms = jiwer.Compose(
            [
                jiwer.RemovePunctuation(),
                jiwer.ToLowerCase(),
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.RemoveKaldiNonWords(),
                jiwer.RemoveWhiteSpace(replace_by_space=True),
            ]
        )
        pred_strs_clean = base_transforms(pred_strs)
        ref_strs_clean = base_transforms(ref_strs)

        eval_path = "eval/results/"
        os.makedirs(eval_path, exist_ok=True)

        file_name = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + ".json"

        with open(os.path.join(eval_path, file_name), "w") as res_file:
            json.dump(
                [
                    {"ref": r, "pred": p, "ref_trans": rt, "pred_trans": pt}
                    for r, p, rt, pt in zip(
                        ref_strs, pred_strs, ref_strs_clean, pred_strs_clean
                    )
                ],
                res_file,
                indent=2,
            )

        return {
            "wer": jiwer.wer(ref_strs_clean, pred_strs_clean),
            "cer": jiwer.cer(ref_strs_clean, pred_strs_clean),
        }
