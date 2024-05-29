import time
from dataclasses import dataclass

import gradio as gr
import pyrallis
import torch
from inference import speechlm_inference


@dataclass
class DemoConfig:
    model: speechlm_inference.SpeechLMInferenceConfig
    default_prompt: str = (
        "Transcribe speech to text and indicate whether user is done talking or they might continue with [END] or [...]: {audio}"
    )
    default_num_beams: int = 4
    default_temp: float = 0.1


class SpeechLMInferenceWithContinuation(speechlm_inference.SpeechLMInference):
    def direct_token_equivalent(self, audio_features):
        # 1 x T x C
        audio_embed = self.model.forward_audio(audio_features)
        # 32000 x C  ->  32000 x 1 x C
        text_embedding_map = self.model.embed_tokens.weight.unsqueeze(1)

        closest_token_ids = [
            torch.nn.functional.cosine_similarity(
                audio_embed[:, idx : idx + 1],
                text_embedding_map,
                dim=-1,
            )
            .argmax(dim=0)
            .item()
            for idx in range(audio_embed.shape[1])
        ]
        return self.tokenizer.decode(closest_token_ids)

    # @torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False)
    @torch.inference_mode
    def transcribe(self, audio_path, prompt, num_beams=4, temp=0.7):
        stats = {}
        start = time.time()

        audio_only_inputs = self.prep_audio(audio_path, prompt)

        stats["audio_prep_time"] = round(time.time() - start, 2)

        stats["audio_direct_token_equivalent"] = self.direct_token_equivalent(
            audio_only_inputs["audio_features"]
        )

        ## pass 0: endpoint detection
        eou_pred_time = time.time()
        audio_logits = self.get_audio_logits(audio_only_inputs)
        audio_preds = audio_logits.argmax(dim=-1).cpu()
        is_eou = audio_preds == self.data_prep_fn.eou_token_id
        stats["epd_early"] = is_eou[-1].item()
        eou_probs = audio_logits.softmax(dim=-1)[..., self.data_prep_fn.eou_token_id]
        stats["epd_early_last_frames"] = self.tokenizer.decode(audio_preds[-20:])
        stats["epd_early_last_probs"] = [
            round(x, 2) for x in eou_probs[-2:].cpu().tolist()
        ]
        if is_eou.any():
            first_eou = is_eou.int().argmax().item()
            stats["first_epd_early_timestamp"] = (
                first_eou * self.processor.total_audio_stride / 16_000
            )

        stats["epd_early_pred_time"] = round(time.time() - eou_pred_time, 2)

        transcript_start = time.time()
        ## pass 1: get the transcript
        tokens = self.model.generate(
            **audio_only_inputs,
            max_new_tokens=100,
            num_beams=num_beams,
            do_sample=True,
            temperature=temp,
            top_k=10,
            top_p=0.95,
        )
        text_res = self.tokenizer.decode(tokens.tolist()[0][:-1])
        transcript = text_res.rsplit("<|assistant|>", 1)[-1]
        epd_late_end = "[END]" in transcript
        epd_late_mid = "[...]" in transcript
        stats["epd_late"] = (
            True if epd_late_end else False if epd_late_mid else "unknown"
        )
        stats["transcription_time"] = round(time.time() - transcript_start, 2)
        stats["num_transcript_tokens"] = len(tokens[0]) - len(
            audio_only_inputs["input_ids"][0]
        )
        stats["transcript_tps"] = round(
            stats["num_transcript_tokens"] / stats["transcription_time"]
        )
        generation_start = time.time()

        ## pass 2: generate the response by forcing response to be generated
        text = text_res + " Response:"
        # text = text_res

        updated_input = self.tokenizer([text], return_tensors="pt")
        audio_only_inputs = {**audio_only_inputs, **updated_input}
        audio_only_inputs = {
            k: v.to(device=self.config.device) for k, v in audio_only_inputs.items()
        }

        tokens = self.model.generate(
            **audio_only_inputs,
            max_new_tokens=100,
            num_beams=num_beams,
            do_sample=True,
            temperature=temp,
            top_k=10,
            top_p=0.95,
        )
        text = self.tokenizer.decode(tokens.tolist()[0])
        text = text.rsplit("<|assistant|>", 1)[-1].strip()
        stats["generation_time"] = round(time.time() - generation_start, 2)
        stats["num_generation_tokens"] = len(tokens[0]) - len(
            audio_only_inputs["input_ids"][0]
        )
        stats["generation_tps"] = round(
            stats["num_generation_tokens"] / stats["generation_time"]
        )
        return stats, text
        # return {"stats": {"audio": audio}, "Text": "Hello, world!", "audio output (may be empty)": audio}


def main():
    cfg = pyrallis.parse(config_class=DemoConfig)

    infer = SpeechLMInferenceWithContinuation(cfg.model)

    demo = gr.Interface(
        infer.transcribe,
        [
            gr.Audio(type="filepath", show_download_button=True),
            gr.Text(
                label="Prompt",
                value=cfg.default_prompt,
            ),
            gr.Number(
                label="Num Beams", value=cfg.default_num_beams, minimum=1, maximum=8
            ),
            gr.Number(
                label="temperature",
                value=cfg.default_temp,
                minimum=0.1,
                maximum=2.0,
                step=0.1,
            ),
        ],
        [
            gr.JSON(label="stats"),
            gr.Text(label="Text"),
        ],
        title="ASR via LLM",
        description=f"""This is a demo of the LLM model for ASR. It will transcribe the audio and then generate a response.
It has not been trained on instruction following, so it's not very good at it.

EPD: endpoint detection\\
EPD Early: detection right after the last audio token\\
EPD Late: detection after the transcription\\
Multi-channel audio: both channels are simply added together, but Gradio cannot trim them correctly.

Model name: {cfg.model.path.replace("runs/", "").replace("/", " ").strip()}""",
    )
    demo.launch(share=True)


if __name__ == "__main__":
    main()
