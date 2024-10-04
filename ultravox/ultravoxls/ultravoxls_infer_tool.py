import torchaudio

from ultravox.data import datasets
from ultravox.inference import base
from ultravox.ultravoxls import ultravoxls_infer


def run_infer(
    inference: base.VoiceInference,
    sample: datasets.VoiceSample,
):
    audio_out = inference.infer(sample, max_tokens=10).text
    output_path = "./data/infer-out.wav"
    torchaudio.save(output_path, audio_out, 24000)
    print("Audio saved to", output_path)


def oneshot_infer(inference: base.VoiceInference, audio_file: str):
    with open(audio_file, "rb") as file:
        sample = datasets.VoiceSample.from_prompt_and_buf("", file.read())
    run_infer(inference, sample)


inference = ultravoxls_infer.UltravoxLSInference(
    "meta-llama/Llama-3.2-1B",
    device="cpu",
    data_type=None,
)

audio_file = "./data/input-130s.wav"

oneshot_infer(inference, audio_file)
