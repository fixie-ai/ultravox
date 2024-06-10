from unittest import mock

import numpy as np
import pytest
import torch
import transformers

from ultravox.data import datasets
from ultravox.inference import base as infer_base
from ultravox.inference import infer
from ultravox.model import ultravox_processing


# We cache these files in our repo to make CI faster and also
# work properly for external contributions (since Llama 3 is gated).
@pytest.fixture(scope="module")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(
        "./assets/hf/Meta-Llama-3-8B-Instruct", local_files_only=True
    )


@pytest.fixture(scope="module")
def audio_processor():
    return transformers.AutoProcessor.from_pretrained(
        "./assets/hf/wav2vec2-base-960h", local_files_only=True
    )


class FakeInference(infer.LocalInference):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        audio_processor: transformers.ProcessorMixin,
    ):
        processor = ultravox_processing.UltravoxProcessor(
            audio_processor, tokenizer=tokenizer
        )
        super().__init__(
            mock.MagicMock(),
            processor=processor,
            tokenizer=tokenizer,
            device="cpu",
            dtype=torch.float32,
        )
        self.model.device = "cpu"
        self.model.generate = mock.MagicMock(return_value=[range(25)])


EXPECTED_TOKEN_IDS_START = [128000, 128006, 882, 128007]
EXPECTED_TOKEN_IDS_END = [128009, 128006, 78191, 128007, 271]


def test_infer_16kHz(tokenizer, audio_processor):
    """Ensure we handle 16kHz float32 audio properly."""
    inference = FakeInference(tokenizer, audio_processor)
    array = np.ones(16000, dtype=np.float32)
    sample = datasets.VoiceSample.from_prompt_and_raw(
        "Transcribe <|audio|>", array, 16000
    )
    output = inference.infer(sample)
    assert output.input_tokens == 20
    assert output.output_tokens == 5
    assert output.text == "56789"
    generate_args = inference.model.generate.call_args[1]
    call_audio_values = generate_args["audio_values"]
    assert call_audio_values.shape == (1, 16000)
    call_input_ids = generate_args["input_ids"]
    assert call_input_ids.shape == (1, 20)
    assert call_input_ids[0, :4].tolist() == EXPECTED_TOKEN_IDS_START
    assert call_input_ids[0, -5:].tolist() == EXPECTED_TOKEN_IDS_END
    assert torch.all(call_input_ids[0, 8:15] == inference.tokenizer.eos_token_id)
    assert generate_args["audio_token_len"].item() == 7
    assert generate_args["audio_token_start_idx"].item() == 8


def test_infer_48kHz(tokenizer, audio_processor):
    """Ensure we resample 48KHz to 16kHz properly."""
    inference = FakeInference(tokenizer, audio_processor)
    array = np.ones(48000, dtype=np.float32)
    sample = datasets.VoiceSample.from_prompt_and_raw(
        "Transcribe <|audio|>", array, 48000
    )
    output = inference.infer(sample)
    assert output.input_tokens == 20
    assert output.output_tokens == 5
    assert output.text == "56789"
    generate_args = inference.model.generate.call_args[1]
    call_audio_values = generate_args["audio_values"]
    assert call_audio_values.shape == (1, 16000)
    call_input_ids = generate_args["input_ids"]
    assert call_input_ids.shape == (1, 20)
    assert call_input_ids[0, :4].tolist() == EXPECTED_TOKEN_IDS_START
    assert call_input_ids[0, -5:].tolist() == EXPECTED_TOKEN_IDS_END
    assert torch.all(call_input_ids[0, 8:15] == inference.tokenizer.eos_token_id)
    assert generate_args["audio_token_len"].item() == 7
    assert generate_args["audio_token_start_idx"].item() == 8


def test_infer_16kHz_stream(tokenizer, audio_processor):
    """Ensure we handle streaming output properly."""
    inference = FakeInference(tokenizer, audio_processor)
    array = np.ones(16000, dtype=np.float32)
    sample = datasets.VoiceSample.from_prompt_and_raw(
        "Transcribe <|audio|>", array, 16000
    )
    gen = inference.infer_stream(sample)
    text = ""
    stats = None
    for msg in gen:
        if isinstance(msg, infer_base.InferenceChunk):
            text += msg.text
        elif isinstance(msg, infer_base.InferenceStats):
            stats = msg
    assert text == "56789"
    assert stats.input_tokens == 20
    assert stats.output_tokens == 5
    generate_args = inference.model.generate.call_args[1]
    call_audio_values = generate_args["audio_values"]
    assert call_audio_values.shape == (1, 16000)
    call_input_ids = generate_args["input_ids"]
    assert call_input_ids.shape == (1, 20)
    assert call_input_ids[0, :4].tolist() == EXPECTED_TOKEN_IDS_START
    assert call_input_ids[0, -5:].tolist() == EXPECTED_TOKEN_IDS_END
    assert torch.all(call_input_ids[0, 8:15] == inference.tokenizer.eos_token_id)
    assert generate_args["audio_token_len"].item() == 7
    assert generate_args["audio_token_start_idx"].item() == 8


def test_infer_text_only(tokenizer, audio_processor):
    """Ensure we handle text without audio properly."""
    inference = FakeInference(tokenizer, audio_processor)
    sample = datasets.VoiceSample.from_prompt("Hello?")
    output = inference.infer(sample)
    assert output.input_tokens == 12
    assert output.output_tokens == 13
    assert output.text == "-./0123456789"
    generate_args = inference.model.generate.call_args[1]
    assert generate_args.get("audio_values") is None
    call_input_ids = generate_args["input_ids"]
    assert call_input_ids.shape == (1, 12)
    assert call_input_ids[0, :4].tolist() == EXPECTED_TOKEN_IDS_START
    assert call_input_ids[0, -5:].tolist() == EXPECTED_TOKEN_IDS_END
    assert generate_args.get("audio_token_len") is None
    assert generate_args.get("audio_token_start_idx") is None
