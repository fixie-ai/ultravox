from unittest import mock

import numpy as np
import pytest
import torch

from ultravox import data as datasets
from ultravox.model import ultravox_data_proc

TEST_USER_MESSAGE = {
    "role": "user",
    "content": "Listen to <|audio|> and respond.",
}
TEST_ASSISTANT_MESSAGE = {
    "role": "assistant",
    "content": "The capital of France is Paris.",
}


def fake_apply_chat_template(messages, tokenize):
    return "\n".join([f"{m['role']}: {m['content']}" for m in messages])


def fake_process(
    text, audio=None, audios=None, return_tensors="pt", sampling_rate=16000
):
    num_messages = len(text.split("\n"))
    input_ids = torch.tensor([range(num_messages * 5)])
    attention_mask = torch.tensor([[1] * num_messages * 5])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "audio_values": torch.tensor([[[0.1, 0.2, 0.3]]]),
        "audio_token_start_idx": torch.tensor([1]),
        "audio_token_len": torch.tensor([2]),
        "audio_lens": torch.tensor([10]),
    }


@pytest.fixture
def mock_processor():
    processor = mock.Mock()
    processor.tokenizer.apply_chat_template.side_effect = fake_apply_chat_template
    processor.side_effect = fake_process
    return processor


@pytest.fixture
def fake_dataset():
    return [
        datasets.VoiceSample(
            messages=[TEST_USER_MESSAGE, TEST_ASSISTANT_MESSAGE],
            audio=np.array([0.1, 0.2, 0.3]),
            sample_rate=16000,
        )
    ]


def test_process(mock_processor, fake_dataset):
    dataproc = ultravox_data_proc.UltravoxDataproc(fake_dataset, mock_processor)
    processed = next(iter(dataproc))

    assert processed["input_ids"].shape == torch.Size([10])
    assert processed["attention_mask"].shape == torch.Size([10])
    assert torch.tensor(processed["labels"]).equal(
        torch.tensor([-100, -100, -100, -100, -100, 5, 6, 7, 8, 9])
    )
    assert processed["audio_values"].shape == torch.Size([1, 1, 3])
    assert "audio_token_start_idx" in processed
    assert "audio_token_len" in processed


def test_process_inference_mode(mock_processor, fake_dataset):
    dataproc = ultravox_data_proc.UltravoxDataproc(
        fake_dataset, mock_processor, inference_mode=True
    )
    processed = next(iter(dataproc))

    assert processed["input_ids"].shape == torch.Size([5])
    assert processed["attention_mask"].shape == torch.Size([5])
    assert processed["audio_values"].shape == torch.Size([1, 1, 3])
    assert "audio_token_start_idx" in processed
    assert "audio_token_len" in processed
