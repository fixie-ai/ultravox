from unittest import mock

import numpy as np
import pytest
import torch

from ultravox import data as datasets
from ultravox.model import ultravox_config
from ultravox.model import ultravox_data_proc

TEST_USER_MESSAGE = {
    "role": "user",
    "content": "Listen to <|audio|> and respond.",
}
TEST_ASSISTANT_MESSAGE = {
    "role": "assistant",
    "content": "The capital of France is Paris.",
}


def fake_apply_chat_template(messages, tokenize, chat_template=None):
    return "\n".join([f"{m['role']}: {m['content']}" for m in messages])


def fake_process(
    text, audio=None, audios=None, return_tensors="pt", sampling_rate=16000
):
    # More realistic token count based on text content
    # Roughly approximate 1 token per word + some extra for special tokens
    num_tokens = len(text.split()) + 2  # +2 for special tokens like <|audio|>
    input_ids = torch.tensor([range(num_tokens)])
    attention_mask = torch.tensor([[1] * num_tokens])
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


@pytest.fixture
def fake_augmentation():
    return datasets.AugRegistry.create_augmentation(
        datasets.AugRegistry.get_config("null")
    )


def test_process(mock_processor, fake_dataset, fake_augmentation):
    dataproc = ultravox_data_proc.UltravoxDataproc(
        fake_dataset,
        mock_processor,
        augmentation=fake_augmentation,
        loss_mask_type=ultravox_config.LossMaskType.LAST_ASSISTANT,
    )
    processed = next(iter(dataproc))

    # Full text: "user: Listen to <|audio|> and respond.\nassistant: The capital of France is Paris."
    # 13 words + 2 = 15 tokens total
    # User message: "user: Listen to <|audio|> and respond." = 6 words + 2 = 8 tokens to mask
    assert processed["input_ids"].shape == torch.Size([15])
    assert processed["attention_mask"].shape == torch.Size([15])
    assert torch.tensor(processed["labels"]).equal(
        torch.tensor(
            [-100, -100, -100, -100, -100, -100, -100, -100, 8, 9, 10, 11, 12, 13, 14]
        )
    )
    assert processed["audio_values"].shape == torch.Size([1, 1, 3])
    assert "audio_token_start_idx" in processed
    assert "audio_token_len" in processed


def test_process_inference_mode(mock_processor, fake_dataset, fake_augmentation):
    dataproc = ultravox_data_proc.UltravoxDataproc(
        fake_dataset,
        mock_processor,
        augmentation=fake_augmentation,
        inference_mode=True,
        loss_mask_type=ultravox_config.LossMaskType.LAST_ASSISTANT,
    )
    processed = next(iter(dataproc))

    # In inference mode, only user message: "user: Listen to <|audio|> and respond."
    # 6 words + 2 = 8 tokens
    assert processed["input_ids"].shape == torch.Size([8])
    assert processed["attention_mask"].shape == torch.Size([8])
    assert processed["audio_values"].shape == torch.Size([1, 1, 3])
    assert "audio_token_start_idx" in processed
    assert "audio_token_len" in processed


def test_process_loss_mask_all(mock_processor, fake_dataset, fake_augmentation):
    """Test that LossMaskType.ALL includes all tokens in the loss (no masking)."""
    dataproc = ultravox_data_proc.UltravoxDataproc(
        fake_dataset,
        mock_processor,
        augmentation=fake_augmentation,
        loss_mask_type=ultravox_config.LossMaskType.ALL,
    )
    processed = next(iter(dataproc))

    # Full text: 13 words + 2 = 15 tokens total
    assert processed["input_ids"].shape == torch.Size([15])
    assert processed["attention_mask"].shape == torch.Size([15])
    # With LossMaskType.ALL, no tokens should be masked (no -100 values)
    assert torch.tensor(processed["labels"]).equal(
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    )
    assert processed["audio_values"].shape == torch.Size([1, 1, 3])
    assert "audio_token_start_idx" in processed
    assert "audio_token_len" in processed


def test_process_loss_mask_after_audio(mock_processor, fake_dataset, fake_augmentation):
    """Test that LossMaskType.AFTER_AUDIO masks tokens up to and including the audio token."""
    dataproc = ultravox_data_proc.UltravoxDataproc(
        fake_dataset,
        mock_processor,
        augmentation=fake_augmentation,
        loss_mask_type=ultravox_config.LossMaskType.AFTER_AUDIO,
    )
    processed = next(iter(dataproc))

    # Full text: 13 words + 2 = 15 tokens total
    # Text up to audio: "user: Listen to <|audio|>" = 4 words + 2 = 6 tokens to mask
    assert processed["input_ids"].shape == torch.Size([15])
    assert processed["attention_mask"].shape == torch.Size([15])
    # With LossMaskType.AFTER_AUDIO, first 6 tokens should be masked
    assert torch.tensor(processed["labels"]).equal(
        torch.tensor(
            [-100, -100, -100, -100, -100, -100, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        )
    )
    assert processed["audio_values"].shape == torch.Size([1, 1, 3])
    assert "audio_token_start_idx" in processed
    assert "audio_token_len" in processed
