import numpy as np
import pytest
import transformers

from ultravox.model import ultravox_processing


@pytest.fixture(scope="module")
def processor():
    audio_processor = transformers.AutoProcessor.from_pretrained(
        "./assets/hf/openai-whisper-tiny", local_files_only=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "./assets/hf/Meta-Llama-3-8B-Instruct", local_files_only=True
    )
    return ultravox_processing.UltravoxProcessor(audio_processor, tokenizer=tokenizer)


@pytest.fixture
def sampling_rate(processor):
    return processor.audio_processor.feature_extractor.sampling_rate


@pytest.fixture
def short_audio(sampling_rate):
    return np.random.randn(sampling_rate)


@pytest.fixture
def long_audio(sampling_rate):
    return np.random.randn(sampling_rate * 10)


@pytest.fixture
def overflowing_audio(sampling_rate):
    # 35 seconds of audio which is longer than the default max length of 30 seconds.
    return np.random.randn(sampling_rate * 35)


def test_processor_text_only(processor):
    result = processor("Hello, how are you?")
    assert result.input_ids.tolist() == [[9906, 11, 1268, 527, 499, 30]]
    assert result.attention_mask.tolist() == [[1, 1, 1, 1, 1, 1]]


def test_processor_single_audio(processor, short_audio, sampling_rate):
    placeholder_token_id = processor.tokenizer.get_vocab()[
        processor.audio_token_replacement
    ]

    result = processor(
        "Test with <|audio|>",
        audio=short_audio,
        sampling_rate=sampling_rate,
    )
    assert result.audio_lens.tolist() == [100]
    assert result.audio_token_len.tolist() == [7]
    assert result.audio_token_start_idx.tolist() == [3]
    assert result.input_ids.tolist() == [[2323, 449, 220] + [placeholder_token_id] * 7]
    assert result.attention_mask.tolist() == [[1] * len(result.input_ids[0])]
    assert result.audio_batch_size.tolist() == [1]

    result = processor(
        "Test with <|audio|>",
        audios=[short_audio],
        sampling_rate=sampling_rate,
    )
    assert result.audio_lens.tolist() == [100]
    assert result.audio_token_len.tolist() == [7]
    assert result.audio_token_start_idx.tolist() == [3]
    assert result.input_ids.tolist() == [[2323, 449, 220] + [placeholder_token_id] * 7]
    assert result.attention_mask.tolist() == [[1] * len(result.input_ids[0])]
    assert result.audio_batch_size.tolist() == [1]


def test_processor_overflowing_audio(processor, overflowing_audio, sampling_rate):
    placeholder_token_id = processor.tokenizer.get_vocab()[
        processor.audio_token_replacement
    ]

    result = processor(
        "Test with <|audio|>",
        audios=[overflowing_audio],
        sampling_rate=sampling_rate,
    )
    assert result.audio_lens.tolist() == [3000, 500]
    assert result.audio_token_len.tolist() == [188, 32]
    assert result.audio_token_start_idx.tolist() == [3, 3 + 188]
    assert result.input_ids.tolist() == [
        [2323, 449, 220] + [placeholder_token_id] * (188 + 32)
    ]
    assert result.attention_mask.tolist() == [[1] * len(result.input_ids[0])]
    assert result.audio_batch_size.tolist() == [2]


def test_processor_multiple_audios(processor, short_audio, long_audio, sampling_rate):
    placeholder_token_id = processor.tokenizer.get_vocab()[
        processor.audio_token_replacement
    ]

    result = processor(
        "Test with <|audio|> and <|audio|>",
        audios=[short_audio, long_audio],
        sampling_rate=sampling_rate,
    )
    assert result.audio_lens.tolist() == [100, 1000]
    assert result.audio_token_len.tolist() == [7, 63]
    assert result.audio_token_start_idx.tolist() == [3, 12]
    assert result.input_ids.tolist() == [
        [2323, 449, 220]
        + [placeholder_token_id] * 7
        + [323, 220]
        + [placeholder_token_id] * 63
    ]
    assert result.attention_mask.tolist() == [[1] * len(result.input_ids[0])]
    assert result.audio_batch_size.tolist() == [2]


def test_processor_multiple_audios_with_overflowing_audio(
    processor, short_audio, long_audio, overflowing_audio, sampling_rate
):
    placeholder_token_id = processor.tokenizer.get_vocab()[
        processor.audio_token_replacement
    ]

    result = processor(
        "Test with <|audio|> and <|audio|> and <|audio|>",
        audios=[short_audio, overflowing_audio, long_audio],
        sampling_rate=sampling_rate,
    )
    assert result.audio_lens.tolist() == [100, 3000, 500, 1000]
    assert result.audio_token_len.tolist() == [7, 188, 32, 63]
    assert result.audio_token_start_idx.tolist() == [3, 12, 200, 234]
    assert result.input_ids.tolist() == [
        [2323, 449, 220]
        + [placeholder_token_id] * 7
        + [323, 220]
        + [placeholder_token_id] * 188
        + [placeholder_token_id] * 32
        + [323, 220]
        + [placeholder_token_id] * 63
    ]
    assert result.attention_mask.tolist() == [[1] * len(result.input_ids[0])]
    assert result.audio_batch_size.tolist() == [4]


def test_processor_fails_with_too_many_audio_tokens(
    processor, short_audio, overflowing_audio, sampling_rate
):
    with pytest.raises(ValueError):
        processor("Hello, how are you? <|audio|>")

    with pytest.raises(ValueError):
        processor(
            "Hello, how are you? <|audio|><|audio|>",
            audios=[short_audio],
            sampling_rate=sampling_rate,
        )

    with pytest.raises(ValueError):
        processor(
            "Hello, how are you? <|audio|><|audio|>",
            audios=[overflowing_audio],
            sampling_rate=sampling_rate,
        )


def test_processor_fails_with_too_few_audio_tokens(
    processor, short_audio, sampling_rate
):
    with pytest.raises(ValueError):
        processor(
            "Hello, how are you?", audios=[short_audio], sampling_rate=sampling_rate
        )

    with pytest.raises(ValueError):
        processor(
            "Hello, how are you? <|audio|>",
            audios=[short_audio, short_audio],
            sampling_rate=sampling_rate,
        )


@pytest.mark.parametrize("sample_count", [0, 1, 159, 160, 161, 319, 320, 321])
def test_audio_shapes(processor, sampling_rate, sample_count):
    specific_audio = np.random.randn(sample_count)
    result = processor(
        "<|audio|>",
        audios=[specific_audio],
        sampling_rate=sampling_rate,
    )

    assert result["audio_lens"][0] == result["audio_values"][0].shape[-1]
