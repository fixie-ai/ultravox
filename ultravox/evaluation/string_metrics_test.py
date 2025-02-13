import pytest

from ultravox.evaluation import eval_types
from ultravox.evaluation import string_metrics

# Test Cases
samples_en = [
    eval_types.Sample(
        index=0,
        question="",
        transcript="",
        expected_answer="The quick brown fox jumps over the lazy dog",
        generated_answer="The quick brown fox jumps over a lazy dog",
    ),
    eval_types.Sample(
        index=1,
        question="",
        transcript="",
        expected_answer="Hello world!",
        generated_answer="Hello world",
    ),
]

samples_zh = [
    eval_types.Sample(
        index=0,
        question="",
        transcript="",
        expected_answer="今天是个好天气",
        generated_answer="今天是一个好天气",
    ),
    eval_types.Sample(
        index=1,
        question="",
        transcript="",
        expected_answer="我喜欢吃苹果",
        generated_answer="我喜欢苹果",
    ),
]

samples_ja = [
    eval_types.Sample(
        index=0,
        question="",
        transcript="",
        expected_answer="私は日本語を勉強しています",
        generated_answer="私は日本語を学んでいます",
    ),
    eval_types.Sample(
        index=1,
        question="",
        transcript="",
        expected_answer="おはようございます",
        generated_answer="おはよう",
    ),
]


def test_wer_en():
    result = string_metrics.wer(samples_en, {})
    assert result.score == pytest.approx(9.090909090909092, rel=1e-2)


def test_wer_zh():
    result = string_metrics.wer(samples_zh, {"lang_id": "zh"})
    assert result.score == pytest.approx(15.384615384615385, rel=1e-2)


def test_wer_ja():
    result = string_metrics.wer(samples_ja, {"lang_id": "ja"})
    assert result.score == pytest.approx(40.909090909090914, rel=1e-2)


def test_bleu_en():
    result = string_metrics.bleu(samples_en, {})
    assert result.score == pytest.approx(61.216343280457046, rel=1e-2)


def test_bleu_zh():
    result = string_metrics.bleu(samples_zh, {"tokenize": "zh"})
    assert result.score == pytest.approx(45.43741956108463, rel=1e-2)


def test_bleu_ja():
    result = string_metrics.bleu(samples_ja, {"tokenize": "ja-mecab"})
    assert result.score == pytest.approx(29.728070921986767, rel=1e-2)


def test_remove_diacritics():
    test_cases = [
        ("السَّلَامُ عَلَيْكُمْ", "السلام عليكم"),  # Contains Fatha, Damma, and Kasra
        ("كِتَابٌ", "كتاب"),  # Contains Kasra, Fatha, and Tanwin Damma
        ("يَذهَبُ", "يذهب"),  # Contains Fatha and Damma
        ("مُحَمَّدٌ", "محمد"),  # Contains Damma, Shadda, and Tanwin
        # ("قُرْآن", "قران"),  # Contains Sukun and Madda
        # ("ٱلسَّلَامُ", "السلام"),  # Contains Wasla
        ("هَٰذَا", "هذا"),  # Contains Superscript Alef
    ]

    for input_text, expected_output in test_cases:
        output = string_metrics.remove_diacritics(input_text)
        assert (
            output == expected_output
        ), f"Failed: {input_text} -> {output}, Expected: {expected_output}"
