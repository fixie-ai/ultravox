import pytest

from ultravox.data import text_proc


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "I SEE LOTS OF PEOPLE HAVE DRONES HERE <COMMA> MAVERICK AS WELL <PERIOD>  ",
            "I see lots of people have drones here, maverick as well.",
        ),
    ],
)
def test_format_asr_text(text, expected):
    assert text_proc.format_asr_text(text) == expected


def test_garbage_utterance():
    with pytest.raises(text_proc.FormatASRError):
        text_proc.format_asr_text("<NOISE> OH WHAT WAS THAT?")


def test_format_message_history():
    roles = {"user": "user", "assistant": "assistant"}

    messages = {"role": ["user", "assistant"], "content": ["A", "B"]}
    assert text_proc.format_message_history(messages, roles) == [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
    ]

    messages = {"role": ["user", "tool_call", "assistant"], "content": ["A", "B", "C"]}
    assert text_proc.format_message_history(messages, roles) == [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "C"},
    ]
