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
