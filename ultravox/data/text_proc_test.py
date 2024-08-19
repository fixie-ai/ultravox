import pytest

from ultravox.data import text_proc


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "I SEE LOTS OF PEOPLE HAVE DRONES HERE <COMMA> MAVERICK AS WELL <PERIOD>  ",
            "I see lots of people have drones here, maverick as well.",
        ),
        # truecase messes with the case of special tags too, but we probably don't care about that
        ("<NOISE> OH WHAT WAS THAT?", "Oh what was that?"),
    ],
)
def test_no_space_punctuation(text, expected):
    assert text_proc.format_asr_text(text) == expected
