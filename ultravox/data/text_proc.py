import os
import sys
from typing import Dict, List

# Temporary fix for an issue where importing NLTK breaks PyTorch multiprocessing on MacOS.
# For more details, see: https://github.com/nltk/nltk/issues/2949
sys.modules["tkinter"] = None  # type: ignore
import nltk  # needed for truecase # noqa: E402
import truecase  # noqa: E402


class FormatASRError(ValueError):
    pass


# only in master thread per node to avoid
# other threads overwriting the downloaded .zip
if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    try:
        truecase.get_true_case("test")
    except LookupError:
        nltk.download("punkt", quiet=True)


def format_asr_text(text: str) -> str:
    """
    Cleans the text for training. First one is Gigaspeech-specific, but the second one is useful for LibriSpeech as well.
    - Convert punctuations
    - Convert to true case
        - This is not perfect, but it's better than nothing
    - Strip leading/trailing spaces

    Example:
        "I SEE LOTS OF PEOPLE HAVE AH DRONES HERE <COMMA> AH MAVERICK AH AS WELL <PERIOD>  "
        --> "I see lots of people have drones here, maverick as well."
    """
    remaining_words = []
    for word in text.split():
        if word in GIGASPEECH_GARBAGE_UTTERANCE_TAGS:
            raise FormatASRError(f"Garbage utterance tag found: {word}")
        if word in GIGASPEECH_PUNCTUATIONS:
            word = GIGASPEECH_PUNCTUATIONS[word]
        remaining_words.append(word)

    text = " ".join(remaining_words)
    text = truecase.get_true_case(text)
    text_stripped = text.strip()
    if len(text_stripped) == 0:
        raise FormatASRError("Empty text after processing")
    return text_stripped


def format_message_history(
    messages: Dict[str, List[str]],
    roles: Dict[str, str],
) -> List[Dict[str, str]]:
    """
    Format the message history from the dataset into a list of messages for ASR.
    Args:
        messages: A dictionary with keys "role" and "content".
        roles: A dictionary of roles.
    Returns:
        A list of messages.
    """
    messages_list = [
        dict(zip(messages.keys(), values)) for values in zip(*messages.values())
    ]
    formatted_messages = []
    for message in messages_list:
        # drop any messages that are not defined in the roles dict
        if message["role"] in roles:
            formatted_messages.append(
                {"role": roles[message["role"]], "content": message["content"]}
            )
    return formatted_messages


CONVERSATIONAL_FILLER = [
    "UH",
    "UHH",
    "UM",
    "EH",
    "MM",
    "HM",
    "AH",
    "HUH",
    "HA",
    "ER",
    "OOF",
    "HEE",
    "ACH",
    "EEE",
    "EW",
]
SPECIAL_TAGS = ["<UNK>", "<unk>", "</s>"]
GIGASPEECH_PUNCTUATIONS = {
    "<COMMA>": ",",
    "<PERIOD>": ".",
    "<QUESTIONMARK>": "?",
    "<EXCLAMATIONPOINT>": "!",
}

# These are the words that are not scored in the Gigaspeech dataset
# Right now we are not using them, but for testing we should use them so that our numbers match other implementations
GIGASPEECH_GARBAGE_UTTERANCE_TAGS = ["<SIL>", "<NOISE>", "<MUSIC>", "<OTHER>"]
NON_SCORING_WORDS = set(
    CONVERSATIONAL_FILLER
    + SPECIAL_TAGS
    + list(GIGASPEECH_PUNCTUATIONS.keys())
    + GIGASPEECH_GARBAGE_UTTERANCE_TAGS
)
