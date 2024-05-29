import nltk  # needed for truecase
import truecase

nltk.download("punkt")


def clean_text_for_training(text: str):
    """
    Cleans the text for training. Most of these are for Gigaspeech:
    - Convert punctuations
    - Remove non-scoring words to reduce noise
    - Convert to true case
        - This is not perfect, but it's better than nothing

    Example:
        "I SEE LOTS OF PEOPLE HAVE AH DRONES HERE <COMMA> AH MAVERICK AH AS WELL <PERIOD>"
        --> "I see lots of people have drones here, maverick as well."
    """
    remaining_words = []
    for word in text.split():
        if word in gigaspeech_punctuations:
            word = gigaspeech_punctuations[word]
        elif word in non_scoring_words:
            continue
        remaining_words.append(word)

    text = " ".join(remaining_words)
    text = truecase.get_true_case(text)

    return {"text": text}


# Source: https://github.com/SpeechColab/GigaSpeech/blob/main/utils/gigaspeech_scoring.py
def asr_text_post_processing(text):
    # 1. convert to uppercase
    text = text.upper()

    # 2. remove hyphen
    #   "E-COMMERCE" -> "E COMMERCE", "STATE-OF-THE-ART" -> "STATE OF THE ART"
    text = text.replace("-", " ")

    # 3. remove non-scoring words from evaluation
    remaining_words = []
    for word in text.split():
        if word in non_scoring_words:
            continue
        remaining_words.append(word)

    return " ".join(remaining_words)


conversational_filler = [
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
special_tags = ["<UNK>", "<unk>", "</s>"]
gigaspeech_punctuations = {
    "<COMMA>": ",",
    "<PERIOD>": ".",
    "<QUESTIONMARK>": "?",
    "<EXCLAMATIONPOINT>": "!",
}
gigaspeech_garbage_utterance_tags = ["<SIL>", "<NOISE>", "<MUSIC>", "<OTHER>"]
non_scoring_words = set(
    conversational_filler
    + special_tags
    + list(gigaspeech_punctuations.keys())
    + gigaspeech_garbage_utterance_tags
)
