import re
import typing as t

import numpy as np
import regex
import transformers


def find_prefix_match(full: str, partial: str, cer_limit: float = 0) -> t.Optional[str]:
    allowed_errors = int(round(len(partial) * cer_limit)) + 10
    # Remove whitespaces and trailing punctuation
    partial = partial.strip()
    if partial[-1] in ".,;:!?":
        partial = partial[:-1]
    # Fuzzy search for a prefix match
    ## (?i) - case insensitive   ## (?e) - enhance match  ## {e<=10} - number of errors allowed  ## .* - any characters after the match
    partial_regex = "^(?i)(?e)(" + partial + "){e<=" + str(allowed_errors) + "}.*$"
    match: t.Optional[regex.Match[str]] = regex.fullmatch(partial_regex, full)
    if match is None:
        return basic_prefix_match(full, partial)
    return match.groups()[0].strip()


def basic_prefix_match(full: str, partial: str):
    """
    Simply want to do full[:len(partial)], but make sure we don't cut off in the middle of a word.
    """
    words = re.split(r" |,|\.|;|:|\?|!", full)
    word_end_lens = np.cumsum(np.ones(len(words)) + [len(w) for w in words])
    word_end_lens = np.insert(word_end_lens, 0, 0)
    closest_ind = np.abs(word_end_lens - len(partial)).argmin()
    return full[: int(word_end_lens[closest_ind])].strip()


class CroppedTranscriber:
    def __init__(self, model_name="openai/whisper-tiny", device="cpu"):
        self.model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        self.model.to(device)

        self.processor = transformers.WhisperProcessor.from_pretrained(model_name)

    def __call__(self, cropped_raw_audio: list, sr: int, full_transcript: str):
        input_features = self.processor(
            cropped_raw_audio, sampling_rate=sr, return_tensors="pt"
        ).input_features
        predicted_ids = self.model.generate(input_features)
        partial_transcript = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        return basic_prefix_match(full_transcript, partial_transcript)
        # return find_prefix_match(full_transcript, partial_transcript)


def test_cropped_transcriber():
    ct = CroppedTranscriber()
    assert ct(np.zeros(1000), 16000, "hello there, how's it going?!") == "hello"
    assert ct(np.zeros(1000), 16000, "helloooo there, how's it going?!") == ""


if __name__ == "__main__":
    ct = CroppedTranscriber()
    print(ct(np.zeros(1000), 16000, "hello there, how's it going?!"))
