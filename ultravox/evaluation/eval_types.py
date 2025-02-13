import dataclasses
from typing import Dict, List, Union

import dataclasses_json


@dataclasses.dataclass
class Sample(dataclasses_json.DataClassJsonMixin):
    index: int  # index of the sample in the dataset, used for preserving order after ddp all_gather
    question: str
    transcript: str
    generated_answer: str
    expected_answer: str
    history: List[Dict[str, str]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class InstructResult:
    """Score is a 0-1 evaluation of the accuracy of the generated answer, or None if an error occurred."""

    score: float
    reason: str


@dataclasses.dataclass
class WerResult:
    """Score is Word Error Rate (%) for the generated transcript."""

    score: float


@dataclasses.dataclass
class ExactMatchResult:
    """Score is the 0-1 evaluation of the accuracy of the generated answer being equal to expected answer."""

    score: float
    reason: str


@dataclasses.dataclass
class BleuResult:
    """
    Score is the BLEU score for the generated answer.
    Note: BLEU is supposed to be computed on a corpus level, not on a single sample.
    """

    score: float


@dataclasses.dataclass
class MeanResult:
    """Score is the mean of the scores of the samples."""

    score: float


Result = Union[InstructResult, WerResult, ExactMatchResult, BleuResult, MeanResult]
