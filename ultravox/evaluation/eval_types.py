import dataclasses
from typing import Optional, Union

import dataclasses_json


@dataclasses.dataclass
class Sample(dataclasses_json.DataClassJsonMixin):
    question: str
    generated_answer: str
    expected_answer: str


@dataclasses.dataclass
class InstructResult:
    """Score is a 0-1 evaluation of the accuracy of the generated answer, or None if an error occurred."""

    score: Optional[float]
    reason: str


@dataclasses.dataclass
class WerResult:
    """Score is the 0-1 Word Error Rate for the generated transcript."""

    score: float


@dataclasses.dataclass
class ExactMatchResult:
    """Score is the 0-1 evaluation of the accuracy of the generated answer being equal to expected answer."""

    score: float
    reason: str


Result = Union[InstructResult, WerResult, ExactMatchResult]
