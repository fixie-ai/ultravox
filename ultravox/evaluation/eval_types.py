import dataclasses
from typing import Dict, List, Optional, Union

import dataclasses_json


@dataclasses.dataclass
class Sample(dataclasses_json.DataClassJsonMixin):
    question: str
    generated_answer: str
    expected_answer: str
    history: List[Dict[str, str]] = dataclasses.field(default_factory=list)


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
class CometResult:
    """Score is computed by comparing the source, reference, and translation using a trained regression model."""

    score: float


@dataclasses.dataclass
class ExactMatchResult:
    """Score is the 0-1 evaluation of the accuracy of the generated answer being equal to expected answer."""

    score: float
    reason: str


Result = Union[InstructResult, WerResult, ExactMatchResult, CometResult]

