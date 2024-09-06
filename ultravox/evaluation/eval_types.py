import dataclasses
from typing import Dict, List, Optional, Union
from pydantic import BaseModel

import dataclasses_json

class EvalConfig(BaseModel):
    metric: str
    args: Optional[Dict[str, str]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Sample(dataclasses_json.DataClassJsonMixin):
    index: int
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


Result = Union[InstructResult, WerResult, ExactMatchResult]
