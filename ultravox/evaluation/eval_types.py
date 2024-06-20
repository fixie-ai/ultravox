import dataclasses
from typing import Optional, Union, List

import dataclasses_json


@dataclasses.dataclass
class Sample(dataclasses_json.DataClassJsonMixin):
    question: str
    generated_answer: str
    expected_answer: str
    history: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class InstructResult:
    """Score is a 0-1 evaluation of the accuracy of the generated answer, or None if an error occurred."""

    score: Optional[float]
    reason: str


@dataclasses.dataclass
class WerResult:
    """Score is the 0-1 Word Error Rate for the generated transcript."""

    score: float


Result = Union[InstructResult, WerResult]
