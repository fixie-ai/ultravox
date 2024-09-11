import dataclasses
from typing import Any, Dict, List, Optional, Union

import dataclasses_json
from pydantic import BaseModel


class EvalConfig(BaseModel):
    metric: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Sample(dataclasses_json.DataClassJsonMixin):
    index: int
    question: str
    hypothesis: str
    reference: str
    history: List[Dict[str, Any]] = dataclasses.field(default_factory=list)


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


Result = Union[InstructResult, WerResult, ExactMatchResult, BleuResult]
