import abc
import dataclasses
from typing import Generator, Optional

from ultravox.data import datasets


@dataclasses.dataclass
class VoiceOutput:
    text: str
    input_tokens: int
    output_tokens: int


class InferenceMessage:
    pass


@dataclasses.dataclass
class InferenceChunk(InferenceMessage):
    text: str


@dataclasses.dataclass
class InferenceStats(InferenceMessage):
    input_tokens: int
    output_tokens: int


InferenceGenerator = Generator[InferenceMessage, None, None]


class VoiceInference(abc.ABC):
    @abc.abstractmethod
    def infer(
        self,
        sample: datasets.VoiceSample,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> VoiceOutput:
        pass

    def infer_stream(
        self,
        sample: datasets.VoiceSample,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> InferenceGenerator:
        """Streaming polyfill, if not supported directly in derived classes."""
        output = self.infer(sample, max_tokens, temperature)
        yield InferenceChunk(output.text)
        yield InferenceStats(output.input_tokens, output.output_tokens)
