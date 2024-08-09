import abc
import dataclasses
from typing import Dict, Generator, List, Optional, Tuple, Union

import transformers

from ultravox.data import datasets


@dataclasses.dataclass
class VoiceOutput:
    text: str
    input_tokens: int
    output_tokens: int
    audio_token_len: int
    past_key_values: Union[Tuple, transformers.cache_utils.Cache]


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
        past_key_values: Optional[Union[Tuple, transformers.cache_utils.Cache]] = None,
    ) -> VoiceOutput:
        pass

    def infer_stream(
        self,
        sample: datasets.VoiceSample,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        past_key_values: Optional[Union[Tuple, transformers.cache_utils.Cache]] = None,
    ) -> InferenceGenerator:
        """Streaming polyfill, if not supported directly in derived classes."""
        output = self.infer(sample, max_tokens, temperature, past_key_values)
        yield InferenceChunk(output.text)
        yield InferenceStats(output.input_tokens, output.output_tokens)


class History:
    def __init__(self, audio_token_replacement: str = "<|eot_token|>"):
        self.audio_token_replacement: str = audio_token_replacement
        self.audio_placeholder = "<|audio|>"
        self.messages: List[Dict[str, str]] = []
        self.key_values: Optional[Union[Tuple, transformers.cache_utils.Cache]] = None

    def add_message(self, message: Dict[str, str], audio_token_len: int):
        message = message.copy()
        content = message["content"]
        if audio_token_len > 0:
            if content.count(self.audio_placeholder) != 1:
                raise ValueError(
                    f"Expected 1 audio placeholder, found {content.count(self.audio_placeholder)}"
                )
            message["content"] = content.replace(
                self.audio_placeholder, self.audio_token_replacement * audio_token_len
            )

        if self.messages:
            self.messages.append(message)
        else:
            self.messages = [message]

    def update_key_values(
        self, key_values: Union[Tuple, transformers.cache_utils.Cache]
    ):
        self.key_values = key_values

    @property
    def past_messages(self) -> List[Dict[str, str]]:
        return self.messages

    @property
    def past_key_values(self) -> Optional[Union[Tuple, transformers.cache_utils.Cache]]:
        return self.key_values

    def reset(self):
        self.messages = []
        self.key_values = None
