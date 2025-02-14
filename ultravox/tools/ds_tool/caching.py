import hashlib
import json
import os
from typing import List, Optional, Union, overload

import openai
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_fixed

from ultravox.tools.ds_tool import tts


class CachingChatWrapper:
    def __init__(self, client: openai.Client, unique_id: str, prefix_length: int = 1):
        super().__init__()
        self._client = client
        self._base_path = os.path.join(
            ".cache/ds_tool/textgen",
            unique_id.replace("https://", "").replace("/", "__"),
        )
        self._prefix_length = prefix_length

    def _get_prefixed_path(self, text_hash: str) -> str:
        prefix = text_hash[: self._prefix_length]
        prefixed_path = os.path.join(self._base_path, prefix)
        os.makedirs(prefixed_path, exist_ok=True)
        return os.path.join(prefixed_path, f"{text_hash}.txt")

    @retry(wait=wait_fixed(3), stop=stop_after_attempt(3))
    def chat_completion(self, **kwargs) -> str:
        text_hash = hashlib.sha256(json.dumps(kwargs).encode()).hexdigest()

        # Try to read from cache
        cache_path = self._get_prefixed_path(text_hash)
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return f.read()

        # If not found, create new response
        response = self._client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content

        # Write to cache
        try:
            with open(cache_path, "w") as f:
                f.write(text)
        except IOError as e:
            print(f"Warning: Unable to cache response: {e}")

        return text


class CachingTtsWrapper:
    def __init__(self, client: tts.Client, implementation: str):
        super().__init__()
        self._client = client
        self._base_path = os.path.join(".cache/ds_tool/tts", implementation)

    @overload
    def tts(self, text: str, voice: Optional[str] = None) -> bytes: ...

    @overload
    def tts(self, text: List[str], voice: Optional[str] = None) -> List[bytes]: ...

    def tts(
        self, text: Union[str, List[str]], voice: Optional[str] = None
    ) -> Union[bytes, List[bytes]]:
        text_hash = hashlib.sha256(str(text).encode()).hexdigest()
        voice = self._client.resolve_voice(voice)

        if isinstance(text, list):
            return [self.tts(t, voice) for t in text]

        path = os.path.join(self._base_path, voice)
        os.makedirs(path, exist_ok=True)

        cache_path = os.path.join(path, f"{text_hash}.wav")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return f.read()

        wav = self._client.tts(text, voice)

        with open(cache_path, "wb") as f:
            f.write(wav)

        return wav
