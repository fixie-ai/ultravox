import hashlib
import json
import os
import random
from typing import List, Optional, Union, overload

import openai

from ultravox.tools.ds_tool import tts


class CachingChatWrapper:
    def __init__(self, client: openai.Client, unique_id: str):
        super().__init__()
        self._client = client
        self._base_path = os.path.join(
            ".cache/ds_tool/textgen",
            unique_id.replace("https://", "").replace("/", "__"),
        )
        os.makedirs(self._base_path, exist_ok=True)

    def chat_completion(self, **kwargs) -> str:
        text_hash = hashlib.sha256(json.dumps(kwargs).encode()).hexdigest()

        cache_path = os.path.join(self._base_path, f"{text_hash}.txt")

        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return f.read()

        response = self._client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content

        with open(cache_path, "w") as f:
            f.write(text)

        return text


class CachingTtsWrapper:
    def __init__(self, client: tts.Client, provider: str):
        super().__init__()
        self._client = client
        self._base_path = os.path.join(".cache/ds_tool/tts", provider)

    @overload
    def tts(self, text: str, voice: Optional[str] = None) -> bytes: ...

    @overload
    def tts(self, text: List[str], voice: Optional[str] = None) -> List[bytes]: ...

    def tts(
        self, text: Union[str, List[str]], voice: Optional[str] = None
    ) -> Union[bytes, List[bytes]]:
        text_hash = hashlib.sha256(str(text).encode()).hexdigest()

        if isinstance(text, list):
            if voice == tts.RANDOM_VOICE_KEY and hasattr(self._client, "ALL_VOICES"):
                voice = random.Random(int(text_hash, 16)).choice(
                    self._client.ALL_VOICES
                )
            return [self.tts(t, voice) for t in text]

        path = os.path.join(self._base_path, voice or "default")
        os.makedirs(path, exist_ok=True)

        cache_path = os.path.join(path, f"{text_hash}.wav")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return f.read()

        wav = self._client.tts(text, voice)

        with open(cache_path, "wb") as f:
            f.write(wav)

        return wav
