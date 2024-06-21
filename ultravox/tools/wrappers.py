import hashlib
import json
import os

import openai


class CachingChatWrapper:
    def __init__(self, client: openai.Client, base_url: str):
        super().__init__()
        self._client = client
        self._base_path = os.path.join(
            ".cache/ds_tool/textgen", base_url.replace("/", "__")
        )
        os.makedirs(self._base_path, exist_ok=True)

    def chat_completion(self, **kwargs) -> str:
        text_hash = hashlib.sha256(json.dumps(kwargs).encode()).hexdigest()

        cache_path = os.path.join(self._base_path, f"{text_hash}.wav")

        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return f.read()

        response = self._client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content

        with open(cache_path, "w") as f:
            f.write(text)

        return text
