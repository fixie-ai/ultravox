import io
import os
from typing import Optional
from xml.sax import saxutils

import numpy as np
import requests
import soundfile as sf


def _make_ssml(voice: str, text: str):
    return f"""
    <speak version="1.0" xml:lang="en-US">
        <voice xml:lang="en-US" name="{voice}">
            {saxutils.escape(text)}
        </voice>
    </speak>"""


class AzureTts:
    DEFAULT_VOICE = "en-US-JennyNeural"

    def __init__(self, voice: Optional[str] = None, sample_rate: int = 16000):
        self._session = requests.Session()
        self._voice = voice or self.DEFAULT_VOICE
        self._sample_rate = sample_rate

    def tts(self, text: str):
        region = "westus"
        api_key = os.environ.get("AZURE_TTS_API_KEY") or os.environ.get(
            "AZURE_WESTUS_TTS_API_KEY"
        )
        output_format = f"raw-{self._sample_rate // 1000}khz-16bit-mono-pcm"
        url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
        headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": output_format,
            "User-Agent": "MyTTS",
        }
        body = _make_ssml(self._voice, text)
        response = self._session.post(url, headers=headers, data=body)
        response.raise_for_status()

        pcm_array = np.frombuffer(response.content, dtype=np.int16)
        wav_bytes = io.BytesIO()
        sf.write(wav_bytes, pcm_array, self._sample_rate, format="wav")
        return wav_bytes.getvalue()
