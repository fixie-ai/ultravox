import abc
import hashlib
import io
import logging
import os
from typing import Any, Dict, Optional
from xml.sax import saxutils

import numpy as np
import requests
import soundfile as sf

RANDOM_VOICE_KEY = "random"


def _make_ssml(voice: str, text: str):
    return f"""
    <speak version="1.0" xml:lang="en-US">
        <voice xml:lang="en-US" name="{voice}">
            {saxutils.escape(text)}
        </voice>
    </speak>"""


class Client(abc.ABC):
    def __init__(self, sample_rate: int = 16000):
        self._session = requests.Session()
        self._sample_rate = sample_rate

    @abc.abstractmethod
    def tts(self, text: str, voice: Optional[str] = None):
        raise NotImplementedError

    def _post(self, url: str, headers: Dict[str, str], json: Dict[str, Any]):
        response = self._session.post(url, headers=headers, json=json)
        response.raise_for_status()
        return response

    def _handle_pcm_response(self, response: requests.Response):
        pcm_array = np.frombuffer(response.content, dtype=np.int16)
        wav_bytes = io.BytesIO()
        sf.write(wav_bytes, pcm_array, self._sample_rate, format="wav")
        return wav_bytes.getvalue()


class AzureTts(Client):
    DEFAULT_VOICE = "en-US-JennyNeural"
    ALL_VOICES = [
        "en-US-AvaNeural",
        "en-US-AndrewNeural",
        "en-US-EmmaNeural",
        "en-US-BrianNeural",
        "en-US-JennyNeural",
        "en-US-GuyNeural",
        "en-US-AriaNeural",
        "en-US-DavisNeural",
        "en-US-JaneNeural",
        "en-US-JasonNeural",
        "en-US-SaraNeural",
        "en-US-TonyNeural",
        "en-US-NancyNeural",
        "en-US-AmberNeural",
        "en-US-AnaNeural",
        "en-US-AshleyNeural",
        "en-US-BrandonNeural",
        "en-US-ChristopherNeural",
        "en-US-CoraNeural",
        "en-US-ElizabethNeural",
        "en-US-EricNeural",
        "en-US-JacobNeural",
        "en-US-MichelleNeural",
        "en-US-MonicaNeural",
        "en-US-RogerNeural",
    ]

    def tts(self, text: str, voice: Optional[str] = None):
        voice = voice or self.DEFAULT_VOICE
        if voice == RANDOM_VOICE_KEY:
            voice = np.random.choice(self.ALL_VOICES)
        assert voice
        region = "westus"
        api_key = os.environ.get("AZURE_TTS_API_KEY") or os.environ.get(
            "AZURE_WESTUS_TTS_API_KEY"
        )
        assert api_key, "Please set the AZURE_TTS_API_KEY environment variable."
        output_format = f"raw-{self._sample_rate // 1000}khz-16bit-mono-pcm"
        url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
        headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": output_format,
            "User-Agent": "MyTTS",
        }
        body = _make_ssml(voice, text)
        return self._handle_pcm_response(self._post(url, headers, body))


class ElevenTts(Client):
    DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"
    DEFAULT_MODEL = "eleven_multilingual_v2"
    ALL_VOICES = [
        "21m00Tcm4TlvDq8ikWAM",
        "29vD33N1CtxCmqQRPOHJ",
        "2EiwWnXFnvU5JabPnv8n",
        "5Q0t7uMcjvnagumLfvZi",
        "AZnzlk1XvdvUeBnXmlld",
        "CYw3kZ02Hs0563khs1Fj",
        "D38z5RcWu1voky8WS1ja",
        "EXAVITQu4vr4xnSDxMaL",
        "ErXwobaYiN019PkySvjV",
        "GBv7mTt0atIp3Br8iCZE",
        "IKne3meq5aSn9XLyUdCD",
        "JBFqnCBsd6RMkjVDRZzb",
        "LcfcDJNUP1GQjkzn1xUU",
        "MF3mGyEYCl7XYWbV9V6O",
        "N2lVS1w4EtoT3dr4eOWO",
        "ODq5zmih8GrVes37Dizd",
        "SOYHLrjzK2X1ezoPC6cr",
        "TX3LPaxmHKxFdv7VOQHJ",
        "ThT5KcBeYPX3keUQqHPh",
        "TxGEqnHWrfWFTfGW9XjX",
        "VR6AewLTigWG4xSOukaG",
        "XB0fDUnXU5powFXDhCwa",
        "Xb7hH8MSUJpSbSDYk0k2",
        "XrExE9yKIg1WjnnlVkGX",
        "ZQe5CZNOzWyzPSCn5a3c",
        "Zlb1dXrM653N07WRdFW3",
    ]

    def tts(self, text: str, voice: Optional[str] = None):
        voice = voice or self.DEFAULT_VOICE
        if voice == RANDOM_VOICE_KEY:
            # Every process has same random seed, so we mix in the PID here for more variation.
            i = np.random.randint(len(self.ALL_VOICES)) + os.getpid()
            voice = self.ALL_VOICES[i % len(self.ALL_VOICES)]
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream?output_format=pcm_16000"
        logging.debug(f"url {url}")
        headers = {"xi-api-key": os.environ["ELEVEN_API_KEY"]}
        body = {
            "text": text,
            "model_id": self.DEFAULT_MODEL,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": False,
            },
        }
        return self._handle_pcm_response(self._post(url, headers, body))


class CachedClientWrapper:
    def __init__(self, client: Client, provider: str):
        super().__init__()
        self._client = client
        self._base_path = os.path.join(".cache/ds_tool", provider)

    def tts(self, text: str, voice: Optional[str] = None):
        path = os.path.join(self._base_path, voice or "default")
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        os.makedirs(path, exist_ok=True)

        cache_path = os.path.join(path, f"{text_hash}.wav")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return f.read()

        pcm = self._client.tts(text, voice)

        with open(cache_path, "wb") as f:
            f.write(pcm)

        return pcm


def create_client(implementation: str, sample_rate: int):
    client: Client
    if implementation == "azure":
        client = AzureTts(sample_rate=sample_rate)
    elif implementation == "eleven":
        client = ElevenTts(sample_rate=sample_rate)
    else:
        raise ValueError(f"Unknown TTS implementation: {implementation}")

    return CachedClientWrapper(client, provider=implementation)
