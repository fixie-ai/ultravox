import abc
import base64
import io
import json
import os
from typing import Any, Dict, List, Optional, Literal
from xml.sax import saxutils
from google.cloud import aiplatform
import numpy as np
import requests
import soundfile as sf
import tempfile

from cambai import CambAI
from cambai.models.output_type import OutputType

RANDOM_VOICE_KEY = "random"
REQUEST_TIMEOUT = 30
NUM_RETRIES = 3

def _make_ssml(voice: str, text: str):
    return f"""
    <speak version="1.0" xml:lang="en-US">
        <voice xml:lang="en-US" name="{voice}">
            {saxutils.escape(text)}
        </voice>
    </speak>"""


class Client(abc.ABC):
    DEFAULT_VOICE: str
    ALL_VOICES: List[str]

    def __init__(self, sample_rate: int = 16000):
        if not hasattr(self, "DEFAULT_VOICE"):
            raise ValueError("DEFAULT_VOICE must be defined in subclasses.")
        if not hasattr(self, "ALL_VOICES"):
            raise ValueError("ALL_VOICES must be defined in subclasses.")

        self._session = requests.Session()
        retries = requests.adapters.Retry(total=NUM_RETRIES)
        self._session.mount(
            "https://", requests.adapters.HTTPAdapter(max_retries=retries)
        )
        self._sample_rate = sample_rate

    @abc.abstractmethod
    def tts(self, text: str, voice: Optional[str] = None) -> bytes:
        raise NotImplementedError

    def _post(self, url: str, headers: Dict[str, str], json: Dict[str, Any]):
        response = self._session.post(
            url, headers=headers, json=json, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response

    def _handle_pcm_response(self, response: requests.Response) -> bytes:
        pcm_array = np.frombuffer(response.content, dtype=np.int16)
        wav_bytes = io.BytesIO()
        sf.write(wav_bytes, pcm_array, self._sample_rate, format="WAV")
        return wav_bytes.getvalue()

    def resolve_voice(self, voice: Optional[str]) -> str:
        voice = voice or self.DEFAULT_VOICE
        if voice == RANDOM_VOICE_KEY:
            # Every process has same random seed, so we mix in the PID here for more variation.
            i = np.random.randint(len(self.ALL_VOICES)) + os.getpid()
            voice = self.ALL_VOICES[i % len(self.ALL_VOICES)]
        return voice


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
        voice = self.resolve_voice(voice)
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
        voice = self.resolve_voice(voice)
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream?output_format=pcm_16000"
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


class CambAIVertexTTS(Client):
    DEFAULT_VOICE = "reference_voice"
    ALL_VOICES = ["reference_voice"]
    # Mars7 supported languages
    Mars7Language = Literal["de-de", "en-gb", "en-us", "es-us", "es-es", "fr-ca", "fr-fr", "ja-jp", "ko-kr", "zh-cn"]
    def __init__(self, sample_rate: int = 24000):
        super().__init__(sample_rate)     
        # Initialize Google Cloud AI Platform
        self.project_id = os.environ.get("PROJECT_ID")
        self.location = os.environ.get("LOCATION")
        self.endpoint_id = os.environ.get("ENDPOINT_ID")
        
        if not self.endpoint_id:
            raise ValueError("ENDPOINT_ID environment variable is required for Camb AI Vertex TTS")
        
        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is required for Camb AI Vertex TTS")
        
        aiplatform.init(project=self.project_id, location=self.location)
        
        # Cache for base64-encoded reference audio to avoid re-encoding
        self._reference_audio_cache = {}

    def _encode_audio_to_base64(self, audio_path: str) -> str:
        """Encode audio file to base64 string with caching."""
        if audio_path in self._reference_audio_cache:
            return self._reference_audio_cache[audio_path]
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Reference audio file not found: {audio_path}")
        
        with open(audio_path, "rb") as f:
            audio_bytes = base64.b64encode(f.read()).decode("utf-8")
        
        self._reference_audio_cache[audio_path] = audio_bytes
        return audio_bytes

    def tts(self, text: str, reference_audio_path: str, reference_text: str, voice: Optional[str] = None, language: Mars7Language = "en-us") -> bytes:
        """Synthesize text to speech using Camb AI MARS7 model.
        
        Args:
            text: Text to synthesize (required)
            reference_audio_path: Path to reference audio file (required) 
            reference_text: Transcription of reference audio (required)
            voice: Voice parameter (not used in MARS7, kept for compatibility)
            language: Target language code (default: "en-us")
            
        Returns:
            WAV audio bytes
            
        Raises:
            ValueError: If required parameters are missing
        """
        
        ref_audio_path = reference_audio_path
        if not ref_audio_path:
            raise ValueError(
                "Reference audio path is required for Camb AI Vertex TTS. "
                "Provide it via reference_audio_path parameter"
            )
        
        ref_text = reference_text
        if not ref_text:
            raise ValueError(
                "Reference text is required for Camb AI Vertex TTS. "
                "Provide it via reference_text parameter"
            )
        
        audio_ref_bytes = self._encode_audio_to_base64(ref_audio_path)
        
        # Prepare prediction instances
        instances = {
            "text": text,
            "audio_ref": audio_ref_bytes,
            "ref_text": ref_text,
            "language": language
        }
        
        endpoint = aiplatform.Endpoint(endpoint_name=self.endpoint_id)
        data = {"instances": [instances]}
        
        try:
            response = endpoint.raw_predict(
                body=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            
            response_data = json.loads(response.content)
            if "predictions" not in response_data or not response_data["predictions"]:
                raise ValueError("Invalid response from Camb AI Vertex TTS: no predictions found")
            audio_bytes = base64.b64decode(response_data["predictions"][0])
            
            # Write FLAC bytes to temp file first to avoid BytesIO format issues
            with tempfile.NamedTemporaryFile(suffix='.flac') as tmp_flac:
                tmp_flac.write(audio_bytes)
                tmp_flac.flush()
                data, samplerate = sf.read(tmp_flac.name)
            
            # Convert to target sample rate if needed
            if samplerate != self._sample_rate:
                ratio = self._sample_rate / samplerate
                new_length = int(len(data) * ratio)
                data = np.interp(np.linspace(0, len(data), new_length), np.arange(len(data)), data)
            
            wav_bytes = io.BytesIO()
            sf.write(wav_bytes, data, self._sample_rate, format="WAV")
            return wav_bytes.getvalue()
            
        except Exception as e:
            raise RuntimeError(f"Error calling Camb AI Vertex TTS: {str(e)}")


class CambAITTS(Client):
    DEFAULT_VOICE = "20303"
    ALL_VOICES = []

    def __init__(self, sample_rate: int = 16000):
        if not CambAI:
            raise ImportError(
                "CambAI SDK is not available. Install it with: pip install cambai"
            )
        
        super().__init__(sample_rate)
        
        api_key = os.environ.get("CAMB_API_KEY")
        if not api_key:
            raise ValueError("CAMB_API_KEY environment variable is required for CambAI TTS")
        
        self._client = CambAI(api_key=api_key)
        self._populate_voices()

    def _populate_voices(self):
        """Populate ALL_VOICES by fetching available voices from CambAI API"""
        try:
            voices = self._client.list_voices()
            self.ALL_VOICES = [str(voice.id) for voice in voices]
            if not self.ALL_VOICES:
                self.ALL_VOICES = [self.DEFAULT_VOICE]
        except Exception as e:
            print(f"Warning: Could not fetch CambAI voices, using default: {e}")
            self.ALL_VOICES = [self.DEFAULT_VOICE]

    def tts(self, text: str, voice: Optional[str] = None) -> bytes:
        """Synthesize text to speech using CambAI SDK.
        
        Args:
            text: Text to synthesize
            voice: Voice ID to use (string number like "20303")
            
        Returns:
            WAV audio bytes
        """
        voice = self.resolve_voice(voice)
        
        if not voice.isnumeric():
            raise ValueError(f"Invalid CambAI voice ID: {voice}")
        voice_id = int(voice)
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp_file:
                temp_path = tmp_file.name
            
                self._client.text_to_speech(
                    text=text,
                    voice_id=voice_id,
                    output_type=OutputType.RAW_BYTES,
                    save_to_file=temp_path
                )
                
                # Read the saved audio file and convert to WAV at target sample rate
                data, samplerate = sf.read(temp_path)
                
                # Convert to target sample rate if needed
                if samplerate != self._sample_rate:
                    ratio = self._sample_rate / samplerate
                    new_length = int(len(data) * ratio)
                    data = np.interp(np.linspace(0, len(data), new_length), np.arange(len(data)), data)
                
                # Convert to WAV format
                wav_bytes = io.BytesIO()
                sf.write(wav_bytes, data, self._sample_rate, format="WAV")
            
            return wav_bytes.getvalue()
            
        except Exception as e:
            raise RuntimeError(f"Error calling CambAI TTS: {str(e)}")


def create_client(implementation: str, sample_rate: int):
    if implementation == "azure":
        return AzureTts(sample_rate=sample_rate)
    elif implementation == "eleven":
        return ElevenTts(sample_rate=sample_rate)
    elif implementation == "cambai":
        return CambAIVertexTTS(sample_rate=sample_rate)
    elif implementation == "cambai-sdk":
        return CambAITTS(sample_rate=sample_rate)
    raise ValueError(f"Unknown TTS implementation: {implementation}")
