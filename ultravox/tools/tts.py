import os
import requests


def _make_ssml(voice: str, text: str):
    return f"""
    <speak version="1.0" xml:lang="en-US">
        <voice xml:lang="en-US" name="{voice}">
            <prosody rate="100%">{text}</prosody>
        </voice>
    </speak>"""


class AzureTts:
    DEFAULT_VOICE = "en-US-JennyNeural"

    def __init__(self):
        self._voice = self.DEFAULT_VOICE
        self._session = requests.ClientSession()

    def _tts(self, text: str):
        region = "westus"
        api_key = os.environ["AZURE_TTS_API_KEY"]
        output_format = "raw-48khz-16bit-mono-pcm"
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
        return response.content
