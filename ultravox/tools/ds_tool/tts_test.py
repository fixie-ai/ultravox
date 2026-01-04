import os
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

from ultravox.tools.ds_tool import tts


class TestCambTts(unittest.TestCase):
    @patch.dict(os.environ, {"CAMB_API_KEY": "test_key"})
    def test_init_with_api_key(self):
        client = tts.CambTts(sample_rate=16000)
        self.assertEqual(client._sample_rate, 16000)
        self.assertEqual(client._api_key, "test_key")

    def test_init_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(AssertionError):
                tts.CambTts()

    @patch.dict(os.environ, {"CAMB_API_KEY": "test_key"})
    def test_resolve_voice_default(self):
        client = tts.CambTts()
        voice = client.resolve_voice(None)
        self.assertEqual(voice, "2681")

    @patch.dict(os.environ, {"CAMB_API_KEY": "test_key"})
    def test_resolve_voice_custom(self):
        client = tts.CambTts()
        voice = client.resolve_voice("1234")
        self.assertEqual(voice, "1234")

    @patch.dict(os.environ, {"CAMB_API_KEY": "test_key"})
    def test_tts_success(self):
        client = tts.CambTts(sample_rate=16000)

        # Mock PCM response (24kHz, 1 second of silence)
        mock_response = MagicMock()
        mock_response.content = np.zeros(24000, dtype=np.int16).tobytes()
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._session, "post", return_value=mock_response):
            result = client.tts("Hello world")

            self.assertIsInstance(result, bytes)
            # Should be WAV format (starts with RIFF)
            self.assertTrue(result.startswith(b"RIFF"))

    @patch.dict(os.environ, {"CAMB_API_KEY": "test_key"})
    def test_tts_resampling(self):
        """Test that 24kHz audio is resampled to target sample rate."""
        client = tts.CambTts(sample_rate=16000)

        mock_response = MagicMock()
        # 1 second of 24kHz audio (sine wave to verify resampling)
        t = np.linspace(0, 1, 24000)
        audio = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        mock_response.content = audio.tobytes()
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._session, "post", return_value=mock_response):
            result = client.tts("Hello")

            # Result should be WAV with 16kHz sample rate
            self.assertTrue(result.startswith(b"RIFF"))

    @patch.dict(os.environ, {"CAMB_API_KEY": "test_key"})
    def test_tts_no_resampling_at_24khz(self):
        """Test that 24kHz audio is NOT resampled when target is 24kHz."""
        client = tts.CambTts(sample_rate=24000)

        mock_response = MagicMock()
        mock_response.content = np.zeros(24000, dtype=np.int16).tobytes()
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._session, "post", return_value=mock_response):
            result = client.tts("Hello")

            self.assertTrue(result.startswith(b"RIFF"))

    @patch.dict(os.environ, {"CAMB_API_KEY": "test_key"})
    def test_tts_request_payload(self):
        """Test that the request payload is correctly formed."""
        client = tts.CambTts(sample_rate=16000)

        mock_response = MagicMock()
        mock_response.content = np.zeros(24000, dtype=np.int16).tobytes()
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._session, "post", return_value=mock_response) as mock_post:
            client.tts("Hello world", voice="2681", language="fr-fr", speed=1.5)

            # Verify the request was made with correct parameters
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            self.assertEqual(
                call_args.kwargs["headers"]["x-api-key"], "test_key"
            )
            payload = call_args.kwargs["json"]
            self.assertEqual(payload["text"], "Hello world")
            self.assertEqual(payload["voice_id"], 2681)
            self.assertEqual(payload["language"], "fr-fr")
            self.assertEqual(payload["voice_settings"]["speed"], 1.5)


class TestCreateClient(unittest.TestCase):
    @patch.dict(os.environ, {"CAMB_API_KEY": "test_key"})
    def test_create_camb_client(self):
        client = tts.create_client("camb", 16000)
        self.assertIsInstance(client, tts.CambTts)

    @patch.dict(os.environ, {"AZURE_TTS_API_KEY": "test_key"})
    def test_create_azure_client(self):
        client = tts.create_client("azure", 16000)
        self.assertIsInstance(client, tts.AzureTts)

    @patch.dict(os.environ, {"ELEVEN_API_KEY": "test_key"})
    def test_create_eleven_client(self):
        client = tts.create_client("eleven", 16000)
        self.assertIsInstance(client, tts.ElevenTts)

    def test_create_unknown_client(self):
        with self.assertRaises(ValueError) as ctx:
            tts.create_client("unknown", 16000)
        self.assertIn("Unknown TTS implementation", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
