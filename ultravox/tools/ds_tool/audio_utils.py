import io

import numpy as np
import soundfile as sf


def numpy_audio_to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    with io.BytesIO() as buffer:
        sf.write(buffer, audio, sample_rate, format="wav")
        return buffer.getvalue()
