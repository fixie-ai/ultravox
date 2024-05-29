Recognize this speech, this is input: prompt/gs_172_cropped.wav

Input audio: gs_172_cropped.wav (14 seconds long)
Total generation time (in seconds):

# Whisper on Mac CPU

tiny (40M) 0.36
large-v2 (1.5B) 10.59
============

# Whisper on GPU

tiny (40M) 0.23
medium (0.8B) 1.02
large-v2 (1.5B) 1.37

---

Cuda v12.3
medium 0.88
large-v2 1.17

---

SDP + Flash + Float16
medium 0.72
large-v2 0.94
============

# SpeechGPT (7B) on GPU 2.9

SDP + Flash 2.0
507 input tokens, 60 output tokens
============
