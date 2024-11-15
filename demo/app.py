import gradio as gr
from gradio_webrtc import WebRTC, ReplyOnPause, AdditionalOutputs
import transformers
import numpy as np
from twilio.rest import Client
import os
import torch
import librosa

pipe = transformers.pipeline(
    model="fixie-ai/ultravox-v0_4_1-llama-3_1-8b",
    trust_remote_code=True,
    device=torch.device("cuda"),
)
whisper = transformers.pipeline(
    model="openai/whisper-large-v3-turbo", device=torch.device("cuda")
)

# only needed if deploying on a cloud environment like EC2, spaces
account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
auth_token = os.environ.get("TWILIO_AUTH_TOKEN")

if account_sid and auth_token:
    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    rtc_configuration = {
        "iceServers": token.ice_servers,
        "iceTransportPolicy": "relay",
    }
else:
    rtc_configuration = None


def transcribe(
    audio: tuple[int, np.ndarray],
    transformers_chat: list[dict],
    conversation: list[dict],
):
    original_sr = audio[0]
    target_sr = 16000

    audio_sr = librosa.resample(
        audio[1].astype(np.float32) / 32768.0, orig_sr=original_sr, target_sr=target_sr
    )

    tf_input = [d for d in transformers_chat]

    output = pipe(
        {"audio": audio_sr, "turns": tf_input, "sampling_rate": target_sr},
        max_new_tokens=512,
    )
    transcription = whisper({"array": audio_sr.squeeze(), "sampling_rate": target_sr})

    conversation.append({"role": "user", "content": transcription["text"]})
    conversation.append({"role": "assistant", "content": output})
    transformers_chat.append({"role": "user", "content": transcription["text"]})
    transformers_chat.append({"role": "assistant", "content": output})

    yield AdditionalOutputs(transformers_chat, conversation)


with gr.Blocks() as demo:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    Talk to Ultravox Llama 3.1 8b (Powered by WebRTC ⚡️)
    </h1>
    <p style='text-align: center'>
    Once you grant access to your microphone, you can talk naturally to Ultravox.
    When you stop talking, the audio will be sent for processing.
    </p>
    <p style='text-align: center'>
    Each conversation is limited to 90 seconds. Once the time limit is up you can rejoin the conversation.
    </p>
    """
    )
    with gr.Row():
        transformers_chat = gr.State(
            value=[
                {
                    "role": "system",
                    "content": "You are a friendly and helpful character. You love to answer questions for people.",
                }
            ]
        )
        with gr.Group():
            transcript = gr.Chatbot(label="transcript", type="messages")
            audio = WebRTC(
                rtc_configuration=rtc_configuration,
                label="Stream",
                mode="send",
                modality="audio",
            )

    audio.stream(
        ReplyOnPause(transcribe),
        inputs=[audio, transformers_chat, transcript],
        outputs=[audio],
        time_limit=90,
    )
    audio.on_additional_outputs(
        lambda t, g: (t, g),
        outputs=[transformers_chat, transcript],
        queue=False,
        show_progress="hidden",
    )

if __name__ == "__main__":
    demo.launch()
