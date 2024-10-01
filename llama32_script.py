# Before starting, comment the following lines in the forward function of MllamaForConditionalGeneration in modeling_mllama.py
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/modeling_mllama.py#L2152-L2155
# These lines don't allow you to specify inputs_embeds when vision input is present.

# %%
import os
import glob
import requests
import librosa
from PIL import Image
import transformers
import safetensors.torch

from ultravox.model import ultravox_model
from ultravox.model import ultravox_config
from ultravox.model import ultravox_processing
from ultravox.model import wandb_utils


audio_model_id = "openai/whisper-medium"
text_model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"


audio_processor = transformers.AutoProcessor.from_pretrained(audio_model_id)
text_processor = transformers.AutoProcessor.from_pretrained(text_model_id)
processor = ultravox_processing.UltravoxProcessor(audio_processor, text_processor)

config = ultravox_config.UltravoxConfig(
    text_model_id=text_model_id,
    audio_model_id=audio_model_id,
)

model = ultravox_model.UltravoxModel(config)

load_path = "wandb://fixie/ultravox/model-zhuang.2024-08-21-ultravox.medium-1e:v5"
if wandb_utils.is_wandb_url(load_path):
    # We assume that the weights are already downloaded via prefetch_weights.py
    # and hence this is just resolving the path. If the weights are not downloaded,
    # we might see a race condition here when using DDP.
    load_path = wandb_utils.download_model_from_wandb(load_path)
if os.path.isdir(load_path):
    load_path = os.path.join(load_path, "model*.safetensors")
paths = glob.glob(load_path)
assert len(paths) > 0, f"No model files found at {load_path}"
for path in glob.glob(load_path):
    state_dict = safetensors.torch.load_file(path)
    mismatch = model.load_state_dict(state_dict, strict=False)
    if mismatch.unexpected_keys:
        raise ValueError(f"Unexpected keys in state dict: {mismatch.unexpected_keys}")

# %%

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)


audio_path = "ttsmaker-file-2024-9-30-14-53-45.mp3"
audio, _ = librosa.load(audio_path, sr=22050)


messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "\n<|audio|>\n"},
            # {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
        ],
    }
]


input_text = processor.tokenizer.apply_chat_template(
    messages, add_generation_prompt=False, tokenize=False
)
inputs = processor(
    images=image,
    text=input_text,
    audio=audio,
    return_tensors="pt",
    sampling_rate=16000,
)
outputs = model.generate(**inputs, max_new_tokens=100)
print("Response for Audio + Image as input:")
print(processor.decode(outputs[0].tolist()))

# %%

### Text only example
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": "If I had to write a haiku for this one, it would be: ",
            },
        ],
    }
]

input_text = processor.tokenizer.apply_chat_template(
    messages, add_generation_prompt=False, tokenize=False
)
simple_inputs = text_processor(
    images=image, text=[input_text], add_special_tokens=False, return_tensors="pt"
)
outputs = model.language_model.generate(**simple_inputs, max_new_tokens=100)
print("Response for Text + Image as input:")
print(processor.decode(outputs[0].tolist()))

# %%

### It's important to send input_ids to .generate even if inputs_embeds is provided
### o.w. the model will ignore the vision input
inputs_embeds = model.get_input_embeddings()(simple_inputs["input_ids"])
outputs = model.language_model.generate(
    # **{**simple_inputs, "inputs_embeds": inputs_embeds, "input_ids": None},  # correct
    **{**simple_inputs, "inputs_embeds": inputs_embeds},  # incorrect
    max_new_tokens=100,
)
print("Incorrect response for Text + Image as input, ignoring Image entirely:")
print(processor.decode(outputs[0].tolist()))
