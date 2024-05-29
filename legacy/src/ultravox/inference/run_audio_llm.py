import safetensors.torch
import torch
from train import data
from train.models import multimodal as multimodal_models

# from train.models import text as text_models


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

audio_llm_config = multimodal_models.SpeechLMConfig(
    audio_enc_name="wav2vec2",
    llm_name="tinyllama",
    # llm_name="llama2-7b",
    audio_stride=6,
    # torch_dtype=dtype,
    use_cpu=device.type == "cpu",
)

audio_llm_model = multimodal_models.SpeechLM(audio_llm_config)
processor = multimodal_models.SpeechLMProcessor.from_config(audio_llm_config)

tokenizer = processor.tokenizer
audio_proc = processor.audio_processor
total_audio_stride = processor.total_audio_stride


tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
# tokenizer.padding_side = "left"


# print(load_result)
for param in audio_llm_model.parameters():
    param.requires_grad = False

audio_llm_model.eval()
audio_llm_model.llm.config.use_cache = True

train_ds, _ = data.get_dataset(
    dataset_name=data.DatasetType.GIGASPEECH,
    dev_env=True,
    # max_duration_in_seconds=config.max_audio_duration_in_seconds,
)

tokenize_fn = data.AudioTextTokenizer(
    audio_proc,
    tokenizer,
    total_audio_stride,
    cfg=data.AudioTextTokenizerConfig(inference_mode=True),
)

data_collator = data.DataCollatorForSeq2SeqWithAudio(
    tokenizer,
    pad_to_multiple_of=8,
    return_tensors="pt",
    padding=True,
    audio_dtype=dtype,
)


train_ds = train_ds.map(tokenize_fn)

# For LibriSpeech
# atrain_ds = train_ds.remove_columns(["file", "speaker_id", "chapter_id", "id"])
# For GigaSpeech
# atrain_ds = train_ds.remove_columns(["audio", "text", "id"])

it = iter(train_ds)
pt_x = next(it)
pt_x = next(it)
pt_x = next(it)
text = pt_x.pop("text")
audio = pt_x.pop("audio")
# x = data_collator([pt_x])


import torch

# audio_llm_model.can_generate = lambda: True
# audio_llm_model.generation_config = audio_llm_model.llm.generation_config
# audio_llm_model.config = argparse.Namespace(is_encoder_decoder=False)
# audio_llm_model.main_input_name = "input_ids"
# audio_llm_model.device = device

audio_llm_model.to(device=device)


# audio_llm_model = torch.compile(audio_llm_model)

# import pdb

# pdb.set_trace()


# tokenize_fn.instructions = ""
audio_only_input = tokenize_fn({"audio": audio})
# audio_only_input = tokenize_fn({"audio": audio, "text": " ".join(text.split()[:1])})
# audio_only_input = tokenize_fn({"audio": audio})
print(tokenizer.decode(audio_only_input["input_ids"]))

# print(tokenizer.decode(audio_only_input["input_ids"][:20]))
# audio_only_input["input_ids"] = audio_only_input["input_ids"][:20]
# audio_only_input["attention_mask"] = audio_only_input["attention_mask"][:-3]
# audio_only_input["input_ids"] = audio_only_input["input_ids"][:20]
# audio_only_input["attention_mask"] = audio_only_input["attention_mask"][:20]
# audio_only_input["audio_features"] = audio_only_input["audio_features"][
#     ..., : 4 * total_audio_stride + 1
# ]
# audio_only_input["audio_token_mask"] = audio_only_input["audio_token_mask"][:20]
audio_only_input.pop("attention_mask")
# print("Labels:", audio_only_input.pop("labels"))
audio_only_input.pop("audio")
print("Full text:", text)
print("Text provided:", audio_only_input.pop("text", ""))
audio_only_inputs = data_collator([audio_only_input])
# audio_only_inputs.pop("audio_features")
audio_only_inputs = {k: v.to(device=device) for k, v in audio_only_inputs.items()}

# audio_llm_model = combined_model
with torch.no_grad():
    outputs = audio_llm_model(**audio_only_inputs)

tokens = outputs["logits"].argmax(dim=-1)
print("Loss before: ", outputs["loss"])
print(tokenizer.decode(tokens.tolist()[0]))


state_dict = safetensors.torch.load_file(
    # "runs/audiollm-tinyllamaR0-wav2vec2frozen-GigaSpeech-template-smooth0.1-8bs/model.safetensors",
    "runs/audiollm-tinyllamaR0-wav2vec2frozen-GigaSpeech-template-smooth0.1-shiftfix-64bs/model.safetensors",
    # "../../audiollm-tinyllamaR0-wav2vec2frozen-GigaSpeech-template-smooth0.1-model.safetensors",
    # "../../audiollm-llama7bR0-wav2vec2frozen-GigaSpeech-template-smooth0.1-model.safetensors",
    device=device.type,
)

state_dict = {
    # k.replace("audio_to_embed", "audio_to_embed.1"): v
    k: v
    for k, v in state_dict.items()
    if "llm" not in k
}

load_result = audio_llm_model.load_state_dict(state_dict, False)

if load_result.unexpected_keys:
    raise ValueError(f"Unexpected keys: {load_result.unexpected_keys}")

outputs = audio_llm_model(**audio_only_inputs)
tokens = outputs["logits"].argmax(dim=-1)
print("Loss after: ", outputs["loss"])
print(tokenizer.decode(tokens.tolist()[0]))

tokens = audio_llm_model.generate(
    **audio_only_inputs,
    max_new_tokens=20,
    # num_beams=4,
    do_sample=True,
    temperature=0.7,
    top_k=10,
    top_p=0.95,
)
print(tokenizer.decode(tokens.tolist()[0]))

import pdb

pdb.set_trace()

# TODO: stop criterion?

outputs = audio_llm_model(**audio_only_inputs)
tokens = outputs["logits"].argmax(dim=-1)

# audio_len = outputs["audio_embed"].shape[-2]
# logits = outputs["logits"][..., audio_len:, :]

# next_logits = outputs["logits"][..., 1:, :]
# expected = x["labels"][..., :-1]

# ccr = next_logits.argmax(dim=-1) == expected
# assisted_cer = 1 - ccr.sum() / (expected != -100).sum()


predicted = tokenizer.decode(tokens.tolist()[0])


# def transcribe():
#     sr, y = audio
#     y = y.astype(np.float32)
#     y /= np.max(np.abs(y))
#     if sr != 16_000:
#         print(f"Got wrong sampling rate {sr}, resampling.")
#         y = librosa.resample(y, orig_sr=sr, target_sr=16_000)

#     last_audio = y
#     input_features = processor(
#         y, sampling_rate=16_000, return_tensors="pt"
#     ).input_features
#     predicted_ids = model.generate(input_features)
#     transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
#     return transcription[0]


# # with gr.Blocks() as demo:
# #     with gr.Column():
# #         for _ in range(4):
# #             audio, label = generate_audio()
# #             output = gr.Audio(sources=["microphone"])

# demo = gr.Interface(
#     transcribe,
#     gr.Audio(sources=["microphone"]),
#     "text",
# )
# _ = demo.launch(inline=False, inbrowser=True, debug=True)

# demo.launch(debug=True)
