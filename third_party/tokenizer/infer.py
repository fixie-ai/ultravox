import torchaudio
import torch
from third_party.tokenizer.decoder.pretrained import WavTokenizer
from third_party.tokenizer.encoder.utils import convert_audio
from huggingface_hub import hf_hub_download

device=torch.device('cpu')

config_path = "../tokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
hf_model_name = "novateur/WavTokenizer"
checkpoint_file = "WavTokenizer_small_600_24k_4096.ckpt"

model_path = hf_hub_download(repo_id=hf_model_name, filename=checkpoint_file)

audio_file_name = "input-130s"
audio_path = f"./data/{audio_file_name}.wav"

print("model path:", model_path)

# Constructing discrete code from audio
wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device)

wav, sr = torchaudio.load(audio_path)
wav = convert_audio(wav, sr, 24000, 1) 
bandwidth_id = torch.tensor([0])
wav=wav.to(device)
print("wav shape", wav.shape)
_,discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
print("discrete code", discrete_code)
print("discrete code shape", discrete_code.shape)

# Constructing audio from discrete code 
features = wavtokenizer.codes_to_features(discrete_code)
bandwidth_id = torch.tensor([0])
audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)


# Export the wave file
output_path = f"./data/{audio_file_name}-out.wav"
torchaudio.save(output_path, audio_out.cpu(), 24000)
print(f"Audio saved to {output_path}")
