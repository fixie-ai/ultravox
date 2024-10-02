from transformers import PreTrainedTokenizer
import torch
from ultravox.tokenizer.decoder.pretrained import WavTokenizer
import os
import json

class CustomWavTokenizer(PreTrainedTokenizer):

    def __init__(self, config_path, model_path, sample_rate=16000, vocab_size=4096,
                 padding_side="left", truncation_side="right", **kwargs):
        self.config_path = config_path
        self.model_path = model_path
        self.sample_rate = sample_rate
        self._vocab_size = vocab_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
        self.wavtokenizer = self.wavtokenizer.to(self.device)

        # Initialize vocabulary
        self.vocab = {str(i): i for i in range(self._vocab_size)}
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        
        super().__init__(
            padding_side=padding_side,
            truncation_side=truncation_side,
            **kwargs
        )
        self.pad_token_id = 0 
        self.pad_token = "0"

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def _tokenize(self, wav):
        if not isinstance(wav, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        wav = wav.to(self.device)
        bandwidth_id = torch.tensor([0], device=self.device)
        _, discrete_code = self.wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
        
        return discrete_code

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def _convert_tokens_to_wav(self, discrete_code):
        features = self.wavtokenizer.codes_to_features(discrete_code)
        bandwidth_id = torch.tensor([0])
        audio_out = self.wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
        return audio_out.cpu()

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Save the configuration
        config = {
            "config_path": self.config_path,
            "model_path": self.model_path,
            "vocab_size": self._vocab_size,
            "padding_side": self.padding_side,
            "truncation_side": self.truncation_side
        }
        with open(os.path.join(save_directory, self.vocab_files_names["config_file"]), "w") as f:
            json.dump(config, f)
        
        # Save the model weights
        torch.save(self.wavtokenizer.state_dict(), os.path.join(save_directory, self.vocab_files_names["model_file"]))

        # Save vocabulary
        with open(os.path.join(save_directory, self.vocab_files_names["vocab_file"]), "w") as f:
            json.dump(self.vocab, f)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        # Load the configuration
        config_file = os.path.join(pretrained_model_name_or_path, cls.vocab_files_names["config_file"])
        with open(config_file, "r") as f:
            config = json.load(f)
        
        # Create an instance of the tokenizer
        tokenizer = cls(config["config_path"], config["model_path"], config["vocab_size"], 
                        padding_side=config.get("padding_side", "right"),
                        truncation_side=config.get("truncation_side", "right"),
                        **kwargs)
        
        # Load the model weights
        model_file = os.path.join(pretrained_model_name_or_path, cls.vocab_files_names["model_file"])
        tokenizer.wavtokenizer.load_state_dict(torch.load(model_file, map_location=tokenizer.device))
        
        # Load vocabulary
        vocab_file = os.path.join(pretrained_model_name_or_path, cls.vocab_files_names["vocab_file"])
        with open(vocab_file, "r") as f:
            tokenizer.vocab = json.load(f)
        tokenizer.ids_to_tokens = {v: k for k, v in tokenizer.vocab.items()}
        return tokenizer

    def encode(self, wav, **kwargs):
        return self._tokenize(wav)

    def decode(self, discrete_code, **kwargs):
        return self._convert_tokens_to_wav(discrete_code)

# # Example usage:
# config_path = "./configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
# hf_model_name = "novateur/WavTokenizer"
# checkpoint_file = "WavTokenizer_small_600_24k_4096.ckpt"
# model_path = hf_hub_download(repo_id=hf_model_name, filename=checkpoint_file)

# tokenizer = CustomWavTokenizer(config_path, model_path)

# # Encoding
# audio_path = "./data/input3.wav"
# wav, sr = torchaudio.load(audio_path)
# wav = convert_audio(wav, sr, 24000, 1) 

# print("wav:", wav)
# token_ids = tokenizer.encode(wav)

# print("token_ids:", token_ids)

# # Decoding
# audio_out = tokenizer.decode(token_ids)



# # Save the output audio
# output_path = "./data/input3-out.wav"
# torchaudio.save(output_path, audio_out, 24000)
# print(f"Audio saved to {output_path}")

# # Save the tokenizer
# tokenizer.save_pretrained("./saved_tokenizer")

# # Load the tokenizer
# loaded_tokenizer = CustomWavTokenizer.from_pretrained("./saved_tokenizer")