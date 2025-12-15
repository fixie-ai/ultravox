from typing import Optional

import torch
import transformers
from transformers import BitsAndBytesConfig

from ultravox.model import ultravox_model


def load_ultravox_quantized(
    model_path: str,
    load_in_nbit: Optional[int] = None,
    device_map: str = "auto",
):
    """
    Load an Ultravox model with quantization.

    Args:
        model_path: HuggingFace model path
        load_in_nbit: Number of bits for quantization (4 or 8)
        device_map: Device distribution for model layers

    Returns:
        Quantized Ultravox model
    """
    config = ultravox_model.UltravoxConfig.from_pretrained(model_path)
    print("Creating empty model...")
    model = ultravox_model.UltravoxModel(config)

    quant_config = None
    if load_in_nbit:
        if load_in_nbit not in [4, 8]:
            raise ValueError(f"load_in_nbit must be 4 or 8, got {load_in_nbit}")
        quant_config = BitsAndBytesConfig(
            load_in_8bit=(load_in_nbit == 8),
            load_in_4bit=(load_in_nbit == 4),
        )

    print(f"Loading language model (quantized={load_in_nbit}-bit)" if load_in_nbit else "Loading language model...")
    if config.text_model_id:
        lm_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch.float16,
        }
        if quant_config:
            lm_kwargs["quantization_config"] = quant_config
            lm_kwargs["low_cpu_mem_usage"] = True

        model.language_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.text_model_id,
            **lm_kwargs
        )

    print("Loading audio tower...")
    if not config.llm_only_training:
        if config.audio_model_id:
            audio_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto" if device_map == "auto" else device_map,
            }
            if quant_config:
                audio_kwargs["quantization_config"] = quant_config
                audio_kwargs["low_cpu_mem_usage"] = True

            if "whisper" in config.audio_model_id.lower():
                from ultravox.model.ultravox_model import ModifiedWhisperEncoder
                model.audio_tower = ModifiedWhisperEncoder.from_pretrained(
                    config.audio_model_id,
                    **audio_kwargs
                )
            else:
                model.audio_tower = transformers.AutoModel.from_pretrained(
                    config.audio_model_id,
                    **audio_kwargs
                )
            print(f"   Loaded audio tower from {config.audio_model_id}")
        else:
            print("   Loading embedded audio tower from Ultravox weights...")
            from safetensors import safe_open
            from huggingface_hub import hf_hub_download
            import json

            audio_weights = {}
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            try:
                try:
                    shard_file = hf_hub_download(repo_id=model_path, filename="model.safetensors.index.json")
                    with open(shard_file) as f:
                        index = json.load(f)

                    for key, shard_name in index.get("weight_map", {}).items():
                        if "audio_tower" in key:
                            shard_path = hf_hub_download(repo_id=model_path, filename=shard_name)
                            with safe_open(shard_path, framework="pt", device="cpu") as f:
                                tensor = f.get_tensor(key)
                                new_key = key.replace("audio_tower.", "")
                                audio_weights[new_key] = tensor.to(device)
                except:
                    model_file = hf_hub_download(repo_id=model_path, filename="model.safetensors")
                    with safe_open(model_file, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            if "audio_tower" in key:
                                tensor = f.get_tensor(key)
                                new_key = key.replace("audio_tower.", "")
                                audio_weights[new_key] = tensor.to(device)

                if audio_weights:
                    model.audio_tower.load_state_dict(audio_weights, strict=False, assign=True)
                    print(f"   Loaded audio tower ({len(audio_weights)} tensors)")
                else:
                    print("   Warning: No audio_tower weights found")
            except Exception as e:
                print(f"   Warning: Error loading audio tower: {e}")
                try:
                    model.audio_tower = model.audio_tower.to_empty(device=device)
                    model.audio_tower.load_state_dict(audio_weights, strict=False)
                    print(f"   Loaded with to_empty ({len(audio_weights)} tensors)")
                except Exception as e2:
                    print(f"   Error: Failed to load audio tower: {e2}")

    print("Loading projector...")
    try:
        from safetensors import safe_open
        from huggingface_hub import hf_hub_download
        import json

        projector_weights = {}
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        try:
            shard_file = hf_hub_download(repo_id=model_path, filename="model.safetensors.index.json")
            with open(shard_file) as f:
                index = json.load(f)

            for key, shard_name in index.get("weight_map", {}).items():
                if "multi_modal_projector" in key:
                    shard_path = hf_hub_download(repo_id=model_path, filename=shard_name)
                    with safe_open(shard_path, framework="pt", device="cpu") as f:
                        tensor = f.get_tensor(key)
                        new_key = key.replace("multi_modal_projector.", "")
                        projector_weights[new_key] = tensor.to(device)
        except:
            try:
                model_file = hf_hub_download(repo_id=model_path, filename="model.safetensors")
                with safe_open(model_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if "multi_modal_projector" in key:
                            tensor = f.get_tensor(key)
                            new_key = key.replace("multi_modal_projector.", "")
                            projector_weights[new_key] = tensor.to(device)
            except Exception as e:
                print(f"   Warning: No model.safetensors found: {e}")

        if projector_weights:
            model.multi_modal_projector.load_state_dict(projector_weights, strict=False, assign=True)
            model.multi_modal_projector = model.multi_modal_projector.to(device=device, dtype=torch.float16)
            print(f"   Loaded {len(projector_weights)} projector tensors on {device} (float16)")
        else:
            print("   Warning: No projector weights found - using random weights")
    except Exception as e:
        print(f"   Warning: Error loading projector: {e}")

    print("Model loaded successfully")
    return model
