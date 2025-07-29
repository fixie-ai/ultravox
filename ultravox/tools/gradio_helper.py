from typing import Optional

from ultravox.inference import ultravox_infer

inference: Optional[ultravox_infer.UltravoxInference] = None


def get_inference(args) -> ultravox_infer.UltravoxInference:
    global inference
    if not inference:
        inference = ultravox_infer.UltravoxInference(
            args.model_path,
            device=args.device,
            data_type=args.data_type,
            conversation_mode=True,
            enable_thinking=args.enable_thinking,
            thinking_regex=args.thinking_regex,
        )
    return inference
