import wandb

WANDB_PREFIX = "wandb://"


def is_wandb_url(model_path: str) -> bool:
    return model_path.startswith(WANDB_PREFIX)


def download_model_from_wandb(model_url: str) -> str:
    assert is_wandb_url(model_url)
    api = wandb.Api()
    # example artifact name: "fixie/ultravox/model-llama2_asr_gigaspeech:v0"
    artifact = api.artifact(model_url[len(WANDB_PREFIX) :])
    model_path = artifact.download()
    return model_path
