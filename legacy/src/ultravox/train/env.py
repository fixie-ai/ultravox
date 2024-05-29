import logging
import os

try:
    from azureml.core import Run
except ImportError:
    Run = None


def set_env_vars_azure():
    if "TOKENIZERS_PARALLELISM" not in os.environ:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if os.environ.get("WANDB_PROJECT", default=None) is None:
        os.environ["WANDB_PROJECT"] = "ultravox"

    try:
        run = Run.get_context()

        os.environ["HF_ACCESS_TOKEN"] = run.get_secret(name="hf-access-token")
        os.environ["WANDB_API_KEY"] = run.get_secret("wandb-api-key")

        os.environ["CLEARML_WEB_HOST"] = "https://app.clear.ml"
        os.environ["CLEARML_API_HOST"] = "https://api.clear.ml"
        os.environ["CLEARML_FILES_HOST"] = "https://files.clear.ml"
        os.environ["CLEARML_API_ACCESS_KEY"] = run.get_secret("clearml-api-access-key")
        os.environ["CLEARML_API_SECRET_KEY"] = run.get_secret("clearml-api-secret-key")
    except:
        logging.warning("Failed to set environment variables from Azure Key Vault")
