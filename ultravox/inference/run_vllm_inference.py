import dataclasses
import os
import subprocess
import time

import pandas as pd
import requests
import simple_parsing
import wandb

from ultravox.tools import push_to_hub
from ultravox.training.helpers import prefetch_weights


@dataclasses.dataclass
class InferenceArgs:
    model: str
    evalset: str
    push_to_hub: bool = True

    # HuggingFace Hub model_id to push to
    def __post_init__(self):
        clean_name = (
            os.path.basename(self.model)
            .replace(":", "_")
            .replace(".", "_")
            .replace("/", "_")
        )
        while "--" in clean_name or "__" in clean_name:
            clean_name = clean_name.replace("--", "_").replace("__", "_")
        self.hf_upload_model = f"fixie-ai-dev/{clean_name}"

        self.exp_name = f"eval__{clean_name}__{self.evalset}"


def main():
    args = simple_parsing.parse(InferenceArgs)

    model_path = prefetch_weights.download_weights(
        [], args.model, include_models_from_load_dir=True
    )

    args.model = model_path or args.model

    if args.push_to_hub:
        push_to_hub.main(
            push_to_hub.UploadToHubArgs(
                model=args.model,
                hf_upload_model=args.hf_upload_model,
                verify=False,
                device="cpu",
            )
        )
        args.model = args.hf_upload_model

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "ultravox"),
        config=dataclasses.asdict(args),
        name=args.exp_name,
        tags=["eval", "vllm"],
        dir="logs",
    )

    log_dir = os.path.join(run.dir, "oaieval")

    vllm_process = start_vllm(args)

    try:
        wait_for_vllm_to_start()
        metrics_df = run_oaievalset(args, log_dir)
        run.log({"eval": wandb.Table(data=metrics_df)})
        run.log({x["eval"]: x["score"] for x in metrics_df.iloc})

    finally:
        # Make sure the VLLM server is stopped
        vllm_process.terminate()
        try:
            vllm_process.wait(10)
        except:
            vllm_process.kill()
            vllm_process.wait(2)


def run_oaievalset(args: InferenceArgs, log_dir: str):
    env = os.environ.copy()
    env["EVALS_THREADS"] = "256"
    env["ULTRAVOX_API_KEY"] = "..."  # API key is not neede for local evaluation

    command = [
        "oaievalset",
        "--record_dir",
        log_dir,
        "generation/direct/vllm",
        args.evalset,
        "--registry_path",
        "ultravox/inference/oaieval_registry",
    ]

    # Run the evaluation set
    subprocess.run(command, check=True, env=env)

    # Extract the results from the log directory
    subprocess.run(
        [
            "python",
            "-m",
            "evals.elsuite.audio.make_table",
            "--out_dir",
            log_dir,
            "--log_dir",
            log_dir,
        ],
        check=True,
    )

    df = pd.read_csv(os.path.join(log_dir, "results.csv"))

    return df


def start_vllm(args: InferenceArgs) -> subprocess.Popen:
    env = os.environ.copy()
    env["VLLM_CONFIGURE_LOGGING"] = "0"
    return subprocess.Popen(
        [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            args.model,
            "--enable-chunked-prefill=False",
            "--max-model-len=8192",
            "--served-model-name=fixie-ai/ultravox",
            "--tensor-parallel-size=8",
            "--uvicorn-log-level=warning",
        ],
        env=env,
        preexec_fn=os.setsid,
    )


def wait_for_vllm_to_start(port: int = 8000):
    while True:
        try:
            response = requests.get(f"http://localhost:{port}/health")
            response.raise_for_status()
            break
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.HTTPError:
            pass

        print("Waiting for server to start...")
        time.sleep(2)


if __name__ == "__main__":
    main()
