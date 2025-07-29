import dataclasses
import os
import subprocess
import sys
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
    clean_name: str | None = None
    push_to_hub: bool = True
    max_samples: int = -1
    chat_template: str | None = None

    # HuggingFace Hub model_id to push to
    def __post_init__(self):
        if self.clean_name is None:
            self.clean_name = os.path.basename(self.model)

        self.clean_name = (
            self.clean_name.replace(":", "_").replace(".", "_").replace("/", "_")
        )

        while "--" in self.clean_name or "__" in self.clean_name:
            self.clean_name = self.clean_name.replace("--", "_").replace("__", "_")
        self.hf_upload_model = f"fixie-ai/dev-{self.clean_name}"

        self.exp_name = f"eval__{self.clean_name}__{self.evalset}"


def main() -> None:
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
    else:
        print(
            "WARNING: `push_to_hub` is set to False, so unmerged LoRA modules may not be properly evaluated."
        )

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "ultravox"),
        config=dataclasses.asdict(args),
        name=args.exp_name,
        tags=["eval", "vllm"],
        dir="logs",
    )

    log_dir = os.path.join(run.dir, "oaieval")

    eval_process: subprocess.Popen | None = None
    vllm_process = start_vllm(args)

    try:
        wait_for_vllm_to_start(vllm_process)
        eval_process = run_oaievalset(args, log_dir)
        assert eval_process is not None
        monitor_processes_and_update_results(vllm_process, eval_process, log_dir)
    finally:
        print(
            "Terminating VLLM and OAIEVAL, please wait a few seconds for the processes to finish..."
        )
        # Make sure the VLLM server is stopped
        vllm_process.terminate()
        if eval_process is not None:
            eval_process.terminate()
        try:
            vllm_process.wait(2)
            if eval_process is not None:
                eval_process.wait(2)
        except:  # noqa: E722
            vllm_process.kill()
            if eval_process is not None:
                eval_process.kill()


def run_oaievalset(args: InferenceArgs, log_dir: str) -> subprocess.Popen:
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

    if args.max_samples > 0:
        command.append("--max_samples")
        command.append(str(args.max_samples))

    # Run the evaluation set in a separate process
    return subprocess.Popen(command, env=env, stdout=sys.stdout, stderr=sys.stderr)


def monitor_processes_and_update_results(
    vllm_process: subprocess.Popen, eval_process: subprocess.Popen, log_dir: str
) -> None:
    # Monitor both processes
    while eval_process.poll() is None:
        update_results(log_dir, silent=True)

        if vllm_process.poll() is not None:
            # VLLM died - kill eval process and raise error
            eval_process.terminate()
            eval_process.wait(2)
            raise RuntimeError(
                f"VLLM server died with return code {vllm_process.returncode}"
            )

        time.sleep(5)

    update_results(log_dir)

    # Check if eval process failed
    if eval_process.returncode != 0:
        raise RuntimeError(
            f"Evaluation failed with return code {eval_process.returncode}"
        )


def update_results(log_dir: str, silent: bool = False) -> None:
    # Extract the results from the log directory
    command = [
        "python3",
        "-m",
        "evals.elsuite.audio.make_table",
        "--out_dir",
        log_dir,
        "--log_dir",
        log_dir,
    ]
    if silent:
        command.append("--silent")

    subprocess.run(command, check=True)

    try:
        df = pd.read_csv(os.path.join(log_dir, "results.csv"))
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("No results found")
        return

    if wandb.run is not None:
        wandb.run.log({"eval": wandb.Table(data=df)})
        wandb.run.log({x["eval"]: x["score"] for x in df.iloc})
        for file in os.listdir(log_dir):
            if file.endswith(".log") or file.endswith(".csv"):
                wandb.run.save(os.path.join(log_dir, file))


def start_vllm(args: InferenceArgs) -> subprocess.Popen:
    env = os.environ.copy()
    env["VLLM_CONFIGURE_LOGGING"] = "0"
    env["VLLM_USE_V1"] = "1"
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    command = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--no-enable-chunked-prefill",
        "--max-model-len=8192",
        "--served-model-name=fixie-ai/ultravox",
        "--tensor-parallel-size=8",
        "--uvicorn-log-level=warning",
        '--override-generation-config={"attn_temperature_tuning": true}',
        "--trust-remote-code",
        "--enforce-eager",
    ]
    if args.chat_template:
        command.append("--chat-template=" + args.chat_template)

    return subprocess.Popen(
        command,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def wait_for_vllm_to_start(
    vllm_process: subprocess.Popen, port: int = 8000, max_wait_time: int = 600
) -> None:
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get(f"http://localhost:{port}/health")
            response.raise_for_status()
            break
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.HTTPError:
            pass

        if vllm_process.poll() is not None:
            raise Exception("VLLM server failed to start")

        print("Waiting for server to start...")
        time.sleep(2)


if __name__ == "__main__":
    main()
