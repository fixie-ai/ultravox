export WANDB_PROJECT:="ultravox"
export WANDB_LOG_MODEL:="checkpoint"
export PROJECT_DIR:="ultravox"
export MCLOUD_CLUSTER:="r7z22p1"
export MCLOUD_INSTANCE:="oci.bm.gpu.b4.8"
export MFA_ENV_NAME:="aligner"


#!/usr/bin/env bash

# Install dependencies
install() {
    pip install poetry==1.7.1
    poetry install
}

# Format code
format() {
    local dir="${PROJECT_DIR}"
    poetry run autoflake "$dir" --remove-all-unused-imports --quiet --in-place -r
    poetry run isort "$dir" --force-single-line-imports
    poetry run black "$dir"
}

# Check code formatting and type correctness
check() {
    local dir="${PROJECT_DIR}"
    poetry run black "$dir" --check
    poetry run isort "$dir" --check --force-single-line-imports
    poetry run autoflake "$dir" --check --quiet --remove-all-unused-imports -r
    poetry run mypy "$dir"
}

# Run tests with optional verbosity
test() {
    local args="${1:-.}"
    cd "$PROJECT_DIR" && poetry run coverage run --source="$PROJECT_DIR" -m pytest --ignore third_party "$args"
    print_coverage
}

test_verbose() {
    local args="${1:-.}"
    cd "$PROJECT_DIR" && poetry run coverage run --source="$PROJECT_DIR" -m pytest --ignore third_party "$args" -vv --log-cli-level=INFO
    print_coverage
}

# Print coverage report
print_coverage() {
    local args="$1"
    cd "$PROJECT_DIR" && poetry run coverage report --omit "*_test.py" --sort miss "$args"
}

# Run Python scripts with arguments
py_exec() {
    poetry run python "$@"
}

# Training and evaluation commands
train() {
    poetry run python -m ultravox.training.train "$@"
}

train_asr() {
    train --config_path ultravox/training/configs/asr_tinyllama.yaml "$@"
}

browse() {
    poetry run python -m ultravox.tools.data_tool "$@"
}

infer() {
    poetry run python -m ultravox.tools.infer_tool "$@"
}

eval() {
    poetry run python -m ultravox.evaluation.eval "$@"
}

tts() {
    poetry run python -m ultravox.tools.ds_tool.ds_tool tts "$@"
}

ds_tool() {
    poetry run python -m ultravox.tools.ds_tool.ds_tool "$@"
}

mds() {
    poetry run python -m ultravox.tools.mds_tool "$@"
}

gradio() {
    poetry run gradio ultravox/tools/gradio_demo.py "$@"
}

# Cloud execution commands
run() {
    poetry run mcli run -f mcloud.yaml --follow "$@"
}

vllm_eval() {
    poetry run mcli run -f mcloud_eval.yaml --follow "$@"
}

mcloud() {
    poetry run mcli interactive "$@" --cluster "${MCLOUD_CLUSTER}" --instance "${MCLOUD_INSTANCE}" --name "$(whoami)" --command "bash -c \"$(cat setup.sh)\""
}

# Check and install Conda if missing
check_conda() {
    if ! command -v conda &> /dev/null; then
        echo "Conda is not installed. Installing Miniconda..."
        mkdir -p ~/miniconda3
        case "$(uname)" in
            Darwin)
                curl -o ~/miniconda3/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
                ;;
            Linux)
                wget -O ~/miniconda3/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
                ;;
            *)
                echo "Unknown operating system."; return 1;
                ;;
        esac
        bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && rm ~/miniconda3/miniconda.sh
    else
        echo "Conda is already installed."
    fi
}

# Install Montreal Forced Aligner
install_mfa() {
    check_conda
    if conda env list | grep -q "$MFA_ENV_NAME"; then
        echo "Environment '$MFA_ENV_NAME' already exists."
    else
        echo "Creating environment '$MFA_ENV_NAME'..."
        conda create --name "$MFA_ENV_NAME" python=3.8 -y
        conda install -n "$MFA_ENV_NAME" -c conda-forge montreal-forced-aligner -y
        conda run -n "$MFA_ENV_NAME" mfa model download acoustic english_mfa
        conda run -n "$MFA_ENV_NAME" mfa model download dictionary english_mfa
    fi
}

# Default action
if [[ "$1" == "" ]]; then
    format
    check
fi
