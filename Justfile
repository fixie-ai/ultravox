export WANDB_PROJECT:="ultravox"
export WANDB_LOG_MODEL:="checkpoint"
export PROJECT_DIR:="ultravox"
export MCLOUD_CLUSTER:="r7z22p1"
export MCLOUD_INSTANCE:="oci.bm.gpu.b4.8"
export MFA_ENV_NAME:="aligner"

default: format check test

install:
    pip install poetry==1.7.1
    poetry install

format:
    poetry run autoflake {{PROJECT_DIR}} --remove-all-unused-imports --quiet --in-place -r 
    poetry run isort {{PROJECT_DIR}} --force-single-line-imports 
    poetry run black {{PROJECT_DIR}} 

check:
    poetry run black {{PROJECT_DIR}} --check 
    poetry run isort {{PROJECT_DIR}} --check --force-single-line-imports
    poetry run autoflake {{PROJECT_DIR}} --check --quiet --remove-all-unused-imports -r 
    poetry run mypy {{PROJECT_DIR}} 

test *ARGS=".":
    cd ${PROJECT_DIR} && poetry run coverage run --source=${PROJECT_DIR} -m pytest --ignore third_party {{ARGS}}
    just print-coverage

test-verbose *ARGS=".":
    cd ${PROJECT_DIR} && poetry run coverage run --source=${PROJECT_DIR} -m pytest --ignore third_party {{ARGS}} -vv --log-cli-level=INFO
    just print-coverage

# the following assumes the coverage report is already created by the test command
print-coverage *ARGS:
    cd ${PROJECT_DIR} && poetry run coverage report --omit "*_test.py" --sort miss {{ARGS}}

@python *FLAGS:
    poetry run python {{FLAGS}}

train *FLAGS:
    poetry run python -m ultravox.training.train {{FLAGS}}

train_asr *FLAGS:
    just train --config_path ultravox/training/configs/asr_tinyllama.yaml {{FLAGS}}

browse *FLAGS:
    poetry run python -m ultravox.tools.data_tool {{FLAGS}}

infer *FLAGS:
    poetry run python -m ultravox.tools.infer_tool {{FLAGS}}

eval *FLAGS:
    poetry run python -m ultravox.evaluation.eval {{FLAGS}}

tts *FLAGS:
    poetry run python -m ultravox.tools.ds_tool.ds_tool tts {{FLAGS}}

ds_tool *FLAGS:
    poetry run python -m ultravox.tools.ds_tool.ds_tool {{FLAGS}}

mds *FLAGS:
    poetry run python -m ultravox.tools.mds_tool {{FLAGS}}

gradio *FLAGS:
    poetry run gradio ultravox/tools/gradio_demo.py {{FLAGS}}

run *FLAGS:
    poetry run mcli run -f mcloud.yaml --follow {{FLAGS}}

vllm_eval *FLAGS:
    poetry run mcli run -f mcloud_eval.yaml --follow {{FLAGS}}

mcloud *FLAGS:
    poetry run mcli interactive {{FLAGS}} --cluster ${MCLOUD_CLUSTER} --instance ${MCLOUD_INSTANCE}  --name `whoami` --command "bash -c \"$(cat setup.sh)\"" 

@check_conda:
    if ! command -v conda &> /dev/null; then  \
        echo "Conda is not installed.";  \
        mkdir -p ~/miniconda3;  \
        if [ "$(uname)" = "Darwin" ]; then  \
            echo "Downloading MacOS Miniconda.";  \
            curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh;  \
        elif [ "$(uname)" = "Linux" ]; then  \
            echo "Downloading Linux Miniconda.";  \
            wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh  \
        else  \
            echo "Unknown operating system.";  \
        fi;  \
        bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3;  \
        rm ~/miniconda3/miniconda.sh;  \
    else  \
        echo "Conda is installed.";  \
    fi

@install_mfa: check_conda
    if conda env list | grep -q "$MFA_ENV_NAME"; then  \
        echo "Environment '$MFA_ENV_NAME' already exists.";  \
    else  \
        echo "Creating environment '$MFA_ENV_NAME'.";  \
        conda create --name "$MFA_ENV_NAME" python=3.8 -y;  \
        conda create -n "$MFA_ENV_NAME" -c conda-forge montreal-forced-aligner;  \
        conda run -n "$MFA_ENV_NAME" mfa model download acoustic english_mfa;  \
        conda run -n "$MFA_ENV_NAME" mfa model download dictionary english_mfa;  \
    fi