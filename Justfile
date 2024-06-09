export WANDB_PROJECT:="ultravox"
export WANDB_LOG_MODEL:="checkpoint"
export PROJECT_DIR:="ultravox"
export VENV_NAME:="venv"
export MCLOUD_CLUSTER:="r7z22"
export MCLOUD_INSTANCE:="oci.bm.gpu.b4.8"

default: format check test

create-venv:
    pip install --upgrade virtualenv  # older virtualenv had some issues in Debian
    python -m venv ${VENV_NAME}
    just install

install:
    # Install torch 2.2.1 if needed, not present in requirements.txt
    just python -c \"import torch\" 2>/dev/null || just pip install torch==2.2.1
    just pip install -r requirements.txt
    just pip install -r requirements-dev.txt
    just python -m pip install types-requests

format:
    . ./activate ${VENV_NAME} && autoflake ${PROJECT_DIR} --remove-all-unused-imports --quiet --in-place -r --exclude third_party --exclude ultravox/model/gazelle
    . ./activate ${VENV_NAME} && isort ${PROJECT_DIR} --force-single-line-imports
    . ./activate ${VENV_NAME} && black ${PROJECT_DIR}

check:
    . ./activate ${VENV_NAME} && black ${PROJECT_DIR} --check
    . ./activate ${VENV_NAME} && isort ${PROJECT_DIR} --check --force-single-line-imports
    . ./activate ${VENV_NAME} && autoflake  ${PROJECT_DIR} --check --quiet --remove-all-unused-imports -r --exclude third_party --exclude ultravox/model/gazelle
    . ./activate ${VENV_NAME} && mypy ${PROJECT_DIR}    

test *ARGS=".":
    . ./activate ${VENV_NAME} && cd ${PROJECT_DIR} && pytest --ignore third_party {{ARGS}}

@python *FLAGS:
    . ./activate ${VENV_NAME} && python {{FLAGS}}

@pip *FLAGS:
    . ./activate ${VENV_NAME} && pip {{FLAGS}}

train *FLAGS:
    just python -m ultravox.training.train {{FLAGS}}

train_asr *FLAGS:
    just train --config_path ultravox/training/configs/asr_tinyllama.yaml {{FLAGS}}

browse *FLAGS:
    just python -m ultravox.tools.data_tool {{FLAGS}}

infer *FLAGS:
    just python -m ultravox.tools.infer_tool {{FLAGS}}

eval *FLAGS:
    just python -m ultravox.tools.eval_tool {{FLAGS}}

tts *FLAGS:
    just python -m ultravox.tools.tts_tool {{FLAGS}}

mds *FLAGS:
    just python -m ultravox.tools.mds_tool {{FLAGS}}

gradio *FLAGS:
    just python -m ultravox.tools.gradio_demo {{FLAGS}}

run *FLAGS:
    mcli run -f mcloud.yaml --follow {{FLAGS}}

mcloud *FLAGS:
    mcli interactive {{FLAGS}} --cluster ${MCLOUD_CLUSTER} --instance ${MCLOUD_INSTANCE}  --name `whoami` --command "bash -c \"$(cat setup.sh)\"" 
