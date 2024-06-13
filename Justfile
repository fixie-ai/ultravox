export WANDB_PROJECT:="ultravox"
export WANDB_LOG_MODEL:="checkpoint"
export PROJECT_DIR:="ultravox"
export MCLOUD_CLUSTER:="r7z22p1"
export MCLOUD_INSTANCE:="oci.bm.gpu.b4.8"

default: format check test

install:
    pip install poetry==1.7.1
    poetry install

format:
    poetry run autoflake ${PROJECT_DIR} --remove-all-unused-imports --quiet --in-place -r --exclude third_party
    poetry run isort ${PROJECT_DIR} --force-single-line-imports
    poetry run black ${PROJECT_DIR}

check:
    poetry run black ${PROJECT_DIR} --check
    poetry run isort ${PROJECT_DIR} --check --force-single-line-imports
    poetry run autoflake  ${PROJECT_DIR} --check --quiet --remove-all-unused-imports -r --exclude third_party
    poetry run mypy ${PROJECT_DIR}    

test *ARGS=".":
    cd ${PROJECT_DIR} && poetry run pytest --ignore third_party {{ARGS}}

test-verbose *ARGS=".":
    cd ${PROJECT_DIR} && poetry run pytest --ignore third_party {{ARGS}} -vv --log-cli-level=INFO {{ARGS}}

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
    poetry run python -m ultravox.tools.eval_tool {{FLAGS}}

tts *FLAGS:
    poetry run python -m ultravox.tools.ds_tool tts {{FLAGS}}

ds_tool *FLAGS:
    poetry run python -m ultravox.tools.ds_tool {{FLAGS}}

mds *FLAGS:
    poetry run python -m ultravox.tools.mds_tool {{FLAGS}}

gradio *FLAGS:
    poetry run python -m ultravox.tools.gradio_demo {{FLAGS}}

run *FLAGS:
    poetry run mcli run -f mcloud.yaml --follow {{FLAGS}}

mcloud *FLAGS:
    poetry run mcli interactive {{FLAGS}} --cluster ${MCLOUD_CLUSTER} --instance ${MCLOUD_INSTANCE}  --name `whoami` --command "bash -c \"$(cat setup.sh)\"" 
