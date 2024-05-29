#!/bin/bash
set -x
which python
# python -m venv .venv
# source .venv/bin/activate

# pip install poetry
# poetry install --sync --no-root 

pip install -r requirements.txt
echo "Done installing requirements. Running task."
# TODO: nproc depends on the number of GPUs, right? How to set automatically?
torchrun --nproc_per_node=8 --master_port=1234 -m train.train --config_path $@
# python -m train.train --config_path $@
# sleep infinity


sleep 20 && pkill -f wandb