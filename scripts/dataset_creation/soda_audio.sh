#!/bin/bash
# Steps to reproduce the fixie-ai/soda-audio dataset

# Step 0: create the `fixie-ai/soda-audio` dataset in the UI or using huggingface_hub.create_repo

# Step 1: Create an alternative last turn using Llama3-8b Instruct model
# We want the model to generate the same response whether the input is audio or text

just ds_tool textgen -d allenai/soda --shuffle True -s test -n 1000 -u fixie-ai/soda-audio -c alt_last_turn \
    -T @ultravox/tools/ds_tool/soda_alt_last_turn.jinja -j -b https://api.fireworks.ai/inference/v1 \
    -k $FIREWORKS_API_KEY -m accounts/fireworks/models/llama-v3-8b-instruct --token $HF_WRITE_TOKEN

just ds_tool textgen -d allenai/soda --shuffle True -s validation -n 1000 -u fixie-ai/soda-audio -c alt_last_turn \
    -T @ultravox/tools/ds_tool/soda_alt_last_turn.jinja -j -b https://api.fireworks.ai/inference/v1 \
    -k $FIREWORKS_API_KEY -m accounts/fireworks/models/llama-v3-8b-instruct --token $HF_WRITE_TOKEN

just ds_tool textgen -d allenai/soda --shuffle True -s train -n 100000 -u fixie-ai/soda-audio -c alt_last_turn \
    -T @ultravox/tools/ds_tool/soda_alt_last_turn.jinja -j -b https://api.fireworks.ai/inference/v1 \
    -k $FIREWORKS_API_KEY -m accounts/fireworks/models/llama-v3-8b-instruct --token $HF_WRITE_TOKEN


# Step 2: TTS the turn before last: audio input for the model

just ds_tool tts -d fixie-ai/soda-audio -u fixie-ai/soda-audio -c "dialogue[-2]" -a audio_second_last_turn -i eleven -V random --token $HF_WRITE_TOKEN -N 8
