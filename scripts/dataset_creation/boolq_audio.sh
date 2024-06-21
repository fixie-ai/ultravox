#!/bin/bash
# Create the fixie-ai/boolq-audio dataset

# Step 1: Create a plausible explanation for the answer
# This explanation is only used in the `-extended` version of the dataset and is used mainly for better training.
just ds_tool textgen -d google/boolq -u fixie-ai/boolq-audio -c explanation -T @ultravox/tools/ds_templates/boolq_template.jinja --token $HF_WRITE_TOKEN

# Step 2: TTS the question into the audio input for the model
# Note: the original dataset was not created using this script. This is just an example of how to create the audio version of the dataset
just ds_tool tts -d fixie-ai/boolq-audio -u fixie-ai/boolq-audio -c question -a audio -i azure --token $HF_WRITE_TOKEN
