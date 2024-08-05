#!/bin/bash

# Given the tools dataset, we want to create the audio for all user messages
just tts -d fixie-ai/tools -u fixie-ai/tools-audio --private \
    -c @ultravox/tools/ds_tool/user_messages.jinja -j -a user_message_audios \
    -V random --num_workers 20 -i eleven --token $HF_WRITE_TOKEN