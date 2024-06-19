# Synthetically enhanced datasets

## BoolQ

```bash
just ds_tool textgen -d google/boolq -u fixie-ai/boolq-audio -c explanation -T @ultravox/tools/ds_templates/boolq_template.jinja --token $HF_WRITE_TOKEN
```

## SODA

The SODA dataset was slightly modified to allow for training a voice-text model as follows to create the `fixie-ai/soda-audio` dataset:

### Alternative last turn (Llama3-8b)

```bash
just ds_tool textgen -d allenai/soda --shuffle True -s test -n 1000 -u fixie-ai/soda -c alt_last_turn \
    -T @ultravox/tools/ds_templates/soda_template.jinja -b https://api.fireworks.ai/inference/v1 \
    -k $FIREWORKS_API_KEY -m accounts/fireworks/models/llama-v3-8b-instruct --token $HF_WRITE_TOKEN

just ds_tool textgen -d allenai/soda --shuffle True -s validation -n 1000 -u fixie-ai/soda -c alt_last_turn \
    -T @ultravox/tools/ds_templates/soda_template.jinja -b https://api.fireworks.ai/inference/v1 \
    -k $FIREWORKS_API_KEY -m accounts/fireworks/models/llama-v3-8b-instruct --token $HF_WRITE_TOKEN

just ds_tool textgen -d allenai/soda --shuffle True -s train -n 100000 -u fixie-ai/soda -c alt_last_turn \
    -T @ultravox/tools/ds_templates/soda_template.jinja -b https://api.fireworks.ai/inference/v1 \
    -k $FIREWORKS_API_KEY -m accounts/fireworks/models/llama-v3-8b-instruct --token $HF_WRITE_TOKEN
```

### TTS the turn before last

```bash
just ds_tool tts -d fixie-ai/soda -u fixie-ai/soda-copy -c "dialogue[-2]" -a audio_one_but_last -i eleven -V random --token $HF_WRITE_TOKEN
```
