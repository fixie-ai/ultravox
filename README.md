<p align="center">
  <picture>
    <img alt="Ultravox" src="https://zfmrfvimiaqahezndsse.supabase.co/storage/v1/object/public/images/custom/Introducing%20Ultravox%20Wide.jpg">
  </picture>
</p>

<h3 align="center">
An open, fast, and extensible multimodal LLM
</h3>

# About

Ultravox is a new kind of multimodal LLM that can understand text as well as human speech, without the need for a separate Audio Speech Recognition (ASR) stage. Building on research like [AudioLM](https://arxiv.org/abs/2209.03143), [SeamlessM4T](https://ai.meta.com/blog/seamless-m4t/), [Gazelle](https://tincans.ai/slm), [SpeechGPT](https://github.com/0nutation/SpeechGPT/tree/main/speechgpt), and others, we've extended Meta's [Llama 3 model](https://llama.meta.com/llama3/) with a multimodal projector that converts audio directly into the high-dimensional space used by Llama 3. This direct coupling allows Ultravox to respond much more quickly than systems that combine separate ASR and LLM components. In the future this will also allow Ultravox to natively understand the paralinguistic cues of timing and emotion that are omnipresent in human speech.

The current version of Ultravox (v0.1), when invoked with audio content, has a time-to-first-token (TTFT) of approximately 200ms, and a tokens-per-second rate of ~100, all using a Llama 3 8B backbone. While quite fast, we believe there is considerable room for improvement in these numbers. We look forward to working with LLM hosting providers to deliver state-of-the-art performance for Ultravox.

Ultravox currently takes in audio and emits streaming text. As we evolve the model, we'll train it to be able to emit a stream of speech tokens that can then be converted directly into raw audio by an appropriate unit vocoder. We're interested in working with interested parties to build this functionality!

### Demo

See Ultravox in action via a [voice call](https://www.ai.town/characters/a90fcca3-53c0-4111-b30a-4984883a23ef) with an AI in our app, [ai.town](https://ai.town).
(*Note: there's been a lot of traffic to our inference server and we've hit a few bugs. If the demo seems to be erroring out please try again in a bit.*)

### Discord

Join us on our Discord server [here](https://discord.gg/Qw6KHxv8YB).

### Jobs

If you're interested in working on Ultravox fulltime, we're hiring! Check out our jobs page [here](https://www.notion.so/fixieai/Careers-at-Fixie-fc1a7ace4c1e42a8886065bc397aba2d).

### Inference Server

You can try out Ultravox using your own audio content (as a WAV file), using the following curl command:

```shell
curl -X POST -H "Authorization: Bearer $ULTRAVOX_API_KEY" -H "Content-Type: application/json" \
     -d @data.json https://ultravox.api.fixie.ai/v1/chat/completions
```

where `data.json` contains:

```json
{ 
  "model": "fixie-ai/ultravox-v0.1",
  "messages": [{ 
    "role": "user",
    "content": [{
      "type": "text",
      "text": "What’s in <|audio|>?"
    }, {
      "type": "image_url",
      "image_url": {
        "url": "data:audio/wav;base64,{base64_wav}"
      }
    }]
  }],
  "stream": true
}
```

### Model

You can download the latest weights from the [Ultravox Hugging Face page](https://huggingface.co/fixie-ai/ultravox-v0.2).

### Architecture

[![architecture diagram](https://raw.githubusercontent.com/fixie-ai/ultravox/main/docs/assets/Ultravox%20Model%20Architecture.svg)](https://docs.google.com/presentation/d/1ey81xuuMzrJaBwztb_Rq24Cit37GQokD2aAes_KkGVI/edit)

# Contributing

Read on if you're interested in training your own version of Ultravox.

## Environment Setup (Mac)

Install the basic tools:

- [`Homebrew`](https://brew.sh) is a package manager for MacOS that also mostly works for Linux. If you're running Debian or Ubuntu Linux, you can alternatively get by with apt.
- [`Just`](https://just.systems/man/en/) simplifies our shell workflows. It frequently functions as our interface to all the other tools.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew update
brew install just
```

Create a Python virtual environment and install the necessary packages:

```bash
just install
```

We're using Poetry to manage the Python virtual environment.

### Mosaic Environment Setup

You need to setup a few things to run on the Mosaic Platform.

1. Install & login to the Mosaic CLI

```bash
pip install --upgrade mosaicml-cli

mcli init

mcli set api-key <new-value>
```

2. set API keys for tools we use:

```bash
# Huggging Face token for accessing walled data and models
mcli create secret env HF_TOKEN=hf_<your_token>

# WandB token for logging experiments
mcli create secret env WANDB_PROJECT=ultravox
mcli create secret env WANDB_API_KEY=<your_wandb_key>

# GCP credentials for accessing data (e.g. BoolQ)
# Get service_account.json file from Justin/Farzad and put it in the root dir, then
mcli create secret gcp
```

## Training

```bash
just train
```

For DDP training make sure to use:
`torchrun --nproc_per_node=8 -m ultravox.training.train`

### Local Training

```bash
python -m ultravox.training.train --config_path ultravox/training/configs/asr_tinyllama.yaml  --data_set 'dummy' --device cpu --batch_size 1  --exp_name <give_your_experiment_a_name>
```

### MosaicML Training

You need to setup your SSH key in the Mosaic Platform: https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/ssh.html#page-secrets-ssh

```bash
## Create a new SSH key and add it to the Mosaic Platform
# ssh-keygen -f ~/.ssh/mclid_id_rsa
## add the **public** key to Github
# mcli create secret ssh ~/.ssh/mclid_id_rsa

mcli run -f mcloud.yaml --follow
```

Other useful commands:

```bash
mcli get clusters

mcli util r7z2
mcli get runs
mcli get runs --cluster r7z2

mcli run -f mcloud.yaml --follow
```

For interactive runs, we don't recommend using `--interactive`. Instead set the `command` to be something like
`sleep 3600` and then connect to it using `mcli connect <job_name> --tmux`.
This way your environment (code and packages) will be the same as the training environment.
The value `3600` (1 hour), is used as an example.

IMPORTANT: Make sure to stop the machine when you're done with any job, specially interactive ones!

### Running evaluations

1. Use `infer_tool.py --json > file` to create a jsonl output from a given model/dataset combo, where each line contains two values: **question** and **answer**.
2. Use `eval_tool.py -f file` to evaluate the jsonl file, which will produce an average score for the model on the dataset.

## Misc

Useful commands:

```bash
just update    # update dependencies
just format    # run formatting (black, isort, autoflake)
just python    # activate venv and run python
```
