<p align="center">
  <picture>
    <img alt="Ultravox" src="https://zfmrfvimiaqahezndsse.supabase.co/storage/v1/object/public/images/custom/Introducing%20Ultravox%20Wide.jpg">
  </picture>
</p>

<h3 align="center">
A fast multimodal LLM for real-time voice
</h3>

_Latest News_
* 2024/08 — [Ultravox 0.4](https://github.com/fixie-ai/ultravox/releases/tag/v0.4) available
* 2024/08 — [Ultravox 0.3](https://github.com/fixie-ai/ultravox/releases/tag/v0.3) available
* 2024/08 — Preview of Ultravox APIs available, more information [here](https://fixie-ai.github.io/ultradox/)

---

# About

Ultravox is a new kind of multimodal LLM that can understand text as well as human speech, without the need for a separate Audio Speech Recognition (ASR) stage. Building on research like [AudioLM](https://arxiv.org/abs/2209.03143), [SeamlessM4T](https://ai.meta.com/blog/seamless-m4t/), [Gazelle](https://tincans.ai/slm), [SpeechGPT](https://github.com/0nutation/SpeechGPT/tree/main/speechgpt), and others, we've extended Meta's [Llama 3 model](https://llama.meta.com/) with a multimodal projector that converts audio directly into the high-dimensional space used by Llama 3. This direct coupling allows Ultravox to respond much more quickly than systems that combine separate ASR and LLM components. In the future this will also allow Ultravox to natively understand the paralinguistic cues of timing and emotion that are omnipresent in human speech.

The current version of Ultravox (v0.3), when invoked with audio content, has a time-to-first-token (TTFT) of approximately 150ms, and a tokens-per-second rate of ~60, all using a Llama 3.1 8B backbone. While quite fast, we believe there is considerable room for improvement in these numbers. We look forward to working with LLM hosting providers to deliver state-of-the-art performance for Ultravox.

Ultravox currently takes in audio and emits streaming text. As we evolve the model, we'll train it to be able to emit a stream of speech tokens that can then be converted directly into raw audio by an appropriate unit vocoder. We're interested in working with interested parties to build this functionality!

### Demo

See Ultravox in action via a [voice call](https://www.ai.town/characters/a90fcca3-53c0-4111-b30a-4984883a23ef) with an AI in our app, [ai.town](https://ai.town).
(*Note: there's been a lot of traffic to our inference server and we've hit a few bugs. If the demo seems to be erroring out please try again in a bit.*)

### Discord

Join us on our Discord server [here](https://discord.gg/Qw6KHxv8YB).

### Jobs

If you're interested in working on Ultravox fulltime, we're hiring! Check out our jobs page [here](https://www.notion.so/fixieai/Careers-at-Fixie-fc1a7ace4c1e42a8886065bc397aba2d).

### Inference Server

You can try out Ultravox using your own audio content (as a WAV file) by spinning up an Ultravox instance on our partner, BaseTen: [https://www.baseten.co/library/ultravox/](https://www.baseten.co/library/ultravox/). They offer free credits to get started.

If you're interested in running Ultravox in a real-time capacity, we offer a set of managed APIs as well. You can learn more about getting access to those [here](https://fixie-ai.github.io/ultradox/).

### Model

You can download the latest weights from the [Ultravox Hugging Face page](https://huggingface.co/fixie-ai/ultravox-v0_4).

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

### Mosaic Environment Setup (Fixie Internal)

If you want to use [Mosaic](https://docs.mosaicml.com/projects/mcli/en/latest/quick_start/getting_started.html) for trainig , you need to setup a few things to run on the Mosaic Platform.

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

Currently, we keep both the LLM and the audio encoder frozen and only train the adapter/projector. Training Ultraox v0.4 took 2-3 hours on 8xH100 GPUs for 14K training steps.

### Use-Cases for Training Ultravox
Why would you want to (re-) train Ultravox? Here are a few scenarios:

1. You want to use a different LLM or audio encoder backbone.

    a. In this case you need to re-train the adapter. You can use `release_config.yaml`, which contains our config for our latest release, and you should be able to simply change the base LLM or encoder by specifying `--text-model <hf-model-id-for-llm>` and/or `--audio-model <hf-model-id-for-encoder>`.

2. You want to improve the knowledge of the model --> NO NEED TO TRAIN ULTRAVOX!

    a. We suggest to either use RAG on the fly (no training needed), or fine-tune the LLM backbone instead. You might need to re-train Ultravox if you fine-tune the LLM.

3. You want to use your own audio data, for example to add support for a new language.

    a. First step, prepare your dataset: at bare minimum, the samples should have an `audio` and a text `continuation` field.
  
    b. Take a look at [`ds_tool.py`](ultravox/tools/ds_tool/ds_tool.py) and [`continuation.jinja`](ultravox/tools/ds_tool/continuation.jinja) as well as [our variant of Common Voice](https://huggingface.co/datasets/fixie-ai/common_voice_17_0/viewer/fr) that was created using `ds_tool` to add the `continuation` field.
    
    c. Add your dataset to the dataset mix in `release_config.yaml` and train.

There's no one-size fits all. If you need help you can find us on our Discord server [here](https://discord.gg/Qw6KHxv8YB).


### How to Train

We do most of our training on the [MosaicML platform](https://docs.mosaicml.com) and therefore most of our tooling and docs are Mosaic-related. However, you can do the same training on your own GPU without much difficulty. Here we assume you have the environment set up (run `just install`). You can also take a look at [`setup.sh`](setup.sh)

To kick off a training run you can do:
```bash
poetry run python -m ultravox.training.train --config_path ultravox/training/configs/release_config.yaml
```

For DDP training make sure to add `torchrun`. We also recommend prefetching weights in advance:
```bash
TRAIN_ARGS="--config_path ultravox/training/configs/release_config.yaml"
poetry run python -m ultravox.training.helpers.prefetch_weights $TRAIN_ARGS
poetry run torchrun --nproc_per_node=8 -m ultravox.training.train $TRAIN_ARGS
```

For a debug run, you can use smaller models, datasets, or batch size. Here's a config that uses TinyLlama as the LLM backbone:
```bash
poetry run python -m ultravox.training.train --config_path ultravox/training/configs/asr_tinyllama_100s.yaml --batch_size 1 --report_logs_to tensorboard
```


We use [SimpleParsing](https://github.com/lebrice/simpleparsing/) for configs. Configs are composable (i.e. you can specify zero or many configs) and `meta_config.yaml` is always used as the default.
See [`configs_base.py`](ultravox/training/config_base.py) to find the parameters you modify, such as the `--text-model`, `--device`, `--exp-name`, etc.


### MosaicML Training (Fixie Internal)

Before running any training jobs, you need to setup your SSH key in the Mosaic Platform: https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/ssh.html#page-secrets-ssh

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

For interactive runs you can use:
```bash
just mcloud --image mosaicml/composer:latest --max-duration 1
```

IMPORTANT: Make sure to monitor your jobs and stop the machine when you're done with any job, specially interactive ones!

### Running evaluations

1. Use `infer_tool.py --json > file` to create a jsonl output from a given model/dataset combo, where each line contains two values: **question** and **answer**.
2. Use `eval_tool.py -f file` to evaluate the jsonl file, which will produce an average score for the model on the dataset.

## Misc

The [Justfile](Justfile) is a good resource for finding popular commands. Here are a few:

```bash
just update    # update dependencies
just format    # run formatting (black, isort, autoflake)
just test      # run tests
just python    # activate venv and run python
```
