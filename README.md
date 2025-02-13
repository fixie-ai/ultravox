<p align="center">
  <picture>
    <img alt="Ultravox" src="https://zfmrfvimiaqahezndsse.supabase.co/storage/v1/object/public/images/custom/Introducing%20Ultravox%20Wide.jpg">
  </picture>
</p>

<h3 align="center">
A fast multimodal LLM designed for real-time voice interactions
</h3>

_Latest News_
* 2025/02 — [Ultravox 0.5](https://github.com/fixie-ai/ultravox/releases/tag/v0.5) available
* 2024/11 — [Ultravox 0.4.1](https://github.com/fixie-ai/ultravox/releases/tag/v0.4.1) available
* 2024/08 — [Ultravox 0.4](https://github.com/fixie-ai/ultravox/releases/tag/v0.4) available
* 2024/08 — [Ultravox 0.3](https://github.com/fixie-ai/ultravox/releases/tag/v0.3) available
* 2024/08 — Preview of Ultravox APIs available, more information [here](https://fixie-ai.github.io/ultradox/)

_Key Links_
* [Ultravox Realtime](https://ultravox.ai) — Build real-time Voice AI agents on top of the Ultravox model
* [Hugging Face](https://huggingface.co/fixie-ai) — Our Hugging Face page

---

# About

Ultravox is a new kind of multimodal LLM that can understand text as well as human speech, without the need for a separate Audio Speech Recognition (ASR) stage. Building on research like [AudioLM](https://arxiv.org/abs/2209.03143), [SeamlessM4T](https://ai.meta.com/blog/seamless-m4t/), [Gazelle](https://tincans.ai/slm), [SpeechGPT](https://github.com/0nutation/SpeechGPT/tree/main/speechgpt), and others, Ultravox is able to extend any open-weight LLM with a multimodal projector that converts audio directly into the high-dimensional space used by LLM. We've trained versions on Llama 3, Mistral, and Gemma. This direct coupling allows Ultravox to respond much more quickly than systems that combine separate ASR and LLM components. In the future this will also allow Ultravox to natively understand the paralinguistic cues of timing and emotion that are omnipresent in human speech.

Ultravox currently takes in audio and emits streaming text. As we evolve the model, we'll train it to be able to emit a stream of speech tokens that can then be converted directly into raw audio by an appropriate unit vocoder.

Our default model is built on top of Llama 3.3 70B. We also have an 8B variant available on Hugging Face.

Ultravox can be trained against any open-weight model. See below for more details on training.

### Demo

See Ultravox in action on our [demo page](https://demo.ultravox.ai). You can build your own voice-to-voice agents on our Realtime platform at ultravox.ai.


### Discord

Join us on our Discord server [here](https://discord.gg/Qw6KHxv8YB).

### Jobs

If you're interested in working on Ultravox fulltime, we're hiring! Check out our jobs page [here](https://careers.fixie.ai).

### Inference Server

You can try out Ultravox using your own audio content (as a WAV file) by spinning up an Ultravox instance on our partner, BaseTen: [https://www.baseten.co/library/ultravox/](https://www.baseten.co/library/ultravox/). They offer free credits to get started.

If you're interested in running Ultravox in a real-time capacity, we offer a set of managed APIs as well. You can learn more about getting access to those [here](https://docs.ultravox.ai).

### Model

You can download the latest weights from the [Ultravox Hugging Face page](https://huggingface.co/fixie-ai/).

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

It's recommended to use pyenv for managing environments due to the use of Poetry:

```bash
brew install xz
brew install pyenv
pyenv init
pyenv install 3.11
pyenv global 3.11

# Optional
pyenv shell 3.11
```

>**Note**: Use of conda is NOT recommended with Poetry

After creating a virtual environment, install required packages using `just` and `poetry`:

```bash
just install
```

We're using Poetry to manage the Python virtual environment. You can observe your environment with `poetry env info`.

### Mosaic Environment Setup (Fixie Internal)

If you want to use [Mosaic](https://docs.mosaicml.com/projects/mcli/en/latest/quick_start/getting_started.html) for training, you need to setup a few things to run on the Mosaic Platform.

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
mcli create secret env HF_WRITE_TOKEN=hf_<your_token_with_write_access>

# WandB token for logging experiments
mcli create secret env WANDB_PROJECT=ultravox
mcli create secret env WANDB_API_KEY=<your_wandb_key>
```

## Training

Currently, we keep both the LLM and the audio encoder frozen and only train the adapter/projector. Training Ultraox v0.4 took 2-3 hours on 8xH100 GPUs for 14K training steps.

### Use-Cases for Training Ultravox
Why would you want to (re-) train Ultravox? Here are a few scenarios:

1. You want to use a different LLM or audio encoder backbone.

    a. In this case you need to re-train the adapter. You can use `release_config.yaml`, which contains our config for our latest release, and you should be able to simply change the base LLM or encoder by specifying `--text-model <hf-model-id-for-llm>` and/or `--audio-model <hf-model-id-for-encoder>`.

2. You want to improve the knowledge of the model

    a. We suggest to either use RAG on the fly (no training needed), or fine-tune the LLM backbone instead. Fine-tuning the LLM backbone does not require re-training Ultravox (i.e., the existing adapter will work).

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

Before running any training jobs, set up [SSH authentication with MosaicML](https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/ssh.html#page-secrets-ssh):

1. Generate an SSH key:
   ```bash
   ssh-keygen -f ~/.ssh/mclid_id_rsa
   ```

2. Add the public key to your GitHub account

3. Upload the private key to MosaicML (this allows MosaicML to clone the repository and run jobs):
   ```bash
   mcli create secret git-ssh ~/.ssh/mclid_id_rsa
   ```

Then you can run the following command to kick off a training job:

```bash
mcli run -f mcloud_train.yaml --follow
```

Other useful commands:

```bash
mcli get clusters

mcli util r7z2
mcli get runs
mcli get runs --cluster r7z2

mcli run -f mcloud_eval.yaml --follow
```

For interactive runs you can use:
```bash
just mcloud --image mosaicml/composer:latest --max-duration 1
```

IMPORTANT: Make sure to monitor your jobs and stop the machine when you're done with any job, specially interactive ones!

### Running evaluations

For inference or evaluations, you can use:

```bash
just eval --config_path ultravox/evaluation/configs/eval_config.yaml
```

where `eval_config.yaml` is a config file that specifies the model, datasets, and configurations to use for inference or evaluation. If your dataset is not already defined in ultravox, you need to create a config file for your dataset in `ultravox/data/configs/` (with the appropriate `eval_config` field to specify evaluation metrics and arguments), and register it in `ultravox/data/registry.py`. Please refer to examples in `ultravox/data/configs/`.

## Misc

The [Justfile](Justfile) is a good resource for finding popular commands. Here are a few:

```bash
just update    # update dependencies
just format    # run formatting (black, isort, autoflake)
just test      # run tests
just python    # activate venv and run python
```
