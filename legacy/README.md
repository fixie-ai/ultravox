# UltraVox

## AzureML

### Installation and Config

```bash
brew update && brew install azure-cli
az extension add --name ml --yes

az login
az account set --subscription 520aa0b2-6a19-4a45-8c03-4c301d1f847a
az configure --defaults workspace=gpu-supercomput
```

```bash
az ml job create  -f ./azureml/configs/audiollm.yml --web
```

## Random Documentation

### LLM + AudioEnc (ours) vs SpeechGPT

```python
# SpeechGPT adds new tokens to the embedding and then trains them
nn.Embedding(32000, 2048) + nn.Embedding(4000, 2048)  # old text tokens + new audio tokens
nn.Embedding(36000, 2048)

###
# In other words:
###

# SpeechGPT tokenizes audio and text separately, then concatenates the embeddings
llm(embed(concat(audio_tokenizer(audio), text_token)))
## -------------------  vs  -------------------
# We create the audio embeddings directly from the audio and skip embedding the audio tokens
llm(concat(audio_enc(audio) * weight, embed(text_token)))
# This means we can easily propagate gradients to the audio encoder (i.e. train end to end)
```

### How does language modeling work with audio?

```python
# t[n] <- t[1..n-1]
# The brown fox jumps over the fence

# a1 a2 a3 a4 The brown fox jumps over the fence
# samples:
# a1 a2 a3 -> a4
# a1 a2 a3 a4 -> The
# a1 a2 a3 a4 The -> brown
# a1 a2 a3 a4 The brown -> fox
# a1 a2 a3 a4 The brown fox -> jumps
# a1 a2 a3 a4 The brown fox jumps -> over
# a1 a2 a3 a4 The brown fox jumps over -> the
# a1 a2 a3 a4 The brown fox jumps over the -> fence
```

## TODO

- [ ] generation metrics (low_pri: added more metrics to cover shifts)
- [x] more metrics to cover shifts
- [x] torchrun
- [ ] shard dataset
- [ ] datasets.distributed.split_dataset_by_node
- [ ] cache preprocessed data
- [ ] <audio> tokens (start&end)
- [ ] loss for audio ending

- [ ] combine multiple datasets (low priority for now)

- [x] half precision
- [x] LR ramp up
- [x] loss mask
- [x] get and and running with Azure
- [x] bfloat16
- [x] auth: HF, WANDB, ClearML
- [x] store models
- [x] loss: language
- [x] LoRA adapted model
  - [x] freeze for rank=0
- [x] audio model stride: start with ~200ms
- [x] audio stride: stack instead of skip
- [x] multi-GPU
- [ ] fix hyperparams
- [ ] loss not going down: is it a real issue?
- [ ] optimizations (Azure specific): deepspeed, nebula, onnx-runtime
  - [ ] https://huggingface.co/docs/transformers/perf_train_gpu_one
  - [ ] torch.compile
  - [ ] (low) n_shards
  - [ ] increase GPU utilization
- [x] W2v-BERT-2 [not merged to HF]
- [ ] is dataloading happening correctly? no same data?
- [x] WER
