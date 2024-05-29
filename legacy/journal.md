# Training Journal

# Jan - Feb 2024

## W2V-BERT-2

### [LS FT w/ WD 0.05](https://wandb.ai/fixie/ultravox/runs/dvquuhym)

- first time training worked!
  - 500x64BS + 100x64BS + worked at ~1.2Kx32BS -> ~2.4Kx32BS
- train loss lags behind val loss (really high reg?)

### [GS FT w/o WD](https://wandb.ai/fixie/ultravox/runs/9yyh0zd4)

- overfitting: may need to add weight decay back

### [GS FT w/ WD 0.01, constant LR 5e-5](https://wandb.ai/fixie/ultravox/runs/zdq723i5)

- idea: find right WD and reduce overfitting

## continuation exps

To test:

- WD
- LS vs GS
- speech tag or no?
- batch size
- LR
- \*audio stride

### [WD=0 Test](https://wandb.ai/fixie/ultravox/runs/huypf4ez)

### [BS=16](https://wandb.ai/fixie/ultravox/runs/7g9c93ej)

### [BS=4](https://wandb.ai/fixie/ultravox/runs/x97fpqrf)

- the smaller the better?!

### [GS](https://wandb.ai/fixie/ultravox/runs/bazab1jz)

### [Speech Tag](https://wandb.ai/fixie/ultravox/runs/cia60j23)

- no difference

### [4xLR=2e-4](https://wandb.ai/fixie/ultravox/runs/5b9o6l5w)

- too high

### [.4LR=2e-5](https://wandb.ai/fixie/ultravox/runs/)

- too low

### [BS=2](https://wandb.ai/fixie/ultravox/runs/fy6beoy9)

## Train on the fastest parameters

### [Wav2Vec2Bert](https://wandb.ai/fixie/ultravox/runs/l44100jk)

### [Wav2Vec2](https://wandb.ai/fixie/ultravox/runs/a1niw6mj)

# Mar 18, 2024

## EOU method #1

### exp1: end shift left

### exp2: start/end shift right

### exp3: cropped audio

Q: can I increase bs?
Q: other ways to make EOU more nuanced:

- padding with more END?
- noise

TODO: fix WER when no text is provided

## [Bugfix] Freeze Text Embeddings

I finally realized that the text embeddings were not being frozen.

- Params before:
  - trainable params: 80,234,496 || all params: 1,430,185,600 || trainable%: 5.6%
- Params after:
  - trainable params: 14,698,496 || all params: 1,430,185,600 || trainable%: 1.0%

## [AudioTag or not?](https://wandb.ai/fixie/ultravox/runs/coom6x21)
