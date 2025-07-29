# Augmentations

Augmentations can be applied to data for training or evaluation to improve or test robustness of the model to acoustic variations. 

Numerous prior works have applied augmentations to improve robustness. A few examples include:
- [Whisper](https://arxiv.org/pdf/2212.04356) applied [SpecAugment](https://arxiv.org/pdf/1904.08779) during training of V2, dropping time and frequency components of the mel spectrogram. It's unclear whether V3 continued to use these augmentations
- [Moshi](http://arxiv.org/abs/2410.00037) applied multiple randomized augmentations during training with various probabilities `p`:
    - Random gain [-24, +15] dB, `p=0.5`
    - Deep Noise Suppression background noise with SNR between [-30, +6] dB, `p=0.3`
    - Echo and reverb on user audio stream while Moshi speaking, `p=0.3`



We are further motivated by known domain shift in our production environment, specifically related to audio compression over telephony networks and noise cancellation applied prior to Ultravox inference. 

## Augmentations Tutorial

Augmentations are organized by *types* and *configurations*. An augmentation *type* is a particular method of augmentation, while a *configuration* also includes the specific parameters applied during augmentation.

You can find the available augmentation *types* and *configurations* from the `AugRegistry`:

```python
from ultravox.data import AugRegistry

print(AugRegistry._configs.keys())

print(AugRegistry._aug_types.keys())
```


We can get also specific augmentation configs or compose new augmentation configs with the `AugRegistry`:

```python
# Get 10dB gain augmentation
gain_10db_config = AugRegistry.get_config('gain_10db')

# Create 20dB gain augmentation that is applied 50% of the time
gain_20db_config = AugRegistry.get_config('gain_20db', override_args=dict(type="gain", gain_db=20, p=0.5))
```

We then can create the actual augmentation from the configs and call it on audio data.

```python
audio = np.random.normal(0, 0.1, 16000)

gain_10db = AugRegistry.create_augmentation(gain_10db_config)
gain_20db = AugRegistry.create_augmentation(gain_20db_config)

audio_10db = gain_10db(audio)
audio_20db = gain_20db(audio)  # applied with 50% prob per the config
```

We can also compose multiple augmentations into a group:

```python
aug_group_config = AugRegistry.get_config("my_aug", override_args=dict(children=[
    AugRegistry.get_config("gain_10db", override_args=dict(p=0.5)),
    AugRegistry.get_config("8kHz_resample", override_args=dict(p=0.5))
], p=0.5))

aug_group = AugRegistry.create_augmentation(aug_group_config)
audio_augmented = aug_group(audio)  # 25% chance of either 10db gain or 8kHz resample
```

When augmentations are applied in a group with `p<1.0`, their true probability of being applied is $p_{group}*p_{aug}$. In the case above, each child augmentation has a 25% chance of being called, and there is 12.5% chance that both of them are called.

## Training and Evaluation with Augmentations

There are two main approaches to applying augmentations to data:
- On-the-fly: audio is augmented in the dataloader or preprocessing phase 
- Pre-computed: dataset copies are created with a specific augmentation applied. Useful when augmentations are too computationally expensive to perform on-the-fly

### On-the-fly

We can configure augmentations in the training and evaluation `yaml` configs to run on-the-fly. These augmentations are performed in the [Ultravox Dataproc](ultravox/model/ultravox_data_proc.py) during training, and applied directly to batches during evaluations.

An example evaluation configuration can be found at [ultravox/evaluation/configs/eval_config_augmentations.yaml](ultravox/evaluation/configs/eval_config_augmentations.yaml):

```yaml
augmentations:
  - name: null
  - name: my_augmentation
    children:
      - name: 8kHz_resample
      - name: telephone_bandpass
        lowcut: 500
  - name: amr_4_75kbps
```

Each augmentation in the `augmentations` list will be applied separately during evaluation—the script will perform 3x the number of evaluations, recording the results with each augmentation applied. However, we can group augmentations to apply together, like done with `my_augmentation`


For training, augmentations are configured similarly, as in [ultravox/training/configs/tiny_config_augmentations.yaml](ultravox/training/configs/tiny_config_augmentations.yaml):

```yaml
train_augmentations:
  - name: 8kHz_resample
    p: 0.5
  - name: amr_wb
    p: 0.25

eval_augmentations:
  - name: null
  - name: 8kHz_resample
```
Crucially, while the `eval_augmentations` will be applied one at a time, the `train_augmentations` are automatically grouped together under a parent augmentation with `p=1.0`.


### Pre-computed

We can use the `ds_tool` to apply augmentations to a dataset and reupload that dataset to huggingface. 

For example, to create an AMR-NB version of the AMI-IHM dataset:

```bash
just ds_tool augment -d edinburghcstr/ami -u fixie-ai/ami_amr_nb -S ihm -A random_amr_compression -w 8
```


