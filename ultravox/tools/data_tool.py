import argparse

import librosa
import numpy as np
import sounddevice as sd

from ultravox import data as datasets

parser = argparse.ArgumentParser()
parser.add_argument("data_sets", nargs="*", help="List of datasets to use")
parser.add_argument("--data-split", default="train", help="Which split of data to use.")
parser.add_argument(
    "--num-samples", "-n", type=int, default=5, help="Number of samples to display"
)
parser.add_argument(
    "--augmentation",
    default="null",
    help="Which augmentation to apply to the data.",
    choices=list(datasets.AugRegistry._configs.keys()),
)
parser.add_argument("--play", "-p", action="store_true", help="Play the audio samples")
parser.add_argument(
    "--write", "-w", action="store_true", help="Write audio samples out as WAV files"
)
parser.add_argument("--playback-rate", "-r", type=float, help="Playback rate")
parser.add_argument("--shuffle", "-s", action="store_true", help="Shuffle the samples")
parser.add_argument("--seed", type=int, help="Shuffle seed")


def main(args: argparse.Namespace):
    data_args = datasets.VoiceDatasetArgs(
        shuffle=args.shuffle,
        split=args.data_split,
    )
    augmentation = datasets.AugRegistry.create_augmentation(
        datasets.AugRegistry.get_config(name=args.augmentation)
    )
    if args.seed is not None:
        data_args.shuffle_seed = args.seed
    data_sets = [datasets.create_dataset(ds, data_args) for ds in args.data_sets]
    out_set = datasets.Range(datasets.InterleaveDataset(data_sets), args.num_samples)
    for i, sample in enumerate(out_set):
        print(f"--- Sample {i} ---")
        sample = augmentation.apply_sample(sample)
        messages = sample.messages
        assert len(messages) >= 2, f"Bad sample (messages) {len(messages)}"
        assert messages[-2]["role"] == "user", f"Bad sample (Q role): {messages}"
        assert messages[-1]["role"] == "assistant", f"Bad sample (A role): {messages}"
        answer = messages[-1]["content"].replace("\n", "\\n")
        print(f"Q: {messages[-2]['content']} [\"{sample.audio_transcript}\"]")
        print(f"A: {answer}")
        if args.play:
            audio = sample.audio
            if args.playback_rate is not None:
                audio = librosa.effects.time_stretch(audio, rate=args.playback_rate)
            sd.play(audio, sample.sample_rate)
            sd.wait()
        if args.write:
            name = (
                f"sample{i}.wav"
                if args.augmentation == "null"
                else f"sample{i}_{args.augmentation}.wav"
            )
            with open(name, "wb") as f:
                if sample.audio.dtype == np.float64:
                    sample.audio = sample.audio.astype(np.float32)
                f.write(datasets.audio_to_wav(sample.audio, sample.sample_rate))


if __name__ == "__main__":
    main(parser.parse_args())
