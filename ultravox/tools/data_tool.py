import argparse

import librosa
import sounddevice as sd

from ultravox.data import datasets

parser = argparse.ArgumentParser()
parser.add_argument("data_sets", nargs="*", help="List of datasets to use")
parser.add_argument("--data-split", default="train", help="Which split of data to use.")
parser.add_argument(
    "--num-samples", "-n", type=int, default=5, help="Number of samples to display"
)
parser.add_argument(
    "--num-prompts", type=int, default=1, help="Number of prompts to use"
)
parser.add_argument("--play", "-p", action="store_true", help="Play the audio samples")
parser.add_argument(
    "--write", "-w", action="store_true", help="Write audio samples out as WAV files"
)
parser.add_argument("--playback-rate", "-r", type=float, help="Playback rate")
parser.add_argument("--shuffle", "-s", action="store_true", help="Shuffle the samples")
parser.add_argument("--seed", type=int, help="Shuffle seed")
parser.add_argument("--mds", action="store_true", help="Use MDS datasets")


def main(args: argparse.Namespace):
    data_args = datasets.VoiceDatasetArgs(
        num_prompts=args.num_prompts,
        shuffle=args.shuffle,
        use_mds=args.mds,
        split=args.data_split,
    )
    if args.seed is not None:
        data_args.shuffle_seed = args.seed
    data_sets = [datasets.create_dataset(ds, data_args) for ds in args.data_sets]
    out_set = datasets.Range(datasets.InterleaveDataset(data_sets), args.num_samples)
    for i, sample in enumerate(out_set):
        print(f"--- Sample {i} ---")
        messages = sample.messages
        assert len(messages) >= 2, f"Bad sample (messages) {len(messages)}"
        assert messages[-2]["role"] == "user", f"Bad sample (Q role): {messages}"
        assert messages[-1]["role"] == "assistant", f"Bad sample (A role): {messages}"
        answer = messages[-2]["content"].replace("\n", "\\n")
        print(f"Q: {messages[-1]['content']} [\"{sample.audio_transcript}\"]")
        print(f"A: {answer}")
        if args.play:
            audio = sample.audio
            if args.playback_rate is not None:
                audio = librosa.effects.time_stretch(audio, rate=args.playback_rate)
            sd.play(audio, sample.sample_rate)
            sd.wait()
        if args.write:
            with open(f"sample{i}.wav", "wb") as f:
                f.write(datasets.audio_to_wav(sample.audio, sample.sample_rate))


if __name__ == "__main__":
    main(parser.parse_args())
