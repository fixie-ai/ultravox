#!/usr/bin/env python

import argparse
import dataclasses
import json
import os
import time
from typing import IO, List, Optional

import numpy as np
import simple_parsing

from ultravox.data import datasets
from ultravox.evaluation import eval
from ultravox.evaluation import eval_types
from ultravox.inference import base
from ultravox.tools import infer_api

# There are two default modes for this tool, agent mode and ASR mode.
# In agent mode, the answer is a response to the input content and cannot be
# directly compared to the expected answer. In ASR mode, the answer is a
# transcription of the audio content and the tool can perfom a WER calculation.
# Remember to set the --asr flag when using an ASR input.
DEFAULT_PROMPT = "Listen to <|audio|> and respond to it"
DEFAULT_ASR_PROMPT = "Transcribe <|audio|>"


@dataclasses.dataclass
class InferArgs:
    # Model ID to use for the model
    model: str = simple_parsing.field(default="fixie-ai/ultravox-v0.1", alias="-m")
    # Path to the audio file
    audio_file: Optional[IO] = simple_parsing.field(
        default=None, type=argparse.FileType("rb"), alias="-f"
    )
    # Prompt to use for inference
    prompt: Optional[str] = None
    # Inference the model using only the text input or transcript, without audio
    text_only: bool = False
    # Use ASR for the prompt and compute WER
    asr: bool = False
    # URL to use for inference
    url: Optional[str] = simple_parsing.field(default=None, alias="-u")
    # Audio processor ID to use
    audio_processor: Optional[str] = None
    # Tokenizer ID to use
    tokenizer: Optional[str] = None
    # Data sets to use for inference
    data_sets: Optional[List[str]] = simple_parsing.field(default=None, alias="-d")
    # Which dataset split to use
    data_split: datasets.DatasetSplit = simple_parsing.field(
        default=datasets.DatasetSplit.VALIDATION, alias="-s"
    )
    # Directory for existing data
    data_dir: Optional[str] = None
    # Use dataset context
    context: bool = False
    # Load datasets using MDS
    mds: bool = False
    # Number of dataset samples to process
    num_samples: int = simple_parsing.field(default=1, alias="-n")
    # Shuffle the dataset
    shuffle: bool = False
    # Seed for shuffling
    seed: Optional[int] = None
    # Device to use for inference
    device: Optional[str] = simple_parsing.field(default=None, alias="-D")
    # Data type to use for the model
    data_type: Optional[str] = None
    # Temperature for sampling
    temperature: Optional[float] = simple_parsing.field(default=None, alias="-t")
    # Maximum tokens to generate
    max_tokens: Optional[int] = simple_parsing.field(default=None, alias="-T")
    # Evaluate the generated answer
    eval: bool = simple_parsing.field(default=False, alias="-e")
    # Verbose output
    verbose: bool = simple_parsing.field(default=False, alias="-v")
    # JSON output
    json: bool = simple_parsing.field(default=False)


def run_tui(
    index: int,
    inference: base.VoiceInference,
    sample: datasets.VoiceSample,
    args: InferArgs,
    expected_response: Optional[str] = None,
    scores: Optional[List[float]] = None,
):
    if index >= 0:
        print(f"--- Sample {index} ---")
    messages = sample.messages
    transcript = f' ["{sample.audio_transcript}"]' if sample.audio_transcript else ""
    print(f"Q: {messages[0]['content']}{transcript}")
    print(f"A: ", end="")
    start_time = time.time()
    first_token_time = None
    text = ""
    stats = None

    # Run streaming inference and print the output as it arrives.
    stream = inference.infer_stream(
        sample,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    for msg in stream:
        if isinstance(msg, base.InferenceChunk):
            if first_token_time is None:
                first_token_time = time.time()
            text += msg.text
            print(msg.text, end="", flush=True)
        elif isinstance(msg, base.InferenceStats):
            stats = msg
    if first_token_time is None or stats is None:
        raise ValueError("No tokens received")

    # If we're in verbose mode, print some stats about the inference.
    if args.verbose:
        ttft = first_token_time - start_time
        total_time = time.time() - start_time
        tokens = stats.output_tokens
        tps = tokens / (total_time - ttft)
        print(
            f" [ttft: {ttft:.2f} s, tok: {tokens}, tps: {tps:.2f}, tot: {total_time:.2f} s]",
            end="",
        )

    # Print the expected response (and eval if desired).
    print()
    if expected_response is not None:
        eval_str = ""
        if scores is not None:
            assert args.data_sets
            ds_name = args.data_sets[0]
            eval_sample = eval_types.Sample(
                sample.audio_transcript or sample.messages[0]["content"],
                expected_answer=expected_response,
                generated_answer=text,
            )
            eval_metric = (
                "asr" if args.asr else "boolq" if ds_name == "boolq" else "instruct"
            )
            result = eval.evaluate_answer(eval_sample, eval_metric)
            if result.score is not None:
                scores.append(result.score)
                eval_name = "score"
                reason_str = ""
                mean = np.mean(scores)
                if isinstance(result, eval_types.WerResult):
                    eval_name = "wer"
                elif isinstance(result, eval_types.InstructResult) and args.verbose:
                    reason_str = f" ({result.reason})"
                eval_str = (
                    f" [{eval_name}: {result.score:.2f}{reason_str}, avg: {mean:.2f}]"
                )
            else:
                eval_str = " [eval failed]"
        print(f"X: {expected_response}{eval_str}")


def oneshot_infer(inference: base.VoiceInference, args: InferArgs):
    prompt = args.prompt or (DEFAULT_ASR_PROMPT if args.asr else DEFAULT_PROMPT)
    if args.audio_file is not None:
        sample = datasets.VoiceSample.from_prompt_and_buf(
            prompt, args.audio_file.read()
        )
    else:
        sample = datasets.VoiceSample.from_prompt(prompt)
    run_tui(-1, inference, sample, args)


def dataset_infer(inference: base.VoiceInference, args: InferArgs):
    assert args.data_sets, "At least one data set must be provided"
    ds_args = datasets.VoiceDatasetArgs(
        data_dir=args.data_dir,
        prompt=args.prompt,
        include_audio=not args.text_only,
        include_context=args.context,
        shuffle=args.shuffle,
        use_mds=args.mds,
        split=args.data_split,
    )
    if args.seed is not None:
        ds_args.shuffle_seed = args.seed
    ds = datasets.create_dataset(args.data_sets[0], ds_args)
    scores: List[float] = []
    for i, sample in enumerate(datasets.Range(ds, args.num_samples)):
        # Store the original question and answer for JSON output.
        question_text = sample.audio_transcript
        expected_answer = sample.messages[1]["content"]
        # Drop any assistant response from the sample.
        sample.messages = sample.messages[:1]
        if not args.json:
            run_tui(i, inference, sample, args, expected_answer, scores)
        else:
            output = inference.infer(
                sample, max_tokens=args.max_tokens, temperature=args.temperature
            )
            obj = {
                "question": question_text,
                "generated_answer": output.text,
                "expected_answer": expected_answer,
            }
            print(json.dumps(obj))


def main(args: InferArgs):
    if args.url is not None:
        api_key = os.environ.get("ULTRAVOX_API_KEY")
        inference = infer_api.create_inference(args.url, args.model, api_key)
    else:
        # Only load our local inference module if we're not using the API.
        from ultravox.inference import ultravox_infer

        inference = ultravox_infer.UltravoxInference(
            args.model,
            tokenizer_id=args.tokenizer,
            audio_processor_id=args.audio_processor,
            device=args.device,
            data_type=args.data_type,
        )
    if args.data_sets is None:
        oneshot_infer(inference, args)
    else:
        dataset_infer(inference, args)


if __name__ == "__main__":
    main(simple_parsing.parse(InferArgs))
