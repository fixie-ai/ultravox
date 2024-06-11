#!/usr/bin/env python

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import requests
import torch
from loguru import logger
from torchaudio import load
from torchaudio.functional import resample
from tqdm import tqdm

from ultravox.model.hubert import HubertWithKmeans

_AUDIO_SUFFIXES = ("*.wav", "*.mp3", "*.flac", "*.ogg")
_MULTILINGUAL_OUTPUT_LAYER = 11
_SAMPLE_RATE = 16000

_SCRIPT_ROOT_DIR = Path(__file__).parent.parent
_MODEL_DIR = _SCRIPT_ROOT_DIR / "models"

# English
_MODEL_CHECKPOINT = "hubert_base_ls960.pt"
_MODEL_KMEANS = "hubert_base_ls960_L9_km500.bin"

# Multilingual
_MODEL_CHECKPOINT_MULTI = "mhubert_base_vp_en_es_fr_it3.pt"
_MODEL_KMEANS_MULTI = "mhubert_base_vp_en_es_fr_it3_L11_km1000.bin"


def _download_file(filename):
    save_path = _MODEL_DIR / filename
    if save_path.exists():
        logger.debug(f"Skipping download of existing file [file={save_path}]")
        return

    logger.debug(f"Downloading [file={save_path}]")
    response = requests.get(
        f"https://dl.fbaipublicfiles.com/hubert/{filename}", stream=True
    )
    response.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def _ensure_models_present():
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _download_file(_MODEL_CHECKPOINT)
    _download_file(_MODEL_KMEANS)
    _download_file(_MODEL_CHECKPOINT_MULTI)
    _download_file(_MODEL_KMEANS_MULTI)


def _load_model(multilingual):
    _ensure_models_present()
    if multilingual:
        checkpoint_path = _MODEL_DIR / _MODEL_CHECKPOINT_MULTI
        kmeans_path = _MODEL_DIR / _MODEL_KMEANS_MULTI
        return HubertWithKmeans(
            checkpoint_path.as_posix(),
            kmeans_path.as_posix(),
            output_layer=_MULTILINGUAL_OUTPUT_LAYER,
        )
    else:
        checkpoint_path = _MODEL_DIR / _MODEL_CHECKPOINT
        kmeans_path = _MODEL_DIR / _MODEL_KMEANS
        return HubertWithKmeans(checkpoint_path.as_posix(), kmeans_path.as_posix())


def _replace_suffix(filename, new_suffix):
    return filename.with_name(filename.stem + new_suffix)


def _process_file(device, hubert, filename):
    try:
        new_suffix = "_hubert.npy" if not args.multilingual else "_mhubert.npy"
        output_filename = _replace_suffix(filename, new_suffix)
        if output_filename.exists():
            logger.debug(f"Skipping pre-extracted file [file={output_filename}].")
            return

        audio, sr = load(filename)
        audio = resample(audio.to(device), sr, _SAMPLE_RATE)

        tokens = hubert(audio)
        tokens = tokens.squeeze(0).cpu().numpy()

        logger.debug(f"Saving [file={output_filename}, token_len={len(tokens)}].")
        np.save(output_filename, tokens)
    except Exception as e:
        logger.warning(f"Failed to process [file={filename}, error={e}]")


def _list_supported_audio_files(directory, extensions=_AUDIO_SUFFIXES):
    return [file for ext in extensions for file in Path(directory).rglob(ext)]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model.")
    hubert = _load_model(args.multilingual).to(device)
    logger.info("Model loaded.")
    # TODO(shaper): Could parallelize with `GPUExecutor` or similar.
    audio_files = _list_supported_audio_files(args.data_dir)
    logger.info(f"Found {len(audio_files)} audio files.")
    for filename in tqdm(
        audio_files, desc="Extracting Hubert tokens", total=len(audio_files)
    ):
        _process_file(device, hubert, filename)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--multilingual", "-m", action="store_true")
    args = parser.parse_args()
    main(args)
