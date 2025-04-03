import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torchaudio
import torchaudio.functional as AF
from datasets import load_dataset
from urhythmic.model import encode


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_dataset(args):
    logging.info("Loading hubert checkpoint")
    hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").cuda()
    # for hungarian multilang hubert: "utter-project/mHuBERT-147"
    dataset = load_dataset("abnerh/TORGO-database", download_mode="reuse_cache_if_exists")["train"]

    logging.info(f"Encoding dataset TORGO")
    for element in dataset:
        #TODO
        wav, sr = torchaudio.load(element)
        wav = wav.unsqueeze(0).cuda()

        with torch.inference_mode():
            units, log_probs = encode(hubert, wav)

        units_out_path = args.out_dir / "soft" / element.relative_to(args.in_dir)
        units_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(units_out_path.with_suffix(".npy"), units.squeeze().cpu().numpy())

        probs_out_path = args.out_dir / "logprobs" / element.relative_to(args.in_dir)
        probs_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(probs_out_path.with_suffix(".npy"), log_probs.squeeze().cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode an audio dataset into soft speech units and the log probabilities of the associated discrete units."
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    args = parser.parse_args()
    encode_dataset(args)
