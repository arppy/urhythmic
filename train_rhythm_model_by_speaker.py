import argparse
import logging
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from urhythmic.rhythm import RhythmModelFineGrained, RhythmModelGlobal


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOP_LENGTH = 320
SAMPLE_RATE = 16000


def train_rhythm_model(args):
    logger.info(f"Training {args.model} rhythm model on {args.dataset_dir}")

    model_type = RhythmModelFineGrained if args.model == "fine" else RhythmModelGlobal
    rhythm_model = model_type(hop_length=HOP_LENGTH, sample_rate=SAMPLE_RATE)

    utterances = {}
    for path in args.dataset_dir.iterdir():
        file = np.load(path, allow_pickle=True)
        uid = path.name.split("_")[0]
        segments = list(file["segments"])
        boundaries = list(file["boundaries"])
        if uid not in utterances :
            utterances[uid] = []
        utterances[uid].append((segments, boundaries))

    for uid in utterances :
        save_path = args.checkpoint_path / (uid+"_rhythm-"+args.model+"_WavLM.pt")
        dists = rhythm_model._fit(utterances)
        logger.info(f"Saving checkpoint to {save_path}")
        torch.save(dists, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the FineGrained or Global rhythm model."
    )
    parser.add_argument(
        "model",
        help="type of rhythm model (fine-grained or global).",
        type=str,
        choices=["fine", "global"],
    )
    parser.add_argument(
        "dataset_dir",
        metavar="dataset-dir",
        help="path to the directory of segmented speech.",
        type=Path,
    )
    parser.add_argument(
        "checkpoint_path",
        metavar="checkpoint-path",
        help="path to save checkpoint.",
        type=Path,
    )
    args = parser.parse_args()
    train_rhythm_model(args)
