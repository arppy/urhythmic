import argparse
import logging
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import torch
import numpy as np
import itertools

from urhythmic.segmenter import Segmenter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def segment_file(segmenter, in_path, out_path):
    log_probs = np.load(in_path)
    segments, boundaries = segmenter(log_probs)
    np.savez(out_path.with_suffix(".npz"), segments=segments, boundaries=boundaries)
    return log_probs.shape[0], np.mean(np.diff(boundaries))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment an audio dataset with LJSpeech segmenter.")
    parser.add_argument(
        "path",
        metavar="path",
        type=Path,
        help="Path of dataset.",
    )
    args = parser.parse_args()
    logging.info("Loading segmenter checkpoint")
    segmenter_state_dict = torch.load(Path("LJSpeech") / "segmenter_LJSpeech_WavLM.pt")
    segmenter = Segmenter(num_clusters=3)
    segmenter.load_state_dict(segmenter_state_dict)
    in_dir = args.path / "logprobs"
    out_dir = args.path / "segments"
    in_paths = list(in_dir.rglob("*.npy"))
    out_paths = [out_dir / path.relative_to(in_dir) for path in in_paths]

    logger.info("Setting up folder structure")
    for path in tqdm(out_paths):
        path.parent.mkdir(exist_ok=True, parents=True)

    logger.info("Segmenting dataset")
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(
            tqdm(
                executor.map(
                    segment_file,
                    itertools.repeat(segmenter),
                    in_paths,
                    out_paths,
                ),
                total=len(in_paths),
            )
        )

    frames, boundary_length = zip(*results)
    logger.info(f"Segmented {sum(frames) * 0.02 / 60 / 60:.2f} hours of audio")
    logger.info(
        f"Average segment length: {np.mean(boundary_length) * 0.02:.4f} seconds"
    )

