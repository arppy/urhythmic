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

from transformers import AutoModel
from transformers import AutoFeatureExtractor
from accelerate import Accelerator

accelerator = Accelerator()
DEVICE = accelerator.device
torch.backends.cudnn.allow_tf32 = False

UTTER_MHUBERT = "mhubert"
BSHALL_HUBERT = "hubert"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def filter_Torgo_dataset(element) :
    file_name = element['audio']['path']
    if "headMic" in file_name:
        return True
    else :
        return False

def encode_dataset(args):
    logging.info("Loading hubert checkpoint")

    if args.hubert == BSHALL_HUBERT :
        hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True).to(DEVICE)
    else :
        feature_extractor = AutoFeatureExtractor.from_pretrained("utter-project/mhubert-147")
        hubert = AutoModel.from_pretrained("utter-project/mhubert-147").to(DEVICE)
        # for hungarian multilang hubert: "utter-project/mHuBERT-147"
        torch.backends.cudnn.allow_tf32 = False

    dataset = load_dataset("abnerh/TORGO-database", download_mode="reuse_cache_if_exists")["train"]
    dataset = dataset.filter(filter_Torgo_dataset)
    logging.info(f"Encoding dataset TORGO")
    for element in dataset:
        if args.hubert == BSHALL_HUBERT :
            wav = torch.tensor(element["audio"]["array"], dtype=torch.float32)
            wav = wav.unsqueeze(0).unsqueeze(0).cuda()
            with torch.inference_mode():
                units, log_probs = encode(hubert, wav)
        else :
            wav = feature_extractor(
                element["audio"]["array"],
                return_tensors="pt",
                sampling_rate=16000,
            ).to(DEVICE)
            with torch.inference_mode():
                outputs = hubert(wav['input_values'])
            units = outputs.last_hidden_state.transpose(1, 2)
            log_probs = torch.zeros(units.shape[0], units.shape[2], 100)  # K=100 dummy clusters

        wav_name = element["audio"]["path"]
        units_out_path = args.out_dir / "soft" / wav_name
        units_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(units_out_path.with_suffix(".npy"), units.squeeze().cpu().numpy())

        probs_out_path = args.out_dir / "logprobs" / wav_name
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
        default=Path("."),
        type=Path,
    )
    parser.add_argument(
        "hubert",
        metavar="hubert",
        help="name of encoder",
        default=BSHALL_HUBERT,
        type=str,
    )
    args = parser.parse_args()
    encode_dataset(args)
