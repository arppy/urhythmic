import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from datasets import load_dataset
from urhythmic.model import encode

from transformers import AutoModel
from transformers import AutoFeatureExtractor
from accelerate import Accelerator
import tempfile
import soundfile as sf  # You may need to install this library: pip install soundfile
import os
from sklearn.cluster import KMeans

accelerator = Accelerator()
DEVICE = accelerator.device
torch.backends.cudnn.allow_tf32 = False

UTTER_MHUBERT = "mhubert"
BSHALL_HUBERT = "hubert"
WAVLM = "wavlm"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def encode_dataset(args):
    logging.info("Loading wavlm checkpoint")

    if args.hubert == BSHALL_HUBERT :
        encoder_model = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True).to(DEVICE)
    elif args.hubert == WAVLM :
        encoder_model = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
        # call cluster_LJSpeech.py to generate ljspeech_wavlm_kmeans_100.pt
        kmeans_import = torch.load("ljspeech_wavlm_kmeans_100.pt")
        cluster_centers = torch.from_numpy(kmeans_import["cluster_centers_"]).to(DEVICE)
    else :
        feature_extractor = AutoFeatureExtractor.from_pretrained("utter-project/mhubert-147")
        encoder_model = AutoModel.from_pretrained("utter-project/mhubert-147").to(DEVICE)
        # for hungarian multilang hubert: "utter-project/mHuBERT-147"
        torch.backends.cudnn.allow_tf32 = False

    lj_speech = load_dataset("keithito/lj_speech", trust_remote_code=True)["train"]

    logging.info(f"Encoding dataset LJSpeech")
    for element in lj_speech:
        if args.hubert == WAVLM:
            query_seq = encoder_model.get_features(element["audio"]["path"])
            units = query_seq.unsqueeze(0)
            logits = torch.cosine_similarity(units.unsqueeze(2), cluster_centers.unsqueeze(0).unsqueeze(0), dim=-1)
            log_probs = F.log_softmax(logits/0.1, dim=-1)

        wav_name = element["audio"]["path"].split('/')[-1]
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
        default=WAVLM,
        type=str,
    )
    args = parser.parse_args()
    encode_dataset(args)
