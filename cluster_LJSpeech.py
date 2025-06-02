import argparse
import logging
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from datasets import load_dataset, Audio

from transformers import AutoModel
from transformers import AutoFeatureExtractor
from accelerate import Accelerator
import os
from sklearn.cluster import KMeans
from knnvc.hubconf import wavlm_large, hifigan_wavlm

accelerator = Accelerator()
DEVICE = accelerator.device
torch.backends.cudnn.allow_tf32 = False

UTTER_MHUBERT = "mhubert"
WAVLM = "wavlm"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_dataset(args):
    if args.hubert == WAVLM:
        logging.info("Loading WavLM checkpoint")
        cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "WavLM-Large.pt"
        model_url = "https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt"
        if not model_path.exists():
            urlretrieve(model_url, model_path)
        state_dict = torch.load(model_path, map_location=DEVICE)
        encoder_model = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
    else:
        logging.info("Loading mkhubert checkpoint")
        feature_extractor = AutoFeatureExtractor.from_pretrained("utter-project/mhubert-147")
        encoder_model = AutoModel.from_pretrained("utter-project/mhubert-147").to(DEVICE)
        # for hungarian multilang hubert: "utter-project/mHuBERT-147"
        torch.backends.cudnn.allow_tf32 = False

    lj_speech = load_dataset("keithito/lj_speech", trust_remote_code=True)["train"]
    lj_speech = lj_speech.cast_column("audio", Audio(sampling_rate=16000))

    lj_features = []
    for element in lj_speech:
        with torch.no_grad():
            query_seq = encoder_model.get_features(element["audio"]["path"])
            query_seq_np = query_seq.detach().cpu().numpy()  #.transpose(0, 1).unsqueeze(0)
            lj_features.append(query_seq_np)

    lj_features = np.concatenate(lj_features, axis=0)
    kmeans = KMeans(n_clusters=100).fit(lj_features)

    checkpoint_path = Path("LJSpeech") / "kmeans_100_LJSpeech_WavLM.pt"
    torch.save({"n_features_in_": kmeans.n_features_in_, "_n_threads": kmeans._n_threads,
                "cluster_centers_": kmeans.cluster_centers_}, checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode an audio dataset into soft speech units and the log probabilities of the associated discrete units."
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
