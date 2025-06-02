import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, Audio

from transformers import AutoModel
from transformers import AutoFeatureExtractor
from accelerate import Accelerator
import os
from sklearn.cluster import MiniBatchKMeans
from knnvc.hubconf import wavlm_large, hifigan_wavlm
from knnvc.matcher import KNeighborsVC

accelerator = Accelerator()
DEVICE = accelerator.device
torch.backends.cudnn.allow_tf32 = False

BATCH_SIZE = 512

UTTER_MHUBERT = "mhubert"
WAVLM = "wavlm"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_dataset(args):
    if args.hubert == WAVLM:
        #logging.info("Loading WavLM checkpoint")
        #cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        #cache_dir.mkdir(parents=True, exist_ok=True)
        #model_path = cache_dir / "WavLM-Large.pt"
        #model_url = "https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt"
        #if not model_path.exists():
        #    urlretrieve(model_url, model_path)
        #state_dict = torch.load(model_path, map_location=DEVICE)
        wavlm = wavlm_large(pretrained=True, device=DEVICE)
        hifigan, hifigan_cfg = hifigan_wavlm(prematched=True, pretrained=True, device=DEVICE)
        encoder_model = KNeighborsVC(wavlm=wavlm, hifigan=hifigan, hifigan_cfg=hifigan_cfg, device=DEVICE)
    else:
        logging.info("Loading mkhubert checkpoint")
        feature_extractor = AutoFeatureExtractor.from_pretrained("utter-project/mhubert-147")
        encoder_model = AutoModel.from_pretrained("utter-project/mhubert-147").to(DEVICE)
        # for hungarian multilang hubert: "utter-project/mHuBERT-147"
        torch.backends.cudnn.allow_tf32 = False

    lj_speech = load_dataset("keithito/lj_speech", trust_remote_code=True)["train"]
    lj_speech = lj_speech.cast_column("audio", Audio(sampling_rate=16000))
    kmeans = MiniBatchKMeans(n_clusters=100, batch_size=BATCH_SIZE)
    feature_list = []
    i = 0
    for element in lj_speech:
        audio = torch.from_numpy(element['audio']['array']).to(torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            features = encoder_model.get_features(audio).cpu().numpy()  # .transpose(0, 1).unsqueeze(0)
        feature_list.append(features)
        if len(feature_list) >= BATCH_SIZE or i == len(lj_speech) - 1:
            lj_features_batches = np.concatenate(feature_list, axis=0)
            kmeans.partial_fit(lj_features_batches)
            feature_list = []
            torch.cuda.empty_cache()
        logging.info(str(len(feature_list)) + ":" + element['audio']['path'])
        i += 1

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
