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

def filter_Torgo_dataset(element) :
    file_name = element['audio']['path']
    if "headMic" in file_name:
        return True
    else :
        return False

def encode_dataset(args):
    logging.info("Loading hubert checkpoint")

    if args.hubert == BSHALL_HUBERT :
        encoder_model = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True).to(DEVICE)
    elif args.hubert == WAVLM :
        encoder_model = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
    else :
        feature_extractor = AutoFeatureExtractor.from_pretrained("utter-project/mhubert-147")
        encoder_model = AutoModel.from_pretrained("utter-project/mhubert-147").to(DEVICE)
        # for hungarian multilang hubert: "utter-project/mHuBERT-147"
        torch.backends.cudnn.allow_tf32 = False

    dataset = load_dataset("abnerh/TORGO-database", download_mode="reuse_cache_if_exists")["train"]
    dataset = dataset.filter(filter_Torgo_dataset)

    lj_speech = load_dataset("keithito/lj_speech", trust_remote_code=True)["train"]

    lj_features = []
    for element in lj_speech :
        with torch.no_grad():
            query_seq = encoder_model.get_features(element["audio"]["path"])
            query_seq_np = query_seq.detach().cpu().numpy() #.transpose(0, 1).unsqueeze(0)
            lj_features.append(query_seq_np)
    lj_features = np.concatenate(lj_features, axis=0)
    kmeans = KMeans(n_clusters=args.n_clusters).fit(lj_features)

    logging.info(f"Encoding dataset TORGO")
    for element in dataset:
        if args.hubert == BSHALL_HUBERT :
            wav = torch.tensor(element["audio"]["array"], dtype=torch.float32)
            wav = wav.unsqueeze(0).unsqueeze(0).cuda()
            with torch.inference_mode():
                units, log_probs = encode(encoder_model, wav)
        elif args.hubert == WAVLM:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                sf.write(tmp_path, element['audio']['array'], element['audio']['sampling_rate'])
            try:
                with torch.no_grad():
                    query_seq = encoder_model.get_features(tmp_path)
                    query_seq = query_seq.transpose(0, 1).unsqueeze(0)
            except torch.cuda.OutOfMemoryError :
                continue
            finally:
                os.unlink(tmp_path) # Clean up the temporary file after use
        else :
            wav = feature_extractor(
                element["audio"]["array"],
                return_tensors="pt",
                sampling_rate=16000,
            ).to(DEVICE)
            with torch.inference_mode():
                outputs = encoder_model(wav['input_values'])
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
