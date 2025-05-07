import argparse
from pathlib import Path

import numpy as np

import itertools
from urhythmic.utils import SoundType, SILENCE

import torch
import torchaudio

from urhythmic.stretcher import TimeStretcherGlobal, TimeStretcherFineGrained
from urhythmic.rhythm import RhythmModelFineGrained, RhythmModelGlobal
from accelerate import Accelerator

accelerator = Accelerator()
DEVICE = accelerator.device

HOP_LENGTH = 320
SAMPLE_RATE = 16000
LJSPEECH_PATH = Path("LJSpeech")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert audio samples using Urhythmic."
    )
    parser.add_argument(
        "model",
        help="available models (Urhythmic-Fine or Urhythmic-Global).",
        choices=["fine", "global"],
    )
    parser.add_argument(
        "path",
        metavar="path",
        help="path to the dataset directory.",
        type=Path,
    )
    args = parser.parse_args()

    units_dir = args.path / "soft"
    segments_dir = args.path / "segments"

    out_path = args.path / "converted_wav"
    out_path.mkdir(parents=True, exist_ok=True)

    model_type = RhythmModelFineGrained if args.model == "fine" else RhythmModelGlobal
    time_stretcher = TimeStretcherFineGrained() if args.model == "fine" else TimeStretcherGlobal()
    rhythm_model = model_type(hop_length=HOP_LENGTH, sample_rate=SAMPLE_RATE)
    tgt_rhythm_model_path = LJSPEECH_PATH / "rhythm-fine-LJSpeech_WavLM.pt" if args.model == "fine" else LJSPEECH_PATH / "rhythm-global-LJSpeech_WavLM.pt"

    encoder_model = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
    for file in segments_dir.iterdir():
        units = torch.from_numpy(np.load(units_dir / file.name)).T.unsqueeze(0)
        uid = file.name.split("_")[0]
        src_rhythm_model_path = args.path / (uid+"_rhythm-"+args.model+"_WavLM.pt")
        if args.model == "fine":
            rhythm_state_dict = {"source": torch.load(src_rhythm_model_path),
                                 "target": torch.load(tgt_rhythm_model_path)}
            rhythm_model.load_state_dict(rhythm_state_dict)
            segments = np.load(segments_dir / file.name, allow_pickle=True)
            clusters = list(segments["segments"])
            boundaries = list(segments["boundaries"])
            tgt_durations = rhythm_model(clusters, boundaries)
            try :
                units_stretched = time_stretcher(units, clusters, boundaries, tgt_durations)
            except RuntimeError :
                units = [
                    units[..., t0:tn]
                    for cluster, (t0, tn) in zip(clusters, itertools.pairwise(boundaries))
                    if cluster not in SILENCE or tn - t0 > 3
                ]
                print(src_rhythm_model_path, units.shape, clusters, boundaries, tgt_durations)
                continue
        else :
            rhythm_state_dict = {"source_rate": torch.load(src_rhythm_model_path),
                                 "target_rate": torch.load(tgt_rhythm_model_path)}
            rhythm_model.load_state_dict(rhythm_state_dict)
            ratio = rhythm_model()
            units_stretched = time_stretcher(units, ratio)
        units_stretched = units_stretched[0].T.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            wav = encoder_model.hifigan(units_stretched)

        out_file_path = out_path / file.name
        out_file_path = out_file_path.with_suffix(".wav")
        torchaudio.save(out_file_path, wav.squeeze(0).cpu(), 16000)
