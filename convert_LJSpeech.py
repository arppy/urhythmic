import argparse
from pathlib import Path

import numpy as np

import torch
import torchaudio
import torchaudio.functional as AF

from urhythmic.stretcher import TimeStretcherGlobal, TimeStretcherFineGrained
from urhythmic.rhythm import RhythmModelFineGrained, RhythmModelGlobal

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
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    args = parser.parse_args()

    logprobs_dir = args.path / "logprobs"
    units_dir = args.path / "soft"
    segments_dir = args.path / "segments"

    src_rhythm_model_path = args.path / "rhythm-fine-Torgo_WavLM.pt" if args.model == "fine" else args.path / "rhythm-global-Torgo_WavLM.pt"
    tgt_rhythm_model_path = LJSPEECH_PATH / "rhythm-fine-LJSpeech_WavLM.pt" if args.model == "fine" else LJSPEECH_PATH / "rhythm-global-LJSpeech_WavLM.pt"

    model_type = RhythmModelFineGrained if args.model == "fine" else RhythmModelGlobal
    time_stretcher = TimeStretcherFineGrained() if args.model == "fine" else TimeStretcherGlobal()
    rhythm_model = model_type(hop_length=HOP_LENGTH, sample_rate=SAMPLE_RATE)
    rhythm_state_dict = {"source": torch.load(src_rhythm_model_path), "target": torch.load(tgt_rhythm_model_path)}
    rhythm_model.load_state_dict(rhythm_state_dict)

    encoder_model = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
    for file in segments_dir.iterdir():
        log_probs = np.load(logprobs_dir / file.name)
        segments = np.load(segments_dir / file.name, allow_pickle=True)
        clusters = list(segments["segments"])
        boundaries = list(segments["boundaries"])
        tgt_durations = rhythm_model(clusters, boundaries)
        units = torch.from_numpy(np.load(units_dir / file.name)).T.unsqueeze(0)
        units_stretched = time_stretcher(units, clusters, boundaries, tgt_durations)

        wav = encoder_model.vocoder(units_stretched)

        out_path = args.out_dir / args.path.relative_to(args.in_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(
            out_path.with_suffix(args.extension), wav.squeeze(0).cpu(), 16000
        )


