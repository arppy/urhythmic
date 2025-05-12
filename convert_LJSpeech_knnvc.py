import argparse
from pathlib import Path

import numpy as np

from datasets import load_dataset

import itertools
from urhythmic.utils import SoundType, SILENCE

import torch
import torchaudio
import torch.nn.functional as F

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

    out_path = args.path / "converted_rythm_wav"
    out_path.mkdir(parents=True, exist_ok=True)

    out_path2 = args.path / "converted_rythmknnvc_wav"
    out_path2.mkdir(parents=True, exist_ok=True)

    model_type = RhythmModelFineGrained if args.model == "fine" else RhythmModelGlobal
    time_stretcher = TimeStretcherFineGrained() if args.model == "fine" else TimeStretcherGlobal()
    rhythm_model = model_type(hop_length=HOP_LENGTH, sample_rate=SAMPLE_RATE)
    tgt_rhythm_model_path = LJSPEECH_PATH / "rhythm-fine-LJSpeech_WavLM.pt" if args.model == "fine" else LJSPEECH_PATH / "rhythm-global-LJSpeech_WavLM.pt"

    dataset_ljspeech = load_dataset("keithito/lj_speech", trust_remote_code=True)
    list_of_problematic_wavs = ['LJ007-0217.wav', 'LJ008-0033.wav', 'LJ009-0126.wav', 'LJ010-0182.wav',
                                'LJ011-0097.wav',
                                'LJ011-0152.wav', 'LJ012-0199.wav', 'LJ013-0148.wav', 'LJ013-0231.wav',
                                'LJ016-0144.wav',
                                'LJ018-0092.wav', 'LJ020-0064.wav', 'LJ028-0208.wav']
    stop_file = 'LJ006-0308.wav'
    # stop_file = 'LJ028-0208.wav'
    dataset_ljspeech_paths = []
    for elem in dataset_ljspeech['train']:
        if stop_file in elem['audio']['path']:
            break
        if all(item not in elem['audio']['path'] for item in list_of_problematic_wavs):
            dataset_ljspeech_paths.append(elem['audio']['path'])

    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)

    with torch.no_grad():
        matching_set = knn_vc.get_matching_set(dataset_ljspeech_paths)
    for file in segments_dir.iterdir():
        torch.cuda.empty_cache()
        units = torch.from_numpy(np.load(units_dir / (file.name[:-1]+'y'))).T.unsqueeze(0)
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
            units_stretched = time_stretcher(units, clusters, boundaries, tgt_durations)
        else :
            rhythm_state_dict = {"source_rate": torch.load(src_rhythm_model_path),
                                 "target_rate": torch.load(tgt_rhythm_model_path)}
            rhythm_model.load_state_dict(rhythm_state_dict)
            ratio = rhythm_model()
            units_stretched = time_stretcher(units, ratio)
        units_stretched = units_stretched[0].T.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            wav = knn_vc.hifigan(units_stretched)

        out_file_path = out_path / file.name
        out_file_path = out_file_path.with_suffix(".wav")
        torchaudio.save(out_file_path, wav.squeeze(0).cpu(), 16000)

        try:
            with torch.no_grad():
                query_seq = knn_vc.get_features(str(out_file_path))
                out_wav = knn_vc.match(query_seq, matching_set, topk=8)

            out_file_path2 = out_path2 / file.name
            out_file_path2 = out_file_path2.with_suffix(".wav")
            torchaudio.save(out_file_path2, out_wav.squeeze(0).cpu(), 16000)
        except RuntimeError:
            pass
        except torch.cuda.OutOfMemoryError:
            pass
