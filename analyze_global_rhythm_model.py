import argparse
from pathlib import Path

import numpy as np

from datasets import load_dataset

import soundfile as sf

import torch
import torchaudio



HOP_LENGTH = 320
SAMPLE_RATE = 16000
LJSPEECH_PATH = Path("LJSpeech")

if __name__ == "__main__":
    lj_global_state_dict = torch.load(Path("LJSpeech") / "rhythm-global-LJSpeech_WavLM.pt")
    print("LJSpeech", lj_global_state_dict)
    path = Path("Torgo")
    for file in list(path.glob('*global_WavLM.pt')) :
        rhythm_state_dict = torch.load(file)
        uid = file.name.split("_")[0]
        print(uid, rhythm_state_dict)


