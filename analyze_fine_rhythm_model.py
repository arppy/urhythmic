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
    lj_fine_state_dict = torch.load(Path("LJSpeech") / "rhythm-fine-LJSpeech_WavLM.pt")
    print("LJSpeech", lj_fine_state_dict)
    mc01_fine_state_dict = torch.load(Path("Torgo") / "MC01_rhythm-fine_WavLM.pt")
    print("LJSpeech", mc01_fine_state_dict)
    m01_fine_state_dict = torch.load(Path("Torgo") / "M01_rhythm-fine_WavLM.pt")
    print("LJSpeech", m01_fine_state_dict)



