import logging
from pathlib import Path

import webrtcvad
import struct
import numpy as np
import librosa
from tqdm import tqdm

import torchaudio
import torch
import torch.nn.functional as F

from datasets import load_dataset
from urhythmic.segmenter import Segmenter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INT16_MAX = (2**15) - 1


def mark_silences(
    vad: webrtcvad.Vad,
    wav: torch.Tensor,
    hop_length: int = 320,
    sample_rate: int = 16000,
    pad: int = 40,
):
    """Marks silent frames using webrtcvad.

    Args:
        vad (webrtcvad.Vad): instance of the webrtcvad.Vad class.
        wav (Tensor): audio waveform of shape (1, T) where T is the number of samples.
        hop_length (int): the hop length measured in number of frames (defaults to 320).
        sample_rate (int): the sample rate (defaults to 16kHz).
        pad (int): padding (defaults to 40)

    Returns:
        NDArray: array of booleans indicating whether each frame is silent.
    """
    win_length = hop_length

    wav = F.pad(wav, (pad, pad))  # add padding to match HuBERT
    wav = wav[:, : wav.size(-1) - (wav.size(-1) % win_length)]

    pcm = struct.pack(
        "%dh" % wav.size(-1),
        *(np.round(wav.squeeze().numpy() * INT16_MAX)).astype(np.int16),
    )

    flags = []
    for window_start in range(0, wav.size(-1), hop_length):
        window_end = window_start + win_length
        flag = vad.is_speech(pcm[window_start * 2 : window_end * 2], sample_rate)
        flags.append(flag)
    return ~np.array(flags)


def mark_voiced(
    wav: torch.Tensor,
    hop_length: int = 320,
    win_length: int = 1024,
    sample_rate: int = 16000,
):
    _, voiced_flags, _ = librosa.pyin(
        wav.squeeze().numpy(),
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C5"),
        sr=sample_rate,
        hop_length=hop_length,
        win_length=win_length,
    )
    return voiced_flags

if __name__ == "__main__":
    logger.info(f"Training Segmenter on LJSpeech")

    segmenter = Segmenter(num_clusters=3)

    # call cluster_LJSpeech.py to generate kmeans_100_LJSpeech_WavLM.pt
    checkpoints = torch.load(Path("LJSpeech") / "kmeans_100_LJSpeech_WavLM.pt")
    codebook = checkpoints["cluster_centers_"]
    segmenter.cluster(codebook)

    vad = webrtcvad.Vad(2)

    lj_speech = load_dataset("keithito/lj_speech", trust_remote_code=True)["train"]
    logprobs_dir = Path("LJSpeech") / "logprobs"
    checkpoint_path = Path("LJSpeech") / "segmenter_LJSpeech_WavLM.pt"
    logger.info("Extracting VAD and voicing flags")

    utterances = []
    for element in lj_speech:
        wav_path = element["audio"]["path"]
        wav_name = wav_path.split('/')[-1]
        basename = wav_name.split('.')[:-1]
        basename_suffix = ".".join(basename)+".npy"

        wav, _ = torchaudio.load(element["audio"]["path"], normalize=True)
        log_probs = np.load(logprobs_dir / basename_suffix)

        segments, boundaries = segmenter._segment(log_probs)
        silences = mark_silences(vad, wav)
        voiced_flags = mark_voiced(wav)

        utterances.append((segments, boundaries, silences, voiced_flags))

    logger.info("Identifying the cluster id corresponding to each sound type")
    sound_types = segmenter.identify(utterances)

    logger.info(f"cluster 0 - {sound_types[0]}")
    logger.info(f"cluster 1 - {sound_types[1]}")
    logger.info(f"cluster 2 - {sound_types[2]}")

    logger.info(f"Saving checkpoint to {checkpoint_path}")
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(segmenter.state_dict(), checkpoint_path)

