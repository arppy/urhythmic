from pathlib import Path
import numpy as np
import torch
import torchaudio
import tempfile

from datasets import load_dataset, Audio

from accelerate import Accelerator
import soundfile as sf

from speechbrain.pretrained import SepformerSeparation as Enhancer

accelerator = Accelerator()
DEVICE = accelerator.device
torch.backends.cudnn.allow_tf32 = False

def filter_Torgo_dataset(element) :
    file_name = element['audio']['path']
    if "headMic" in file_name:
        return True
    else :
        return False

def process_audio(element, model, out_path):
    """Process a single audio element"""
    # Prepare output path
    out_file_path = out_path / Path(element['audio']['path']).name
    out_file_path = out_file_path.with_suffix(".wav")
    audio = torch.from_numpy(element['audio']['array']).to(torch.float32).unsqueeze(0).to(DEVICE)
    if out_file_path.exists():
        return  # Skip already processed files
    try:
        with torch.no_grad():
            # Process audio
            enhanced_speech = model(audio)

            # Convert to numpy and ensure proper shape/dtype
            enhanced_np = enhanced_speech[:,:,0].squeeze().cpu().numpy()

            # Normalize to prevent clipping
            enhanced_np = enhanced_np / np.max(np.abs(enhanced_np))

            # Save final output
            sf.write(out_file_path, enhanced_np, 16000, subtype='PCM_16')

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        print(f"Error processing {element['audio']['path']}: {str(e)}")
        return

# Load the pretrained model
#model = Enhancer.from_hparams(source="speechbrain/sepformer-dns4-16k-enhancement", savedir="pretrained_models/sepformer-dns4-enhancement", run_opts={DEVICE})
#out_path = Path("Torgo") / "SepFormerDNS4_wav"
model = Enhancer.from_hparams(source="speechbrain/sepformer-whamr16k", savedir='pretrained_models/sepformer-whamr16k', run_opts={DEVICE})
out_path = Path("Torgo") / "SepFormerWHAMR_wav"

out_path.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("abnerh/TORGO-database", download_mode="reuse_cache_if_exists")["train"]
dataset = dataset.filter(filter_Torgo_dataset)
#dataset = load_dataset("keithito/lj_speech", trust_remote_code=True)["train"]
#dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

for element in dataset:
    torch.cuda.empty_cache()
    process_audio(element, model, out_path)