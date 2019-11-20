# Dependencies: hparams.py, wavernn_model.py (model.py in orig repo) 

import torch
import numpy as np
import librosa
from wavernn_model import build_model
from hparams import hparams as hp
import sys

CHECKPOINT_PATH = 'wavernn_chkpt.pth'
MEL_PATH = 'gen_spectrum.npy'

use_cuda = torch.cuda.is_available()

def load_path(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model):
    print("Load checkpoint from: {}".format(path))
    checkpoint = load_path(path)
    model.load_state_dict(checkpoint["state_dict"])
    return model

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python convert.py output_path')
        sys.exit(1)
    loaded_model = load_checkpoint(CHECKPOINT_PATH, build_model())
    mel = np.load(MEL_PATH)
    mel = np.clip(mel, 0, 1)
    wav = loaded_model.generate(mel)
    librosa.output.write_wav(sys.argv[1], wav, sr=hp.sample_rate)
