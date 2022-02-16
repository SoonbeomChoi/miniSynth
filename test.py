import torch
import torchaudio
from os import path

import config, preprocess, vocoder
from model import Model


def run(midi_file, savename=None):
    data = preprocess.preprocess(midi_file, test=True)
    model = Model().cuda()

    model.load_state_dict(torch.load(path.join(config.exp_path, 'checkpoint.pt')))
    model.eval()

    audio = []
    for d in data:
        with torch.no_grad():
            note = d['note'].unsqueeze(0).cuda()
            mel_pred = model(note)
            audio.append(vocoder.run(mel_pred))

    audio = torch.cat(audio, dim=-1).cpu()
    if savename:
        torchaudio.save(path.join(config.exp_path, savename + '.wav'), audio, sample_rate=config.sample_rate)

    return audio[0].numpy()


if __name__ == "__main__":
    run(None)