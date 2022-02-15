import torch
from torch.utils.data import Dataset, DataLoader
from os import path
from itertools import islice

import config


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def collate_fn(data):
    batch_size = len(data)
    src_len = torch.zeros(batch_size).long()
    trg_len = torch.zeros(batch_size).long()
    for i in len(data):
        trg_len[i] = data[i]['mel'].size(-1)
        src_len[i] = data[i]['note'].size(-1)

    trg_len, indices = trg_len.sort(descending=True)
    src_len = src_len[indices]

    data_padded = {
        'note': torch.zeros(batch_size, max(src_len)),
        'note_len': src_len,
        'mel': torch.zeros(batch_size, data[0]['mel'].size(1), max(trg_len)),
        'mel_len': trg_len}
    for i in len(data):
        data_padded['note'][i, :src_len[i]] = data[indices[i]]['note']
        data_padded['mel'][i, ..., :src_len[i]] = data[indices[i]]['mel']

    return data_padded['note'], data_padded['mel']


class AudioMIDIDataset(Dataset):
    def __init__(self, data_path):
        super(AudioMIDIDataset, self).__init__(data_path)
        self.data = []
        for filename in data_path:
            self.data += torch.load(filename)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def load():
    dataset = dict()
    for set_name in ['train', 'test']:
        dataset[set_name] = AudioMIDIDataset(path.join(config.data_path, set_name))

    dataloader = dict()
    dataloader['train'] = DataLoader(
        dataset['train'], batch_size=config.batch_size,
        shuffle=True, collate_fn=collate_fn, drop_last=False)
    dataloader['train'] = islice(iter(cycle(dataloader['train']), config.save_step)
    dataloader['test'] = DataLoader(
        dataset['test'], batch_size=1,
        shuffle=False, collate_fn=collate_fn, drop_last=False)

    return dataloader