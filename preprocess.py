import torch
import pretty_midi
from os import listdir, mkdir, path
from tqdm import tqdm

import config
import audio


def load_midi(filename):
    midi_data = pretty_midi.PrettyMIDI(filename)

    return midi_data.instruments[0].notes


def time2frame(x, frame_rate):
    return int(frame_rate*x)


def swap_dict_list(x):
        """ Swap dict of list with the same lengths to list of dict """
        length = len(next(iter(x.values())))
        for key in x:
            if len(x[key]) != length:
                raise AssertionError('All the values should be the same length of lists.')

        y = []
        for i in range(length):
            temp = {key: x[key][i] for key in x}
            y.append(temp)

        return y


class Range:
    def __init__(self, start=0, duration=0, frame_rate=1):
        self.start = time2frame(start, frame_rate)
        self.end = time2frame(start + duration, frame_rate)

    def duration(self):
        return self.end - self.start
    
    def to_arange(self):
        return torch.arange(self.start, self.end)


class SegmentList:
    def __init__(self):
        self.list = []
        self.segment = []

    def extend(self, x):
        self.segment.extend(x)

    def split(self):
        to_type = torch.FloatTensor
        is_list = False
        if type(self.segment[0]) is int:
            to_type = torch.LongTensor
        if type(self.segment[0]) is list:
            is_list = True
            if type(self.segment[0][0]) is int:
                to_type = torch.LongTensor
        
        self.segment = to_type(self.segment)
        if is_list:
            self.segment = self.segment.T

        self.list.append(self.segment)
        self.segment = []


class MIDISegmenter:
    def __init__(self, frame_rate, min_length=1.0, max_length=10.0, min_silence=0.3, max_silence=1.2, release=0.0):
        self.frame_rate = frame_rate
        self.min_length = time2frame(min_length, frame_rate)
        self.max_length = time2frame(max_length - release, frame_rate)
        self.min_silence = time2frame(min_silence, frame_rate)
        self.max_silence = time2frame(max_silence, frame_rate)
        self.release = time2frame(release, frame_rate)

        self.note = SegmentList()
        self.frame_range_list = []

        self.prev_range = None
        self.curr_range = None
        self.next_range = None

    def check_split(self, frame_range):
        # Conditions to split range
        split = False
        if self.next_range.end - frame_range.start > self.max_length: 
            split = True
        elif self.curr_range.end - frame_range.start > self.min_length:
            if self.next_range.start - self.curr_range.end > self.max_silence:
                split = True
            else:
                split = False
        else:
            split = False if self.next_range.start - self.curr_range.end < self.min_silence else True

        return split

    def run(self, note_list, retain_silence=False):
        frame_range = Range()
        for i in range(len(note_list)):
            # Remove MIDI note overlap
            if i < len(note_list) - 1:
                if note_list[i].end > note_list[i+1].start:
                    note_list[i].end = note_list[i+1].start

            # Initialize note ranges
            self.prev_range = Range(0, 0, self.frame_rate)
            self.curr_range = Range(note_list[i].start, note_list[i].end - note_list[i].start, self.frame_rate)
            self.next_range = Range(note_list[-1].end, 0, self.frame_rate)
            if i > 0:
                self.prev_range = Range(note_list[i-1].start, note_list[i-1].end - note_list[i-1].start, self.frame_rate)
            if i < len(note_list) - 1:
                self.next_range = Range(note_list[i+1].start, note_list[i+1].end - note_list[i+1].start, self.frame_rate)

            if len(self.note.segment) == 0:
                if self.curr_range.start - frame_range.start > self.max_silence:
                    if retain_silence:
                        frame_range.end = self.curr_range.start - self.max_silence
                        self.extend_silence(frame_range)

                    frame_range.start = self.curr_range.start - self.max_silence
                self.prev_range = Range(frame_range.start, 0, 1)

            split = self.check_split(frame_range)
            if i == len(note_list) - 1:
                split = True

            silence = self.curr_range.start - self.prev_range.end
            pitch = note_list[i].pitch
            n = silence*[0] + self.curr_range.duration()*[pitch]
            self.note.extend(n)

            if split:
                frame_range.end = self.curr_range.end
                if self.next_range.start - self.curr_range.end > self.release:
                    frame_range.end = self.curr_range.end + self.release
                    self.note.extend(self.release*[0])

                self.note.split()
                self.frame_range_list.append(frame_range.to_arange())

                # Initialize next frame range
                frame_range.start = self.curr_range.end
                if self.next_range.start - self.curr_range.end > self.release:
                    frame_range.start = self.curr_range.end + self.release
                if self.next_range.start - frame_range.start > self.max_silence:
                    if retain_silence:
                        frame_range.end = self.next_range.start - self.max_silence
                        self.extend_silence(frame_range)

                    frame_range.start = self.next_range.start - self.max_silence

        note_list = self.note.list
        frame_range = self.frame_range_list

        return note_list, frame_range


def preprocess(basename):
    note = load_midi(path.join(config.dataset_path, 'mid', basename + '.mid'))
    wave = audio.load(path.join(config.dataset_path, 'wav', basename + '.wav'), config.sample_rate)

    mel_fn = audio.MelSpectrogram(
        config.sample_rate, config.fft_size, config.hop_size,
        config.mel_size, config.f_min, config.f_max, config.min_level_db)
    mel = mel_fn(wave)

    segmenter_fn = MIDISegmenter(config.sample_rate/config.hop_size)
    note, frame_range = segmenter_fn.run(note)
    mel_list = []
    for i in range(len(frame_range)):
        mel_list.append(mel[..., frame_range[i]])

    data = {'note': note, 'mel': mel_list}
    data = swap_dict_list(data)

    return data


def main():
    if not path.exists(path.join(config.data_path, 'train')):
        mkdir(path.join(config.data_path, 'train'))
    if not path.exists(path.join(config.data_path, 'test')):
        mkdir(path.join(config.data_path, 'test'))

    for set_name in ['train', 'test']:
        file_list = path.join(config.data_path, set_name + '_list.txt')
        with open(file_list) as f:
            file_list = f.read().splitlines()

        for basename in tqdm(file_list):
            data = preprocess(basename)
            torch.save(data, path.join(config.data_path, set_name, basename + '.pt'))


if __name__ == "__main__":
    main()