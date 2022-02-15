import math
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F


def load(filename, sample_rate):
    audio, source_rate = torchaudio.load(filename)
    if source_rate != sample_rate:
        resample_fn = torchaudio.transforms.Resample(source_rate, sample_rate)
        audio = resample_fn(audio)

    return audio[0]


def to_tensor(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float)

    return x 


def amp2db(x, min_level_db=None):
    x = to_tensor(x)

    x = 20.0*torch.log10(x)
    if min_level_db is not None:
        min_level_db = torch.tensor(min_level_db).float()
        x = torch.max(x, min_level_db)

    return x


def db2amp(x):
    x = to_tensor(x)

    return torch.pow(10.0, x*0.05)


class dBNorm:
    def __init__(self, min_level_db):
        self.min_level_db = min_level_db

    def __call__(self, x):
        x = amp2db(x)

        return torch.clamp((x - self.min_level_db)/(-self.min_level_db), 0, 1)

    def inv(self, x):
        x = x.clamp(0, 1)*(-self.min_level_db) + self.min_level_db

        return db2amp(x)


class MelScale(nn.Module):
    """
    Turn a normal STFT into non-HTK mel-frequency STFT
    """
    def __init__(self, n_mels=128, sample_rate=16000, f_min=0.0, f_max=None, n_stft=None, norm=None):
        super(MelScale, self).__init__()
        self.f_sp = 200.0/3.0
        self.min_log_hz = 1000.0
        self.logstep = math.log(6.4)/27.0
        self.min_log_mel = (self.min_log_hz - 0.0)/self.f_sp

        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate//2)
        self.f_min = f_min
        self.norm = norm

        assert f_min <= self.f_max, 'Require f_min: {} < f_max: {}'.format(f_min, self.f_max)
        
        fb = torch.empty(0) if n_stft is None else self.create_fb_matrix(
            n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate, self.norm)
        self.register_buffer('fb', fb)

    def mel2hertz(self, mel):
        mel = to_tensor(mel)
        hz = 0.0 + self.f_sp*mel

        return torch.where(
            mel >= self.min_log_mel, 
            self.min_log_hz*torch.exp(self.logstep*(mel - self.min_log_mel)), hz)

    def hertz2mel(self, hz):
        hz = to_tensor(hz)
        mel = (hz - 0.0)/self.f_sp

        return torch.where(
            hz >= self.min_log_hz, 
            self.min_log_mel + torch.log(hz/self.min_log_hz)/self.logstep, mel)

    def create_fb_matrix(self, n_freqs, f_min, f_max, n_mels, sample_rate, norm=None):
        if norm is not None and norm != "slaney":
            raise ValueError("norm must be one of None or 'slaney'")

        all_freqs = torch.linspace(0, sample_rate//2, n_freqs)

        m_min = self.hertz2mel(f_min)
        m_max = self.hertz2mel(f_max)
        m_pts = torch.linspace(m_min, m_max, n_mels + 2)

        f_pts = self.mel2hertz(m_pts)
        f_diff = f_pts[1:] - f_pts[:-1]

        slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)

        down_slopes = (-1.0*slopes[:, :-2])/f_diff[:-1]
        up_slopes = slopes[:, 2:]/f_diff[1:]
        fb = torch.clamp(torch.min(down_slopes, up_slopes), min=0.0)

        if norm is not None and norm == "slaney":
            # Slaney-style mel is scaled to be approx constant energy per channel
            enorm = 2.0/(f_pts[2:n_mels + 2] - f_pts[:n_mels])
            fb *= enorm.unsqueeze(0)

        if (fb.max(dim=0).values == 0.).any():
            warnings.warn(
                "At least one mel filterbank has all zero values. "
                f"The value for `n_mels` ({n_mels}) may be set too high. "
                f"Or, the value for `n_freqs` ({n_freqs}) may be set too low.")

        return fb

    def forward(self, specgram):
        shape = specgram.size()
        specgram = specgram.reshape(-1, shape[-2], shape[-1])

        if self.fb.numel() == 0:
            tmp_fb = self.create_fb_matrix(
                specgram.size(1), self.f_min, self.f_max, self.n_mels, self.sample_rate, self.norm)
            self.fb.resize_(tmp_fb.size())
            self.fb.copy_(tmp_fb)

        mel_specgram = torch.matmul(specgram.transpose(1, 2), self.fb).transpose(1, 2)
        mel_specgram = mel_specgram.reshape(shape[:-2] + mel_specgram.shape[-2:])

        return mel_specgram


class MelSpectrogram(nn.Module):
    def __init__(self, sample_rate, fft_size, hop_size, mel_size, f_min, f_max, min_level_db):
        super(MelSpectrogram, self).__init__()
        self.spec_fn = torchaudio.transforms.Spectrogram(
            n_fft=fft_size, 
            win_length=fft_size, 
            hop_length=hop_size,
            power=1)
        self.mel_scale_fn = MelScale(
            sample_rate=sample_rate, 
            n_mels=mel_size,
            f_min=f_min, 
            f_max=f_max,
            n_stft=fft_size//2 + 1,
            norm="slaney")
        self.norm_fn = dBNorm(min_level_db)

    def forward(self, audio):
        self.spec_fn.to(audio.device)
        self.mel_scale_fn.to(audio.device)
        
        spec = self.spec_fn(audio)
        mel = self.mel_scale_fn(spec)
        mel = self.norm_fn(mel)

        return mel