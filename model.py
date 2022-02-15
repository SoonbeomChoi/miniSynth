import torch
import torch.nn as nn

import config


def get_padding(padding, kernel_size, dilation):
    kernel_size_type = type(kernel_size)
    if kernel_size_type != list and kernel_size_type != tuple:
        kernel_size = [kernel_size]

    padding_list = []
    for k in kernel_size:
        if padding == "same":
            p = ((k - 1)*dilation + 1)//2
        elif padding == "causal":
            p = (k - 1)*dilation
        elif padding == "valid":
            p = 0
        else:
            raise ValueError("Padding should be 'valid', 'same' or 'causal'")

        padding_list.append(p)

    if kernel_size_type != list and kernel_size_type != tuple:
        padding = padding_list[0]
    else:
        padding = padding_list

    return padding


class Conv1d(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=1, stride=1, padding="same", dilation=1, bias=True):
        super(Conv1d, self).__init__()
        padding = get_padding(padding, kernel_size, dilation)
        self.layer = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        length = x.size(-1)

        return self.layer(x)[..., :length]


class HighwayConv1d(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, padding="same", dilation=1, dropout=0.0):
        super(HighwayConv1d, self).__init__()

        self.conv1d = Conv1d(input_size, 2*output_size, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h1, h2 = self.conv1d(x).chunk(2, dim=1)
        h1 = torch.sigmoid(h1)
        h2 = torch.relu(h2)

        h = h1*h2 + (1.0 - h1)*x
        h = self.dropout(h)

        return h


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.min_note = config.min_note
        self.num_note = config.num_note
        hidden_size = config.model_size

        self.embedding = nn.Embedding(config.num_note, hidden_size)
        self.encoder = nn.Sequential(
            Conv1d(hidden_size, hidden_size),
            nn.ReLU(),
            HighwayConv1d(hidden_size, hidden_size, dilation=1, dropout=config.dropout),
            HighwayConv1d(hidden_size, hidden_size, dilation=3, dropout=config.dropout),
            HighwayConv1d(hidden_size, hidden_size, dilation=9, dropout=config.dropout),
            HighwayConv1d(hidden_size, hidden_size, dilation=27, dropout=config.dropout))
        self.decoder = nn.Sequential(
            Conv1d(hidden_size, hidden_size),
            nn.ReLU(),
            HighwayConv1d(hidden_size, hidden_size, padding="causal", dilation=1, dropout=config.dropout),
            HighwayConv1d(hidden_size, hidden_size, padding="causal", dilation=3, dropout=config.dropout),
            HighwayConv1d(hidden_size, hidden_size, padding="causal", dilation=9, dropout=config.dropout),
            HighwayConv1d(hidden_size, hidden_size, padding="causal", dilation=27, dropout=config.dropout),
            HighwayConv1d(hidden_size, hidden_size, padding="causal", dilation=1, dropout=config.dropout),
            HighwayConv1d(hidden_size, hidden_size, padding="causal", dilation=1, dropout=config.dropout),
            Conv1d(hidden_size, hidden_size),
            nn.ReLU(),
            Conv1d(hidden_size, config.mel_size))

    def normalize(self, x):
        x[x > 0] = x[x > 0] - self.min_note
        
        return x.clamp(0, self.num_note - 1)
    
    def forward(self, x):
        x = self.normalize(x)

        y = self.embedding(x).transpose(1, 2)
        y = self.encoder(y)
        y = self.decoder(y)

        return y