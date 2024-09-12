import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        max_len: int = 4096,
        device: torch.device = None,
        dtype: torch.dtype = None
    ):
        super(PositionalEncoding, self).__init__()
        assert hidden_size % 2 == 0, \
            f"Cannot use sin/cos positional encoding with odd hidden_size (go size={hidden_size})."
        
        device = device if device is not None else torch.device('cuda')
        dtype = dtype if dtype is not None else torch.float16

        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size)))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False

        self.pe = self.pe.to(dtype)
        self.pe = self.pe.to(device)
    

    def forward(self, seq_len):
        return self.pe[:, :seq_len, :]