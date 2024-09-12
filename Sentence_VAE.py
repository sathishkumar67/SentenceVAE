from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings import PositionalEncoding


@dataclass
class SentenceEncoderConfig:
    hidden_size: int = 128
    vocab_size: int = 64
    device: str = 'cpu'
    dtype: torch.dtype = torch.float32
    max_seq_len: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 2
    pad_id: int = 0
    dropout: float = 0.0


class SentenceEncoder(nn.Module):
    def __init__(self,
                hidden_size: int,
                vocab_size: int,
                device: str,
                dtype: torch.dtype,
                max_seq_len: int,
                num_hidden_layers: int,
                num_attention_heads: int,
                pad_id: int,
                dropout: float = 0.0,
                load_embedding_weights: bool = False,
                embedding_weights_path: Optional[str] = None) -> None: 
        
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_id)
        if load_embedding_weights and embedding_weights_path is not None:
            embedding_weights = torch.load(embedding_weights_path)
            self.embedding.load_state_dict(embedding_weights)

        self.positional_encoding = PositionalEncoding(hidden_size, max_seq_len, dtype=dtype, device=device)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_hidden_layers
        )

        self.layer_norm = nn.LayerNorm(hidden_size, device=device, dtype=dtype)


    def forward(self, input_ids, attention_mask):
        _, seq_len = input_ids.shape
        attention_mask = ~attention_mask.to(torch.bool)

        input_embeddings = self.embedding(input_ids)
        positional_embeddings = self.positional_encoding(seq_len)
        embeddings = input_embeddings + positional_embeddings

        hidden_states = self.encoder(embeddings, src_key_padding_mask=attention_mask)
        hidden_states[attention_mask] = 0

        # need to add sum

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states
    
# # pass the config to the model
# config = SentenceEncoderConfig()

# # create the model
# model = SentenceEncoder(
#     hidden_size=config.hidden_size,
#     vocab_size=config.vocab_size,
#     device=config.device,
#     dtype=config.dtype,
#     max_seq_len=config.max_seq_len,
#     num_hidden_layers=config.num_hidden_layers,
#     num_attention_heads=config.num_attention_heads,
#     pad_id=config.pad_id,
#     dropout=config.dropout
# )

# # pass a sample input to the model

# input_ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
# attention_mask = input_ids != config.pad_id
# hidden_states = model(input_ids, attention_mask)
# print(hidden_states)
