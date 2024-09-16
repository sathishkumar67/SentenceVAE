from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings import PositionalEncoding
import os


@dataclass
class SentenceDecoderConfig:
    word_embed_proj_dim: Optional[int] = 64
    hidden_size: int = 512
    vocab_size: int = 64
    max_seq_len: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 2
    pad_id: int = 0
    dropout: float = 0.0
    load_embedding_weights: bool = False
    embedding_weights_path: Optional[str] = None
    do_finetune: bool = False
    learnable_add: bool = False
    seed: int = 42

torch.manual_seed(42)


class SentenceEncoder(nn.Module):
    def __init__(self,
                word_embed_proj_dim: int,
                hidden_size: int,
                vocab_size: int,
                max_seq_len: int,
                num_hidden_layers: int,
                num_attention_heads: int,
                pad_id: int,
                dropout: float = 0.0,
                load_embedding_weights: bool = False,
                embedding_weights_path: Optional[str] = None,
                do_finetune: bool = False) -> None: 
        
        super().__init__()

        self.pad_id = pad_id
        word_embed_proj_dim = word_embed_proj_dim if word_embed_proj_dim is not None else hidden_size

        self.embedding = nn.Embedding(vocab_size, word_embed_proj_dim, padding_idx=pad_id)
        self.positional_encoding = PositionalEncoding(hidden_size, max_seq_len)

        if word_embed_proj_dim != hidden_size:
            self.projection = nn.Linear(word_embed_proj_dim, hidden_size, bias=False)
        else:
            self.projection = None

        if load_embedding_weights and embedding_weights_path is not None:
            if os.path.exists(embedding_weights_path):
                embedding_weights = torch.load(embedding_weights_path)
                self.embedding.load_state_dict(embedding_weights)
            else:
                raise FileNotFoundError(f"Embedding weights not found at {embedding_weights_path}")
            
        if do_finetune:
            self.embedding.weight.requires_grad = True
        else:
            self.embedding.weight.requires_grad = False

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dropout=dropout,
            dim_feedforward=hidden_size * 2,
            batch_first=True)

        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=num_hidden_layers
        )

        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, sentence_embedding: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        attention_mask = (input_ids != self.pad_id)
        attention_mask = ~attention_mask.to(torch.bool)

        input_embeddings= self.embedding(input_ids)
        positional_encoding = self.positional_encoding(seq_len)

        if self.projection is not None:
            input_embeddings = self.projection(input_embeddings)

        embeddings = input_embeddings + positional_encoding

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len) == -torch.inf

        hidden_states = self.decoder(
            embeddings,
            sentence_embedding,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=attention_mask,
            tgt_is_causal=True
        )

        output = self.linear(hidden_states)

        return output