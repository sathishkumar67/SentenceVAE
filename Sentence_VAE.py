from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings import PositionalEncoding
import os


@dataclass
class SentenceEncoderConfig:
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
                do_finetune: bool = False,
                learnable_add: bool = False) -> None: # need to implement learnable_add: bool = False 
        
        super().__init__()
        self.learnable_add = learnable_add
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
            self.embedding.requires_grad = True
        else:
            self.embedding.requires_grad = False

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

        if learnable_add:
            self.la = nn.Linear(hidden_size, 1)

        self.layer_norm = nn.LayerNorm(hidden_size)


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        attention_mask = (input_ids != self.pad_id)
        attention_mask = ~attention_mask.to(torch.bool)

        input_embeddings = self.embedding(input_ids)
        positional_embeddings = self.positional_encoding(seq_len)

        if self.projection is not None:
            input_embeddings = self.projection(input_embeddings)

        embeddings = input_embeddings + positional_embeddings

        hidden_states = self.encoder(embeddings, src_key_padding_mask=attention_mask)
        hidden_states[attention_mask] = 0
        if self.learnable_add:
            alpha = self.la(hidden_states)
            sentence_embedding = torch.sum(hidden_states * alpha, dim=-2, keepdim=True)
        else:
            sentence_embedding = torch.sum(hidden_states, dim=-2, keepdim=True)

        sentence_embedding = self.layer_norm(sentence_embedding)
        
        return sentence_embedding

config = SentenceEncoderConfig()

# create the model
model = SentenceEncoder(
    word_embed_proj_dim=config.word_embed_proj_dim,
    hidden_size=config.hidden_size,
    vocab_size=config.vocab_size,
    max_seq_len=config.max_seq_len,
    num_hidden_layers=config.num_hidden_layers,
    num_attention_heads=config.num_attention_heads,
    pad_id=config.pad_id,
    dropout=config.dropout,
    load_embedding_weights=config.load_embedding_weights,
    embedding_weights_path=config.embedding_weights_path,
    do_finetune=config.do_finetune,
    learnable_add=True
)