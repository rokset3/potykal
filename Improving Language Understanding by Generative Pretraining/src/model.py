
import os
import random
import math


import torch
import torch.nn as nn
import torch.nn.functional as F


from einops import rearrange, reduce, repeat
from torch import einsum 

import numpy as np



class GPT(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_layers,
                 num_heads,
                 hidden_dim,
                 ffc_hidden_dim,
                 attn_dropout_p=0.1,
                 ffc_dropout_p=0.1,
                 max_seq_len=512,
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.ffc_hidden_dim = ffc_hidden_dim
        self.attn_dropout_p = attn_dropout_p
        self.ffc_dropout_p = ffc_dropout_p
        self.max_seq_len = max_seq_len

        self.decoder_block = nn.ModuleList([DecoderLayer(self.num_heads,
                                                         self.hidden_dim,
                                                         self.ffc_hidden_dim,
                                                         self.attn_dropout_p,
                                                         self.ffc_dropout_p) for _ in range(self.num_layers)])
        
        self.pos_embeddings = nn.Embedding(self.max_seq_len, self.hidden_dim)
        self.token_embeddings = nn.Embedding(self.vocab_size, self.hidden_dim)

        self.proj_layer = nn.Linear(self.hidden_dim, self.vocab_size)
        
        self.register_buffer('tril',
                             torch.tril(torch.ones(self.max_seq_len, self.max_seq_len)).bool())
        self.register_buffer('pos_ids',
                             torch.arange(self.max_seq_len))
    


    def forward(self,
                input_tokens,
                tokenizer_mask=None):
        seq_len = input_tokens.shape[-1]
        b_size = input_tokens.shape[0]
        
        mask = self.make_attn_mask(seq_len, b_size, tokenizer_mask)
        
        x = self.pos_embeddings(self.pos_ids[:seq_len]) + self.token_embeddings(input_tokens)

        for layer in self.decoder_block:
            x = layer(x)
        x = self.proj_layer(x)
        return x


    def make_attn_mask(self, seq_len, b_size, tokenizer_mask=None):
        mask = self.tril[:seq_len, :seq_len].unsqueeze(0).repeat(b_size, 1, 1)
        
        if tokenizer_mask is not None:
            mask = mask & tokenizer_mask.bool().unsqueeze(1)
        return mask

class MSALayer(nn.Module):
    def __init__(self,
                 num_heads,
                 hidden_dim,
                 attn_dropout_p=0.1
                 ):
        
        assert hidden_dim % num_heads == 0
        
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = self.hidden_dim // self.num_heads
        self.attn_dropout_p = attn_dropout_p

        self.toq = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.tok = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.tov = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ffc = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.layer_norm = nn.LayerNorm(normalized_shape=self.hidden_dim)
        self.attn_dropout = nn.Dropout(p=self.attn_dropout_p if self.attn_dropout_p else 0)

    def forward(self,
                x,
                mask=None):
        # shape of input is [b_size, seq_len, hidden_dim]
        q = self.toq(x)
        k = self.tok(x)
        v = self.tov(x)
        
        q = rearrange(q, 'b s (num_heads h) -> (b num_heads) s h', num_heads=self.num_heads)
        k = rearrange(k, 'b s (num_heads h) -> (b num_heads) s h', num_heads=self.num_heads)
        v = rearrange(v, 'b s (num_heads h) -> (b num_heads) s h', num_heads=self.num_heads)

        output, probs = attn_function(q, k, v, mask=mask, attn_dropout=self.attn_dropout)

        output = rearrange(output, '(b num_heads) s h -> b s (num_heads h)', num_heads=self.num_heads)
        output = self.ffc(output)

        output = self.layer_norm(output + x)
        return output, probs
        
        
class DecoderLayer(nn.Module):
    def __init__(self, 
                 num_heads,
                 hidden_dim,
                 ffc_hidden_dim,
                 attn_dropout_p=0.1,
                 ffc_dropout_p=0.1,
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.ffc_hidden_dim = ffc_hidden_dim
        self.attn_dropout_p = attn_dropout_p
        self.ffc_dropout_p = ffc_dropout_p

        self.ffc_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ffc_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=self.ffc_dropout_p),
            nn.Linear(self.ffc_hidden_dim, self.hidden_dim)
        ) 
        self.ffc_layer_norm = nn.LayerNorm(normalized_shape=self.hidden_dim)

        self.msalayer = MSALayer(self.num_heads,
                                 self.hidden_dim,
                                 self.attn_dropout_p,)

    def forward(self,
                x,
                mask=None):
        res = x
        x, _ = self.msalayer(x, mask=mask)
        
        return self.ffc_layer_norm(self.ffc_layer(x) + res)
    

def attn_function(q, k, v, mask=None, attn_dropout=None):
    
    #q, k, v shape is [b, s, h]
    b_size = q.shape[0]
    seq_len = q.shape[1]
    hidden_dim = q.shape[2]


    scaled_dot_product = einsum('bsh, bvh -> bsv', [q, k])/math.sqrt(hidden_dim)

    if mask:
        scaled_dot_product = scaled_dot_product.masked_fill(mask==False, 1e-9)
    
    if attn_dropout:
        scaled_dot_product = attn_dropout(scaled_dot_product)
    
    attn_probs = F.softmax(scaled_dot_product, dim=-1)
    attn_output = einsum('bsv, bvd -> bsd', [attn_probs, v])

    return attn_output, attn_probs
