
___all__ = ['PatchTST']

from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import numpy as np

class PatchTST(nn.Module):
    def __init__(self, c_in: int, target_dim: int, patch_len: int, stride: int, num_patch: int, 
                 n_layers: int = 3, d_model: int = 128, n_heads: int = 8, shared_embedding: bool = True, 
                 d_ff: int = 256, norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0., 
                 act: str = "gelu", res_attention: bool = True, pre_norm: bool = False, 
                 store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True, 
                 head_dropout=0, head_type="prediction", individual=False, 
                 y_range: Optional[tuple] = None, verbose: bool = False, **kwargs):

        super().__init__()

        assert head_type in ['pretrain', 'prediction', 'regression', 'classification'], 'head type should be either pretrain, prediction, regression, or classification'

        # Backbone
        self.backbone = PatchTSTEncoder(c_in, num_patch=num_patch, patch_len=patch_len, 
                                        n_layers=n_layers, d_model=d_model, n_heads=n_heads, 
                                        shared_embedding=shared_embedding, d_ff=d_ff,
                                        attn_dropout=attn_dropout, dropout=dropout, act=act, 
                                        res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                        pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.n_vars = c_in
        self.head_type = head_type

        if head_type == "pretrain":
            self.head = PretrainHead(d_model, patch_len, head_dropout)
        elif head_type == "prediction":
            self.head = PredictionHead(individual, self.n_vars, d_model, num_patch, target_dim, head_dropout)
        elif head_type == "regression":
            self.head = RegressionHead(self.n_vars, d_model, target_dim, head_dropout, y_range)
        elif head_type == "classification":
            self.head = ClassificationHead(self.n_vars, d_model, target_dim, head_dropout)

    def forward(self, z):                             
        z = self.backbone(z)
        z = self.head(z)                                                                    
        return z


class PatchTSTEncoder(nn.Module):
    def __init__(self, c_in, num_patch, patch_len, 
                 n_layers=3, d_model=128, n_heads=8, shared_embedding=True,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", 
                 store_attn=False, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding        

        # Input encoding
        if not shared_embedding: 
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars): self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model)      

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, 
                                  dropout=dropout, pre_norm=pre_norm, activation=act, 
                                  res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

    def forward(self, x) -> Tensor:          
        bs, num
