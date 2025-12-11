__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from ..models.layers.pos_encoding import *
from ..models.layers.basics import *
from ..models.layers.attention import *

            
# Cell
class PatchTST(nn.Module):
    """
    Output dimension: 
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """
    def __init__(self, c_in:int, target_dim:int, patch_len:int, stride:int, num_patch:int, 
                 n_layers:int=3, d_model=128, n_heads=16, shared_embedding=True, d_ff:int=256, 
                 norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", 
                 res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, head_dropout = 0, 
                 head_type = "prediction", individual = False, 
                 y_range:Optional[tuple]=None, verbose:bool=False, **kwargs):

        super().__init__()

        assert head_type in ['pretrain', 'prediction', 'regression', 'classification'], 'head type should be either pretrain, prediction, or regression'
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
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        """   
        z = self.backbone(z)
        z = self.head(z)
        return z


class RegressionHead(nn.Module):
    def __init__(self, n_vars, d_model, output_dim, head_dropout, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, output_dim)

    def forward(self, x):
        x = x[:,:,:,-1]
        x = self.flatten(x)
        x = self.dropout(x)
        y = self.linear(x)
        if self.y_range: y = SigmoidRange(*self.y_range)(y)        
        return y


class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, n_classes)

    def forward(self, x):
        x = x[:,:,:,-1]
        x = self.flatten(x)
        x = self.dropout(x)
        y = self.linear(x)
        return y


class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model*num_patch

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)


    def forward(self, x):                     
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.dropout(x)
            x = self.linear(x)
        return x.transpose(2,1)


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        x = x.transpose(2,3)
        x = self.linear( self.dropout(x) )
        x = x.permute(0,2,1,3)
        return x


# ============== NEW: Cross-Channel Attention Classes ==============

class CrossChannelMultiheadAttention(nn.Module):
    """6 cross-channel + 2 same-channel heads for 2 channels"""
    def __init__(self, d_model, n_heads=8, n_cross_heads=6, attn_dropout=0., proj_dropout=0.):
        super().__init__()
        self.n_heads = n_heads
        self.n_cross_heads = n_cross_heads
        self.n_same_heads = n_heads - n_cross_heads
        self.d_k = d_model // n_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)
        
    def forward(self, x):
        """x: [bs*2, num_patch, d_model]"""
        bs2, num_patch, d_model = x.shape
        bs = bs2 // 2
        
        x = x.view(bs, 2, num_patch, d_model)
        
        Q = self.W_Q(x).view(bs, 2, num_patch, self.n_heads, self.d_k).permute(0,1,3,2,4)
        K = self.W_K(x).view(bs, 2, num_patch, self.n_heads, self.d_k).permute(0,1,3,2,4)
        V = self.W_V(x).view(bs, 2, num_patch, self.n_heads, self.d_k).permute(0,1,3,2,4)
        
        Q_cross, Q_same = Q[:,:,:self.n_cross_heads], Q[:,:,self.n_cross_heads:]
        K_cross, K_same = K[:,:,:self.n_cross_heads], K[:,:,self.n_cross_heads:]
        V_cross, V_same = V[:,:,:self.n_cross_heads], V[:,:,self.n_cross_heads:]
        
        K_swap = torch.stack([K_cross[:,1], K_cross[:,0]], dim=1)
        V_swap = torch.stack([V_cross[:,1], V_cross[:,0]], dim=1)
        
        attn_cross = self.attn_dropout(F.softmax(Q_cross @ K_swap.transpose(-2,-1) / (self.d_k**0.5), dim=-1))
        out_cross = attn_cross @ V_swap
        
        attn_same = self.attn_dropout(F.softmax(Q_same @ K_same.transpose(-2,-1) / (self.d_k**0.5), dim=-1))
        out_same = attn_same @ V_same
        
        out = torch.cat([out_cross, out_same], dim=2)
        out = out.permute(0,1,3,2,4).reshape(bs, 2, num_patch, d_model)
        out = self.proj_dropout(self.W_O(out))
        
        return out.view(bs2, num_patch, d_model)


class CrossChannelTSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads=8, n_cross_heads=6, d_ff=256,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation="gelu"):
        super().__init__()
        
        self.self_attn = CrossChannelMultiheadAttention(d_model, n_heads, n_cross_heads, attn_dropout, dropout)
        self.dropout_attn = nn.Dropout(dropout)
        
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout_ffn = nn.Dropout(dropout)
        
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)
        
    def forward(self, src):
        src2 = self.self_attn(src)
        src = src + self.dropout_attn(src2)
        src = self.norm_attn(src)
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        src = self.norm_ffn(src)
        return src


class CrossChannelTSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads=8, n_cross_heads=6, d_ff=256, 
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu', n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossChannelTSTEncoderLayer(d_model, n_heads, n_cross_heads, d_ff, norm, attn_dropout, dropout, activation)
            for _ in range(n_layers)
        ])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src

# ============== END: Cross-Channel Attention Classes ==============


class PatchTSTEncoder(nn.Module):
    def __init__(self, c_in, num_patch, patch_len, 
                 n_layers=3, d_model=128, n_heads=8, n_cross_heads=6, shared_embedding=True,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding        

        if not shared_embedding: 
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars): self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model)      

        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)
        self.dropout = nn.Dropout(dropout)

        # CHANGED: Use CrossChannelTSTEncoder instead of TSTEncoder
        self.encoder = CrossChannelTSTEncoder(d_model, n_heads, n_cross_heads, d_ff=d_ff, norm=norm, 
                                               attn_dropout=attn_dropout, dropout=dropout,
                                               activation=act, n_layers=n_layers)

    def forward(self, x) -> Tensor:          
        bs, num_patch, n_vars, patch_len = x.shape
        
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars): 
                z = self.W_P[i](x[:,:,i,:])
                x_out.append(z)
            x = torch.stack(x_out, dim=2)
        else:
            x = self.W_P(x)
        x = x.transpose(1,2)

        u = torch.reshape(x, (bs*n_vars, num_patch, self.d_model))
        u = self.dropout(u + self.W_pos)

        z = self.encoder(u)
        z = torch.reshape(z, (-1, n_vars, num_patch, self.d_model))
        z = z.permute(0,1,3,2)

        return z
