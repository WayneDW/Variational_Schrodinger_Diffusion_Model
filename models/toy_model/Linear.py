import numpy as np
import torch
import torch.nn as nn
from models.utils import *
from sde import compute_vp_diffusion

def build_linear(zero_out_last_layer):
    return LinearPolicy(zero_out_last_layer=zero_out_last_layer)

def create_orthogonal_layer(data_dim):
    linear_layer = nn.Linear(data_dim, data_dim, bias=False)
    linear_layer.weight.data = torch.eye(data_dim)
    return nn.utils.parametrizations.orthogonal(linear_layer) 


class LinearPolicy(torch.nn.Module):
    def __init__(self, data_dim=2, hidden_dim=256, time_embed_dim=128, zero_out_last_layer=False):
        super(LinearPolicy,self).__init__()

        self.Lambda = nn.Parameter(torch.zeros(data_dim))
        self.U = create_orthogonal_layer(data_dim)
        self.V = create_orthogonal_layer(data_dim)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype


    @property
    def A(self):
        Lambda_mat = torch.diag(self.Lambda)
        return self.U.weight @ Lambda_mat @ self.V.weight.T


    def forward(self, x, t, beta_min=0.1, beta_max=10., beta_r=1., interval=100.):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        """
        # make sure t.shape = [T]
        if len(t.shape)==0:
            t=t[None]

        out = torch.einsum('ij,bj->bi', self.A, x)
        out = compute_vp_diffusion(t, b_min=beta_min, b_max=beta_max, b_r=beta_r, T=interval).unsqueeze(dim=1) * out

        return out
