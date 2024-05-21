import torch
import torch.nn as nn
from typing import Optional, List
from torchdrug.models import ProteinCNN, ESM, GearNet
from torchdrug.layers import MLP
import numpy as np
import torch.nn.functional as F


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # seq
                 seq_model: nn.Module,
                 # graph
                 graph_model: nn.Module,
                 graph_struct_crop: nn.Module,
                 graph_attr_crop: nn.Module
                 ):
        super().__init__()

        self.seq_model = seq_model
        self.graph_model = graph_model

        self.proj_seq = nn.Parameter(torch.rand(self.seq_model.output_dim, embed_dim))
        self.proj_graph = nn.Parameter(torch.rand(self.graph_model.output_dim, embed_dim))

        self.graph_struct_crop = graph_struct_crop
        self.graph_attr_crop = graph_attr_crop

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_seq(self, batch):
        seq_repr = self.seq_model(batch)
        return seq_repr

    def encode_graph(self, batch):
        arg_graph_x = self.graph_struct_crop(batch)
        arg_graph_x = self.graph_attr_crop(arg_graph_x)
        arg_graph_x_repr = self.graph_model(arg_graph_x)

        arg_graph_y = self.graph_struct_crop(batch)
        arg_graph_y = self.graph_attr_crop(arg_graph_y)
        arg_graph_y_repr = self.graph_model(arg_graph_y)
        return arg_graph_x_repr, arg_graph_y_repr

    def forward(self, batch):
        seq_repr = self.encode_seq(batch)
        arg_graph_x_repr, arg_graph_y_repr = self.encode_graph(batch)
        # self-supervison loss for the structure view
        loss_contrast = self.contrast.forward(arg_graph_x_repr, arg_graph_y_repr)

        proj_seq_repr = seq_repr @ self.proj_seq
        proj_arg_graph_x_repr = arg_graph_x_repr @ self.proj_graph
        proj_arg_graph_y_repr = arg_graph_y_repr @ self.proj_graph

        # normalized features
        proj_seq_repr = proj_seq_repr / proj_seq_repr.norm(dim=1, keepdim=True)
        proj_arg_graph_x_repr = proj_arg_graph_x_repr / proj_arg_graph_x_repr.norm(dim=1, keepdim=True)
        proj_arg_graph_y_repr = proj_arg_graph_y_repr / proj_arg_graph_y_repr.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        label = torch.arange(seq_repr.shape[0], device=seq_repr.device)
        logits_per_graph_x_to_seq = logit_scale * proj_arg_graph_x_repr @ proj_seq_repr.t()
        logits_per_seq_to_graph_x = logits_per_graph_x_to_seq.t()

        loss_modal_graph_x_to_seq = F.cross_entropy(logits_per_graph_x_to_seq, label)
        loss_modal_seq_to_graph_x = F.cross_entropy(logits_per_seq_to_graph_x, label)

        logits_per_graph_y_to_seq = logit_scale * proj_arg_graph_y_repr @ proj_seq_repr.t()
        logits_per_seq_to_graph_y = logits_per_graph_y_to_seq.t()

        loss_modal_graph_y_to_seq = F.cross_entropy(logits_per_graph_y_to_seq, label)
        loss_modal_seq_to_graph_y = F.cross_entropy(logits_per_seq_to_graph_y, label)

        # multi-view supervision loss for the sequence-structure view
        loss_clip = 1 / 4 * (loss_modal_graph_x_to_seq + loss_modal_seq_to_graph_x
                             + loss_modal_graph_y_to_seq + loss_modal_seq_to_graph_y)

        total_loss = 0.5 * loss_clip + 0.5 * loss_contrast
