import torch
import torch.nn as nn
from torchdrug.metrics import f1_max
from torchdrug.layers import MLP


class PropertyPredictionModel(nn.Module):
    def __init__(self, model, cls_num, graph_construct, num_mlp_layer):
        super(PropertyPredictionModel, self).__init__()
        self.graph_construct = graph_construct
        self.model = model
        hidden_dims = [self.model.output_dim] * (num_mlp_layer - 1)
        self.mlp = MLP(self.model.output_dim, hidden_dims + [cls_num])

    def forward(self, batch):
        graph = self.graph_construct(batch["graph"])
        node_feature = graph.node_feature.type(torch.float)
        output = self.model(graph, node_feature)
        pred = self.mlp(output["graph_feature"])
        return pred

    def target(self, batch):
        return batch["targets"]

    def evaluate(self, pred, target):
        metric = {}
        metric["f1_max"] = f1_max(pred, target)
        return metric