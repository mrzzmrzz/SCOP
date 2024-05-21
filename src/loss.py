import torch
import torch.nn as nn
from torch.nn import functional as F
from torchdrug.layers import MLP

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False


def gather_features(
        seq_features,
        graph_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,

):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'

    # We gather tensors from all gpus
    if gather_with_grad:
        all_seq_features = torch.cat(torch.distributed.nn.all_gather(seq_features), dim=0)
        all_graph_features = torch.cat(torch.distributed.nn.all_gather(graph_features), dim=0)
    else:
        gathered_seq_features = [torch.zeros_like(seq_features) for _ in range(world_size)]
        gathered_graph_features = [torch.zeros_like(graph_features) for _ in range(world_size)]
        dist.all_gather(gathered_seq_features, seq_features)
        dist.all_gather(gathered_graph_features, graph_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_seq_features[rank] = seq_features
            gathered_graph_features[rank] = graph_features
        all_seq_features = torch.cat(gathered_seq_features, dim=0)
        all_graph_features = torch.cat(gathered_graph_features, dim=0)

    return all_seq_features, all_graph_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=True,
            gather_with_grad=True,
            cache_labels=False,
            rank=0,
            world_size=1
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, seq_features, graph_features, logit_scale):
        device = seq_features.device
        if self.world_size > 1:
            all_seq_features, all_graph_features = gather_features(
                seq_features, graph_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size)

            if self.local_loss:
                logits_per_seq = logit_scale * seq_features @ all_graph_features.T
                logits_per_graph = logit_scale * graph_features @ all_seq_features.T
            else:
                logits_per_seq = logit_scale * all_seq_features @ all_graph_features.T
                logits_per_graph = logits_per_seq.T
        else:
            logits_per_seq = logit_scale * seq_features @ graph_features.T
            logits_per_graph = logit_scale * graph_features @ seq_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_seq.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (F.cross_entropy(logits_per_seq, labels) +
                      F.cross_entropy(logits_per_graph, labels)) / 2
        return total_loss


class ContrastLoss(nn.Module):

    def __init__(self, output_dim, num_mlp_layer=2, activation="relu", tau=0.07,
                 local_loss=True, gather_with_grad=True, rank=0, world_size=1):
        super(ContrastLoss, self).__init__()

        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size

        # projector layer
        # project the representation into one common space
        self.mlp = MLP(output_dim, [output_dim] * num_mlp_layer, activation=activation)

        # temperature parameter
        self.tau = tau

    def forward(self, batch_x, batch_y):
        def contrast_loss(i, j):
            similarity_matrix = F.cosine_similarity(i.unsqueeze(1), j.unsqueeze(0), dim=-1)
            similarity_matrix = similarity_matrix / self.tau
            is_positive = torch.eye(similarity_matrix.shape[0], similarity_matrix.shape[1], dtype=torch.bool)
            mutual_info = (similarity_matrix[is_positive] - similarity_matrix.logsumexp(dim=-1)).mean()
            return -mutual_info

        assert batch_x.shape[1] == batch_y.shape[1], \
            "input features dimension should be in the same shape"

        proj_batch_x = self.mlp(batch_x)
        proj_batch_y = self.mlp(batch_y)

        if self.world_size > 1:
            all_batch_x, all_batch_y = gather_features(
                batch_x, batch_y, self.local_loss, self.gather_with_grad, self.rank, self.world_size)
            proj_all_batch_x = self.mlp(all_batch_x)
            proj_all_batch_y = self.mlp(all_batch_y)
            loss_x_to_y = contrast_loss(proj_batch_x, proj_all_batch_y)
            loss_y_to_x = contrast_loss(proj_batch_y, proj_all_batch_x)
        else:
            loss_x_to_y = contrast_loss(proj_batch_x, proj_batch_y)
            loss_y_to_x = contrast_loss(proj_batch_y, proj_batch_x)

        consistency_loss = 1 / 2 * (loss_x_to_y + loss_y_to_x)

        return consistency_loss

