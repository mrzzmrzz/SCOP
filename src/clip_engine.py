import math
from engine import Engine
from loss import ClipLoss

import torch
from torch import distributed as dist
from torch.cuda import amp as amp
from torch import nn
from torch.utils import data as torch_data

from torchdrug import data, core, utils
from torchdrug.utils import comm, pretty
import torch.nn.functional as F


class ClipEngine(Engine):
    def __init__(self, task, train_set, valid_set, test_set, optimizer, scheduler=None, 
                 gpus=None, batch_size=1, num_worker=0, logger="logging", log_interval=100, 
                 half_precision=False, criterion=None, local_loss=True, gather_with_grad=True):

        super(ClipEngine, self).__init__(task, train_set, valid_set, test_set, optimizer, 
                                         scheduler, gpus, batch_size, num_worker, logger, 
                                         log_interval, half_precision)

        self.criterion = ClipLoss(local_loss=local_loss, gather_with_grad=gather_with_grad,
                                  rank=self.rank, world_size=self.world_size)
        self.criterion_name = "clip loss"

    def train(self, num_epoch=1):
        sampler = torch_data.DistributedSampler(self.train_set, self.world_size, self.rank)
        dataloader = data.DataLoader(self.train_set, self.batch_size, 
                                     sampler=sampler, num_workers=self.num_worker)
        model = self.model

        if self.world_size > 1:
            if self.device.type == "cuda":
                model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device],
                                                            find_unused_parameters=True)
            else:
                model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

        model.train()

        for epoch in self.meter(num_epoch):
            sampler.set_epoch(epoch)
            losses = []

            for i, batch in enumerate(dataloader):
                if self.device.type == "cuda":
                    batch = utils.cuda(batch, device=self.device)

                with torch.cuda.amp.autocast(enabled=self.half_precision):
                    # TODO: Be Sure to Use `model` (Not `Self.Model`)
                    seq_repr, arg_graph_repr, logit_scale = model(batch)
                    loss = self.criterion(seq_repr, arg_graph_repr, logit_scale)

                if self.half_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                else:
                    loss.backward()
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Note: we clamp to 4.6052 = ln(100), as in the original paper
                with torch.no_grad():
                    if hasattr(model, "module"):
                        model.module.logit_scale.clamp_(0, math.log(100))
                    else:
                        model.logit_scale.clamp_(0, math.log(100))

                # Here We Can Add Self-Designed Loss to Update the Model
                # Key: Loss Name (Support Multiple Loss Record)
                cur_loss = {self.criterion_name: loss}
                losses.append(cur_loss)

                cur_loss = utils.stack(losses, dim=0)
                cur_loss = utils.mean(cur_loss, dim=0)

                if self.world_size > 1:
                    cur_loss = comm.reduce(cur_loss, op="mean")
                self.meter.update(cur_loss)

                losses = []

            if self.scheduler:
                self.scheduler.step()


