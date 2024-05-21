import os
import torch
from engine import Engine
from property_prediction import PropertyPredictionModel
import torch.nn as nn
from torchdrug import transforms, utils
from torchdrug.layers.geometry import GraphConstruction
from enzyme_commission import EnzymeCommission
from torchdrug.layers import geometry
from torchdrug.models import ProteinCNN, GearNet

import util
import logging

# Utlize the GPU Float Computing
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark=True

# Set Random Seed
rank = utils.comm.get_rank()
torch.manual_seed(1024 + rank)


# Set Experimental Parameters
epoch = 50
batch_size = 8
gpu=[0, 1, 2, 3]


# Set the File Path to Save Results and Weight Files
task_name = "task_name"
save_folder_name = "save_folder_name"
ckpt_path = os.path.join(os.getcwd(), "ckpt", task_name, save_folder_name)
if not os.path.exists(ckpt_path):
    os.mkdir(ckpt_path)


# Set Log Formatter
format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")
log_path = os.path.join(ckpt_path, "log.txt")
handler = logging.FileHandler(log_path)
handler.setFormatter(format)
logger = logging.getLogger("")
logger.addHandler(handler)





if __name__ == '__main__':
    # you can change the hyper-paramters to observe the model performance
    model = GearNet(input_dim=21, hidden_dims=[512, 512, 512, 512, 512, 512], 
                    num_relation=7, edge_input_dim=59, num_angle_bin=8, 
                    batch_norm=True, concat_hidden=True,
                    short_cut=True, readout="sum")


    pretrained_ckpt_path = "file_path_for_pretrained_structure_encoder"
    model.load_state_dict(torch.load(pretrained_ckpt_path, map_location="cpu"))
    logger.info("mc_gearnet_edge ckpt file load successfully!")
 
    transform = transforms.ProteinView(view="residue")
   
    dataset = EnzymeCommission(path="dataset_file_path", transform=transform)
    train_set, valid_set, test_set = dataset.split()

    graph_construction_model = GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                 edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                              geometry.KNNEdge(k=10, min_distance=5),
                                                              geometry.SequentialEdge(max_distance=2)],
                                                 edge_feature="gearnet")
    
    property_prediction_model = PropertyPredictionModel(model, len(dataset.tasks), 
                                                        graph_construction_model, num_mlp_layer=3)

    property_prediction_model = property_prediction_model.cuda()
    optimizer = torch.optim.AdamW(property_prediction_model.parameters(), lr=1e-4, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, patience=5, verbose=True)

    # construct the task
    task = Engine(task=property_prediction_model, train_set=train_set, valid_set=valid_set, 
                  test_set=test_set, optimizer=optimizer, scheduler=None, batch_size=batch_size, 
                  criterion="bce", gpus=gpu, log_interval=1000, half_precision=True)
    
    best_metric = float("-inf")
    for i in range(epoch):
        task.train(1)
        metric = task.evaluate("test")
        metric = metric["f1_max"]
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        if metric > best_metric:
            best_metric = metric
            best_epoch = task.epoch
            ckpt_name = "model_epoch_%d.pth" % task.epoch
            save_path = os.path.join(ckpt_path, ckpt_name)
            if rank == 0:
                print(save_path)
            task.save(save_path)

    print(best_metric.item())
