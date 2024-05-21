import os
import torch
from torchdrug import transforms, utils
from torchdrug.layers.geometry import GraphConstruction
from torchdrug.layers import geometry
from torchdrug.models import ProteinCNN, GearNet
from torchdrug.metrics import f1_max
from clip_engine import ClipEngine
from clip import CLIP
from alphafold import AlphaFoldDB
import gc


# to Utlize the GPU Float Computing
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark=True

# Set Random Seed
rank = utils.comm.get_rank()
torch.manual_seed(1024 + rank)


# Set Experimental Parameters
epoch = 30
steps = 5
batch_size = 128
gpu=[0, 1, 2, 3]


# Set the File Path to Save Results and Weight Files
task_name = "pretraining"
save_folder_name = "save_folder_name"
ckpt_path = os.path.join(os.getcwd(), "ckpt", task_name, save_folder_name)
if not os.path.exists(ckpt_path):
    os.mkdir(ckpt_path)


# Here are the two separate models to learn protein embeddings
seq_model = ProteinCNN(input_dim=21, hidden_dims=[1024, 1024], kernel_size=5, padding=2)
graph_model = GearNet(input_dim=21, hidden_dims=[512, 512, 512, 512, 512, 512], 
                      num_relation=7, edge_input_dim=59, num_angle_bin=8,  
                      batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")

structure_crop = geometry.SubspaceNode(entity_level="residue", min_radius=15.0, min_neighbor=15)
attribute_crop = geometry.RandomEdgeMask()

graph_construction_model = GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                             edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                             geometry.KNNEdge(k=10, min_distance=5),
                                             geometry.SequentialEdge(max_distance=2)],
                                             edge_feature="gearnet")

task = CLIP(embed_dim=3072, graph_construct=graph_construction_model, 
                  seq_model=seq_model, graph_model=graph_model, 
                  graph_struct_crop=structure_crop, graph_attr_crop=attribute_crop)

optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
transform = transforms.ProteinView(view="residue")

solver = ClipEngine(task=task, train_set=None, valid_set=None, test_set=None, optimizer=optimizer,
                    batch_size=192, gpus=[0, 1], half_precision=True)




if __name__ == '__main__':
    for index in range(0, 30):
        print("Current Dataset ID:{}".format(index))
        dataset = AlphaFoldDB(split_id=index, transform=transform)
        solver.train_set = dataset
        solver.train(steps)
        ckpt_name = "model_epcoh_{}.pth".format(index)
        save_path = os.path.join(ckpt_path, ckpt_name)
        solver.save(save_path)
        solver.train_set = None
        torch.cuda.empty_cache()
        del dataset
        print(gc.collect())
        dataset = None
