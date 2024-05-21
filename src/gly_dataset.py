import os
import csv
import sys
import glob
import torch
import os.path as osp
from torch.utils import data as torch_data
from torchdrug import data, utils
from torchdrug import utils
from torchdrug.data import Protein


class GlyDataset(data.ProteinDataset):
    processed_file = "gly.pkl.gz"

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
       
        pkl_file = os.path.join(path, self.processed_file)
        csv_file = os.path.join(path, "gly_test.csv")


        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)

        else:
            pdb_files = []
            for split in ["train",  "test"]:
                split_path = utils.extract(os.path.join(path, "%s.zip" % split))
                pdb_files += sorted(glob.glob(os.path.join(split_path, split, "*.pdb")))
            
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)


        tsv_file = os.path.join(path, "gly_annot.tsv")
        pdb_ids = [os.path.basename(pdb_file).replace(".pdb","") for pdb_file in self.pdb_files]
        

        self.load_annotation(tsv_file, pdb_ids)

        splits = [os.path.basename(os.path.dirname(pdb_file)) for pdb_file in self.pdb_files]
        self.num_samples = [splits.count("train"), splits.count("test")]


    def load_annotation(self, tsv_file, pdb_ids):

        with open(tsv_file, "r") as fin:
            reader = csv.reader(fin, delimiter="\t")
            _ = next(reader)
            tasks = next(reader)
            task2id = {task: i for i, task in enumerate(tasks)}
            _ = next(reader)
            pos_targets = {}
            

            for pdb_id, pos_target in reader:
                pos_target = [task2id[t] for t in pos_target.split(",")]
                pos_target = torch.tensor(pos_target)
                pos_targets[pdb_id] = pos_target
                
        
        # fake targets to enable the property self.tasks
        self.targets = task2id
        self.pos_targets = []
        for pdb_id in pdb_ids:
            self.pos_targets.append(pos_targets[pdb_id])

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits

    def get_item(self, index):
        protein = self.data[index].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {"graph": protein}
        if self.transform:
            item = self.transform(item)
        item["targets"] = self.pos_targets[index]
        return item


    
   


