import os
import glob
from torchdrug import data
from dataset import ProteinDataset


class AlphaFoldDB(ProteinDataset):
    
    def __init__(self, split_id=0, verbose=1, **kwargs):
        alphafold_pdb_fold = "alphafold_pdb_fold_save_path"
        self.processed_file = "swissprot_{}.pkl.gz".format(split_id)
        pkl_file = os.path.join(alphafold_pdb_fold, self.processed_file)
        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            pdb_files = sorted(glob.glob(os.path.join(alphafold_pdb_fold, "*.pdb")))
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)

    def get_item(self, index):
        if getattr(self, "lazy", False):
            protein = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
        else:
            protein = self.data[index].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {"graph": protein}
        if self.transform:
            item = self.transform(item)
        return item

    def __repr__(self):
        lines = ["#sample: %d" % len(self)]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))


