import os
import csv
import math
from multiprocessing import Pool

import lmdb
import pickle
import logging
import warnings
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from torchdrug.data import MoleculeDataset
from torchdrug import core, data, utils

logger = logging.getLogger(__name__)


class ProteinDataset(MoleculeDataset, core.Configurable):
    """
    Protein dataset.

    Each sample contains a protein graph, and any number of prediction targets.
    """

    @utils.copy_args(data.Protein.from_sequence)
    def load_sequence(self, sequences, targets, attributes=None, transform=None, lazy=False, verbose=0, **kwargs):
        """
        Load the dataset from protein sequences and targets.

        Parameters:
            sequences (list of str): protein sequence strings
            targets (dict of list): prediction targets
            attributes (dict of list): protein-level attributes
            transform (Callable, optional): protein sequence transformation function
            lazy (bool, optional): if lazy mode is used, the proteins are processed in the dataloader.
                This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
            verbose (int, optional): output verbose level
            **kwargs
        """
        num_sample = len(sequences)
        if num_sample > 1000000:
            warnings.warn("Preprocessing proteins of a large dataset consumes a lot of CPU memory and time. "
                          "Use load_sequence(lazy=True) to construct molecules in the dataloader instead.")
        for field, target_list in targets.items():
            if len(target_list) != num_sample:
                raise ValueError("Number of target `%s` doesn't match with number of molecules. "
                                 "Expect %d but found %d" % (field, num_sample, len(target_list)))
        self.transform = transform
        self.lazy = lazy
        self.kwargs = kwargs
        self.sequences = []
        self.data = []
        self.targets = defaultdict(list)

        if verbose:
            sequences = tqdm(sequences, "Constructing proteins from sequences")
        for i, sequence in enumerate(sequences):
            if not self.lazy or len(self.data) == 0:
                protein = data.Protein.from_sequence(sequence, **kwargs)
            else:
                protein = None
            if attributes is not None:
                with protein.graph():
                    for field in attributes:
                        setattr(protein, field, attributes[field][i])
            self.data.append(protein)
            self.sequences.append(sequence)
            for field in targets:
                self.targets[field].append(targets[field][i])

    @utils.copy_args(load_sequence)
    def load_lmdbs(
            self, lmdb_files, sequence_field="primary", target_fields=None,
            number_field="num_examples", transform=None, lazy=False, verbose=0, **kwargs
    ):
        """
        Load the dataset from lmdb files.

        Parameters:
            lmdb_files (list of str): list of lmdb files
            sequence_field (str, optional): name of the field of protein sequence in lmdb files
            target_fields (list of str, optional): name of target fields in lmdb files
            number_field (str, optional): name of the field of sample count in lmdb files
            transform (Callable, optional): protein sequence transformation function
            lazy (bool, optional): if lazy mode is used, the proteins are processed in the dataloader.
                This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
            verbose (int, optional): output verbose level
            **kwargs
        """
        if target_fields is not None:
            target_fields = set(target_fields)

        sequences = []
        num_samples = []
        targets = defaultdict(list)
        for lmdb_file in lmdb_files:
            env = lmdb.open(
                lmdb_file,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
                )
            with env.begin(write=False) as txn:
                num_sample = pickle.loads(txn.get(number_field.encode()))
                for i in range(num_sample):
                    item = pickle.loads(txn.get(str(i).encode()))
                    sequences.append(item[sequence_field])
                    if target_fields:
                        for field in target_fields:
                            value = item[field]
                            if isinstance(value, np.ndarray) and value.size == 1:
                                value = value.item()
                            targets[field].append(value)
                num_samples.append(num_sample)

        self.load_sequence(sequences, targets, attributes=None, transform=transform, lazy=lazy, verbose=verbose, **kwargs)
        self.num_samples = num_samples

    @utils.copy_args(data.Protein.from_molecule)
    def load_pdbs_single_process(self, pdb_files, transform=None, lazy=False, verbose=0, **kwargs):
        """
        Load the dataset from pdb files.

        Parameters:
            pdb_files (list of str): pdb file names
            transform (Callable, optional): protein sequence transformation function
            lazy (bool, optional): if lazy mode is used, the proteins are processed in the dataloader.
                This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
            verbose (int, optional): output verbose level
            **kwargs
        """

        self.transform = transform
        self.lazy = lazy
        self.kwargs = kwargs

        data_list = []
        pdb_files_list = []
        sequences_list = []

        if verbose:
            pdb_files = tqdm(pdb_files, "Constructing proteins from pdbs")
        for i, pdb_file in enumerate(pdb_files):
            if not lazy or i == 0:
                mol = Chem.MolFromPDBFile(pdb_file)
                if not mol:
                    logger.debug("Can't construct molecule from pdb file `%s`. Ignore this sample." % pdb_file)
                    continue
                protein = data.Protein.from_molecule(mol, **kwargs)
                if not protein:
                    logger.debug("Can't construct protein from pdb file `%s`. Ignore this sample." % pdb_file)
                    continue
            else:
                protein = None

            # if hasattr(protein, "residue_feature"):
            #     with protein.residue():
            #         protein.residue_feature = protein.residue_feature

            sequences_list.append(protein.to_sequence() if protein else None)
            data_list.append(protein)
            pdb_files_list.append(pdb_file)

        return data_list, pdb_files_list, sequences_list


    @utils.copy_args(data.Protein.from_molecule)
    def load_pdbs_new(self, pdb_files, transform=None, lazy=False, verbose=0, num_process=10, **kwargs):
        """
        Load the dataset from pdb files.

        Parameters:
            pdb_files (list of str): pdb file names
            transform (Callable, optional): protein sequence transformation function
            lazy (bool, optional): if lazy mode is used, the proteins are processed in the dataloader.
                This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
            verbose (int, optional): output verbose level
            num_process (int): the number of thread to transfer the pdb files
            **kwargs
        """
        num_sample = len(pdb_files)
        if num_sample > 1000000:
            warnings.warn("Preprocessing proteins of a large dataset consumes a lot of CPU memory and time. "
                          "Use load_pdbs(lazy=True) to construct molecules in the dataloader instead.")

        self.transform = transform
        self.lazy = lazy
        self.kwargs = kwargs
        self.data = []
        self.pdb_files = []
        self.sequences = []

        if num_process > 1:
            warnings.warn("Multiprocessing is applied to tranform the pdb files.")

        multi_process_pool = []
        samples_per_process = math.ceil(num_sample / num_process)
        pool = Pool(num_process)

        for i in range(num_process):
            start_index = i * samples_per_process
            if i != num_process - 1:
                end_index = (i + 1) * samples_per_process
            else:
                end_index = num_sample

            cur_pdb_files = pdb_files[start_index: end_index]
            multi_process_pool.append(pool.apply_async(self.load_pdbs_single_process,
                                                       args=(cur_pdb_files, transform, lazy, verbose)))

        pool.close()
        pool.join()

        for r in multi_process_pool:
            r_data, r_pdb_files, r_sequences = r.get()
            self.data.extend(r_data)
            self.pdb_files.extend(r_pdb_files)
            self.sequences.extend(r_sequences)


    @utils.copy_args(data.Protein.from_molecule)
    def load_pdbs(self, pdb_files, transform=None, lazy=False, verbose=0, **kwargs):
        """
        Load the dataset from pdb files.

        Parameters:
            pdb_files (list of str): pdb file names
            transform (Callable, optional): protein sequence transformation function
            lazy (bool, optional): if lazy mode is used, the proteins are processed in the dataloader.
                This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
            verbose (int, optional): output verbose level
            **kwargs
        """
        num_sample = len(pdb_files)
        if num_sample > 1000000:
            warnings.warn("Preprocessing proteins of a large dataset consumes a lot of CPU memory and time. "
                          "Use load_pdbs(lazy=True) to construct molecules in the dataloader instead.")

        self.transform = transform
        self.lazy = lazy
        self.kwargs = kwargs
        self.data = []
        self.pdb_files = []
        self.sequences = []

        if verbose:
            pdb_files = tqdm(pdb_files, "Constructing proteins from pdbs")
        for i, pdb_file in enumerate(pdb_files):
            if not lazy or i == 0:
                mol = Chem.MolFromPDBFile(pdb_file)
                if not mol:
                    logger.debug("Can't construct molecule from pdb file `%s`. Ignore this sample." % pdb_file)
                    continue
                protein = data.Protein.from_molecule(mol, **kwargs)
                if not protein:
                    logger.debug("Can't construct protein from pdb file `%s`. Ignore this sample." % pdb_file)
                    continue
            else:
                protein = None
            if hasattr(protein, "residue_feature"):
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()
            self.data.append(protein)
            self.pdb_files.append(pdb_file)
            self.sequences.append(protein.to_sequence() if protein else None)

    @utils.copy_args(load_sequence)
    def load_fasta(self, fasta_file, verbose=0, **kwargs):
        """
        Load the dataset from a fasta file.

        Parameters:
            fasta_file (str): file name
            verbose (int, optional): output verbose level
            **kwargs
        """
        with open(fasta_file, "r") as fin:
            if verbose:
                fin = tqdm(
                    fin,
                    "Loading %s" % fasta_file,
                    utils.get_line_count(fasta_file)
                    )
            sequences = []
            lines = []
            for line in fin:
                line = line.strip()
                if line.startswith(">") and lines:
                    sequence = "".join(lines)
                    sequences.append(sequence)
                    lines = []
                else:
                    lines.append(line)
            if lines:
                sequence = "".join(lines)
                sequences.append(sequence)

        return self.load_sequence(sequences, verbose=verbose, **kwargs)

    @utils.copy_args(data.Protein.from_molecule)
    def load_pickle(self, pkl_file, transform=None, lazy=False, verbose=0, **kwargs):
        """
        Load the dataset from a pickle file.

        Parameters:
            pkl_file (str): file name
            transform (Callable, optional): protein sequence transformation function
            lazy (bool, optional): if lazy mode is used, the proteins are processed in the dataloader.
                This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
            verbose (int, optional): output verbose level
            **kwargs
        """
        with utils.smart_open(pkl_file, "rb") as fin:
            num_sample = pickle.load(fin)

            self.transform = transform
            self.lazy = lazy
            self.kwargs = kwargs
            self.sequences = []
            self.pdb_files = []
            self.data = []
            indexes = range(num_sample)
            if verbose:
                indexes = tqdm(indexes, "Loading %s" % pkl_file)
            for i in indexes:
                pdb_file, sequence, protein = pickle.load(fin)
                self.sequences.append(sequence)
                self.pdb_files.append(pdb_file)
                self.data.append(protein)

    def save_pickle(self, pkl_file, verbose=0):
        with utils.smart_open(pkl_file, "wb") as fout:
            num_sample = len(self.data)
            pickle.dump(num_sample, fout)

            indexes = range(num_sample)
            if verbose:
                indexes = tqdm(indexes, "Dumping to %s" % pkl_file)
            for i in indexes:
                pdb_dir, pdb_name = os.path.split(self.pdb_files[i])
                split = os.path.basename(pdb_dir)
                pdb_file = os.path.join(split, pdb_name)
                pickle.dump((pdb_file, self.sequences[i], self.data[i]), fout)

    @property
    def residue_feature_dim(self):
        """Dimension of residue features."""
        return self.data[0].residue_feature.shape[-1]
