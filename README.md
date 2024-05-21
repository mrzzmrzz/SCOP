# This is the code implementation for SCOP.


## Dataset
#### Pre-training dataset
- AlphaFold Swiss-Prot: A protein structure database predicted by the AlphaFold model, containing approximately 542K protein 3D structure information. *You can download this dataset clicking the [website](https://www.alphafold.com/download)*.



#### Benchmark datasets
1. EC: A public dataset used to predict the protein's enzyme commission (EC) numbers, which describe their catalysis of biochemical reactions. *You can download this dataset clicking the [website](https://zenodo.org/record/6622158/files/EnzymeCommission.zip)*.
2. GO-BP: A dataset related to the biological process (BP) terms of a protein, representing a specific objective that the organism is genetically programmed to achieve. *You can download this dataset clicking the [website](https://zenodo.org/record/6622158/files/GeneOntology.zip)*.
3. GO-MF: A dataset about the protein's molecular functions (MF), which correspond to activities that can be performed by individual gene products. *You can download this dataset clicking the [website](https://zenodo.org/record/6622158/files/GeneOntology.zip)*
4. GO-CC: A dataset with the cellular component (CC) terms of a protein, referring to the locations about cellular structures where a gene product performs a function. *You can download this dataset clicking the [website](https://zenodo.org/record/6622158/files/GeneOntology.zip)*
5. Glycoprotein Dataset (self-built dataset used in Section 6 `Case Study`): *You can find this dataset in  the folder `datasets`*


## Requirements
In order to reproduce our code, you need to install the following pip dependencies:
```python
numpy==1.22.4
pandas==1.4.3
rdkit-pypi==2022.3.5
torch==1.12.0
torch-cluster==1.6.0
torch-geometric==2.0.4
torch-scatter==2.0.9
torch-sparse==0.6.14
torch-spline-conv==1.2.1
torchdrug==0.2.0
```

## Usage
1. Preprocess the pre-training dataset
    - After downloading the full  AlphaFold Swiss-Prot dataset, you are supposed to leverage the `src\alphafold.py`  to transform the original protein PDB file into a `NetworkX` file. 
2. Pretrain the protein structure model and sequence model.
   - Run the file `src\pretrain_model.py` to pretrain two models simultaneously.
3. Evaluate the pretrained structure model on benchmark datasets.
    - Transform the downloaded dataset to the `NetworkX` format.
    - Run the file `src\pipeline.py` to evaluate the model on downstream tasks.
