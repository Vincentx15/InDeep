# InDeep : 3D convolutional neural networks to assist in silico drug design on protein-protein interactions 

## Description

This repository offers a toolbox for PPI drug discovery. 
It enables prediction of interactibility and ligandability sites 
over a protein structure. 
The scores associated with those values can also be monitored along a MD analysis.
These predictions can be visualised easily.

## Setup

### Getting the weights
You can download the wights used for the neural network prediction through this link :
https://mega.nz/file/mEVQUJwR#tqLX-UGo3a8yORmQJB9tYxx7teXF3PyEUBbyVYEnCt0

You should copy them to a directory named ROOT_OF_PROJECT/results/weights/HPO/

### Packages

To run this code, one need to use python with several packages. 
The key packages to install are PyTorch (we recommend using the gpu version if one
is equiped with a gpu) and PyMol.
Eventhough users can install these packages manually, we recommend using a conda environment.
One should have conda installed and then run one of the following :

For a cpu-only installation

`conda create env -f hd_cpu.yml`

Respectively for gpu usage :

`conda create env -f hd_gpu.yml`

And to activate this environment, run :
```
conda activate indeep
```

### PyMol plugin setup 

To setup the PyMol plugin, just activate your environment and type pymol in the 
commandline to launch PyMol.
Then click on the tab *Plugin* and *Plugin Manager* and on the window that pops, click on the tab *Settings*
In this tab, one can *Add new directory...* and select :
```ROOT_OF_PROJECT/DeepPPI_pymol_script```

Finally, one should quit PyMol and relaunch it by running pymol in the command line.
In the tab *Plugin*, there should be a new plugin : *deepPPI Plugin* that one can use directly.

## Command line usage 

After activating the conda environment, one can also directly run a prediction on
a PDB file or a MD trajectory in the DCD format.
```
python learning/predict.py pdb -p PATH_TO_PDB
python learning/predict.py traj -p PATH_TO_PDB -d PATH_TO_DCD
```

For an explanation of all options, run :
```
python learning/predict.py --help
```

## Reproducing the results

### Data
The data is the one of the IPPI-DB, that can be downloaded from the website.

The data should be downloaded in a data/ folder at the root from : 
TODO : insert link

### Learning
Read the README in learning to launch the training of a new model.
This new model should then appear in the PyMol plugin or be called explicitly 
using the command line.

### Benchmark
The scripts necessary for reproducing both the PL and HD plots and results are 
present in the benchmark/ folder.





