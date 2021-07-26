# Overview
This module contains several Python scripts to create and train a neural net. 
It can also be used for inference and a few other utilities, interfaced in the main.py

### Training

The learning directory contains a default.ini with some default values for the 
network. Additional ini files can be built and should be stored in 
results/inis with the .ini extension, following the example of the default.ini.
Additional arguments can be passed directly through the argparse mecanism 
(try running `python main.py train -h`), including extra ini files. The 
argparse arguments have priority over the ini (so adding an argparse section
in the ini will have no effect).
An example of the line to run a custom architecture, stored in 
results/inis/YOUR_INI_NAME.ini, with a batch size of 3 is :

`python main.py train -ini YOUR_INI_NAME -bs 3`

This will first dump a .exp file in results.exp with a summary of all the parameters.
It will then launch a training experiment and log metrics in results/logs, enabling
vizualisation of all runs with tensorboard. Each epoch, it will save the model
 in results/weights. 

## Predicting

One can put weights in an .envrc or add a weights path to the argparse and then 
simply run :

`python main.py predict NAME_OF_THE_PDB`


## Removing conf files

After an unsuccessful or erroneous experiment, one can remove it and all of the
saved models / logs / .exp files by using : 
`python main.py rm NAME_TO_REMOVE`


# Files descriptions

`learn.py` contains the script for testing and training a model

`main.py` is a script used only for command line interfacing call to functions.
 It can be called from either the root of the project or the directory. It can be 
used either in train mode or in predict mode, using argv[1] as :
`python main.py train -ep 50 -n example ...`

`Nnet.py` contains the model as well as inference and optimizations functions using
 this instance of a model

`utils.py` contains the Metrics object useful for logging, the ConfParser object
used for experiments management and some small functions



