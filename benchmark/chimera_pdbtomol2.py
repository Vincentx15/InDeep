import chimera
from chimera import runCommand as rc
from DockPrep import prep
from WriteMol2 import writeMol2

import sys
in_pdb, out_name = sys.argv[1], sys.argv[2]
opened = chimera.openModels.open(in_pdb)
models = chimera.openModels.list()
rc('del solvent')
rc('del ligand')
# rc('addCharge all')
prep(models)
writeMol2(models, out_name)
