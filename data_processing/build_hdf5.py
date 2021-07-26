#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2019-10-16 21:53:55 (UTC+0200)

import sys
import h5py
sys.path.append('../')
from data_processing import Density, utils


def add_to_h5(dbfilename, database_path, h5filename, hetatm):
    """
    Add the given data to a hdf5 file.
    - dbfilename: name of the text file of the DB
    - database_path: path to the database in PDB format
    - h5filename: name of the hdf5 file
    """
    with h5py.File(h5filename, 'a') as h5file:
        with open(dbfilename) as pldb:
            protlig = utils.get_protlig_list(pldb, database_path)
            n = len(protlig)
            for i, (prot, lig) in enumerate(protlig):
                sys.stdout.write('%.2f %%: %s-%s' % ((i+1)*100./n, prot, lig))
                #try:
                Density.Coords_channel(prot, lig, h5file=h5file, hetatm=hetatm)
                sys.stdout.write(utils.bcolors.OKGREEN + ' [OK]' + utils.bcolors.ENDC + '\n')
                #except:
                #    sys.stdout.write(utils.bcolors.FAIL + ' [FAILED]' + utils.bcolors.ENDC + '\n')

if __name__ == '__main__':
    # Add PL-database
    DBFILENAME = '../data/PL-database.txt'
    DATABASE_PATH = '../data/PL-database'
    H5FILENAME = '../data/PLHD-database.hdf5'
    add_to_h5(DBFILENAME, DATABASE_PATH, H5FILENAME, hetatm=True)
    # Add HD-database
    DBFILENAME = '../data/HD-database_chunk.txt'
    DATABASE_PATH = '../data/HD-database'
    add_to_h5(DBFILENAME, DATABASE_PATH, H5FILENAME, hetatm=False)
