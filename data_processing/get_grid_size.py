#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2019-09-26 08:31:43 (UTC+0200)

"""
Returns the grid size for each system in HD-database
"""

import sys
import os
from data_processing import utils, Complex
import pymol.cmd as cmd


def get_grid_size(ligand, padding, spacing):
    cmd.delete('all')
    cmd.load(ligand, 'ligand')
    cmd.remove('hydrogens')
    coords = cmd.get_coords(selection='ligand')
    nx, ny, nz = Complex.get_grid_shape(coords.min(axis=0), coords.max(axis=0), spacing, padding)
    return nx, ny, nz


if __name__ == "__main__":
    padding = 6
    spacing = 1
    dbfile = sys.argv[1]
    pldb = open(dbfile, 'r')
    protlig = utils.get_protlig_list(pldb, 'HD-database')
    nsystem = len(protlig)
    outfilename = os.path.splitext(dbfile)[0] + '_gridsize.txt'
    with open(outfilename, 'w') as outfile:
        for i, (protein, ligand) in enumerate(protlig):
            xs, ys, zs = get_grid_size(ligand, padding, spacing)
            gridsize = xs * ys * zs
            line = protein + " " + ligand + " %d" % gridsize
            print('%.2f' % ((i + 1) * 100. / nsystem) + '% ' + line)
            outfile.write(line + "\n")
