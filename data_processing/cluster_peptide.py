#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2019-09-26 10:09:46 (UTC+0200)

import sys
import os
import numpy
from sklearn.cluster import KMeans
from pymol import cmd
from data_processing.Density import get_topology


def update_selection(chunk, selection):
    if len(chunk) > 1:
        selection += '%d-%d+'%(chunk[0], chunk[-1])
    else:
        selection += '%d+'%chunk[0]
    return selection

def cluster(pdbfilename, n_clusters):
    cluster = KMeans(n_clusters=n_clusters)
    cmd.load(pdbfilename, 'peptide')
    cmd.remove('hydrogens')
    coords, topology = get_topology('peptide', 'all')
    labels = cluster.fit_predict(coords)
    print(pdbfilename)
    clusters = []
    for label in numpy.unique(labels):
        resids = numpy.unique(numpy.int_(topology['resids'][labels == label]))
        selection = 'resi '
        chunk = [resids[0]]
        for resid in resids[1:]:
            if resid - chunk[-1] == 1:
                chunk.append(resid)
            else:
                selection = update_selection(chunk, selection)
                chunk = [resid, ]
        selection = update_selection(chunk, selection)
        print("Peptide %d: "%label + selection)
        outpdbfilename = os.path.splitext(pdbfilename)[0] + '_%d.pdb'%label
        cmd.save(outpdbfilename, selection)
        clusters.append(cmd.get_coords(selection))
    return clusters

if __name__ == "__main__":
    pdbfilename = sys.argv[1]
    n_clusters = int(sys.argv[2])
    clusters = cluster(pdbfilename, n_clusters)
