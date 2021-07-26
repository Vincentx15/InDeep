#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2019-09-19 09:05:29 (UTC+0200)

import os
import numpy as np
import scipy.ndimage as ndi
import skimage.morphology
import skimage.segmentation
from skimage.feature import peak_local_max
import random
import string
from pymol import cmd
import mrcfile


def read_atomtypes():
    """
    Read the atomtype_mapping.txt file and return the
    mapping as a python dictionary
    """
    atomtype_mapping = {}
    with open(os.path.join(os.path.dirname(__file__), 'atomtype_mapping.txt'), 'r') as atomtypefile:
        for line in atomtypefile:
            line = line.strip()
            resname, pdbatomtype, atomtype = line.split(" ")
            if atomtype in atomtype_mapping:
                atomtype_mapping[atomtype].append((resname, pdbatomtype))
            else:
                atomtype_mapping[atomtype] = [(resname, pdbatomtype), ]
    return atomtype_mapping


class bcolors:
    UNDERLINE = '\033[4m'
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def save_density(density, outfilename, spacing, origin, padding):
    """
    Save the density file as mrc for the given atomname
    """
    density = density.astype('float32')
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(density.T)
        mrc.voxel_size = spacing
        mrc.header['origin']['x'] = origin[0] - padding
        mrc.header['origin']['y'] = origin[1] - padding
        mrc.header['origin']['z'] = origin[2] - padding
        mrc.update_header_from_data()
        mrc.update_header_stats()


def save_coords(coords, topology, outfilename, selection=None):
    """
    Save the coordinates to a pdb file
    • coords: coordinates
    • topology: topology
    • outfilename: name of the oupyt pdb
    • selection: Boolean array to select atoms
    """
    object_name = 'struct_save_coords'
    cmd.delete(object_name)
    if selection is None:
        selection = np.ones(len(topology['resids']), dtype=bool)
    for i, coords_ in enumerate(coords):
        if selection[i]:
            name = topology['names'][i]
            resn = topology['resnames'][i]
            resi = topology['resids'][i]
            chain = topology['chains'][i]
            elem = name[0]
            cmd.pseudoatom(object_name,
                           name=name,
                           resn=resn,
                           resi=resi,
                           chain=chain,
                           elem=elem,
                           hetatm=0,
                           segi=chain,
                           pos=list(coords_))
    cmd.save(outfilename, selection=object_name)
    cmd.delete(object_name)


def get_protlig_list(pldb, database_path):
    """
    Returns the list of protein ligand system:
    - pldb: file object of the protein-ligand database
    - database_path: path to the protein-ligand database
    """
    protlig_dict = {}
    for line in pldb:
        line = line.strip()
        pdbcode = line.split(" ")[1][:5]
        if pdbcode not in protlig_dict:
            protlig_dict[pdbcode] = {}
        if line[0] == "P":
            protein = line.split(" ")[1]
            protein = protein[:4] + protein[5:]
            protlig_dict[pdbcode].update({"P": protein})
        if line[0] == "L":
            ligand = line.split(" ")[1]
            ligand = ligand[:4] + ligand[5:]
            if "L" in protlig_dict[pdbcode]:
                protlig_dict[pdbcode]["L"].append(ligand)
            else:
                protlig_dict[pdbcode]["L"] = [ligand, ]
    protlig = []
    for pdbcode in protlig_dict:
        pdbpath = database_path + '/' + '/'.join([s for s in pdbcode[:4]]) + '/'
        protein = protlig_dict[pdbcode]['P']
        for ligand in protlig_dict[pdbcode]['L']:
            protlig.append((pdbpath + protein, pdbpath + ligand))
    del protlig_dict
    return protlig


def save_prediction(pred, spacing, origin, padding, name):
    """
    Save the prediction (pred) as a mrc file (mrcfilename) for each channel
    - pred: the prediction (np.ndarray)
    - mrcfilename: A string for the output mrc file name
    """
    # Remove extension
    name = os.path.splitext(name)[0]
    atomtypes = read_atomtypes().keys()
    # Remove empty batch size
    pred = np.squeeze(pred)
    if len(pred.shape) == 3:
        mrcfilename_ = name + '_PL.mrc'
        save_density(pred, mrcfilename_, spacing, origin, padding)
    elif len(pred.shape) == 4:
        for i, atomtype in enumerate(atomtypes):
            mrcfilename_ = f"{name}_{atomtype}.mrc"
            save_density(pred[i, ...], mrcfilename_, spacing, origin, padding)
        mrcfilename_ = f"{name}_ALL.mrc"
        save_density(1. - pred[-1, ...], mrcfilename_, spacing, origin, padding)


def watershed(grid, min_distance=6):
    """
    Watershed algorithm on the predicted 3D grids.
    returns filter labels or blobs in grid format, max and max_locations sorted by pmax value
    """
    local_maxi = peak_local_max(grid, indices=False, min_distance=min_distance)
    markers, npockets = ndi.label(local_maxi)
    labels = skimage.segmentation.watershed(-grid, markers, watershed_line=True)
    labels_list = np.unique(labels)

    # If no blobs were found, labels_list will only contain zeros
    if len(labels_list) < 2:
        return None, None, None

    # Watershed line adds a "0" label for the line that parts the basins
    labels_list = labels_list[labels_list > 0]

    # Funnily enough, this smart version is slower than the following iteration... and less readable
    # pmax = ndi.measurements.labeled_comprehension(grid, labels, labels_list, np.max, float, 0)
    # pmax_locs = ndi.measurements.labeled_comprehension(grid, labels, labels_list, np.argmax, float, 0)
    pmax, pmax_locs = list(), list()
    for label in labels_list:
        grid_label = (labels == label) * grid
        index = np.argmax(grid_label)
        index = np.unravel_index(index, grid.shape)
        label_pmax = grid_label[index]
        pmax.append(label_pmax)
        pmax_locs.append(index)
    pmax_locs = np.vstack(np.array(pmax_locs))
    pmax = np.array(pmax)
    # p_values = grid[pmax_locs.T[0], pmax_locs.T[1], pmax_locs.T[2]] # This should be equal to pmax

    sorter = pmax.argsort()[::-1]
    pmax = pmax[sorter]
    pmax_locs = pmax_locs[sorter]
    labels_list = labels_list[sorter]
    # Renumber the labels by decreasing probabilities:
    labels_new = np.zeros_like(labels)
    for i, label in enumerate(labels_list):
        labels_new[labels == label] = i + 1
    labels = labels_new
    return labels, pmax, pmax_locs


def split_grid(grid, labels):
    """
    Split the grid in grids given the labels
    See watershed.
    """
    grids = []
    labels_list = np.unique(labels)
    for label in labels_list:
        grid_ = np.zeros_like(grid)
        sel = (labels == label)
        grid_[sel] = grid[sel]
        grids.append(grid_)
    return grids


def get_random_string(length=8):
    return ''.join(random.choices(string.ascii_uppercase + string.digits,
                                  k=length))


def get_rmsd(pdbfilename1, pdbfilename2):
    """
    Compute the RMSD between 2 pdbs
    """
    pdb1 = get_random_string()
    pdb2 = get_random_string()
    cmd.load(pdbfilename1, pdb1)
    cmd.load(pdbfilename2, pdb2)
    (rmsd_refine,
     n_aligned_atoms,
     n_refine_cycle,
     rmsd_ori,
     n_aligned_ori,
     aln_score,
     n_res_aligned) = cmd.align(pdb1, pdb2)
    cmd.delete(pdb1)
    cmd.delete(pdb2)
    return rmsd_refine


if __name__ == '__main__':
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
