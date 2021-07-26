#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2019-09-12 09:38:26 (UTC+0200)

import os
import pymol.cmd as cmd
import numpy

import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from data_processing import utils


def get_topology(object_name, selection):
    """
    • filename: pdb file name
    """
    selection = "%s and %s" % (object_name, selection)
    coords = cmd.get_coords(selection=selection)
    topology = {'resnames': [],
                'names': [],
                'chains': [],
                'resids': [],
                'vdws': []}
    cmd.iterate(selection,
                'resnames.append(resn);names.append(name);chains.append(chain);resids.append(resi);vdws.append(vdw)',
                space=topology)
    for k in topology:
        topology[k] = numpy.asarray(topology[k])
        # if k == 'resids': # Removed to handle residue insertion
        #    topology[k] = numpy.int_(topology[k])
    topology['counts'] = numpy.arange(len(topology['resids']))
    # cmd.delete(object_name)
    return coords, topology


def get_selection(atomtype, atomtype_mapping):
    sel_list = atomtype_mapping[atomtype]
    selection = ''
    for i, ra in enumerate(sel_list):
        resname, atomname = ra
        if i > 0 and i < len(sel_list):
            selection += ' | '
        selection += '(resn %s & name %s)' % (resname, atomname)
    return selection


ATOMTYPES = utils.read_atomtypes()
ATOMTYPES = {atomtype: get_selection(atomtype, ATOMTYPES) for atomtype in ATOMTYPES}


class Coords_channel(object):
    def __init__(self, pdbfilename_p, pdbfilename_l=None,
                 hetatm=False, h5file=None):
        """
        The goal is to go from pdb files and optionnally some selections to the (n,4) or (n,1) format of coordinates
        annotated by their channel id (or 1 for hetatm). This can be used to fill an hdf5 if it is not None

        - pdbfilename_p: PDB file name for the protein
        - pdbfilename_l: PDB file name for the ligand
        - hetatm: Boolean. Set to True for HETATM ligand
        - h5file: HDF5 file object from h5py.File
        """
        a = cmd.get_object_list('all')
        safe_sel = ['None', 'None'] + list({'prot', 'lig'}.intersection(set(a)))
        safe_sel = ' or '.join(safe_sel)
        cmd.delete(safe_sel)
        self.pdbfilename_p = pdbfilename_p
        self.pdbfilename_l = pdbfilename_l
        self.hetatm = hetatm
        self.h5file = h5file
        # self.atomtypes = utils.read_atomtypes()
        self.atomtypes = ATOMTYPES
        cmd.load(pdbfilename_p, 'prot')
        if pdbfilename_l is not None:
            cmd.load(pdbfilename_l, 'lig')
            self.ligand_selection = 'lig'
        else:
            self.ligand_selection = None
        cmd.remove('hydrogens')
        self.prot_selection = 'prot'

        a = cmd.get_object_list('all')
        safe_sel = ['None', 'None'] + list({'prot', 'lig'}.intersection(set(a)))
        safe_sel = ' or '.join(safe_sel)

        _, self.topology = get_topology('prot', safe_sel)
        self.coords = None
        self.set_coords()

    def set_coords(self, coords=None, selection='prot or lig'):
        if coords is not None:
            cmd.load_coords(coords, selection)
        self.coords = cmd.get_coords(selection='%s or %s' % (self.prot_selection,
                                                             self.ligand_selection))
        self.coords_prot = cmd.get_coords(selection=self.prot_selection)
        if self.ligand_selection is not None:
            self.coords_lig = cmd.get_coords(selection=self.ligand_selection)
        else:
            self.coords_lig = None

        self.coords_channels_prot = self.split_coords_by_channels()
        self.coords_channels_lig = self.split_coords_by_channels(ligand=True,
                                                                 hetatm=self.hetatm)

    def split_coords_by_channels(self, selection=None, hetatm=False, ligand=False, load_coords=None):
        """
        Build the grid.
        - selection: pymol selection string to compute the grid on
        - hetatm: if True consider the ligand as hetero-atoms and
                  do not decompose in channels
        """
        initial_selection = self.ligand_selection if ligand else self.prot_selection
        if selection is not None:
            selection = initial_selection + ' and ' + selection
        else:
            selection = initial_selection

        if load_coords is not None:
            cmd.load_coords(load_coords, selection)
            # raw_coords = cmd.get_coords(selection=selection)
            # print('raw coords : ', raw_coords.shape)

        coords_all = None
        atomtypes = {'ALL': 'all'} if hetatm else self.atomtypes
        for cid, atomtype in enumerate(atomtypes):  # cid: channel id
            if hetatm:
                cid = -1
                coords = cmd.get_coords(selection=selection)
            else:  # Channels
                coords = cmd.get_coords(selection='%s and (%s)' % (selection,
                                                                   atomtypes[atomtype]))
            if coords is not None:
                if coords_all is None:
                    # Store the coordinates in a (n, 4) array (coords_all)
                    # with the last column corresponding to the channel id
                    coords_all = numpy.c_[coords,
                                          numpy.repeat(cid, coords.shape[0])]
                else:
                    coords_all = numpy.r_[coords_all,
                                          numpy.c_[coords,
                                                   numpy.repeat(cid, coords.shape[0])]]
        if self.h5file is not None:
            groupname = os.path.basename(self.pdbfilename_p)
            if groupname not in self.h5file:
                group = self.h5file.create_group(groupname)
            else:
                group = self.h5file[groupname]
            if selection == 'lig':  # create group for the ligand
                grouplig = group.create_group(os.path.basename(self.pdbfilename_l))
                grouplig.create_dataset('coords', data=coords_all)
            if selection == 'prot' and 'coords' not in group:
                group.create_dataset('coords', data=coords_all)
        return coords_all
