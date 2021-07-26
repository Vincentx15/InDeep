#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2019-10-17 11:04:53 (UTC+0200)

import os
import sys
import numpy as np
import h5py
from scipy import ndimage
from sklearn.gaussian_process.kernels import RBF

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from data_processing import utils

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    print("Cannot import scipy.spatial.transform.Rotation")

ATOMTYPES = utils.read_atomtypes()

"""
This script makes the conversion from (n,4) matrices to the grid format.
It also introduces the 'Complex' class that is fulling the Database object
"""


def get_coords(h5file, path):
    """
    Return the coordinates for the given hdf5 path:
    - path: a string containing the path the system, e.g.:
        - '7r1r-AD-P00452-P69924-D.pdb' for a protein
        - '7r1r-AD-P00452-P69924-D.pdb/7r1r-AD-P00452-P69924-A-short_0.pdb'
          for a ligand
    """
    return h5file[path]['coords'][:]


def get_grid_shape(xyz_min, xyz_max, spacing, padding):
    xm, ym, zm = xyz_min - (padding,) * 3
    xM, yM, zM = xyz_max + (padding,) * 3
    X = np.arange(xm, xM, spacing)
    Y = np.arange(ym, yM, spacing)
    Z = np.arange(zm, zM, spacing)
    nx, ny, nz = len(X) - 1, len(Y) - 1, len(Z) - 1
    return nx, ny, nz


def just_one(coord, xi, yi, zi, sigma, total_grid, use_multiprocessing=False):
    """

    :param coord: x,y,z
    :param grid:
    :param sigma:
    :return:
    """
    #  Find subgrid
    nx, ny, nz = xi.size, yi.size, zi.size

    bound = int(4 * sigma)
    x, y, z = coord
    binx = np.digitize(x, xi)
    biny = np.digitize(y, yi)
    binz = np.digitize(z, zi)
    min_bounds_x, max_bounds_x = max(0, binx - bound), min(nx, binx + bound)
    min_bounds_y, max_bounds_y = max(0, biny - bound), min(ny, biny + bound)
    min_bounds_z, max_bounds_z = max(0, binz - bound), min(nz, binz + bound)

    X, Y, Z = np.meshgrid(xi[min_bounds_x: max_bounds_x],
                          yi[min_bounds_y: max_bounds_y],
                          zi[min_bounds_z:max_bounds_z],
                          indexing='ij')
    X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()

    #  Compute RBF
    rbf = RBF(sigma)
    subgrid = rbf(coord, np.c_[X, Y, Z])
    subgrid = subgrid.reshape((max_bounds_x - min_bounds_x,
                               max_bounds_y - min_bounds_y,
                               max_bounds_z - min_bounds_z))

    #  Add on the first grid
    if not use_multiprocessing:
        total_grid[min_bounds_x: max_bounds_x, min_bounds_y: max_bounds_y, min_bounds_z:max_bounds_z] += subgrid
    else:
        return min_bounds_x, max_bounds_x, min_bounds_y, max_bounds_y, min_bounds_z, max_bounds_z, subgrid


def gaussian_blur(coords, xi, yi, zi, sigma=1., use_multiprocessing=False):
    """
    Compute RBF on a set of coords,
    We loop over each coord to compute only a neighborhood and add it to the right grid
    """

    nx, ny, nz = xi.size, yi.size, zi.size
    total_grid = np.zeros(shape=(nx, ny, nz))

    if use_multiprocessing:
        import multiprocessing
        args = [(coord, xi, yi, zi, sigma, None, True) for coord in coords]
        pool = multiprocessing.Pool()
        grids_to_add = pool.starmap(just_one, args)
        for min_bounds_x, max_bounds_x, min_bounds_y, max_bounds_y, min_bounds_z, max_bounds_z, subgrid in grids_to_add:
            total_grid[min_bounds_x: max_bounds_x, min_bounds_y: max_bounds_y, min_bounds_z:max_bounds_z] += subgrid
    else:
        for coord in coords:
            just_one(coord, xi=xi, yi=yi, zi=zi, sigma=sigma, total_grid=total_grid)
    return total_grid


def get_grid(coords, spacing, padding, xyz_min=None, xyz_max=None, sigma=1.):
    """
    Generate a grid without channels from the coordinates
    """

    def get_bins(coords, spacing, padding, xyz_min=None, xyz_max=None):
        """
        Compute the 3D bins from the coordinates
        """
        if xyz_min is None:
            xm, ym, zm = coords.min(axis=0) - padding
        else:
            xm, ym, zm = xyz_min - padding
        if xyz_max is None:
            xM, yM, zM = coords.max(axis=0) + padding
        else:
            xM, yM, zM = xyz_max + padding

        xi = np.arange(xm, xM, spacing)
        yi = np.arange(ym, yM, spacing)
        zi = np.arange(zm, zM, spacing)
        return xi, yi, zi

    xi, yi, zi = get_bins(coords, spacing, padding, xyz_min, xyz_max)
    grid = gaussian_blur(coords, xi, yi, zi, sigma=sigma)
    return grid


def get_grid_channels(coords, spacing, padding, xyz_min, xyz_max, sigma=1., is_target=False):
    """
    Compute the 3D grids per channel from the coordinates
    - coords: coordinates in the format [x, y, z, channel_id]
    Remarks: - channel_id is zero-based numbering of the channels
             - if channel_id is -1 only one channel is present (HETATM ligand
               and not a protein partner)
    """
    channel_ids = coords[:, -1]
    channel_ids_u = np.unique(channel_ids)
    if len(channel_ids_u) == 1 and channel_ids_u[0] == -1:
        # Its an organic HETATM ligand (from PL)
        grid = get_grid(coords[:, :3], spacing, padding,
                        sigma=sigma, xyz_max=xyz_max, xyz_min=xyz_min)
        grid = grid[None, ...]
    else:
        grid = []
        for channel_id, _ in enumerate(ATOMTYPES):
            sel = (channel_ids == channel_id)
            coords_channel = coords[sel]
            grid_ = get_grid(coords_channel[:, :3], spacing, padding,
                             xyz_min=xyz_min, xyz_max=xyz_max,
                             sigma=sigma)
            grid.append(grid_)
        grid = np.asarray(grid)

    if is_target:
        # If this ligand has several channels, we add a 'void' channel
        if grid.shape[0] > 1:
            grid = np.concatenate([grid, (1. - np.sum(grid, axis=0))[None, ...]],
                                  axis=0)
            grid = grid.clip(min=0)
            grid /= grid.sum(axis=0)[None, ...]

        # Cap each voxel because RBF can exceed one
        else:
            grid = grid.clip(max=1)
    return grid


def get_mask_grid_prot(grid_prot, gaussian_cutoff=0.2, iterations=6):
    """
    Returns a grid with a binary value corresponding to ones in the enveloppe of the input protein.
    :param grid_prot:
    :param gaussian_cutoff:
    :param iterations:
    :return:
    """
    grid_prot = grid_prot.copy()
    summed_grid_prot = np.sum(grid_prot, axis=0)
    initial_mask = summed_grid_prot > gaussian_cutoff
    mask = ndimage.binary_dilation(initial_mask, iterations=iterations)
    enveloppe = mask - initial_mask.astype(np.int)
    return enveloppe


class Complex(object):
    """
    Object containing a protein-ligand system
    """

    def __init__(self, h5file, protein, ligand, spacing=1., padding=8, rotate=True):
        """
        - h5file: the hdf5 database File object opened by h5py
        - protein: a string identifying the protein
        - ligand: a string identifying the ligand
        - rotate: if True, randomly rotate the coordinates of the complex

        Test the rotation:
        >>> cp_rotate_PL = Complex(h5file, P, L)
        >>> cp_rotate_HD = Complex(h5file, H, D)

        The coordinates should change:
        >>> (cp_rotate_HD.coords_prot[:, :3] != cp_HD.coords_prot[:, :3]).any()
        True
        >>> (cp_rotate_PL.coords_prot[:, :3] != cp_PL.coords_prot[:, :3]).any()
        True
        
        But not the channel id columns (the fourth)
        >>> (cp_rotate_HD.coords_prot[:, 3] == cp_HD.coords_prot[:, 3]).all()
        True
        >>> (cp_rotate_PL.coords_prot[:, 3] == cp_PL.coords_prot[:, 3]).all()
        True

        And consequently the size of the grids should be different:
        >>> cp_rotate_HD.grid_prot.shape != cp_HD.grid_prot.shape
        True
        >>> cp_rotate_PL.grid_prot.shape != cp_PL.grid_prot.shape
        True

        Save mrc files:
        >>> cp_rotate_HD.save_mrc_prot()
        >>> cp_rotate_HD.save_mrc_lig()
        """
        self.h5file = h5file
        self.protein = protein
        self.ligand = ligand
        self.spacing = spacing
        self.padding = padding
        self.coords_prot = self._get_coords_prot()
        self.coords_lig = self._get_coords_lig()
        if rotate:
            self.random_rotate()
        self.xyz_min = self.coords_lig[:, :3].min(axis=0)
        self.xyz_max = self.coords_lig[:, :3].max(axis=0)

        # # TO LEARN ON THE WHOLE COMPLEX :
        # self.xyz_min = np.min(np.r_[self.coords_prot[:, :3], self.coords_lig[:, :3]], axis=0)
        # self.xyz_max = np.max(np.r_[self.coords_prot[:, :3], self.coords_lig[:, :3]], axis=0)

        self.grid_shape = get_grid_shape(self.xyz_min, self.xyz_max,
                                         self.spacing, self.padding)

    def _get_coords_prot(self):
        """
        Returns the coordinates of the protein

        >>> cp_PL._get_coords_prot().shape
        (1078, 4)
        >>> cp_HD._get_coords_prot().shape
        (1624, 4)
        """
        return get_coords(self.h5file, self.protein)

    def _get_coords_lig(self):
        """
        Returns the coordinates of the ligand
        with the corresponding channel ids (x, y, z, channel_id)

        >>> cp_PL._get_coords_lig().shape
        (28, 4)
        >>> cp_HD._get_coords_lig().shape
        (116, 4)
        """
        key = self.protein + '/' + self.ligand
        return get_coords(self.h5file, key)

    def _concat_coords(self):
        """
        Concatenate the coordinates of the protein and the ligand together
        and store in the fifth column a zero for the protein and a 1 for the ligand

        >>> cp_PL._concat_coords().shape
        (1106, 5)

        Store in the last columns (the fifth) a zero tag for the protein coordinates:
        >>> (cp_PL._concat_coords()[:, 4] == 0).sum() == cp_PL._get_coords_prot().shape[0]
        True

        And a 1 for the ligand coordinates:
        >>> (cp_PL._concat_coords()[:, 4] == 1).sum() == cp_PL._get_coords_lig().shape[0]
        True
        """
        coords_prot, coords_lig = (self._get_coords_prot(),
                                   self._get_coords_lig())
        coords_cat = np.r_[coords_prot, coords_lig]
        protlig = np.r_[np.zeros(coords_prot.shape[0]),
                        np.ones(coords_lig.shape[0])]
        coords_cat = np.c_[coords_cat, protlig]
        return coords_cat

    def random_rotate(self, mutate=True):
        """
        Randomly rotate the coordinates of both proteins and ligands
        - mutate: if True, update the values of self.coords_lig and self.coords_prot
        
        >>> coords_prot, coords_lig = cp_PL.random_rotate(mutate=False)
        >>> coords_prot.shape == cp_PL.coords_prot.shape
        True
        >>> coords_lig.shape == cp_PL.coords_lig.shape
        True

        The channel ids are preserved:
        >>> (coords_prot[:, 3] == cp_PL.coords_prot[:, 3]).all()
        True
        >>> (coords_lig[:, 3] == cp_PL.coords_lig[:, 3]).all()
        True

        But the coordinates are rotated:
        >>> (coords_prot[:, :3] != cp_PL.coords_prot[:, :3]).any()
        True
        >>> (coords_lig[:, :3] != cp_PL.coords_lig[:, :3]).any()
        True

        Check that the distance matrices before and after rotation are equal:
        >>> np.allclose(scipy.spatial.distance.pdist(coords_prot[:, :3]),\
                           scipy.spatial.distance.pdist(cp_PL.coords_prot[:, :3]))
        True
        >>> np.allclose(scipy.spatial.distance.pdist(coords_lig[:, :3]),\
                           scipy.spatial.distance.pdist(cp_PL.coords_lig[:, :3]))
        True
        >>> np.allclose(scipy.spatial.distance.cdist(coords_prot[:, :3], coords_lig[:, :3]),\
                           scipy.spatial.distance.cdist(cp_PL.coords_prot[:, :3], cp_PL.coords_lig[:, :3]))
        True
        """
        coords_cat = self._concat_coords()
        coords = coords_cat[:, :3]
        channel_ids = coords_cat[:, 3]
        prot_lig = coords_cat[:, 4]
        alpha, beta, gamma = np.random.uniform(low=0., high=360., size=3)

        def rotate(coords, alpha, beta, gamma):
            """
            - coords: the coordinates to rotate
            - alpha, beta, gamma: the Euler angles in degree
            """
            r = R.from_euler('zyx', [alpha, beta, gamma], degrees=True)
            com = coords.mean(axis=0)
            coords_0 = coords - com
            return r.apply(coords_0) + com

        coords = rotate(coords, alpha, beta, gamma)
        coords_prot = np.c_[coords[prot_lig == 0],
                            channel_ids[prot_lig == 0]]
        coords_lig = np.c_[coords[prot_lig == 1],
                           channel_ids[prot_lig == 1]]
        if mutate:
            self.coords_prot = coords_prot
            self.coords_lig = coords_lig
        return coords_prot, coords_lig

    @property
    def is_PL(self):
        """
        Return True if the system is a PL system

        >>> cp_PL.is_PL
        True
        >>> cp_HD.is_PL
        False
        """
        return self.grid_lig.shape[0] == 1

    @property
    def is_HD(self):
        """
        Return True if the system is a PL system

       >>> cp_PL.is_HD
        False
        >>> cp_HD.is_HD
        True
        """
        return self.grid_lig.shape[0] > 1

    @property
    def grid_prot(self):
        """
        Return the grid (with channels) for the protein
        with the corresponding channel ids (x, y, z, channel_id)
        """
        grid = get_grid_channels(self.coords_prot,
                                 self.spacing,
                                 self.padding,
                                 self.xyz_min,
                                 self.xyz_max,
                                 is_target=False)
        return grid

    @property
    def mask_grid_prot(self):
        """
        Return the grid (with channels) for the protein
        with the corresponding channel ids (x, y, z, channel_id)
        """
        return get_mask_grid_prot(self.grid_prot)

    @property
    def grid_lig(self):
        """
        Return the grid (with channels) for the ligand
        """
        grid = get_grid_channels(self.coords_lig,
                                 self.spacing, self.padding,
                                 self.xyz_min, self.xyz_max,
                                 is_target=True)

        return grid

    def save_mrc_prot(self):
        """
        Save all the channels of the protein in separate mrc files
        """
        outbasename = os.path.splitext(self.protein)[0]
        for channel_id, atomtype in enumerate(ATOMTYPES):
            utils.save_density(self.grid_prot[channel_id, ...],
                               '%s_%s.mrc' % (outbasename, atomtype),
                               self.spacing, self.xyz_min, self.padding)
        utils.save_density(self.grid_prot.sum(axis=0),
                           '%s_ALL.mrc' % outbasename,
                           self.spacing, self.xyz_min, self.padding)

    def save_mrc_lig(self):
        """
        Save all the channels of the ligand in separate mrc files
        """
        if self.is_PL:
            outbasename = os.path.splitext(self.ligand)[0]
            utils.save_density(self.grid_lig[0, ...],
                               '%s_ALL.mrc' % outbasename,
                               self.spacing, self.xyz_min, self.padding)
        else:
            outbasename = os.path.splitext(self.ligand)[0]
            for channel_id, atomtype in enumerate(ATOMTYPES):
                utils.save_density(self.grid_lig[channel_id, ...],
                                   '%s_%s.mrc' % (outbasename, atomtype),
                                   self.spacing, self.xyz_min, self.padding)
            utils.save_density(1. - self.grid_lig[-1, ...],
                               '%s_ALL.mrc' % outbasename,
                               self.spacing, self.xyz_min, self.padding)


if __name__ == '__main__':
    import doctest
    import scipy.spatial.distance  # For the tests

    # TEST FUNCTIONS
    from data_processing import Density

    pdbfile = '1ycr.cif'
    selection = 'chain A'
    spacing = 1
    padding = 8
    density = Density.Coords_channel(pdbfilename_p=pdbfile)
    coords = density.split_coords_by_channels(selection=selection)
    xyz_min = coords[:, :3].min(axis=0)
    xyz_max = coords[:, :3].max(axis=0)
    grid = get_grid_channels(coords, spacing=spacing, padding=padding,
                             xyz_min=xyz_min, xyz_max=xyz_max)

    enveloppe = get_mask_grid_prot(grid, iterations=7, gaussian_cutoff=0.2)
    utils.save_density(enveloppe,
                       'mask.mrc',
                       spacing,
                       xyz_min,
                       padding)
    print(enveloppe.sum())
    i, j, k = enveloppe.shape
    print(i * j * k)
    # TEST COMPLEX

    # data_path = '../data/20201120_PLHD-database_chunks.hdf5'
    # H5FILE = h5py.File(data_path, 'r')
    #
    # P, L = '1a29-A-P62157.pdb', '1a29-A-TFP-153.pdb'
    # H, D = '15c8-LH-P01837-P01869-H.pdb', '15c8-LH-P01837-P01869-L-short_0.pdb'
    # cp = Complex(H5FILE, P, L, rotate=False)
    # mask = cp.mask_grid_prot
    # print(mask.shape)
    # utils.save_density(mask,
    #                    'mask.mrc',
    #                    cp.spacing,
    #                    cp.xyz_min,
    #                    cp.padding)
    # print(mask.sum())
    # i, j, k = mask.shape
    # print(i * j * k)
    # doctest.testmod(extraglobs={'db': Database(H5FILE),
    #                             'cp_PL': Complex(H5FILE, P, L, rotate=False),
    #                             'cp_HD': Complex(H5FILE, H,
    #                                              D, rotate=False),
    #                             'h5file': H5FILE,
    #                             'H': H, 'D': D,
    #                             'P': P, 'L': L},
    #                 # verbose=True,
    #                 optionflags=doctest.ELLIPSIS)
