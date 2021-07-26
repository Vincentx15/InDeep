#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2021 Institut Pasteur                                       #
#                 				                            #
#                                                                           #
#  Redistribution and use in source and binary forms, with or without       #
#  modification, are permitted provided that the following conditions       #
#  are met:                                                                 #
#                                                                           #
#  1. Redistributions of source code must retain the above copyright        #
#  notice, this list of conditions and the following disclaimer.            #
#  2. Redistributions in binary form must reproduce the above copyright     #
#  notice, this list of conditions and the following disclaimer in the      #
#  documentation and/or other materials provided with the distribution.     #
#  3. Neither the name of the copyright holder nor the names of its         #
#  contributors may be used to endorse or promote products derived from     #
#  this software without specific prior written permission.                 #
#                                                                           #
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      #
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        #
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    #
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     #
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   #
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         #
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    #
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    #
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      #
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    #
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     #
#                                                                           #
#  This program is free software: you can redistribute it and/or modify     #
#                                                                           #
#############################################################################

import sys
import numpy as np
import scipy
import time


def get(dmap, beta=50., func=lambda x, dmap, beta: np.exp(-beta * (dmap - x)), normalize=True):
    """
    The probabilities are computed with: exp(-beta*(dj-di)) with dj and di the
    densities in cells i and j
    • func: function to compute the probabilities or weights
    """
    nx, ny, nz = dmap.shape
    # transition_matrix_data = np.zeros((np.prod(dmap.shape), 26))
    transition_matrix = scipy.sparse.coo_matrix((np.prod(dmap.shape),) * 2)
    transition_matrix.data = np.zeros(nx * ny * nz * 26)
    transition_matrix.row = np.zeros(nx * ny * nz * 26, dtype=int)
    transition_matrix.col = np.zeros(nx * ny * nz * 26, dtype=int)
    mgrid = np.asarray(np.meshgrid(*[np.arange(_) for _ in (nx, ny, nz)], indexing='ij'))
    # Matrix that the give the rows in the transition matrix
    rowmat = np.ravel_multi_index(mgrid, (nx, ny, nz))
    n = nx * ny * nz
    ind = -1
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                if (i, j, k) != (0, 0, 0):
                    ind += 1
                    # transition_matrix_data[:, ind] = (1 / (dmap*move(dmap, (i, j, k)))).flatten()
                    # Matrix that the give the columns in the transition matrix
                    colmat = np.copy(mgrid)
                    colmat[:, 1:-1, 1:-1, 1:-1] += np.asarray((i, j, k))[:, None, None, None]
                    colmat = np.ravel_multi_index(colmat, (nx, ny, nz))
                    # transition_matrix.data[ind * n:(ind + 1) * n] = (1 / (dmap * move(dmap, (i, j, k)))).flatten()
                    transition_matrix.data[ind * n:(ind + 1) * n] = (func(move(dmap, (i, j, k)), dmap, beta)).flatten()
                    transition_matrix.row[ind * n:(ind + 1) * n] = rowmat.flatten()
                    transition_matrix.col[ind * n:(ind + 1) * n] = colmat.flatten()
    transition_matrix = transition_matrix.tocsr()
    if normalize:
        # Normalize the transition matrix
        sq = np.squeeze(np.asarray(1 / transition_matrix.sum(axis=0)))
        diagsparse = scipy.sparse.csr_matrix((sq, (np.arange(n), np.arange(n))), shape=(n, n))
        transition_matrix = transition_matrix.dot(diagsparse)
    return transition_matrix


def move(M, direction):
    """
    Get the map to move to the given direction.
    Returns: an array with the same dimensions as M
    To get the respective transition just do:
    1/(M*move(M, direction))
    M: density_map
    direction: Tuple given the direction: (1,0,0), (0,1,0), (0,0,1), (-1,0,0),
    ..., (0,0,-1)
    """
    nx, ny, nz = M.shape
    if direction[0] == -1:
        # Up
        M = np.concatenate((-np.ones((1, ny, nz)) * np.inf, M[:-1]), axis=0)
    if direction[0] == 1:
        # Down
        M = np.concatenate((M[1:], -np.ones((1, ny, nz)) * np.inf), axis=0)
    if direction[1] == -1:
        # Left
        M = np.concatenate((-np.ones((nx, 1, nz)) * np.inf, M[:, :-1, :]), axis=1)
    if direction[1] == 1:
        # Right
        M = np.concatenate((M[:, 1:, :], -np.ones((nx, 1, nz)) * np.inf), axis=1)
    if direction[2] == -1:
        # Front
        M = np.concatenate((-np.ones((nx, ny, 1)) * np.inf, M[:, :, :-1]), axis=2)
    if direction[2] == 1:
        # Back
        M = np.concatenate((M[:, :, 1:], -np.ones((nx, ny, 1)) * np.inf), axis=2)
    return M


def flood(A, source, level, timing=False):
    """
    Flood a bassin from the given source (i, j, k) until the given level
    """
    t0 = time.perf_counter()
    adj = get(A, beta=1.)
    if timing:
        print(f'Adjmat: {time.perf_counter() - t0}')
    # adj.data = -np.log(adj.data)
    # adj.data -= adj.data.min()
    start = np.ravel_multi_index(source, A.shape)
    cell = start
    visited_cell = set([cell, ])
    t0 = time.perf_counter()
    D = scipy.sparse.csgraph.dijkstra(adj, indices=list(visited_cell), min_only=True)
    if len(D) < level + 1:
        limit = np.inf
    else:
        limit = np.sort(D)[level + 1]
    for i in range(level):
        D = scipy.sparse.csgraph.dijkstra(adj, indices=list(visited_cell), min_only=True, limit=limit)
        D = np.atleast_2d(D)
        D[:, tuple(visited_cell)] = np.inf
        D_shape = D.shape
        D = D.flatten()
        cell = np.argmin(D)
        cell = np.unravel_index(cell, D_shape)[1]
        visited_cell.add(cell)
    if timing:
        print(f'Flooding: {time.perf_counter() - t0}')
    blob = np.zeros_like(A)
    blob = blob.flatten()
    blob[list(visited_cell)] = 1.
    blob = blob.reshape(A.shape)
    return blob


if __name__ == '__main__':
    import argparse
    import gaussian_grid3
    import matplotlib.pyplot as plt
    import mrc
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    args = parser.parse_args()

    shape = (30, 20, 22)
    A = gaussian_grid3.random(ncenters=10, scale=5, shape=shape)
    # A = np.random.uniform(0, 1, size=(10, 10, 10))
    source = tuple(np.asarray(shape) // 2)
    blob = flood(A, source=(5, 5, 5), level=150)
    mrc.save_density(blob, 'blob.mrc')
    # plt.matshow(D)
    # plt.colorbar()
    # plt.show()
