import numpy as np
import scipy as sp
from scipy.spatial import distance_matrix
import scipy.ndimage as im

import os
import sys

import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
from multiprocessing import Pool

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from data_processing import utils


def zero_grid(grid, labels=None, label=None, min_cutoff=None):
    """
    Compute a modified grid that selects residues from a given basin or over a min cutoff by setting all others to zero
    :param grid:
    :param label:
    :param min_cutoff:
    :return:
    """
    if label is not None:
        grid = grid * (labels == label)
    if min_cutoff is not None:
        grid = grid * (grid > min_cutoff)
    return grid


def num_feat(grid, prob_value):
    """
    Takes a grid and returns the number of 'blobs' that are above a certain threshold : prob_value

    Can optionnaly use different patterns for merging than the default
    :param grid:
    :param prob_value:
    :return:
    """
    zeroed = zero_grid(grid, min_cutoff=prob_value)
    return im.label(zeroed)[1]
    # return im.label(zeroed, structure=np.ones((3, 3, 3)))[1]


def get_neighbours(p, exclude_p=True, shape=None):
    """
    Get the n-d grid neighbors of a pixel
    :param p:
    :param exclude_p:
    :param shape:
    :return:
    """
    ndim = len(p)
    offset_idx = np.indices((3,) * ndim).reshape(ndim, -1).T
    offsets = np.r_[-1, 0, 1].take(offset_idx)
    if exclude_p:
        offsets = offsets[np.any(offsets, 1)]
    neighbours = p + offsets  # apply offsets to p
    if shape is not None:
        valid = np.all((neighbours < np.array(shape)) & (neighbours >= 0), axis=1)
        neighbours = neighbours[valid]
    return neighbours


def find_neighbors_line(coords_line, distrib_line, grid, labels):
    """
    Get neighborhoods to merge contiguous blobs

    :param coords_line: The coords of the line to follow shape (n,3)
    :param distrib_line: The values along that line shape n
    :param grid:
    :param labels:
    :return:
    """

    adjacency_lookup = defaultdict(lambda: -1)

    # Now following this division line we find the highest point of division between any two blobs
    interpolate_missing = list()
    for i, pixel in enumerate(coords_line):
        value_pix = distrib_line[i]
        neighbors = get_neighbours(pixel, shape=grid.shape)
        neighbors_labels = labels[neighbors.T[0], neighbors.T[1], neighbors.T[2]]
        neighbors_values = grid[neighbors.T[0], neighbors.T[1], neighbors.T[2]]

        # We need this to fill the line with continuous labels
        pix_label = neighbors_labels[np.argmax(neighbors_values * (neighbors_labels > 0))]
        interpolate_missing.append((pixel, pix_label))

        # Compute adjacency
        uniques = np.unique(neighbors_labels)
        uniques = set(uniques)
        uniques.remove(0)
        for a, b in itertools.combinations(uniques, r=2):
            if a > b:
                a, b = b, a
            # Don't include in the adjacency table the very low values
            if adjacency_lookup[(a, b)] < value_pix:
                adjacency_lookup[(a, b)] = value_pix
    # This is necessary to pickle the results and use in a multiprocessing fashion
    adjacency_lookup = dict(adjacency_lookup)
    return adjacency_lookup, interpolate_missing


def find_line_proc(args):
    """
    Distribute the line computation over chunks of the list.
    We use chunks and not trivial parallelisation to avoid creating a lot of return dictionaries
    :param args:
    :return:
    """
    chunk_min, chunk_max, coords_line, distrib_line, grid, labels = args
    coords_line_proc, distrib_line_proc = coords_line[chunk_min: chunk_max], distrib_line[chunk_min:chunk_max]
    adjacency_lookup_proc, interpolate_missing_proc = find_neighbors_line(coords_line_proc,
                                                                          distrib_line_proc,
                                                                          grid,
                                                                          labels)
    return adjacency_lookup_proc, interpolate_missing_proc


def filter_by_condition(grid, condition):
    """
    takes a grid and returns the coordinates of the non_zero elements corresponding to a condition
     as well as the associated values
    :param grid:
    :param condition: something like grid >0 : a numpy array of booleans and same size that grid
    :return:
    """
    zeroed = grid * condition
    coords = np.argwhere(zeroed)
    distrib = grid[condition]
    # Additional filtering
    distrib = distrib[distrib > 0]
    return coords, distrib


def find_cutoff_blob(bassin):
    """
    Get the meaningful blob out of a watershed basin :
    Looks at the profile of the number of blobs and look at the maxima as they correspond to having fulled all lower
    proba crevasses. One can comment_uncomment on finding either the lowest of such events or the highest
    :param bassin: (labels == label) * grid
    :return:
    """

    def find_cutoff_list(list_p):
        """
        Get the cutoff parameter for a list of number of blobs
        Intuitively, we start by getting the most probable max that corresponds to overflowing the main crevasse
        :param list_p: list of number of blobs, ordered by growing probability
        :return:
        """
        # Get first one : coherent blob and then move towards the rights
        try:
            index_one = list_p.index(1)
            while index_one < len(list_p) and list_p[index_one] == 1:
                index_one += 1
            return max(0, index_one - 4)
        except ValueError:
            return 0
        # return len(list_p) - 1 - np.argmax(list(reversed(list_p)))

    linspace = np.linspace(0, np.max(bassin), 50)
    linvalues = [num_feat(bassin, value) for value in linspace]

    proba_cutoff = linspace[find_cutoff_list(linvalues)]
    proba_cutoff = max(proba_cutoff, 0.2)

    # volumes = [np.sum(bassin > value) for value in linspace]
    # print('vols : ', volumes)
    # plt.plot(linspace, linvalues, color='g')
    # plt.plot(linspace, volumes, color='b')
    # plt.axvline(proba_cutoff)
    # plt.show()

    return proba_cutoff


def crop_blob_parallel(args):
    label, labels, grid = args
    bassin = ((labels == label) * grid)
    blob_cutoff = find_cutoff_blob(bassin=bassin)
    return blob_cutoff


def score_blob(score_array, volume=150):
    """
    just  a placeholder operation to have easier modification
    :param score_array: a sorted array
    :return:
    """
    # return score_array[-volume]
    return np.mean(score_array[-volume:])


def filter_blob(grid_blob, volume=150, global_min=0, global_max=1):
    """
    To remove very small or very low probability blobs
    :param grid_blob:
    :param volume:
    :param proba_min:
    :return: Boolean
    """
    selection = grid_blob > 0
    volume_blob = np.sum(selection)
    if volume_blob < volume:
        return 0

    selected_distrib = grid_blob[selection]
    selected_distrib = np.sort(selected_distrib)
    sel_value = score_blob(selected_distrib, volume=volume)

    normalized_sel_value = (sel_value - global_min) / (global_max - global_min)
    # print(f"{volume_blob} A, mean value at 150 is {sel_value}, normalized is {normalized_sel_value}")

    return normalized_sel_value


def to_blobs(grid,
             vol_min=150,
             simultaneous=False,
             hetatm=False,
             min_merging_value_hd=0.95,
             min_merging_value_pl=0.675,
             min_euclidean_distance_hd=1,
             min_euclidean_distance_pl=1,
             max_euclidean_distance_hd=20,
             max_euclidean_distance_pl=15,
             blob_score_min=0.1,
             serial=False,
             use_p_min=False):
    """
        For each watershed basin, find a proba cutoff and then turns it into a small blob. The steps are as follow :

    Computes watershed with a min distance between max of min_euclidean distance.

    Then finds both the global maximum and a global min value used to normalize the further processings.
    The min value is approximately the one before the maximum number of disjoint blobs.

    Then some basin need to get merged if they are neighbors and have only a small energy barrier between them
    To do so, an adjacency matrix is computed. This computation relies on following the lines that delimit the
    basins and linking neighboring pixels.
    This computation is quite long and computed in parallel

    Edge weights are then normalized with the global min/max values.
    Then two algorithms are implemented to make the graph coarser :
    - simultaneous just groups all nodes that are closer to a cutoff.
    - iterative merge the two closest components if the blob that it creates does not merge maximas that are
    further away than max_euclidean_distance.

    Finally, for each of those merged blobs, we find a cutoff that best suits it by applying watershed again
    and stopping when it yields one connected component. Each blob then is zeroed beyond that threshold and scored
    Blobs that end up being too small or having too weak a score are discarded.

    The blobs are returned in a sparse matrix format, as lists of coordinates and values, ordered by their score

    The longest steps are following the split line and finding the watershed cutoff values in the end.
    For small systems, following the line is the longest but for big one, the last is much longer (quadratic as one
    need to do more blob finding operations on a bigger grid)

    :param grid: the proba values
    :param vol_min: min volume of a blob
    :param simultaneous: The algorithm to follow for merging the blobs
    :param min_merging_value_hd: The minimum normalized value for conducting a merge operation
    :param min_merging_value_pl: same for pl
    :param min_euclidean_distance_hd: The distance below which we consider two maximas to belong to the same blob
    :param min_euclidean_distance_pl: same for pl
    :param max_euclidean_distance_hd: The distance beyond which two neighbors cannot be merged
    :param max_euclidean_distance_pl: same for pl
    :param blob_score_min: The minimum score of a blob allowed
    :param serial: Whether to use parallelism for faster computations
    :param use_p_min: A speedup that ignores the threshold per blob computation and just uses a fixed one.
    :return: list of coordinates, the corresponding grid values and their id and label attributions
    """

    min_merging_value = min_merging_value_hd if not hetatm else min_merging_value_pl
    min_euclidean_distance = min_euclidean_distance_hd if not hetatm else min_euclidean_distance_pl
    max_euclidean_distance = max_euclidean_distance_hd if not hetatm else max_euclidean_distance_pl

    # import time
    # time_total = time.perf_counter()
    # time1 = time.perf_counter()

    # Perform the watershed and check if the output contains something.
    labels, pmax, pmax_locs = utils.watershed(grid, min_distance=min_euclidean_distance)
    if labels is None:
        return None, None, None
    label_uniques = np.unique(labels[labels > 0])
    global_max = max(pmax)

    dm_pmax_locs = distance_matrix(pmax_locs, pmax_locs)

    # # Get min relevant value above which we mostly have one blob
    # # First cut on volume : we want to avoid absurd situation so we keep 20% of the total mass
    linspace = np.linspace(0, np.max(grid), 50)
    volume_ratio = np.array([np.sum(grid >= value) / grid.size for value in linspace])
    fifth_ratio = linspace[np.argmax(volume_ratio < 0.2)]

    # print('volume ratio', fifth_ratio)

    # enveloppe
    linspace = np.linspace(0, np.max(grid), 50)
    non_zero_grid = grid[grid > 0]
    volume_ratio = np.array([np.sum(non_zero_grid >= value) / non_zero_grid.size for value in linspace])
    fifth_ratio_env = linspace[np.argmax(volume_ratio < 0.4)]

    # print('env ratio', fifth_ratio_env)
    # The current intuition is that this has the same global_min in the end.

    # plt.hist(grid[grid > 0], bins=100, cumulative=True)
    # plt.show()
    # sys.exit()
    # print(linspace[1])
    # print(fifth_ratio)
    # print(grid.size)
    # print(len(grid[grid > 0]))
    # new_min = np.min(grid[grid > 0])
    # print(new_min)
    # plt.plot(linspace, volume_ratio)
    # plt.axvline(fifth_ratio, color='r')
    # plt.axvline(new_min, color='g')
    # plt.show()

    # We now select on the number of blobs : We want the argmax before the peak
    # Sometimes this value would fall below the fifth ratio, we then take the fifth ratio as global_min
    try:
        linspace = np.linspace(fifth_ratio, np.max(grid), 50)
        linvalues = np.array([num_feat(grid, value) for value in linspace])
        num_blob_thresh = max(linvalues) / 4 + 1
        amax = np.argmax(linvalues)
        selected_linvalues = linvalues[:amax]
        temp_reversed = selected_linvalues[::-1]
        index = len(temp_reversed) - np.argmax(temp_reversed < num_blob_thresh) - 1
        global_min = linspace[index]
    except ValueError:
        global_min = fifth_ratio

    # print("global min is ", global_min)

    # plt.plot(linspace, linvalues)
    # plt.axvline(fifth_ratio, color='r')
    # plt.axvline(fifth_ratio_env, color='g')
    # plt.axvline(global_min, color='b')
    # plt.show()

    # print(global_min)
    # print(global_max)
    # print("time before the line : ", time.perf_counter() - time1)
    # time1 = time.perf_counter()

    coords_line = np.argwhere(labels == 0)
    distrib_line = grid[labels == 0]
    if serial:
        adjacency_lookup, interpolate_missing = find_neighbors_line(coords_line,
                                                                    distrib_line,
                                                                    grid=grid,
                                                                    labels=labels)
    else:
        # Chunk the input line in smaller pieces for parallel processing.
        # It's a bit tricky to deal with integer division
        N = len(coords_line)
        num_procs = os.cpu_count()
        chunk_size, rab = N // num_procs, N % num_procs
        chunk_bounds = [(proc_id * (chunk_size + 1), min((proc_id + 1) * (chunk_size + 1), N))
                        for proc_id in range(rab)]
        last = (chunk_size + 1) * rab
        chunk_bounds_2 = [(last + proc_id * chunk_size, min(last + (proc_id + 1) * chunk_size, N))
                          for proc_id in range(num_procs - rab)]
        chunk_bounds.extend(chunk_bounds_2)
        parallel_input = [(chunk_min, chunk_max, coords_line, distrib_line, grid, labels)
                          for chunk_min, chunk_max in chunk_bounds]

        # Then feed it to workers
        with Pool(num_procs) as p:
            all_res = p.map(find_line_proc, parallel_input)
            all_adjacency, all_interpolate_missing = zip(*all_res)
            interpolate_missing = list(itertools.chain(*all_interpolate_missing))
            adjacency_lookup = defaultdict(lambda: -1)
            for adjacency_proc in all_adjacency:
                for (a, b), value in adjacency_proc.items():
                    if adjacency_lookup[(a, b)] < value:
                        adjacency_lookup[(a, b)] = value

    # Fill out the zero line with the highest non-zero neighbor
    for (x, y, z), v in interpolate_missing:
        labels[x, y, z] = v
    # print("time for the line : ", time.perf_counter() - time1)
    # time1 = time.perf_counter()

    # bridges = sorted(adjacency_lookup.values())
    # print('bridges', bridges)
    # plt.hist(bridges)
    # plt.show()
    # print(sorted(adjacency_lookup.items(), key=lambda x: x[1]))

    # Now this highest value should be compared to the max values of the blobs to be merged
    # After normalization low scores ==> should be merged
    for (a, b), value in adjacency_lookup.items():
        # maxa, maxb = max_label[a], max_label[b]
        # minmax, maxmin = min(maxa, maxb), max(maxa, maxb)
        # delta = (minmax - value) / global_max
        # delta = (minmax - value)
        # delta = (minmax - value) / maxmin
        # delta1 = (minmax - value) / (global_max - global_min)
        delta2 = (global_max - value) / (global_max - global_min)
        adjacency_lookup[(a, b)] = delta2
    # print('adjusted', sorted(adjacency_lookup.values()))
    # print('adjusted', sorted(adjacency_lookup.items(), key=lambda x: x[1]))
    # print('labels before merging', label_uniques)

    # # Sparsify : we don't need to keep the values greater than the merging ones
    to_remove = []
    for key, value in adjacency_lookup.items():
        if value > min_merging_value:
            to_remove.append(key)
    for key in to_remove:
        del adjacency_lookup[key]

    # Now merge blobs
    if len(adjacency_lookup) > 0:
        if simultaneous:
            # We have to organize the blobs into sets of unique blobs,
            # because if we merge 3 blobs, there could be a problem

            # This has a limitation though : we sometimes end up with all blobs getting merged at once,
            # which calls for the non-simultaneous, iterative version of the algorithm.
            new_labels = {frozenset([i]): i for i in label_uniques}
            for (a, b), value in adjacency_lookup.items():
                if value < min_merging_value:
                    seta, setb = set(), set()
                    for fset in new_labels:
                        if a in fset:
                            seta = fset
                        if b in fset:
                            setb = fset
                    new_fset = seta.union(setb)
                    if seta != new_fset:
                        mina, minb = new_labels[seta], new_labels[setb]
                        del new_labels[seta]
                        del new_labels[setb]
                        new_labels[new_fset] = min(mina, minb)
            for labelset, labelvalue in new_labels.items():
                if len(labelset) == 1:
                    pass
                else:
                    for old_label in labelset:
                        labels[labels == old_label] = labelvalue
        else:
            value = 0
            list_edges = sorted(adjacency_lookup.items(), key=lambda x: -x[1])
            # print(list_edges)
            while value < min_merging_value and len(list_edges) > 0:
                (a, b), value = list_edges.pop()
                # Just for the first iteration, if we entered with excessive value.
                if value > min_merging_value:
                    break

                euclidean_distance = dm_pmax_locs[a - 1, b - 1]
                if euclidean_distance < max_euclidean_distance:
                    # merge by collapsing labels, updating distance matrix and adjacency values
                    labels[labels == b] = a

                    # This is sufficient as we only ever look at values a<b
                    dm_pmax_locs[a - 1] = np.max((dm_pmax_locs[a - 1], dm_pmax_locs[b - 1]), axis=0)

                    # To be explored if slow :
                    # We could make this linear by adding all neighbors at once
                    # and then looking in the list just once... need to time but I think it's going to be fast
                    # The best option would be to use an actual adjacency matrix and update it in a similar way
                    # to the dm
                    to_remove = set()
                    to_add = list()
                    for (k, l), value_bneigh in list_edges:
                        if b in (k, l):
                            # In any case we will disconnect the edge between b and neigh
                            to_remove.add(((k, l), value_bneigh))
                            neigh = k if b == l else l
                            # We have to order them properly to search for it
                            m, n = (neigh, a) if neigh < a else (a, neigh)
                            found_common = False
                            for (u, v), value_aneigh in list_edges:
                                if (m, n) == (u, v):
                                    found_common = True
                                    # We have found a common neighbor !
                                    # if b is closer (which means smaller here) than a, replace the value
                                    if value_bneigh < value_aneigh:
                                        to_remove.add(((m, n), value_aneigh))
                                        to_add.append(((m, n), value_bneigh))
                            # if only b is linked to this neighbor, now the neighbor is linked to a
                            if not found_common:
                                to_add.append(((m, n), value_bneigh))
                    updated_list = [elt for elt in list_edges if elt not in to_remove] + to_add
                    list_edges = sorted(updated_list, key=lambda x: -x[1])

    label_uniques = np.unique(labels[labels > 0])
    # print('labels after merging', label_uniques)
    # print("time before blob cutting: ", time.perf_counter() - time1)
    # time1 = time.perf_counter()
    all_coords, all_distribs, all_labels, all_ids = list(), list(), list(), list()
    contiguous_index = 1
    score_mapping = list()

    if use_p_min:
        all_cutoffs = [global_min for _ in range(len(label_uniques))]
    else:
        # This step can be run in parallel mode to be faster as it is the bottleneck
        if serial:
            all_cutoffs = list()
            for label in label_uniques:
                bassin = ((labels == label) * grid)
                blob_cutoff = find_cutoff_blob(bassin=bassin)
                all_cutoffs.append(blob_cutoff)
        else:
            parallel_input = [(label, labels, grid) for label in label_uniques]
            with Pool() as p:
                all_cutoffs = p.map(crop_blob_parallel, parallel_input)

    for i, label in enumerate(label_uniques):
        bassin = ((labels == label) * grid)
        blob_cutoff = all_cutoffs[i]
        grid_blob = zero_grid(bassin, min_cutoff=blob_cutoff)
        blob_score = filter_blob(grid_blob, volume=vol_min, global_min=global_min, global_max=global_max)
        if blob_score > blob_score_min:
            # This step is only a flattening of the blob into sparse format
            coords, distrib = filter_by_condition(grid_blob, grid_blob > 0)
            score_mapping.append(blob_score)
            all_coords.extend(coords)
            all_distribs.extend(distrib)
            all_ids.extend([contiguous_index] * len(coords))
            contiguous_index += 1

    # reorder the indices using the value at 150 instead of the pmax
    all_ids = np.array(all_ids)
    score_mapping = np.array(score_mapping)
    new_all_ids = np.zeros_like(all_ids)
    argsort_old = np.argsort(-score_mapping)
    for new_id, old_id in enumerate(argsort_old):
        new_all_ids[all_ids == old_id + 1] = new_id + 1
    all_ids = new_all_ids
    # print("final", np.unique(all_ids))
    # print("time for blob cutting : ", time.perf_counter() - time1)
    # print("total blobbing time : ", time.perf_counter() - time_total)
    if len(all_coords) == 0:
        return None, None, None
    all_coords, all_distribs, all_ids = np.stack(all_coords, axis=0), np.array(all_distribs), np.array(all_ids)
    return all_coords, all_distribs, all_ids


if __name__ == '__main__':
    pass
    import mrcfile

    print('HD')
    # a = np.load("hd_preds.npz")
    # grid = a["grid"]
    # a = mrcfile.open("hd.mrc")
    a = mrcfile.open("pymol_ALL.mrc")
    grid = a.data.T
    to_blobs(grid)

    # print()
    # print('PL')
    # a = np.load("pl_preds.npz")
    # grid = a["grid"]
    # a = mrcfile.open("1t4e_pl_buggy.mrc")
    # a = mrcfile.open("1t4e_pl.mrc")
    # grid = a.data
    # to_blobs(grid)


def npz_to_blob(npzfile_name, overwrite=False):
    """
    Computes blobbing over raw output grids
    :param npzfile_name:
    :param overwrite:
    :return:
    """
    assert isinstance(npzfile_name, str)
    npzfile = np.load(npzfile_name)

    # If some blobs were already written and we don't want to overwrite them
    if not overwrite:
        try:
            npzfile['all_coords']
            return
        except KeyError:
            pass

    grid = np.squeeze(npzfile['hd'])
    grid = grid[-1]
    grid = 1. - grid
    hd_coords, hd_distribs, hd_ids = to_blobs(grid)
    out_pl = npzfile['pl']
    grid = out_pl.squeeze()
    pl_coords, pl_distribs, pl_ids = to_blobs(grid)
    np.savez_compressed(npzfile_name,
                        hd=npzfile['hd'],
                        pl=npzfile['pl'],
                        origin=npzfile['origin'],
                        hd_coords=hd_coords,
                        hd_distribs=hd_distribs,
                        hd_ids=hd_ids,
                        pl_coords=pl_coords,
                        pl_distribs=pl_distribs,
                        pl_ids=pl_ids)


if __name__ == '__main__':
    pass
    # grid, labels, origin = get_grid_labels('results/evaluations/P00533_3w32_4i21.npz')
    # grid, labels, origin = get_grid_labels('results/evaluations/P62826_3gj0_3gj5.npz')
    # print((labels == 1).shape)
    # utils.save_density(grid, 'tmp_all.mrc', 1, origin, 6)
    # utils.save_density((labels == 1) * grid, 'tmp2.mrc', 1, origin, 6)
    # utils.save_density((labels == 37) * grid, 'tmp37.mrc', 1, origin, 6)

    # coords, distrib = filter_by_condition(grid, (labels == 1))
    # print(coords.shape, distrib.shape)
