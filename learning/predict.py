#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-12-07 14:03:04 (UTC+0100)


import os
import sys

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import h5py
import numpy as np
import pickle
from tqdm import tqdm
from collections.abc import Iterable
from pymol import cmd
from sklearn.neighbors import KDTree
import time
from pymol import cmd
import psico.fullinit
import psico.helping
import scipy.ndimage
import skimage.segmentation
from skimage.feature import peak_local_max
import scipy.spatial.distance

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from data_processing import Density, Complex, pytorch_loading, utils
from learning import utils as lutils, model_factory
from post_processing import blobber
from post_processing import adjmat


def filter_outgrid_enveloppe(grid, enveloppe, pl=False):
    """
    Takes squeezed grids and filter them on an enveloppe
    :param grid: [5/1,x,y,z]
    :param enveloppe: [x,y,z]
    :param pl:
    :return:
    """
    if pl:
        return grid * enveloppe
    else:
        for i in range(grid.shape[0] - 1):
            grid[i] = grid[i] * enveloppe
        grid[-1] = np.ones_like(grid[-1]) * (1 - enveloppe) + enveloppe * grid[-1]
        return grid


def predict_frame(model, grid, device=None):
    """
    Predict on a given grid with a given model
    :param model: torch model with 'inference_call' method
    :param grid: numpy grid
    :param device: to force putting the model on a specific device
    :return:
    """
    # Make the prediction
    torch_grid = torch.from_numpy(grid)
    torch_grid = torch_grid.float()[None, ...]
    if device is not None:
        model.to(device)
        torch_grid = torch_grid.to(device)
    else:
        if torch.cuda.is_available():
            model.cuda()
            torch_grid = torch_grid.cuda()

    # traced_script_module = torch.jit.trace(self.model, torch_grid)
    # traced_script_module.save("pl_model.pt")

    out_hd, out_pl = model.inference_call(torch_grid)
    out_hd, out_pl = out_hd.detach().cpu().numpy(), out_pl.detach().cpu().numpy()
    out_hd, out_pl = out_hd.squeeze(), out_pl.squeeze()
    return out_hd, out_pl


def predict_pdb(model, pdbfile,
                selection=None,
                pdbfile_lig=None,
                selection_lig=None,
                hetatm=False,
                spacing=1.,
                padding=8,
                xyz_min=None,
                xyz_max=None,
                enveloppe=True,
                device=None):
    """

    Makes a model prediction from a pdb.
    Returns the prediction for the hd branch, the pl branch and the origin of the grid.
    If there is a ligand, aligns the grid to include it and return a ligand mrc that looks like the prediction

    :param model: A pytorch model
    :param pdbfile: A pdb to be opened
    :param selection: A pymol selection to run the prediction on, instead of the whole protein
    :param pdbfile_lig: A pdb to use as a ligand/target for instance in the HDPL database
    :param selection_lig: A selection to use for the ligand. Is pdbfile_lig is empty,
    the ligand selection is done on 'pdbfile'
    :param hetatm: A flag to indicate that the ligand is not a protein but a small molecule
    :param spacing: The spacing of the grid
    :param padding: ... and its padding
    :param xyz_min: For usage with a fixed box, a tuple of the bottom corner of the box, the origin of an mrc
    :param xyz_max: For usage with a fixed box, a tuple of the top corner of the box
    :param enveloppe: A flag to compute enveloppe around the input and zero predictions outside of this enveloppe.
    :return: out_hd, out_pl : two grids in the squeezed output format [1/6,x,y,z],
                                xyz_min, lig_grid if there is a ligand in the same [1/6,x,y,z] format
    """
    # We can instantiate the object either with 2 pdbs (Olivier and Karen setting)
    # or with a single one with two different selections (usual setting)
    if pdbfile_lig is None and selection_lig is not None:
        pdbfile_lig = pdbfile
    density = Density.Coords_channel(pdbfilename_p=pdbfile, pdbfilename_l=pdbfile_lig)
    coords = density.split_coords_by_channels(selection=selection)
    assert coords.ndim == 2
    assert coords.shape[-1] == 4

    # If we also load a ligand, we then need to create coords_all to compute a common grid
    coords_all = coords  # The size of the grid box is defined as the min max of the protein coordinates
    if pdbfile_lig is not None or selection_lig is not None:
        coords_lig = density.split_coords_by_channels(selection=selection_lig, ligand=True, hetatm=hetatm)
        # coords_all = np.r_[coords, coords_lig] # The size of the grid box is defined as the min max of the protein+ligand coordinates
    if xyz_min is None:
        xyz_min = coords_all[:, :3].min(axis=0)
    if xyz_max is None:
        xyz_max = coords_all[:, :3].max(axis=0)
    grid = Complex.get_grid_channels(coords, spacing=spacing, padding=padding,
                                     xyz_min=xyz_min, xyz_max=xyz_max, is_target=False)

    # Make the prediction
    out_hd, out_pl = predict_frame(model=model, grid=grid, device=device)

    if enveloppe:
        out_enveloppe = Complex.get_mask_grid_prot(grid)
        out_hd = filter_outgrid_enveloppe(out_hd, out_enveloppe, pl=False)
        out_pl = filter_outgrid_enveloppe(out_pl, out_enveloppe, pl=True)

    # Optionnaly also return a grid corresponding to the ligand
    if pdbfile_lig is not None or selection_lig is not None:
        lig_grid = Complex.get_grid_channels(coords_lig, spacing=spacing, padding=padding, xyz_min=xyz_min,
                                             xyz_max=xyz_max, is_target=True)
        if enveloppe:
            lig_grid = filter_outgrid_enveloppe(lig_grid, out_enveloppe, pl=hetatm)

        return out_hd, out_pl, xyz_min, lig_grid.squeeze()
    else:
        return out_hd, out_pl, xyz_min, None


def get_pocket_coords(pockets, padding=8, spacing=1, vol=150, hd=True, small_blob_cutoff=50, return_channels=False):
    """
    Read a npz and return its blobs.
    :param pockets:
    :param padding:
    :param spacing:
    :param vol:
    :param hd:
    :param small_blob_cutoff:
    :param return_channels:
    :return:
    """
    if return_channels and not hd:
        raise ValueError('Cannot compute channels on a pl prediction')
    vol = np.inf if vol is None else vol
    hd_string = 'hd' if hd else 'pl'
    pockets_coords, pockets_probs = list(), list()
    for pocket_label in range(1, pockets[f'{hd_string}_ids'].max() + 1):
        sel = pockets[f'{hd_string}_ids'] == pocket_label
        offset = pockets['origin'] - padding
        grid_coords = pockets[f'{hd_string}_coords'][sel]
        coords = (grid_coords * spacing + offset)

        if not return_channels:
            probs = pockets[f'{hd_string}_distribs'][sel]
        else:
            # We need to go back to the full grid, and we swap it for easy extraction.
            grid_hd = pockets['hd']
            grid_hd = np.swapaxes(grid_hd[..., None], axis1=0, axis2=-1).squeeze()
            grid_coords = grid_coords.T
            selected_vectors = grid_hd[grid_coords[0], grid_coords[1], grid_coords[2]]
            probs = 1. - selected_vectors[:, -1]
            # other_probs = pockets[f'{hd_string}_distribs'][sel]
            # print(np.max(probs-other_probs))
        # If we have less points in the blob than asked, we will return a smaller blob.
        # If this blob is less than small_blob_cutoff, we just skip it.
        cutoff = vol
        if cutoff > len(coords):
            cutoff = len(coords)
            if cutoff < small_blob_cutoff:
                continue
        threshold = probs[probs.argsort()][::-1][cutoff - 1]
        sel = probs > threshold
        coords = coords[sel]
        pockets_coords.append(coords)
        if not return_channels:
            probs = probs[sel]
            pockets_probs.append(probs)
        else:
            selected_vectors = selected_vectors[sel]
            pockets_probs.append(selected_vectors)
    return pockets_coords, pockets_probs


# def v_1(model):
#     score_1 = score(model, "1ycr.cif", selection="chain A", selection_lig="chain B", pl=False)[0]
#     score_2 = score(model, "1t4e.cif", selection="chain A", selection_lig="chain A and resname DIZ", pl=True)[0]
#     return (score_2 + score_1) / 2


def string_blobs(blobs, blobs_channel, P='protein_input', L='ligand', nres=None, time_pred=0, time_metric=0):
    """
    Takes a blobs 'metrics' list and parse it; computing aggregated metrics on it and organizing it into a string.
    Also returns an aggregated metric score
    :param blobs: the output of lutils.get_metrics_blob
    :param blobs_channel: the output of lutils.get_metrics_blob, contains info on the blob channels if HD
    :param P: Name of the protein
    :param L: Name of the ligand
    :param nres: Number of residues
    :param time_pred: Time for prediction
    :param time_metric: Time for metrics computation
    :return: String representing all blobs
    """
    return_string_list = []
    # Print for each blob :
    return_string_list.append(f"P: {P}\n")
    return_string_list.append(f"L: {L}\n")
    # Default surface yields small like system
    surface = 20
    if nres is not None:
        return_string_list.append(f"nres: {nres}\n")
        surface = nres ** (2 / 3)
        return_string_list.append(f"proxy_surface: {surface:.2f}\n")
    try:
        # if True:
        nblobs = len(blobs)
        pl = len(blobs_channel[0]) == 0
        hdpl_tag = 'PL' if pl else 'HD'
        local_blob_return_string_list = list()
        overlaps = []
        distance_sim = []
        jaccards = []
        for rank, blob in enumerate(blobs):
            overlaps.append(blob.metrics['overlap_local_150'])
            distance_sim.append(blob.metrics['distance_sim'])
            jaccards.append(blob.metrics['jaccard_global'])
            local_blob_return_string_list.append(f"del:=====\n")
            local_blob_return_string_list.append(f"blob: {rank + 1}\n")
            local_blob_return_string_list.append(blob.__str__())

        # Retrieve the system metrics in arrays
        local_aggregated_return_string_list = list()
        overlaps = np.asarray(overlaps)
        distance_sim = np.asarray(distance_sim)
        dist_overlap = (distance_sim + overlaps) / 2
        jaccards = np.asarray(jaccards)

        # Add aggregated metrics for the whole system
        maxoverdist = np.max(dist_overlap)
        argmaxoverdist = np.argmax(dist_overlap)
        argmaxoverdist_sim = np.exp(-argmaxoverdist * 20 / surface)
        rankoverdist = np.sum(
            np.asarray([((nblobs - i - 1) / nblobs) * o for i, o in enumerate(dist_overlap)]))
        rankoverdist_corrected = np.sum(
            np.asarray([((nblobs - i) / nblobs) * o for i, o in enumerate(dist_overlap)]))
        rankoverdist_squared = np.sum(
            np.asarray([((nblobs - i) / nblobs) ** 2 * o for i, o in enumerate(dist_overlap)]))
        jacsel = np.sum(dist_overlap * jaccards) / dist_overlap.sum()
        nblobs_metric = - nblobs / surface
        aggmetrics = maxoverdist + rankoverdist_squared + argmaxoverdist_sim + jacsel * 5 + nblobs_metric

        # Add to return string
        local_aggregated_return_string_list.append(f"tag: {hdpl_tag}\n")
        local_aggregated_return_string_list.append(f"nblobs: {nblobs}\n")
        local_aggregated_return_string_list.append(f"rankoverdist: {rankoverdist}\n")
        local_aggregated_return_string_list.append(f"rankoverdist_corrected: {rankoverdist_corrected}\n")
        local_aggregated_return_string_list.append(f"rankoverdist_squared: {rankoverdist_squared}\n")
        local_aggregated_return_string_list.append(f"jacsel: {jacsel}\n")
        local_aggregated_return_string_list.append(f"maxoverdist: {maxoverdist}\n")
        local_aggregated_return_string_list.append(f"argmaxoverdist: {argmaxoverdist}\n")
        local_aggregated_return_string_list.append(f"argmaxoverdist_sim: {argmaxoverdist_sim}\n")
        local_aggregated_return_string_list.append(f"nblobs_metric: {nblobs_metric}\n")
        if time_metric > 0:
            local_aggregated_return_string_list.append(f"time_metric: {time_metric}\n")
        if time_pred > 0:
            local_aggregated_return_string_list.append(f"time_pred: {time_pred}\n")

        # now iterate over blob channels

        if not pl:
            # for i, blob in enumerate(blobs_channel):
            #     for j, blob_channel in enumerate(blob):
            #         return_string_list.append('del: -----\n')
            #         return_string_list.append(f"channel: {j}\n")
            #         return_string_list.append(blob_channel.__str__())

            blob_to_look_for = blobs_channel[argmaxoverdist]
            local_max = []
            for j, blob_channel in enumerate(blob_to_look_for):
                local_blob_return_string_list.append('del: -----\n')
                local_blob_return_string_list.append(f"channel: {j}\n")
                local_blob_return_string_list.append(blob_channel.__str__())
                local_max.append(blob_channel.metrics[f'overlap_local_150_{blob_channel.channel}'])
            channel_score_max = np.max(local_max)
            channel_score_mean = np.mean(local_max)
            local_aggregated_return_string_list.append(f"channel_score_mean: {channel_score_mean}\n")
            local_aggregated_return_string_list.append(f"channel_score_max: {channel_score_max}\n")
            aggmetrics += channel_score_max

        local_aggregated_return_string_list.append(f"aggmetrics: {aggmetrics}\n")
        return_string_list.extend(local_aggregated_return_string_list)
        return_string_list.extend(local_blob_return_string_list)

    except Exception as e:
        # else:
        print(f"Error for {P}-{L}: {e}")
        return_string_list.append("status: failed\n")
        aggmetrics = 1.5
        pass
    return_string_list.append("\n")
    score = aggmetrics
    return ''.join(return_string_list), score


def count_nres(pdbname, selection=None):
    """
    small utils to count the number of residues in a pdb, using pymol
    :param pdbname:
    :return:
    """
    cmd.load(pdbname, 'rescount')
    selection = 'rescount and name CA' if selection is None else f'rescount and name CA and {selection}'
    n = cmd.count_atoms(selection)
    cmd.delete('rescount')
    return n


def v_2(model,
        maxsystems=None,
        outmetricfilename=None,
        hd_txt='../data/HD-database_validation_set.txt',
        hd_dir='../data/HD-database',
        pl_txt='../data/PL-database_validation_set.txt',
        pl_dir='../data/PL-database',
        print_blobs=False,
        blobber_params={},
        device=None,
        no_PL=False,
        no_HD=False):
    """

    :param model:
    :param maxsystems:
    :param outmetricfilename:
    :param hd_txt: list of systems in txt format
    :param hd_dir: hdf5 database of the corresponding pdbs
    :param pl_txt: list of systems in txt format
    :param pl_dir: hdf5 database of the corresponding pdbs
    :param print_blobs: Boolean to print the blobs
    :param blobber_params: Dict_options for model optimization
    :return:
    """

    HD_list = lutils.read_dbfile(hd_txt, hd_dir)
    PL_list = lutils.read_dbfile(pl_txt, pl_dir)
    all_scores = list()
    for i, inlist in enumerate([HD_list, PL_list]):
        hetatm = [False, True][i]
        # Added option to just compute the values on HD/PL
        if no_HD and not hetatm:
            continue
        if no_PL and hetatm:
            continue
        for P, L in inlist[:maxsystems]:
            nres = count_nres(P)
            if nres < 25:
                continue

            # a = time.perf_counter()
            out_hd, out_pl, xyz_min, lig_grid = predict_pdb(model=model,
                                                            pdbfile=P,
                                                            pdbfile_lig=L,
                                                            hetatm=hetatm,
                                                            device=device)

            # time_pred = time.perf_counter() - a
            # print("time for prediction: ", time_pred)

            # a = time.perf_counter()
            out_grid = out_hd if not hetatm else out_pl
            blobs, blobs_channel = lutils.get_metrics_blob(lig_grid,
                                                           out_grid,
                                                           hetatm=hetatm,
                                                           blobber_params=blobber_params)
            # time_metric = time.perf_counter() - a
            # print("time for metrics computing : ", time_metric)

            string_res, score_blob = string_blobs(blobs, blobs_channel, P=P, L=L, nres=nres)
            all_scores.append(score_blob)

            if print_blobs:
                print(string_res)
            print(f'Aggmetrics : {score_blob:3f}')
            if outmetricfilename is not None:
                with open(outmetricfilename, 'w') as outfile:
                    outfile.write(string_res)
                    outfile.flush()
    final_score = np.mean(all_scores)
    return final_score


def predict_file(model, data_file, callbacks=None, dump_path=None, use_true_y=False):
    """

    Runs the prediction on a hdf5 and dumps resulting prediction grids along with a mapping that enables creating
    the correct complex {grid_name : (prot,lig)}

    :param model: a pytorch model
    :param data_file: an hdf5 to compute results on
    :param callbacks: an optional computation to run on the grid. takes the hd and the pl grid as input
                        If not None will be returned as a list of results
    :param dump_path: if not None, the whole prediction is dumped
    :return:
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    if use_true_y:
        dataset = pytorch_loading.HDPLDataset(data_file=data_file, rotate=False)
    else:
        dataset = pytorch_loading.InferenceDataset(data_file=data_file)
    loader = DataLoader(dataset=dataset, num_workers=5)

    # if os.path.exists(os.path.join(dump_path, 'file.h5')):
    #     raise FileExistsError("You almost overwrote the previous prediction !")
    if dump_path is not None:
        os.mkdir(dump_path)
        h5py.File(os.path.join(dump_path, 'prediction.h5'), "w")
    res_callbacks = list() if callbacks is not None else None

    for i, item in enumerate(loader):
        # for i, item in tqdm(enumerate(loader), total=len(loader)):
        if use_true_y:
            grid_prot, grid_lig, is_pl, protlig, enveloppe = item
        else:
            grid_prot, is_pl, protlig, enveloppe = item

        grid_prot = grid_prot.to(device)
        out_hd, out_pl = model.inference_call(grid_prot)

        # Unbatch as its only BS=1 and then remove .pdb... TODO : DIRTY FIX
        prot, lig = protlig
        prot, lig = prot[0][:-4], lig[0][:-4]
        protlig = f"{prot}_INFERENCE_{lig}"

        out_pl = out_pl.cpu().numpy()
        grid_pl = out_pl[0, ..., 0]
        out_hd = out_hd.cpu().numpy()
        grid_hd = np.squeeze(out_hd)
        grid_hd = grid_hd[-1]
        grid_hd = 1. - grid_hd

        if callbacks is not None:
            if use_true_y:
                out_grid = grid_pl if is_pl else grid_hd
                res_callbacks.append((protlig, callbacks(out_grid, grid_lig, is_pl)))
            else:
                res_callbacks.append((protlig, callbacks(out_hd, out_pl)))

        # Serialize the result and save the intermediary mapping
        # np.save(file=os.path.join(dump_path, f"{i}.npy"), arr=grid)
        if dump_path is not None:
            with h5py.File(os.path.join(dump_path, 'prediction.h5'), "a") as f:
                f.create_dataset(protlig + '_hd', data=grid_hd)
                f.create_dataset(protlig + '_pl', data=grid_pl)
    if dump_path is not None:
        with open(os.path.join(dump_path, 'prediction.txt'), 'w') as f:
            for line in res_callbacks:
                protlig_line, callbacks_line = line
                if isinstance(callbacks_line, Iterable):
                    f.write(f"{protlig_line} {' '.join(callbacks_line)}\n")
                else:
                    f.write(f"{protlig_line} {callbacks_line}\n")
    return res_callbacks


def metrics_experiments(experiment, data_file, out_name=None, return_enveloppe=True):
    """
    Just recompute the metrics for a specific experiment and data. Log them in a relog/experiment/ dir
    :param experiment: just the name for instance d_4
    :param data_file:
    :return:
    """
    from torch.utils.tensorboard import SummaryWriter
    from learning.utils import Metrics

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    script_dir = os.path.dirname(os.path.realpath(__file__))
    expfilename = os.path.join(script_dir, '../results/experiments/', f'{experiment}.exp')
    weight_dir = os.path.join(script_dir, '../results/weights/', experiment)
    out_name = experiment if out_name is None else out_name
    log_dir = os.path.join(script_dir, '../results/relog/', out_name)

    # Get the first model
    model = model_factory.model_from_exp(expfilename, load_weights=False)

    writer = SummaryWriter(log_dir)
    metrics_hd_test = Metrics(writer, print_each=None, message='HD', mode="test")
    metrics_pl_test = Metrics(writer, print_each=None, message='PL', mode="test")
    test_metrics = [metrics_hd_test, metrics_pl_test]
    train_dataset, val_dataset, test_dataset = pytorch_loading.Loader(data_file,
                                                                      num_workers=5,
                                                                      rotate=False,
                                                                      return_enveloppe=return_enveloppe).get_data()

    file_list = sorted(os.listdir(weight_dir))
    for epoch, file in enumerate(file_list):
        weights_path = os.path.join(weight_dir, file)
        model.load_state_dict(torch.load(weights_path))
        print(f"done {epoch}/{len(file_list)}")
        model = model.to(device)
        for i, (grid_prot, grid_lig, branch, _, enveloppe) in tqdm(enumerate(test_dataset), total=len(test_dataset)):
            grid_prot, grid_lig, enveloppe = grid_prot.to(device), grid_lig.to(device), enveloppe.to(device)
            out, loss = model.testing_call(grid_prot, grid_lig, branch)
            metric = test_metrics[branch]
            metric.update(loss, grid_lig, out, enveloppe=enveloppe)

        # Aggregate/log the test results and compute mean from pl and hd for early stopping
        metrics_hd_test.print_and_log(epoch)
        metrics_pl_test.print_and_log(epoch)


def get_protein_coords_and_residues(pdbfilename, selection=None):
    cmd.reinitialize()
    cmd.load(pdbfilename, 'inpdb')
    if selection is None:
        pymolsel = 'inpdb and polymer.protein'
    else:
        pymolsel = f'inpdb and polymer.protein and {selection}'
    coords_in = cmd.get_coords(pymolsel)
    pymolspace = {'resids_in': [], 'chains_in': []}
    cmd.iterate(pymolsel,
                'resids_in.append(resi); chains_in.append(chain)',
                space=pymolspace)
    # resids_in = np.int_(pymolspace['resids_in'])
    resids_in = np.asarray(pymolspace['resids_in'])
    chains_in = np.asarray(pymolspace['chains_in'])
    return coords_in, resids_in, chains_in, pymolsel


def aggregate_per_resid(val_per_coords, resids, chains, function, pymolsel, outname=None, return_keys=False):
    """
    Given lists of values, resid and chains, aggregate all values per resids according to function and optionnally
    save a PDB with updated b-factors.
    :param val_per_coords:
    :param resids:
    :param chains:
    :param function:
    :param pymolsel:
    :param outname:
    :param return_keys:
    :return:
    """
    if return_keys:
        val_per_resid = dict()
    else:
        val_per_resid = []
    val_per_coords = np.asarray(val_per_coords)
    for resid, chain in set(zip(resids, chains)):
        sel = np.logical_and(resids == resid, chains == chain)
        val = function(val_per_coords[sel])
        if return_keys:
            val_per_resid[(chain, resid)] = val
        else:
            val_per_resid.append(val)
        cmd.alter(f'{pymolsel} and resi {resid} and chain {chain}', f'b={val}')
    if outname is not None:
        cmd.save(outname, selection=pymolsel)
    return val_per_resid


def project_gridvalues(gridcoords, gridvalues, resids_in, chains_in, coords_in, pymolsel,
                       grid_aggregation_function='max',
                       atom_aggregation_function=np.max, outname=None, radius=3., return_keys=True):
    """
    Project a sparse grid prediction (n,3) and (n,) onto a sequence of residues.
    Returns a dict {(chain, resid): value}
    :param gridcoords:
    :param gridvalues:
    :param resids_in:
    :param chains_in:
    :param coords_in:
    :param pymolsel:
    :param grid_aggregation_function:
    :param atom_aggregation_function:
    :param outname:
    :param radius:
    :param return_keys:
    :return:
    """
    kdtree = KDTree(gridcoords)
    if grid_aggregation_function not in {'max', 'sum', 'onion'}:
        raise ValueError(f'The aggregation function "{grid_aggregation_function}" should be one of max, sum, or onion')

    # Now for each query atom, find the neighbors
    # Then iterate over the atoms and get the max value per atom
    # Finally aggregate atomic values per residue.
    if grid_aggregation_function == 'onion':
        inds_per_coord, dist_per_coords = kdtree.query_radius(X=coords_in, r=radius, return_distance=True)
    else:
        inds_per_coord = kdtree.query_radius(X=coords_in, r=radius)
    projected_per_coord = []
    for i, inds in enumerate(inds_per_coord):
        # coords_aggregated = np.mean(gridvalues[inds]) if len(inds) > 0 else 0
        # coords_aggregated = 0.5*(np.mean(gridvalues[inds]) + np.mean(gridvalues[inds])) if len(inds) > 0 else 0
        if grid_aggregation_function == 'max':
            coords_aggregated = np.max(gridvalues[inds]) if len(inds) > 0 else 0
        if grid_aggregation_function == 'sum':
            coords_aggregated = np.sum(gridvalues[inds]) if len(inds) > 0 else 0
        if grid_aggregation_function == 'onion':
            if len(inds) > 0:
                local_distances = dist_per_coords[i]
                local_similarities = np.exp(-local_distances / 4)
                local_values = gridvalues[inds]
                onion_mean = np.sum(local_similarities * local_values) / np.sum(local_similarities)
                coords_aggregated = onion_mean
            else:
                coords_aggregated = 0
        projected_per_coord.append(coords_aggregated)
    projected_per_resid = aggregate_per_resid(projected_per_coord, resids_in, chains_in,
                                              atom_aggregation_function, pymolsel,
                                              outname=outname, return_keys=return_keys)
    return projected_per_resid


def project_npz(npz_path, pdbfilename,
                selection=None, hd=True,
                grid_aggregation_function='max',
                atom_aggregation_function=np.max, outname=None, radius=3., return_keys=True,
                n_top_blob=1, vol=150):
    """
    Get the above projection based on a npz file.
    We iterate over blobs to return a list of several mappings.
    :param npz_path:
    :param pdbfilename:
    :param selection:
    :param hd:
    :param grid_aggregation_function:
    :param atom_aggregation_function:
    :param outname:
    :param radius:
    :param return_keys:
    :param n_top_blob:
    :param vol:
    :return:
    """
    # Get the pdbfilename side.
    coords_in, resids_in, chains_in, pymolsel = get_protein_coords_and_residues(pdbfilename,
                                                                                selection=selection)

    # Get the pocket predictions in the npz form.
    pockets = np.load(npz_path)
    pocket_coords, pocket_probs = get_pocket_coords(pockets, hd=hd, vol=vol)
    max_blob = len(pocket_coords)
    n_blob_to_do = min(max_blob, n_top_blob)
    reslist = []
    for i in range(n_blob_to_do):
        projected_per_resid = project_gridvalues(pocket_coords[i], pocket_probs[i],
                                                 resids_in, chains_in, coords_in, pymolsel,
                                                 atom_aggregation_function=atom_aggregation_function,
                                                 grid_aggregation_function=grid_aggregation_function,
                                                 outname=outname,
                                                 radius=radius,
                                                 return_keys=return_keys)
        reslist.append(projected_per_resid)
    return reslist


def local_blobber(grid: np.ndarray, vol: float) -> (np.ndarray, float):
    """
    Local blobber
    Filter the input grid by taking the largest blob and filling it up to the
    desired volume.
    Returns a ndarray with the same shape as grid and a threshold to select the
    blob of interest.
    :param grid:
    :type grid: np.ndarray
    :param vol:
    :type vol: float
    :rtype: np.ndarray
    """
    threshold = np.sort(grid, axis=None)[-vol]
    labels, npockets = scipy.ndimage.label(
        grid >= threshold)  # labels with a different integer each distinct blob being above the threshold
    label_ids, blob_volumes = np.unique(labels[labels > 0], return_counts=True)
    label_big = label_ids[blob_volumes.argmax()]  # label of the biggest blob
    labels = skimage.segmentation.watershed(-grid, labels)  # watershed on each blob

    # threshold at the given volume on the biggest blob, if it is bigger than vol. Else threshold is zero
    main_blob_values = grid[labels == label_big]
    if len(main_blob_values) > vol:
        threshold = np.sort(main_blob_values, axis=None)[-vol]
    grid[labels != label_big] = 0.  # keep only the expanded big blob
    return grid, threshold


def find_contours(mask):
    """
    Return the contours from a 3D mask
    """
    edt = scipy.ndimage.distance_transform_edt(mask)
    contour = np.asarray(np.where(edt == 1)).T
    return contour


def centered_local_blobber(grid: np.ndarray, vol: float, mask=None) -> (np.ndarray, float):
    """
    Another local_blobber, based on the center of the grid now.
    This is made to avoid having the predicted blob on the side of the box (i.e. not really in the local pocket)
    mask is the set of voxels in the grid with protein density

    :param grid:
    :type grid: np.ndarray
    :param vol:
    :type vol: float
    :rtype: np.ndarray

    Filter the input grid by taking the largest blob and filling it up to the
    desired volume.
    Returns a ndarray with the same shape as grid and a threshold to select the
    blob of interest.
    """
    mask_contour = find_contours(mask)  # find the surface of the protein on the grid
    solvent = np.asarray(np.where(~mask)).T  # find the solvent on the grid
    # xyz_c = np.asarray(grid.shape) / 2
    all_dist = scipy.spatial.distance.cdist(solvent, mask_contour).mean(axis=1)  # Get the pairwise distances between the solvent points and the protein surface
    # all_dist = np.linalg.norm(solvent - xyz_c, axis=1)
    label_ind = solvent[all_dist.argmin()]  # Define the source of the glob as a the closest point of the solvent to the geometric center of the surface patch
    blob = adjmat.flood(1. - grid, source=label_ind, level=vol)
    main_blob_values = grid[blob == 1]
    if len(main_blob_values) > vol:
        threshold = np.sort(main_blob_values, axis=None)[-vol]
    else:
        threshold = np.min(grid[blob == 1])
    grid[blob != 1] = 0.  # keep only the expanded big blob
    return grid, threshold


def anchor_local_blobber(grid, vol, anchor_xyz, mask=None):
    """
    Another local_blobber, based on the center of the grid now.
    This is made to avoid having the predicted blob on the side of the box (i.e. not really in the local pocket)
    mask is the set of voxels in the grid with protein density

    :param grid:
    :type grid: np.ndarray
    :param vol:
    :type vol: float
    :rtype: np.ndarray

    Filter the input grid by taking the largest blob and filling it up to the
    desired volume.
    Returns a ndarray with the same shape as grid and a threshold to select the
    blob of interest.
    """
    solvent = np.asarray(np.where(~mask)).T  # find the solvent on the grid
    all_dist = np.linalg.norm(solvent - anchor_xyz, axis=1)
    label_ind = solvent[all_dist.argmin()]  # Define the source of the glob as a the closest point of the solvent to the geometric center of the surface patch
    # label_ind = np.int_(anchor_xyz)
    grid_masked = np.copy(grid)
    grid_masked[mask] -= 1
    blob = adjmat.flood(1. - grid_masked, source=label_ind, level=vol)
    main_blob_values = grid[blob == 1]
    if len(main_blob_values) > vol:
        threshold = np.sort(main_blob_values, axis=None)[-vol]
    else:
        threshold = np.min(grid[blob == 1])
    grid[blob != 1] = 0.  # keep only the expanded big blob
    return grid, threshold


class Predicter(object):
    def __init__(self, expfilename, spacing=1, padding=8):
        self.model = model_factory.model_from_exp(expfilename, load_weights=True)
        self.spacing = spacing
        self.padding = padding
        # Store the grid in hd and pl attributes
        self.hd = None
        self.pl = None
        self.origin = None
        self.pdbfilename = None
        self.selection = None
        self.atomtypes = list(utils.read_atomtypes().keys())

    def project(self, pdbfilename=None, pocket=None, use_hd=True, radius=3., outname=None):
        """
                Project the prediction onto the atomic coordinates
        :param pocket: a tuple : grid coords, grid values of shape : ((n,3), (n,)).
        If None is given we take the grid in the object
        :param use_hd: If we take the grid in the object, should it be the hd or pl one ?
        :param radius: The radius beyond which we project
        :return:
        """
        if pdbfilename is None and self.pdbfilename is None:
            raise ValueError('Please for projection, provide a pdb value')
        pdbfilename = pdbfilename if pdbfilename is not None else self.pdbfilename
        coords_in, resids_in, chains_in, pymolsel = get_protein_coords_and_residues(pdbfilename,
                                                                                    selection=self.selection)

        # Now add the prediction points.
        # If it is None, then we do it over the whole grid.
        if pocket is None:
            xm, ym, zm = self.origin - self.padding
            xM, yM, zM = np.asarray([xm, ym, zm]) + np.asarray(self.pl.shape) * self.spacing
            X = np.arange(xm, xM, self.spacing)
            Y = np.arange(ym, yM, self.spacing)
            Z = np.arange(zm, zM, self.spacing)
            grid = np.asarray(np.meshgrid(X, Y, Z, indexing='ij'))
            _, n, p, q = grid.shape
            gridcoords = grid.reshape(3, n * p * q).T
            gridvalues = (1 - self.hd[-1, :]).flatten() if use_hd else self.pl.flatten()
        else:
            gridcoords, gridvalues = pocket

        projected_per_resid = project_gridvalues(gridcoords, gridvalues,
                                                 resids_in, chains_in, coords_in, pymolsel,
                                                 atom_aggregation_function=np.max, outname=outname, radius=radius)
        return projected_per_resid

    def save_mrcs(self):
        pass

    def pred_pdb(self,
                 pdbfilename,
                 selection=None,
                 no_save_mrcs=False,
                 outname=None,
                 no_save_npz=False,
                 pdbfile_lig=None,
                 selection_lig=None,
                 hetatm=False,
                 print_blobs=True,
                 project=False):
        """
        Encapsulates the prediction and then optionally dump mrc or print blobs scores

        :param pdbfilename:
        :param selection:
        :param no_save_mrcs:
        :param outname:
        :param pdbfile_lig:
        :param selection_lig:
        :param hetatm:
        :param print_blobs:
        :return:
        """
        outname = os.path.basename(pdbfilename)[:-4] if outname is None else outname
        out_hd, out_pl, origin, lig_grid = predict_pdb(self.model,
                                                       pdbfilename,
                                                       selection=selection,
                                                       pdbfile_lig=pdbfile_lig,
                                                       selection_lig=selection_lig,
                                                       padding=self.padding,
                                                       spacing=self.spacing,
                                                       hetatm=hetatm,
                                                       enveloppe=True)
        if not no_save_npz or project:
            if not hetatm or not no_save_npz:
                # Watershed on HD
                grid = np.squeeze(out_hd)
                grid = grid[-1]
                grid = 1. - grid
                hd_coords, hd_distribs, hd_ids = blobber.to_blobs(grid, hetatm=False)
            # Watershed on PL
            if hetatm or not no_save_npz:
                grid = out_pl.squeeze()
                pl_coords, pl_distribs, pl_ids = blobber.to_blobs(grid, hetatm=True)
            if not no_save_npz:
                # Save npz file
                np.savez_compressed(f'{outname}.npz',
                                    hd=np.squeeze(out_hd),
                                    pl=np.squeeze(out_pl),
                                    origin=origin,
                                    hd_coords=hd_coords,
                                    hd_distribs=hd_distribs,
                                    hd_ids=hd_ids,
                                    pl_coords=pl_coords,
                                    pl_distribs=pl_distribs,
                                    pl_ids=pl_ids)
            if project:
                # We need to get an npz_like dict
                hd_string = 'hd' if not hetatm else 'pl'
                pockets = dict()
                pockets['origin'] = origin
                pockets[f'{hd_string}_ids'] = hd_ids if not hetatm else pl_ids
                pockets[f'{hd_string}_coords'] = hd_coords if not hetatm else pl_coords
                pockets[f'{hd_string}_distribs'] = hd_distribs if not hetatm else pl_distribs

                pocket_coords, pocket_probs = get_pocket_coords(pockets)
                first_pocket = pocket_coords[0], pocket_probs[0]
                self.project(pdbfilename=pdbfilename,
                             pocket=first_pocket,
                             outname=f"{outname}_projected.pdb")

        if not no_save_mrcs:
            # Dump HD with channels
            out_hd = out_hd.squeeze()
            for i, name in enumerate(self.atomtypes):
                out_hd_channel = out_hd[i]
                utils.save_density(out_hd_channel, f'{outname}_{name}.mrc', self.spacing, origin, self.padding)
            out_hd_void = 1 - out_hd[-1, ...]
            utils.save_density(out_hd_void, f'{outname}_ALL.mrc', self.spacing, origin, self.padding)

            # Dump PL
            utils.save_density(out_pl.squeeze(), f'{outname}_PL.mrc', self.spacing, origin, self.padding)

            # If we have a ligand, also dump the lig
            if lig_grid is not None:
                if hetatm:
                    lig_grid_all = lig_grid.squeeze()
                else:
                    lig_grid_all = 1 - lig_grid[-1, ...]
                utils.save_density(lig_grid_all, f'{outname}_ligand.mrc', self.spacing, origin, self.padding)

        if print_blobs:
            # To get metrics, we need to use the right output and to have a lig grid of the right shape
            out_grid = out_hd if not hetatm else out_pl
            if lig_grid is None:
                raise ValueError('We need a ligand to make a prediction')
            assert len(lig_grid.shape) == 3 if hetatm else len(lig_grid.shape) == 4

            # Now do the blobbing and metrics computation
            blobs, blobs_channel = lutils.get_metrics_blob(lig_grid, out_grid, hetatm=hetatm)
            pdbfile_lig = 'ligand' if pdbfile_lig is None else pdbfile_lig  # If we used a selection over the same PDB
            nres = count_nres(pdbfilename, selection=selection)
            string_res, score = string_blobs(blobs, blobs_channel, P=pdbfilename, L=pdbfile_lig, nres=nres)
            print(string_res)

        self.pdbfilename = pdbfilename
        self.hd = out_hd.squeeze()
        self.pl = out_pl.squeeze()
        self.origin = origin
        self.selection = selection
        return out_hd, out_pl, origin

    def pred_traj(self, pdbfilename, trajfilename,
                  outfilename=None,
                  box=None,
                  chain=None,
                  hetatm=False,
                  spacing=1.,
                  vol=150,
                  radius=6,
                  outmrcname=None,
                  dcdfilename=None,
                  use_centered_blobber=True,
                  ckp=1000, refpdb=None, refsel=None):
        """
        Inference on trajectory file

        Timing on GPU :
        time spent in the loading step : 0.075
        time spent predicting :          0.053
        time spent post-processing :     0.023
        ckp: checkpoint output dcd every ckp frame
        """

        cmd.feedback('disable', 'all', 'everything')

        density = Density.Coords_channel(pdbfilename, hetatm=hetatm)
        cmd.load(pdbfilename, 'traj')
        if trajfilename is not None:
            cmd.load_traj(trajfilename, 'traj', state=1)
        cmd.remove('hydrogens')
        print(cmd.get_object_list())
        print('Number of atoms in prot', cmd.select('prot'))
        print('Number of atoms in traj', cmd.select('traj'))
        nstates = cmd.count_states('traj')

        if outfilename is None:
            outfilename = f'{"ligandability" if hetatm else "interactibility"}_scores_radius-{radius}.txt'

        with open(outfilename, 'w') as outfile:
            outfile.write(f'''# Topology file: {pdbfilename}
# Trajectory file: {trajfilename}
# Number of frames: {nstates}
# Prediction model: {"ligandability" if hetatm else "interactibility"}
# Box selection: {box}
# Chain: {chain}
# Volume: {vol}
# Radius: {radius}
# Frame {"ligandability" if hetatm else "interactibility"} \
{"ligandability_old" if hetatm else "interactibility_old"} \
{"volume"}\n''')
            if chain is None:
                chain = 'polymer.protein'
            if box is None:
                # The maximum box should be the chain
                box = chain
            if trajfilename is not None:
                cmd.intra_fit(f'traj and name CA and {box}', state=1)  # Align the trajectory on the selected atoms on the first frame
            if refpdb is not None:
                cmd.load(refpdb, 'refpdb')
                cmd.align(f'refpdb and name CA and {box}', f'traj and name CA and {box}', target_state=1)
                if refsel is None:
                    refsel = 'hetatm'
                anchor_xyz = cmd.get_coords(f'refpdb and {refsel}').mean(axis=0)  # COM of the reference anchor
            else:
                anchor_xyz = None
            for state in range(nstates):
                # t0 = time.perf_counter()
                # Get the coords of this specific frame from the traj cmd object.
                # Then from this object find the right box based on the box selection
                coords_frame = cmd.get_coords(selection=f'traj and {chain}', state=state + 1)
                coords_box = cmd.get_coords(selection=f'traj and {box} and {chain}', state=state + 1)
                xyz_min = coords_box[:, :3].min(axis=0)
                xyz_max = coords_box[:, :3].max(axis=0)

                # Now we need to use the density object to iterate on the same atoms and use pymol
                #  utils to split these atoms into our channels (for the ones for which we have a mapping)
                # Using load coords, and the same chain selection,
                #  we ensure the coordinates are the ones of the correct frame
                other_coords = density.split_coords_by_channels(selection=chain,
                                                                ligand=False,
                                                                load_coords=coords_frame)

                # Now take these (n,4) data and put them in the grid format using padding.
                # Once the prediction is run, we remove the padding to focus on the relevant part of the output
                #  and we run a local blobber on it to look only at the most relevant pixels.
                # We zero out the rest of the grid and compute the mean over remaining pixels
                grid = Complex.get_grid_channels(other_coords,
                                                 spacing=spacing,
                                                 padding=radius,
                                                 xyz_min=xyz_min,
                                                 xyz_max=xyz_max,
                                                 is_target=False)
                # np.save('out/ingrid_%04d.npy' % state, grid)
                out_hd, out_pl = predict_frame(grid=grid, model=self.model)
                out_grid = 1 - out_hd[-1] if not hetatm else out_pl



                # focus on the initial box selection
                out_grid = out_grid[radius:-radius, radius:-radius, radius:-radius]


                # import matplotlib.pyplot as plt
                # out_zero= out_hd[0].flatten()
                # print('pred, zero', out_zero.max())
                # out_void = out_hd[-1].flatten()
                # print('pred, void', out_void.max())
                # print()
                #
                # plt.hist(out_hd[0].flatten(), label='zero', bins=20)
                # plt.hist(out_hd[1].flatten(), label='one', bins=20)
                # plt.hist(out_hd[-1].flatten(), label='void', bins=20)
                # plt.legend()
                # plt.show()
                # sys.exit()

                mask = grid.sum(axis=0)[radius:-radius, radius:-radius, radius:-radius] > 0.01
                score_old = np.sort(out_grid.flatten())[::-1][:vol].mean()
                # np.save('out/outgrid_%04d.npy' % state, out_grid)

                # Local blobber zeroes out all watershed bassins but one and
                # returns the threshold probability corresponding to either the 150th value or the last one.
                # print(f'Inference: {time.perf_counter() - t0}')
                # t0 = time.perf_counter()
                if use_centered_blobber:
                    if anchor_xyz is not None:
                        anchor_ijk = anchor_xyz - xyz_min
                        out_grid, threshold = anchor_local_blobber(out_grid, vol, anchor_ijk, mask=mask)
                    else:
                        out_grid, threshold = centered_local_blobber(out_grid, vol, mask=mask)
                else:
                    out_grid, threshold = local_blobber(out_grid, vol)
                # print(f'blobber: {time.perf_counter() - t0}')

                # print(threshold)
                # print(out_grid[out_grid >= threshold])

                volume = (out_grid >= threshold).sum()
                score = np.sum(out_grid[out_grid >= threshold]) / vol
                outfile.write(f"{state},{score},{score_old}, {volume}\n")
                outfile.flush()
                sys.stdout.write(f'Inference for frame: {state + 1}/{nstates} {score:.2g} {score_old:.2g}           \r')
                sys.stdout.flush()

                if outmrcname is not None:
                    # Dump HD with channels
                    if not hetatm:
                        out_hd = out_hd.squeeze()
                        out_hd = out_hd[:, radius:-radius, radius:-radius, radius:-radius]
                        out_hd = out_hd * (out_grid >= threshold)[None, ...]
                        for i, name in enumerate(self.atomtypes):
                            out_hd_channel = out_hd[i]
                            utils.save_density(out_hd_channel, f'{outmrcname}_{name}.mrc',
                                               spacing=spacing, origin=xyz_min, padding=0)
                        out_hd_void = 1 - out_hd[-1, ...]
                        utils.save_density(out_hd_void, f'{outmrcname}_ALL.mrc',
                                           spacing=spacing, origin=xyz_min, padding=0)

                    else:
                        # Dump PL
                        out_pl = out_pl[radius:-radius, radius:-radius, radius:-radius]
                        out_pl = out_pl * (out_grid >= threshold)
                        utils.save_density(out_pl, f'{outmrcname}_PL.mrc',
                                           spacing=spacing, origin=xyz_min, padding=0)

                if dcdfilename is not None:
                    whole_coords_frame = cmd.get_coords(selection=f'traj', state=state + 1)
                    grid_point, grid_values = blobber.filter_by_condition(out_grid, out_grid >= threshold)
                    grid_point_xyz = grid_point + xyz_min

                    # We sort the values above threshold so to have best pixels first.
                    # We also need to add :vol in case of equal values.
                    # Finally if we lack points, we add fake ones on the center of mass
                    grid_point_xyz = grid_point_xyz[grid_values.argsort()[::-1]][:vol]
                    if len(grid_point_xyz) < vol:
                        center_of_mass = np.mean(grid_point_xyz, axis=0)
                        fake_ones = np.ones(shape=(vol - len(grid_point_xyz), 3)) * center_of_mass
                        grid_point_xyz = np.concatenate((grid_point_xyz, fake_ones))

                    if state == 0:
                        cmd.create(name='out', selection="traj", source_state=1)
                        for resi, (x, y, z) in enumerate(grid_point_xyz):
                            cmd.pseudoatom(pos=(x, y, z), object='out', state=1, resi=resi + 1, chain="Z", name="H",
                                           elem="H")
                        cmd.save(selection='out', filename=f'{dcdfilename}.pdb', state=1)
                    else:
                        coords_cat = np.concatenate((whole_coords_frame, grid_point_xyz), axis=0)
                        cmd.load_coordset(coords_cat, object='out', state=state + 1)
                    if ((state + 1) % ckp == 0) and dcdfilename is not None:
                        print(f'DCD checkpoint with {state + 1} frames')
                        cmd.save_traj(filename=f'{dcdfilename}.dcd', selection='out')
            if dcdfilename is not None:
                cmd.save_traj(filename=f'{dcdfilename}.dcd', selection='out')

    def pred_file(self, data_file, callbacks=None, dump_path=None, use_true_y=False):
        return predict_file(model=self.model,
                            data_file=data_file,
                            callbacks=callbacks,
                            dump_path=dump_path,
                            use_true_y=use_true_y)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='')
    default_exp_path = os.path.join(script_dir, '../results/experiments/HPO.exp')
    parser.add_argument('-e', '--exp', help='Experiment file name', default=default_exp_path, type=str)

    subparsers = parser.add_subparsers(dest='command')

    # PDB PREDICTION
    parser_pdb = subparsers.add_parser('pdb')
    parser_pdb.add_argument('-p', '--pdb', help='pdb file name', type=str)
    parser_pdb.add_argument('-s', '--sel', help='Pymol selection for the prediction', type=str, default=None)
    parser_pdb.add_argument('--pdb_lig', help='pdb file name of the ligand', type=str, default=None)
    parser_pdb.add_argument('--sel_lig', help='Pymol selection for the ligand', type=str, default=None)
    parser_pdb.add_argument('-o', '--outname', help='Outname for the mrc or npz files', type=str, default=None)
    parser_pdb.add_argument('--pl', help='If the partner a small molecule and not a protein', action='store_true')
    parser_pdb.add_argument('--no_mrc', help='If we do not want mrc', action='store_true')
    parser_pdb.add_argument('--no_npz', help='If we do not want npz', action='store_true')
    parser_pdb.add_argument('--pblob', help='If we want to print the blobs', action='store_true')
    parser_pdb.add_argument('--project', help='Project the prediction onto the sequence', action='store_true')

    # MD TRAJECTORY PREDICTION
    parser_traj = subparsers.add_parser('traj')
    # In args
    parser_traj.add_argument('-p', '--pdb', help='PDB file name for the topology', type=str)
    parser_traj.add_argument('-t', '--traj', help='File name of the trajectory', type=str)
    parser_traj.add_argument('-b', '--box', help='Pymol selection for the prediction box definition', type=str,
                             default=None)
    parser_traj.add_argument('-c', '--chain', help='Pymol selection for the chain used for inference', type=str,
                             default=None)
    parser_traj.add_argument('--pl', help='If the partner a small molecule and not a protein', action='store_true')
    parser_traj.add_argument('-v', '--vol', help='Volume of the predicted pocket', type=int, default=150)
    parser_traj.add_argument('-r', '--radius',
                             help='Radius in Angstrom (but integer as grid spacing) to take around the selection',
                             type=int, default=6)
    parser_traj.add_argument('--blobber_max', help='To use the blobber with the max instead of centered',
                             action='store_true', default=False)
    parser_traj.add_argument('--ref', help='Reference pdb file to locate the pocket to track')
    parser_traj.add_argument('--refsel', help='Atom selection to define pocket location')
    # Out args
    parser_traj.add_argument('-o', '--outname', help='file name for the scores', type=str, default='default_name')
    parser_traj.add_argument('-mrc', '--outmrcname', help='file name for the mrc preds from pdb', type=str,
                             default=None)
    parser_traj.add_argument('-d', '--dcd', help='If we want dcd file of the trajectory', type=str, default=None)

    # METRICS
    parser_metrics = subparsers.add_parser('metrics')
    parser_metrics.add_argument("-ms", "--max_systems", type=int, default=None,
                                help="To set a maximum number of systems to screen")
    parser_metrics.add_argument('--outname', help='Outname for the mrc files', type=str, default='metrics.rec')
    parser_metrics.add_argument('--hd_txt', help='HD text file', type=str, default='HD-database_validation_set.txt')
    parser_metrics.add_argument('--hd_dir', help='HD Database', type=str, default='HD-database')
    parser_metrics.add_argument('--pl_txt', help='PL text file', type=str, default='PL-database_validation_set.txt')
    parser_metrics.add_argument('--pl_dir', help='PL Database', type=str, default='PL-database')
    parser_metrics.add_argument('--pblob', help='If we want to print the blobs', action='store_true')

    # FILE PREDICTION
    parser_file = subparsers.add_parser('file')
    parser_file.add_argument('-f', '--file', help='file name of the hdf5', type=str)

    # V2
    parser_v2 = subparsers.add_parser('v_2')
    parser_v2.add_argument('--no_hd', help='If we do not want hd', action='store_true')
    parser_v2.add_argument('--no_pl', help='If we do not want pl', action='store_true')
    args = parser.parse_args()

    if args.command == 'metrics':
        predicter = Predicter(expfilename=args.exp)
        v_2(predicter.model,
            outmetricfilename=args.outname,
            maxsystems=args.max_systems,
            hd_txt=args.hd_txt,
            hd_dir=args.hd_dir,
            pl_txt=args.pl_txt,
            pl_dir=args.pl_dir,
            print_blobs=args.pblob)

    if args.command == 'pdb':
        # python predict.py pdb -p ../1t4e.cif --no_mrc --project
        predicter = Predicter(expfilename=args.exp)
        predicter.pred_pdb(args.pdb,
                           selection=args.sel,
                           pdbfile_lig=args.pdb_lig,
                           selection_lig=args.sel_lig,
                           outname=args.outname,
                           hetatm=args.pl,
                           print_blobs=args.pblob,
                           no_save_mrcs=args.no_mrc,
                           no_save_npz=args.no_npz,
                           project=args.project)

        # predicter.model = None
        # pickle.dump(predicter, open('tmp.p', 'wb'))
        # predicter = pickle.load(open('tmp.p', 'rb'))
        # predicter.model = model_factory.model_from_exp(args.exp, load_weights=True)

    if args.command == 'traj':
        predicter = Predicter(expfilename=args.exp)
        predicter.pred_traj(args.pdb,
                            args.traj,
                            outfilename=args.outname,
                            box=args.box,
                            chain=args.chain,
                            hetatm=args.pl,
                            dcdfilename=args.dcd,
                            vol=args.vol,
                            radius=args.radius,
                            outmrcname=args.outmrcname,
                            use_centered_blobber=not args.blobber_max,
                            refpdb=args.ref, refsel=args.refsel)
    if args.command == 'v_2':
        model = model_factory.model_from_exp(expfilename=args.exp, load_weights=True)
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        final_score = v_2(model, device=device, no_HD=args.no_hd, no_PL=args.no_pl)
