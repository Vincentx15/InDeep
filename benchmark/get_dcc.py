#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2021-03-16 14:38:02 (UTC+0100)
import os
import sys
import glob
import shutil
import h5py
from collections import Iterable
from tqdm import tqdm

from pymol import cmd

import numpy as np
import scipy.spatial.distance as distance
from skimage.morphology import closing
from skimage.segmentation import clear_border
from skimage.measure import label

import multiprocessing

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from learning.predict import Predicter, get_pocket_coords
from data_processing import Complex


def compute_one_npz_chen(scop_dir, predicter, overwrite=False):
    """
    Runs the prediction of the model over the apo and holo contained in this scop dir.
    Skips if overwrites is False
    :param scop_dir:
    :param predicter:
    :param overwrite:
    :return:
    """
    filename_apo = os.path.join(scop_dir, 'apo_dump.mol2')
    if os.path.exists(filename_apo):
        dump_basename_apo = os.path.join(scop_dir, 'apo_pred_hpo_double')
        dump_filename_apo = dump_basename_apo + '.npz'
        if not os.path.exists(dump_filename_apo) or overwrite:
            predicter.pred_pdb(filename_apo,
                               outname=dump_basename_apo,
                               print_blobs=False,
                               no_save_mrcs=True,
                               no_save_npz=False)

    filename_holo = os.path.join(scop_dir, 'holo_dump.mol2')
    if os.path.exists(filename_holo):
        dump_basename_holo = os.path.join(scop_dir, 'holo_pred_hpo_double')
        dump_filename_holo = dump_basename_holo + '.npz'
        if not os.path.exists(dump_filename_holo) or overwrite:
            predicter.pred_pdb(filename_holo,
                               outname=dump_basename_holo,
                               print_blobs=False,
                               no_save_mrcs=True,
                               no_save_npz=False)


def compute_all_npz_chen(exp_path=os.path.join(script_dir, '../results/experiments/HPO.exp'),
                         data_path='../data/test_set/',
                         overwrite=False):
    """
    Get all apo and holo prediction for the chen res dir.
    :param exp_path:
    :param overwrite:
    :return:
    """
    predicter = Predicter(expfilename=exp_path)
    for scop in tqdm(os.listdir(data_path)):
        scop_dir = os.path.join(data_path, scop)
        try:
            compute_one_npz_chen(scop_dir=scop_dir, predicter=predicter, overwrite=overwrite)
        except:
            print(f'npz computation failed for {scop_dir}')


def get_ligands_coords(scop_dir):
    """
    Provides the centers of mass of each ligand for a given PDB
    :param scop_dir:
    :param holo:
    :return:
    """
    all_ligands = set(glob.glob(os.path.join(scop_dir, f'lig_*.mol2')))
    all_ligands = list(sorted(list(all_ligands)))
    ligand_coords = list()
    ligand_ids = list()
    for ligand in all_ligands:
        cmd.load(ligand)
        coords = cmd.get_coords()
        cmd.reinitialize()
        ligand_coords.append(coords)
        ligand_id = os.path.basename(ligand).split('_')[1].split('.')[0]
        ligand_ids.append(int(ligand_id))
    return ligand_ids, ligand_coords


def get_cavities_coords(scop_dir, holo=True):
    """
    Provides the centers of mass of different cavities detected by volsite around each ligands for a given PDB
    :param scop_dir:
    :param holo:
    :return:
    """
    holo_str = 'holo' if holo else 'apo'
    all_cavities = set(glob.glob(os.path.join(scop_dir, f'{holo_str}_cavity_*.mol2')))
    all_cavities = list(sorted([os.path.basename(cavity) for cavity in all_cavities]))
    cavity_id = 0
    cavity_coords = list()
    cavity_ids = list()
    while all_cavities:
        # Filter in the names the ones for each cavity id and
        cavities_id = [cavity for cavity in all_cavities if cavity.startswith(f'{holo_str}_cavity_{cavity_id}_')]
        if cavity_id > 100:
            return cavity_coords
        if len(cavities_id) == 0:
            cavity_id += 1
            continue
        all_cavities = [cavity for cavity in all_cavities if cavity not in cavities_id]
        for partial_cavity in cavities_id:
            local_cav_name = os.path.join(scop_dir, partial_cavity)
            cmd.load(local_cav_name)
        coords = cmd.get_coords()
        cmd.reinitialize()
        cavity_coords.append(coords)
        cavity_ids.append(cavity_id)
        cavity_id += 1
    return cavity_ids, cavity_coords


def get_cavities_ligands_center(scop_dir, holo=True):
    """
    Provides the centers of mass of different cavities detected by volsite around each ligands for a given PDB
    :param scop_dir:
    :param holo:
    :return:
    """
    holo_str = 'holo' if holo else 'apo'
    all_cavities = set(glob.glob(os.path.join(scop_dir, f'{holo_str}_cavity_*.mol2')))
    all_cavities = [os.path.basename(cavity) for cavity in all_cavities]

    # Get the ligands ids that gave cavities
    all_cavities_ids = sorted(list(set([cavity.split('_')[2] for cavity in all_cavities])))
    ligand_coords = list()
    ligand_ids = list()
    for cavity_id in all_cavities_ids:
        ligand = os.path.join(scop_dir, f'lig_{cavity_id}.mol2')
        cmd.load(ligand)
        coords = cmd.get_coords()
        cmd.reinitialize()
        ligand_coords.append(coords)
        ligand_ids.append(int(cavity_id))
    return ligand_ids, ligand_coords


def get_kala_pockets_old(scop_dir, holo=True):
    """
    This computes a pocket based on the centers of the flagged residues
    :param scop_dir:
    :param holo:
    :return:
    """
    holo_str = 'holo' if holo else 'apo'
    all_pockets = set(glob.glob(os.path.join(scop_dir, f'{holo_str}_pocket*.mol2')))
    # all_pockets = [os.path.basename(cavity) for cavity in all_pockets]
    reslist = list()
    for pocket in all_pockets:
        cmd.load(pocket)
        coords_pockets = cmd.get_coords()
        cmd.reinitialize()
        reslist.append(coords_pockets)
    return reslist


def get_kala_pockets(scop_dir, holo=True, threshold=0.5):
    """
    This computes a DCC based on the raw .cmap file.
    :param scop_dir:
    :param holo:
    :param threshold:
    :return:
    """

    holo_str = 'holo' if holo else 'apo'
    cmap_path = os.path.join(scop_dir, f'{holo_str}_pockets.cmap')
    f = h5py.File(cmap_path, 'r')
    cmap_file = f['Chimera']['image1']
    step = cmap_file.attrs['step']
    origin = cmap_file.attrs['origin']
    assert step[0] == step[1] == step[2]
    density = np.array(cmap_file['data_zyx'])
    density = density.transpose([2, 1, 0])
    bw = closing(density > threshold)
    cleared = clear_border(bw)

    # label regions
    label_image, num_labels = label(cleared, return_num=True)
    for i in range(1, num_labels + 1):
        pocket_idx = (label_image == i)
        pocket_size = pocket_idx.sum() * step[0] * step[1] * step[2]
        if pocket_size < 50:
            label_image[np.where(pocket_idx)] = 0

    pockets_coords = list()
    for pocket_label in range(1, label_image.max() + 1):
        indices = np.argwhere(label_image == pocket_label).astype('float32')
        if len(indices) == 0:
            continue
        indices *= step
        coords = indices + origin
        # with open("output.xyz", "w") as f:
        #     f.write(str(len(indices))+'\n')
        #     f.write('toto\n')
        #     for x, y, z in indices:
        #         f.write(f'C {x} {y} {z}\n')
        pockets_coords.append(coords)

    return pockets_coords


# By default, coords_a is ligand and coords_b is receptor, but this is symmetric.
def compute_dvo(coords_a, coords_b, voxel_size=1.5, gaussian=True):
    coords = np.concatenate((coords_a, coords_b), axis=0)
    xm, ym, zm = coords.min(axis=0)
    xM, yM, zM = coords.max(axis=0)
    xi = np.arange(xm, xM, voxel_size)
    yi = np.arange(ym, yM, voxel_size)
    zi = np.arange(zm, zM, voxel_size)

    if gaussian:
        volume_a = Complex.gaussian_blur(coords_a, xi, yi, zi, sigma=voxel_size)
        volume_b = Complex.gaussian_blur(coords_b, xi, yi, zi, sigma=voxel_size)
        # from scipy.ndimage import label
        # _, n_blobs = label(volume_a < 0.25)
        # print('nblobs', n_blobs)
        # with open('epipred_dcc_hpo_double.rec', 'a') as f:
        #     f.write(f'nblobs: {n_blobs}\n')

        volume_a[volume_a > 1] = 1
        volume_b[volume_b > 1] = 1
        intersection_value = np.sum(volume_a * volume_b)
        union_value = np.sum(volume_a) + np.sum(volume_b) - intersection_value
        dvo = intersection_value / union_value

        # print("gaussian volumes : ")
        # print(np.sum(volume_a))
        # print(np.sum(volume_b))
        # print(intersection_value)
        # print(union_value)
        # print(dvo)
        # print()
        return dvo
    else:
        volume_a, _ = np.histogramdd(coords_a, bins=(xi, yi, zi))
        volume_b, _ = np.histogramdd(coords_b, bins=(xi, yi, zi))

        volume_a = volume_a > 0
        volume_b = volume_b > 0

        intersection_value = np.sum(volume_a * volume_b)
        union_value = np.sum(volume_a) + np.sum(volume_b) - intersection_value
        dvo = intersection_value / union_value
        # print("dicrete volumes : ")
        # print(np.sum(volume_a))
        # print(np.sum(volume_b))
        # print(intersection_value)
        # print(union_value)
        # print(dvo)
        # print()
        return dvo


def compute_score(lig_coords, pocket_coords, pocket_probs):
    def weighted_mean(value_arr, weight_arr):
        # print(value_arr[:5])
        # print(weight_arr[:5])
        # print((value_arr * weight_arr[..., None])[:5])
        return np.sum(value_arr * weight_arr[..., None], axis=0) / weight_arr.sum()

    if len(lig_coords) == 0:
        print('FAILED NOW, LIGS = 0')
        return [100], [0]
    ligcenters = [np.mean(ligand, axis=0) for ligand in lig_coords]
    pocket_centers = [weighted_mean(pocket_coord, pocket_prob) for pocket_coord, pocket_prob in
                      zip(pocket_coords, pocket_probs)]
    dm = distance.cdist(ligcenters, pocket_centers)
    # dccs1 = np.min(dm, axis=1)
    dccs_locs = np.argmin(dm, axis=1)
    dccs = np.take_along_axis(dm, np.expand_dims(dccs_locs, axis=1), axis=1).squeeze()
    dccs = np.atleast_1d(dccs)

    dvos = list()
    for i, ligand_coord in enumerate(lig_coords):
        pocket_coord = pocket_coords[dccs_locs[i]]
        dvo = compute_dvo(coords_a=ligand_coord, coords_b=pocket_coord)
        dvos.append(dvo)
    return dccs, dvos



def compute_one_score(scop_dir, holo=True, precomputed_lig_info=None, cavity='cavity', kalasanty=False):
    """
    Loads the volsite cavities present in a given scop dir in mol2 format
    as well as the predicted blobs and computes the DCC metric
    :param scop_dir:
    :param holo:
    :param precomputed_lig_info: to provide precomputed lig centers, should unwrap into two iterables :
    the first one with ligand ids, the second one with their coords
    :param cavity: If we want to use Volsite cavity instead of ligands coordinates.
    We can also ask for ligands_cavity that are ligands who generated volsite cavities
    :return: iterable of names and results of the same length
    """
    holo_str = 'holo' if holo else 'apo'
    # Get Ligand locations
    if precomputed_lig_info is None:
        if cavity == 'cavity':
            lignames, ligcoords = get_cavities_coords(scop_dir=scop_dir, holo=holo)
        elif cavity == 'ligands':
            lignames, ligcoords = get_ligands_coords(scop_dir=scop_dir)
        elif cavity == 'ligands_cavity':
            lignames, ligcoords = get_cavities_ligands_center(scop_dir=scop_dir, holo=holo)
        else:
            raise ValueError(f'{cavity} is not a correct possiblity for the "cavity" argument in compute_one_score.'
                             f'try using one of : "cavity", "ligands" or  "ligands_cavity"')
    else:
        lignames, ligcoords = precomputed_lig_info
    n = len(ligcoords)

    # Get predictions and if empty return a placeholder
    if kalasanty:
        pocket_coords = get_kala_pockets(scop_dir, holo=holo)
        pocket_probs = [np.ones(len(pocket)) for pocket in pocket_coords]
    else:
        npz_path = os.path.join(scop_dir, f'{holo_str}_pred_hpo_double.npz')
        try:
            pockets = np.load(npz_path)
            pocket_coords, pocket_probs = get_pocket_coords(pockets, hd=False)
        except FileNotFoundError:
            pocket_coords, pocket_probs = [], []
    pocket_coords, pocket_probs = pocket_coords[:n], pocket_probs[:n]
    if len(pocket_coords) == 0:
        return lignames, 100 * np.ones(shape=len(ligcoords)), np.zeros(shape=len(ligcoords))

    # Shape of dccs : (n_cavities, n_pockets)
    dccs, dvos = compute_score(ligcoords, pocket_coords, pocket_probs)
    # for i, dcc in enumerate(dccs):
    #     print(f'{holo_str}_kala_cavity_{i}_dcc: {dcc:.4f}')

    return lignames, dccs, dvos


def compute_all_scores_chen(data_path='../data/chen_res/', rec_name='chen_dcc.rec'):
    out_rec = os.path.join(data_path, rec_name)
    with open(out_rec, 'w') as _:
        pass
    for scop in tqdm(os.listdir(data_path)):
        scop_dir = os.path.join(data_path, scop)
        # Don't try to run on recfiles...
        if not os.path.isdir(scop_dir):
            continue
        filename_apo = os.path.join(scop_dir, 'apo_dump.mol2')
        filename_holo = os.path.join(scop_dir, 'holo_dump.mol2')

        with open(out_rec, 'a') as outfile:
            outfile.write(f'scop: {scop}\n')

        def compute_and_write(outfile_name, ligand_type, precomputed_lig_info, scop_dir, holo):
            holo_str = 'holo' if holo else 'apo'
            for kalasanty in [False, True]:
                results = compute_one_score(scop_dir=scop_dir, holo=holo,
                                            precomputed_lig_info=precomputed_lig_info,
                                            kalasanty=kalasanty)
                kala_str = 'kala' if kalasanty else 'indeep'
                with open(outfile_name, 'a') as outfile:
                    for ligand, dcc, dvo in zip(*results):
                        outfile.write(f'dcc_{holo_str}_{ligand_type}_{kala_str}_{ligand}: {dcc}\n')
                        outfile.write(f'dvo_{holo_str}_{ligand_type}_{kala_str}_{ligand}: {dvo}\n')

        # Compute values for all ligands in the apo setting
        precomputed_lig_info_apocav = get_cavities_coords(scop_dir=scop_dir, holo=False)
        if len(precomputed_lig_info_apocav[0]) > 0 and os.path.exists(filename_apo):
            compute_and_write(outfile_name=out_rec, ligand_type='cav', scop_dir=scop_dir, holo=False,
                              precomputed_lig_info=precomputed_lig_info_apocav)

        precomputed_lig_info_holocav = get_cavities_coords(scop_dir=scop_dir, holo=True)
        if len(precomputed_lig_info_holocav[0]) > 0 and os.path.exists(filename_holo):
            compute_and_write(outfile_name=out_rec, ligand_type='cav', scop_dir=scop_dir, holo=True,
                              precomputed_lig_info=precomputed_lig_info_holocav)

        # Now compute the values for filtered ligands
        precomputed_lig_info_apocavlig = get_cavities_ligands_center(scop_dir=scop_dir, holo=False)
        if len(precomputed_lig_info_apocavlig[0]) > 0 and os.path.exists(filename_apo):
            compute_and_write(outfile_name=out_rec, ligand_type='cavlig', scop_dir=scop_dir, holo=False,
                              precomputed_lig_info=precomputed_lig_info_apocavlig)
        precomputed_lig_info_holocavlig = get_cavities_ligands_center(scop_dir=scop_dir, holo=True)
        if len(precomputed_lig_info_holocavlig[0]) > 0 and os.path.exists(filename_holo):
            compute_and_write(outfile_name=out_rec, ligand_type='cavlig', scop_dir=scop_dir, holo=True,
                              precomputed_lig_info=precomputed_lig_info_holocavlig)

        # Now compute these values for all ligands
        precomputed_lig_info_lig = get_ligands_coords(scop_dir=scop_dir)
        if len(precomputed_lig_info_lig[0]) > 0:
            if os.path.exists(filename_apo):
                compute_and_write(outfile_name=out_rec, ligand_type='lig', scop_dir=scop_dir, holo=False,
                                  precomputed_lig_info=precomputed_lig_info_lig)
            if os.path.exists(filename_holo):
                compute_and_write(outfile_name=out_rec, ligand_type='lig', scop_dir=scop_dir, holo=True,
                                  precomputed_lig_info=precomputed_lig_info_lig)
        with open(out_rec, 'a') as outfile:
            outfile.write(f'\n')


def init(l):
    global lock
    lock = l


def do_one(npzfile, scpdb_path, pred_path, outfilename='scores_scPDB.rec', topn=[1, 3, 5, 10]):
    pdbcode = os.path.splitext(npzfile)[0]
    # print(f'{i+1}/{len(all_preds)}: {pdbcode}')
    scpdb_dir = f'{scpdb_path}/{pdbcode}'
    cavity = f'{scpdb_dir}/cavity6.mol2'
    ligand = f'{scpdb_dir}/ligand.mol2'
    pockets = np.load(f'{pred_path}/{npzfile}')
    pocket_coords, pocket_probs = get_pocket_coords(pockets, hd=False)
    # Pymol stuff
    cmd.load(ligand, 'mylig')
    cmd.load(cavity, 'mycav')
    ligcoords = cmd.get_coords('mylig')
    cavcoords = cmd.get_coords('mycav')
    cmd.reinitialize()
    ##############
    if not isinstance(topn, Iterable):
        topn = [topn, ]
    dccs_lig_topn, dvos_lig_topn = [], []
    dccs_cav_topn, dvos_cav_topn = [], []
    for n in topn:
        pocket_coords_, pocket_probs_ = pocket_coords[:n], pocket_probs[:n]
        dccs_lig, dvos_lig = compute_score([ligcoords, ], pocket_coords_, pocket_probs_)
        dccs_cav, dvos_cav = compute_score([cavcoords, ], pocket_coords_, pocket_probs_)
        dccs_lig_topn.append(dccs_lig[0])
        dvos_lig_topn.append(dvos_lig[0])
        dccs_cav_topn.append(dccs_cav[0])
        dvos_cav_topn.append(dvos_cav[0])
    lock.acquire()
    with open(outfilename, 'a') as outfile:
        for i, n in enumerate(topn):
            dcc_lig = dccs_lig_topn[i]
            dvo_lig = dvos_lig_topn[i]
            dcc_cav = dccs_cav_topn[i]
            dvo_cav = dvos_cav_topn[i]
            if i == 0:
                outfile.write(f'pdb: {pdbcode}\n')
            outfile.write(f'dcc_lig_top_{n}: {dcc_lig}\n')
            outfile.write(f'dvo_lig_top_{n}: {dvo_lig}\n')
            outfile.write(f'dcc_cav_top_{n}: {dcc_cav}\n')
            outfile.write(f'dvo_cav_top_{n}: {dvo_cav}\n')
            if i == len(topn) - 1:
                outfile.write('\n')
    lock.release()


def compute_all_scores_scPDB(scpdb_path, pred_path, outfilename='scores_scPDB.rec', topn=[1, 3, 5, 10]):
    open(outfilename, 'w')
    all_preds = os.listdir(pred_path)
    l = multiprocessing.Lock()
    pool = multiprocessing.Pool(initializer=init, initargs=(l,))
    njobs = len(all_preds)
    pool.starmap(do_one,
                 tqdm(zip(all_preds, [scpdb_path, ] * njobs, [pred_path, ] * njobs, [outfilename, ] * njobs,
                          [topn, ] * njobs), total=njobs))
    # for i, npzfile in enumerate(all_preds):


def move_kala(holo=True):
    """
    Just some small util to put the right preds in the right place
    :param holo:
    :return:
    """
    holo_str = 'holo' if holo else 'apo'
    data_path = f'../data/kala_{holo_str}_preds/'
    dump_path = '../data/chen_res/'
    for scop in tqdm(os.listdir(data_path)):
        scop_dir = os.path.join(data_path, scop)
        dump_dir = os.path.join(dump_path, scop)
        for result in os.listdir(scop_dir):
            file_place = os.path.join(scop_dir, result)
            dump_name = f'{holo_str}_{result}'
            dump_place = os.path.join(dump_dir, dump_name)
            shutil.move(file_place, dump_place)
            # print(file_place, dump_place)


if __name__ == '__main__':
    pass
    # compute_all_scores_scPDB(scpdb_path='/c7/scratch/bougui/sc-PDB/scPDB',
    #                          pred_path='/c7/scratch/bougui/sc-PDB/pockets_indeep', topn=[1, 3, 5, 10])
    '''
    import argparse

    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    
    parser = argparse.ArgumentParser(description='Compute DCC from a npz and a pdb file')
    parser.add_argument('-p', '--pdb')  # -> holo/a.001.001.001_1s69a.pdb
    parser.add_argument('-n', '--npz')
    args = parser.parse_args()

    cmd.load(args.pdb, 'inpdb')
    ligands = get_ligands()
    # print(f"ligands: {ligands}")  # -> [('HEM', 125, 'A'), ('FLC', 227, '')]
    coords_ligs = get_ligands_coords(ligands)
    ligcenters = [c.mean(axis=0) for c in coords_ligs]
    pockets = np.load(args.npz)
    n = len(ligcenters)
    pocket_centers = [get_pocket_center(pockets, i + 1) for i in range(n)]
    dccs = distance.cdist(ligcenters, pocket_centers).min(axis=0)
    print(f'name: {args.pdb}')
    for i, dcc in enumerate(dccs):
        print(f'resname: {ligands[i][0]}')
        print(f'resid: {ligands[i][1]}')
        print(f'chain: {ligands[i][2]}')
        print(f'dcc: {dcc:.4f}')
    print()
    '''

    # compute_all_npz_chen(data_path = '../data/chen_res/')
    # compute_all_npz_chen(data_path = '../data/test_set/')
    # compute_one_score(scop_dir='../data/chen_res/c.069.001.020', cavity='ligands_cavity', kalasanty=True, holo=True)
    # get_cavities_center('../data/chen_res/c.069.001.020')

    # scop_dir_local = '../data/chen_res/c.094.001.001'
    # precomputed_lig_info_apo_cavlig = get_cavities_ligands_center(scop_dir=scop_dir_local, holo=False)
    # precomputed_lig_info_apo_lig = get_ligands_coords(scop_dir=scop_dir_local)
    #
    # results = compute_one_score(scop_dir=scop_dir_local, holo=False,
    #                             precomputed_lig_info=precomputed_lig_info_apo_cavlig,
    #                             kalasanty=False)
    # results2 = compute_one_score(scop_dir=scop_dir_local, holo=False,
    #                              precomputed_lig_info=precomputed_lig_info_apo_lig,
    #                              kalasanty=False)
    # print(results)
    # print(results2)

    compute_all_scores_chen(data_path='../data/chen_res/', rec_name='chen_scores_hpo_double.rec')
    compute_all_scores_chen(data_path='../data/test_set/', rec_name='testset_scores_hpo_double.rec')

    # move_kala(holo=True)
    # move_kala(holo=False)
