#!/usr/bin/env python
import os
import sys

import pickle
import itertools
import csv
from multiprocessing import Pool

import numpy as np
from functools import partial
import scipy.spatial.distance as distance
from tqdm import tqdm
from pymol import cmd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, auc
from collections import defaultdict

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from learning.predict import Predicter, project_npz, get_protein_coords_and_residues, aggregate_per_resid, \
    get_pocket_coords
from benchmark.get_dcc import compute_score
from data_processing.utils import read_atomtypes

ATOMTYPES = read_atomtypes()

# TEST_SET = {'1Y64-r', '1VFB-r', '1FAK-r', '1ZLI-l', '3P57-r', '1I9R-l', '3F1P-l', '2OZA-r', '2IDO-r', '1WDW-l',
#             '1DQJ-r', '1T6B-l', '2J7P-r', '2FJU-l', '1BKD-r', '1KTZ-l', '1K5D-l', '1SBB-l', '3SGQ-l', '1KAC-r',
#             '1DE4-l', '4GXU-l', '1DE4-r', '1Z5Y-l', '1ZM4-l', '1M10-l', '1CLV-l', '1E4K-l', '1OC0-r', '1GP2-l',
#             '3D5S-r', '1ATN-r', '3H11-r', '1ML0-r', '4FQI-r', '1NW9-l', '3AAA-r', '1WQ1-r', '3L89-l', '1DFJ-r',
#             '2JEL-l', '1E6J-l', '4FZA-r', '4DN4-r', '1AK4-r', '2W9E-l', '1CLV-r', '3A4S-r', '1BVK-l', '1RLB-r',
#             '1BUH-r', '1AY7-r', '3BIW-r', '1AKJ-l', '4CPA-r', '2AYO-r', '3AAA-l', '2PCC-r', '1R0R-l', '3L5W-l',
#             '3MXW-l', '2W9E-r', '4FZA-l', '4CPA-l', '2HQS-l', '3VLB-l', '1MAH-l', '2NZ8-l', '1PPE-l', '1PVH-l',
#             '1MQ8-l', '1LFD-r', '2VIS-l', '1AK4-l', '1AHW-l', '3EOA-r', '4LW4-r', '1F51-l', '2X9A-r', '1E96-l',
#             '1Z0K-r', '3FN1-l', '1WEJ-r', '1JTG-r', '1YVB-l', '1EER-l', '1JTG-l', '4DN4-l', '1F34-l', '1R8S-r',
#             '2GAF-l', '2OOB-r', '1K4C-l', '1FQJ-r', '2UUY-l', '2Z0E-r', '2HLE-r', '1QFW-r', '1UDI-l', '4H03-l'}
TEST_SET = {"1HE1-l", "1IQD-l", "1JPS-l", "1E6E-l", "1EWY-l", "3D5S-l", "1M10-l", "3AAA-l", "1SYX-l", "1HCF-l",
            "2UUY-l", "1JZD-l", "1EER-l", "2B4J-l", "2JEL-l", "2FD6-l", "1F51-l", "3BX7-l", "2FJU-l", "1WEJ-l",
            "1YVB-l", "1S1Q-l", "1ZHI-l", "1IRA-l", "2W9E-l", "1NCA-l", "1ZLI-l", "2PCC-l", "1BVN-l", "2J7P-l",
            "1JMO-l", "3MXW-l", "2AYO-l", "1VFB-l", "3PC8-l", "2B42-l", "2OUL-l", "1HE1-r", "1IQD-r", "1JPS-r",
            "1E6E-r", "1EWY-r", "3D5S-r", "1M10-r", "3AAA-r", "1SYX-r", "1HCF-r", "2UUY-r", "1JZD-r", "1EER-r",
            "2B4J-r", "2JEL-r", "2FD6-r", "1F51-r", "3BX7-r", "2FJU-r", "1WEJ-r", "1YVB-r", "1S1Q-r", "1ZHI-r",
            "1IRA-r", "2W9E-r", "1NCA-r", "1ZLI-r", "2PCC-r", "1BVN-r", "2J7P-r", "1JMO-r", "3MXW-r", "2AYO-r",
            "1VFB-r", "3PC8-r", "2B42-r", "2OUL-r"}
ALL_DISTRIB_RESULTS = defaultdict(list)


def custom_list_dir(directory, hidden=False):
    full_listdir = os.listdir(directory)
    if not hidden:
        return full_listdir
    return [file for file in full_listdir if not file.startswith('.')]


def build_database(infile='../data/epipred_data.csv', outpath='../data/epipred'):
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    with open(infile, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader, None)
        # cmd.set('fetch_path', cmd.exp_path('~/pdb'), quiet=1)

        for i, row in enumerate(csv_reader):
            pdb_code, heavy, light, antigene = row

            dump_dir = os.path.join(outpath, pdb_code)
            if not os.path.exists(dump_dir):
                os.mkdir(dump_dir)

            # fetch the pdb
            cmd.set('fetch_path', dump_dir, quiet=1)
            cmd.fetch(pdb_code, name='whole_pdb')

            # Get the receptor selection
            antigene_selection = f'whole_pdb and polymer.protein and chain {antigene}'
            antigene_name = os.path.join(dump_dir, 'receptor.pdb')
            cmd.save(filename=antigene_name, selection=antigene_selection)

            # Get the ligand pdb
            antibody_selection = f'whole_pdb and polymer.protein and (chain {heavy} or chain {light})'
            antibody_name = os.path.join(dump_dir, 'ligand.pdb')
            cmd.save(filename=antibody_name, selection=antibody_selection)
            cmd.reinitialize()


def compute_all_npz_epipred(exp_path=os.path.join(script_dir, '../results/experiments/gaussian_blur.exp'),
                            data_path='../data/epipred/',
                            dump_name='holo_pred',
                            overwrite=False):
    """
    Get all apo and holo prediction for the chen res dir.
    :param exp_path:
    :param overwrite:
    :return:
    """
    predicter = Predicter(expfilename=exp_path)
    for pdb_code in tqdm(custom_list_dir(data_path)):
        pdb_code_dir = os.path.join(data_path, pdb_code)
        try:
            filename_holo = os.path.join(pdb_code_dir, 'receptor.pdb')
            if os.path.exists(filename_holo):
                dump_basename_holo = os.path.join(pdb_code_dir, dump_name)
                dump_filename_holo = dump_basename_holo + '.npz'
                if not os.path.exists(dump_filename_holo) or overwrite:
                    predicter.pred_pdb(filename_holo,
                                       outname=dump_basename_holo,
                                       print_blobs=False,
                                       no_save_mrcs=True,
                                       no_save_npz=False)
        except:
            print(f'npz computation failed for {pdb_code_dir}')


def compute_all_npz_dbd5(exp_path=os.path.join(script_dir, '../results/experiments/gaussian_blur.exp'),
                         data_path='../data/dbd5/',
                         method_suffix='hpo',
                         overwrite=False):
    """
    Get all apo and holo prediction for the npz res dir.
    :param exp_path:
    :param overwrite:
    :return:
    """
    predicter = Predicter(expfilename=exp_path)
    for pdb_code in tqdm(custom_list_dir(data_path)):
        pdb_code_dir = os.path.join(data_path, pdb_code)
        try:
            filename_apo_r = os.path.join(pdb_code_dir, 'receptor_u.pdb')
            if os.path.exists(filename_apo_r):
                dump_name_apo_r = f"receptor_u_pred_{method_suffix}"
                dump_basename_apo_r = os.path.join(pdb_code_dir, dump_name_apo_r)
                dump_filename_apo_r = dump_basename_apo_r + '.npz'
                if not os.path.exists(dump_filename_apo_r) or overwrite:
                    predicter.pred_pdb(filename_apo_r,
                                       outname=dump_basename_apo_r,
                                       print_blobs=False,
                                       no_save_mrcs=True,
                                       no_save_npz=False)
        except:
            print(f'apo receptor npz computation failed for {pdb_code_dir}')
        try:
            filename_holo_r = os.path.join(pdb_code_dir, 'receptor_b.pdb')
            if os.path.exists(filename_holo_r):
                dump_name_holo_r = f"receptor_b_pred_{method_suffix}"
                dump_basename_holo_r = os.path.join(pdb_code_dir, dump_name_holo_r)
                dump_filename_holo_r = dump_basename_holo_r + '.npz'
                if not os.path.exists(dump_filename_holo_r) or overwrite:
                    predicter.pred_pdb(filename_holo_r,
                                       outname=dump_basename_holo_r,
                                       print_blobs=False,
                                       no_save_mrcs=True,
                                       no_save_npz=False)
        except:
            print(f'holo receptor npz computation failed for {pdb_code_dir}')

        try:
            filename_apo_l = os.path.join(pdb_code_dir, 'ligand_u.pdb')
            if os.path.exists(filename_apo_l):
                dump_name_apo_l = f"ligand_u_pred_{method_suffix}"
                dump_basename_apo_l = os.path.join(pdb_code_dir, dump_name_apo_l)
                dump_filename_apo_l = dump_basename_apo_l + '.npz'
                if not os.path.exists(dump_filename_apo_l) or overwrite:
                    predicter.pred_pdb(filename_apo_l,
                                       outname=dump_basename_apo_l,
                                       print_blobs=False,
                                       no_save_mrcs=True,
                                       no_save_npz=False)
        except:
            print(f'apo ligand npz computation failed for {pdb_code_dir}')
        try:
            filename_holo_l = os.path.join(pdb_code_dir, 'ligand_b.pdb')
            if os.path.exists(filename_holo_l):
                dump_name_holo_l = f"ligand_b_pred_{method_suffix}"
                dump_basename_holo_l = os.path.join(pdb_code_dir, dump_name_holo_l)
                dump_filename_holo_l = dump_basename_holo_l + '.npz'
                if not os.path.exists(dump_filename_holo_l) or overwrite:
                    predicter.pred_pdb(filename_holo_l,
                                       outname=dump_basename_holo_l,
                                       print_blobs=False,
                                       no_save_mrcs=True,
                                       no_save_npz=False)
        except:
            print(f'holo ligand npz computation failed for {pdb_code_dir}')


def project_supervision(receptor_path, ligand_path):
    receptor_coords, resids_receptor, chains_receptor, pymolsel = get_protein_coords_and_residues(receptor_path)

    cmd.load(ligand_path, 'ligand')
    ligand_coords = cmd.get_coords(selection='ligand')

    dm = distance.cdist(receptor_coords, ligand_coords)
    receptor_distances = np.min(dm, axis=1)

    # Now for each query atom, find the neighbors
    # Then iterate over the atoms and get the max value per atom
    # Finally aggregate atomic values per residue.
    def binarize(list_to_binarize):
        return int(np.min(list_to_binarize) < 6)

    projected_per_resid = aggregate_per_resid(receptor_distances, resids_receptor, chains_receptor,
                                              binarize, pymolsel, outname=None, return_keys=True)
    return projected_per_resid


def aucpr(y_test, y_score):
    # Data to plot precision - recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    # Use AUC function to calculate the area under the curve of precision recall curve
    auc_precision_recall = auc(recall, precision)
    return auc_precision_recall


def compute_one_score(filename_pdb, filename_supervision, filename_npz, radius=5., n_top_blob=3, vol=None,
                      grid_aggregation_function='max', outname=None, outrec=None):
    #
    # Get the predicted projection

    indeep_prediction_projected = project_npz(npz_path=filename_npz,
                                              pdbfilename=filename_pdb,
                                              radius=radius,
                                              n_top_blob=n_top_blob,
                                              grid_aggregation_function=grid_aggregation_function,
                                              vol=vol, outname=outname)
    # old filename_supervision was a pdb, now its a pickle
    # supervision = project_supervision(filename_pdb, filename_supervision)
    supervision = pickle.load(open(filename_supervision, 'rb'))
    supervision = {(key[0], str(key[1])): val for key, val in supervision.items()}

    # Take all blobs and add them in a consistent order. return an array (nres, nblobs + 1(supervision))
    aligned_predictions = list()
    for key, value in supervision.items():
        res_values = [value]
        for blob_res in indeep_prediction_projected:
            res_values.append(blob_res[key])
        aligned_predictions.append(res_values)
    aligned_predictions = np.asarray(aligned_predictions)
    padded_aligned_predictions = np.zeros((len(aligned_predictions), n_top_blob + 1))
    padded_aligned_predictions[:, :aligned_predictions.shape[1]] = aligned_predictions

    # extract in separated arrays and aggregate values across blobs
    supervised = padded_aligned_predictions[:, 0][None, ...]
    preds = padded_aligned_predictions[:, 1:].T
    agg_preds = np.max(preds, axis=0)[None, ...]
    arange_blob = np.arange(start=n_top_blob, stop=0, step=-1)[:, None]
    agg_preds_2 = np.sum(arange_blob * preds, axis=0)[None, :]

    mean_positive = supervised.mean()
    aucpr_all = distance.cdist(supervised, preds, metric=average_precision_score)
    aucpr_all = np.atleast_1d(np.squeeze(aucpr_all))
    aucpr_max = distance.cdist(supervised, agg_preds, metric=average_precision_score)
    aucpr_max = np.squeeze(aucpr_max)
    auprc_mean = distance.cdist(supervised, agg_preds_2, metric=average_precision_score)
    auprc_mean = np.squeeze(auprc_mean)
    correlation_all = distance.cdist(supervised, preds, metric='correlation')
    correlation_all = np.atleast_1d(np.squeeze(correlation_all))
    correlation_max = distance.cdist(supervised, agg_preds, metric='correlation')
    correlation_max = np.squeeze(correlation_max)
    correlation_mean = distance.cdist(supervised, agg_preds_2, metric='correlation')
    correlation_mean = np.squeeze(correlation_mean)

    supervised = supervised.squeeze()
    agg_preds = agg_preds.squeeze()
    aucpr_2 = average_precision_score(supervised, agg_preds)
    aucpr_3 = aucpr(supervised, agg_preds)
    try:
        auroc = roc_auc_score(supervised, agg_preds)
    except:
        auroc = np.nan

    lines = []
    lines.append(f'mean_positive: {mean_positive}' + '\n')
    lines.append('allpockets: ' + ('{} ' * n_top_blob).format(*aucpr_all) + '\n')
    lines.append(f'all_together: {aucpr_max}' + '\n')
    lines.append(f'sorted_per_blobs: {auprc_mean}' + '\n')
    lines.append('correlation_allpockets: ' + ('{} ' * n_top_blob).format(*correlation_all) + '\n')
    lines.append(f'correlation_all_together: {correlation_max}' + '\n')
    lines.append(f'correlation_sorted_per_blobs: {correlation_mean}' + '\n')
    lines.append(f'average_precision: {aucpr_2}' + '\n')
    lines.append(f'homemade_aucpr: {aucpr_3}' + '\n')
    lines.append(f'auroc: {auroc}' + '\n')
    lines.append('\n')

    lines = ''.join(lines)

    if outrec is not None:
        with open(outrec, 'a') as f:
            f.write(lines)
    # This is to be fine-tuned, we have to choose one final metric and stick to it.
    # Probs follow PInet
    # TODO : choose a scoring function
    final_score = aucpr_max
    return indeep_prediction_projected, supervision, final_score


def get_names_dbd5(pdb_code, method_suffix, receptor=True, holo=True, data_path='../data/dbd5'):
    holo_str = 'b' if holo else 'u'
    receptor_str = 'receptor' if receptor else 'ligand'
    receptor_code = 'r' if receptor else 'l'
    # ligand_str = 'ligand' if receptor else 'receptor'
    pinet_name = f"{pdb_code}-{receptor_code}"
    pdb_code_dir = os.path.join(data_path, pdb_code)
    filename_pdb = os.path.join(pdb_code_dir, f'{receptor_str}_{holo_str}.pdb')
    filename_npz = os.path.join(pdb_code_dir, f'{receptor_str}_{holo_str}_pred_{method_suffix}.npz')
    filename_supervision = os.path.join(pdb_code_dir, f'{receptor_str}_b_patch.p')
    return pinet_name, filename_pdb, filename_npz, filename_supervision


def compute_all_score_dbd5(radius=5., n_top_blob=3, vol=None,
                           grid_aggregation_function='max', method_suffix='hpo_double',
                           outrec=None, compute_holo=True, test_set=TEST_SET, do_test_set=False):
    """
    :param radius:
    :param n_top_blob:
    :param vol:
    :param grid_aggregation_function:
    :param outrec:
    :param compute_holo:
    :param do_test_set: if we want inference on the test set
    :return:
    """
    # create a clean one
    if outrec is not None:
        with open(outrec, 'w') as _:
            pass

    data_path = '../data/dbd5'
    all_score = list()
    done = 0
    for pdb_code in custom_list_dir(data_path):
        # Get the prediction for the receptor and ligand files.
        # We only get it if it is present in the filter set (train/test)
        pinet_name, filename_pdb, filename_npz, filename_supervision = get_names_dbd5(pdb_code=pdb_code,
                                                                                      method_suffix=method_suffix,
                                                                                      receptor=True,
                                                                                      holo=compute_holo,
                                                                                      data_path=data_path)
        in_test = pinet_name in test_set
        if (in_test and do_test_set) or (not in_test and not do_test_set):
            if os.path.exists(filename_npz) and os.path.exists(filename_supervision):
                done += 1
                line = f'pdbcode: {pdb_code}_receptor'
                print(line)
                print(f'done {done}')
                if outrec is not None:
                    with open(outrec, 'a') as f:
                        f.write(line + '\n')
                prediction, supervision, score = compute_one_score(filename_pdb=filename_pdb,
                                                                   filename_npz=filename_npz,
                                                                   filename_supervision=filename_supervision,
                                                                   radius=radius,
                                                                   n_top_blob=n_top_blob,
                                                                   vol=vol,
                                                                   grid_aggregation_function=grid_aggregation_function,
                                                                   outrec=outrec)
                all_score.append(score)

        pinet_name, filename_pdb, filename_npz, filename_supervision = get_names_dbd5(pdb_code=pdb_code,
                                                                                      method_suffix=method_suffix,
                                                                                      receptor=False,
                                                                                      holo=compute_holo,
                                                                                      data_path=data_path)
        in_test = pinet_name in test_set
        if (in_test and do_test_set) or (not in_test and not do_test_set):
            if os.path.exists(filename_npz) and os.path.exists(filename_supervision):
                done += 1
                line = f'pdbcode: {pdb_code}_ligand'
                print(line)
                print(f'done {done}')
                if outrec is not None:
                    with open(outrec, 'a') as f:
                        f.write(line + '\n')
                prediction, supervision, score = compute_one_score(filename_pdb=filename_pdb,
                                                                   filename_npz=filename_npz,
                                                                   filename_supervision=filename_supervision,
                                                                   radius=radius,
                                                                   n_top_blob=n_top_blob,
                                                                   vol=vol,
                                                                   grid_aggregation_function=grid_aggregation_function,
                                                                   outrec=outrec)
                all_score.append(score)
    return np.nanmean(np.asarray(all_score))


def grid_search(outfile='grid_search_3.csv', overwrite_rec=False, method_suffix='hpo_double', parallel=False):
    """
    Scoring options
    - number of blobs
    - volume per blob
    - radius around each atom
    - aggregation in this radius: max, sum, onion
    - aggregation over different atoms : for now its max
    - aggregation of the residue values across blobs : for now its max
    - smoothing ?

    Grid search aims to do that. Each score computation is approx 3 mins (could be less, if only on the train)
    and the number of experiments is 84, so 4 hours. in parallel ~15 mins

    :param outfile:
    :param overwrite_rec:
    :return:
    """

    if overwrite_rec:
        with open(outfile, 'w') as _:
            pass
    all_radius = [4., 6.]
    all_n_top_blob = [1, 3, 5]
    all_vol = [None, 150, 300, 600]
    all_grid_aggregation_function = ['max', 'sum']
    to_explore = (all_radius, all_n_top_blob, all_vol, all_grid_aggregation_function)
    todo = list(itertools.product(*to_explore, repeat=1))
    scorer = partial(compute_all_score_dbd5, method_suffix=method_suffix,
                     outrec=None, compute_holo=True, test_set=TEST_SET, do_test_set=False)

    if not parallel:
        for grid_point in todo:
            radius, n_top_blob, vol, grid_aggregation_function = grid_point
            score = scorer(radius=radius, n_top_blob=n_top_blob, vol=vol,
                           grid_aggregation_function=grid_aggregation_function)
            line = f'{radius}, {n_top_blob}, {vol}, {grid_aggregation_function}, {score}'
            print(line)
            with open(outfile, 'a') as f:
                f.write(line + '\n')
    else:
        with Pool() as p:
            all_scores = p.starmap(scorer, todo)
        with open(outfile, 'a') as f:
            for (grid_point, score) in zip(todo, all_scores):
                radius, n_top_blob, vol, grid_aggregation_function = grid_point
                line = f'{radius}, {n_top_blob}, {vol}, {grid_aggregation_function}, {score}'
                f.write(line + '\n')
    all_radius = [6., 8., 10.]
    all_n_top_blob = [1, 3, 5]
    all_vol = [None, 150, 300, 600]
    all_grid_aggregation_function = ['onion']
    to_explore = (all_radius, all_n_top_blob, all_vol, all_grid_aggregation_function)
    todo = list(itertools.product(*to_explore, repeat=1))
    if not parallel:
        for grid_point in todo:
            radius, n_top_blob, vol, grid_aggregation_function = grid_point
            score = scorer(radius=radius, n_top_blob=n_top_blob, vol=vol,
                           grid_aggregation_function=grid_aggregation_function)
            line = f'{radius}, {n_top_blob}, {vol}, {grid_aggregation_function}, {score}'
            print(line)
            with open(outfile, 'a') as f:
                f.write(line + '\n')
    else:
        with Pool() as p:
            all_scores = p.starmap(scorer, todo)
        with open(outfile, 'a') as f:
            for (grid_point, score) in zip(todo, all_scores):
                radius, n_top_blob, vol, grid_aggregation_function = grid_point
                line = f'{radius}, {n_top_blob}, {vol}, {grid_aggregation_function}, {score}'
                f.write(line + '\n')


def get_names_epipred(pdb_code, method_suffix, data_path='../data/epipred'):
    pdb_code_dir = os.path.join(data_path, pdb_code)
    filename_pdb = os.path.join(pdb_code_dir, f'receptor.pdb')
    filename_npz = os.path.join(pdb_code_dir, f'holo_pred_{method_suffix}.npz')
    filename_supervision = os.path.join(pdb_code_dir, f'receptor_patch.p')
    return filename_pdb, filename_npz, filename_supervision


def compute_all_score_epipred(radius=5., n_top_blob=3, vol=None,
                              grid_aggregation_function='max', method_suffix='hpo_double',
                              outrec=None):
    """
    :param radius:
    :param n_top_blob:
    :param vol:
    :param grid_aggregation_function:
    :param outrec:
    :return:
    """
    # create a clean one
    if outrec is not None:
        with open(outrec, 'w') as _:
            pass

    data_path = '../data/epipred'
    all_score = list()
    done = 0
    for pdb_code in custom_list_dir(data_path):
        filename_pdb, filename_npz, filename_supervision = get_names_epipred(pdb_code=pdb_code,
                                                                             method_suffix=method_suffix,
                                                                             data_path=data_path)
        if os.path.exists(filename_npz) and os.path.exists(filename_supervision):
            done += 1
            line = f'pdbcode: {pdb_code}_receptor'
            print(line)
            print(f'done {done}')
            if outrec is not None:
                with open(outrec, 'a') as f:
                    f.write(line + '\n')
            prediction, supervision, score = compute_one_score(filename_pdb=filename_pdb,
                                                               filename_npz=filename_npz,
                                                               filename_supervision=filename_supervision,
                                                               radius=radius,
                                                               n_top_blob=n_top_blob,
                                                               vol=vol,
                                                               grid_aggregation_function=grid_aggregation_function,
                                                               outrec=outrec)
            all_score.append(score)

    return np.nanmean(np.asarray(all_score))


def compute_one_score_pinet(filename_prediction, filename_supervision, outrec=None):
    pinet_prediction_projected = pickle.load(open(filename_prediction, 'rb'))
    pinet_prediction_projected = {(key[0], str(key[1])): val for key, val in pinet_prediction_projected.items()}
    supervision = pickle.load(open(filename_supervision, 'rb'))
    supervision = {(key[0], str(key[1])): val for key, val in supervision.items()}

    # Take all blobs and add them in a consistent order. return an array (nres, nblobs + 1(supervision))
    aligned_predictions = list()
    for key, value in supervision.items():
        if key in pinet_prediction_projected:
            aligned_predictions.append((value, pinet_prediction_projected[key]))
    aligned_predictions = np.asarray(aligned_predictions)
    supervised = aligned_predictions[:, 0][None, ...]
    preds = aligned_predictions[:, 1][None, ...]
    mean_positive = supervised.mean()
    aucpr_all = distance.cdist(supervised, preds, metric=average_precision_score)
    aucpr_all = np.squeeze(aucpr_all)
    correlation_all = distance.cdist(supervised, preds, metric='correlation')
    correlation_all = np.squeeze(correlation_all)

    supervised = supervised.squeeze()
    agg_preds = preds.squeeze()
    aucpr_2 = average_precision_score(supervised, agg_preds)
    aucpr_3 = aucpr(supervised, agg_preds)
    try:
        auroc = roc_auc_score(supervised, agg_preds)
    except:
        auroc = np.nan

    lines = []
    lines.append(f'mean_positive: {mean_positive}' + '\n')
    lines.append(f'all_together: {aucpr_all}' + '\n')
    lines.append(f'correlation_all_together: {correlation_all}' + '\n')
    lines.append(f'average_precision: {aucpr_2}' + '\n')
    lines.append(f'homemade_aucpr: {aucpr_3}' + '\n')
    lines.append(f'auroc: {auroc}' + '\n')
    lines.append('\n')

    lines = ''.join(lines)

    if outrec is not None:
        with open(outrec, 'a') as f:
            f.write(lines)
    # This is to be fine-tuned, we have to choose one final metric and stick to it.
    # Probs follow PInet
    final_score = aucpr_all
    return pinet_prediction_projected, supervision, final_score


def get_names_dbd5_pinet(pdb_code, receptor=True, holo=True, method='prob_patch', data_path='../data/dbd5'):
    holo_str = 'b' if holo else 'u'
    receptor_str = 'receptor' if receptor else 'ligand'
    receptor_code = 'r' if receptor else 'l'
    pinet_name = f"{pdb_code}-{receptor_code}"
    pdb_code_dir = os.path.join(data_path, pdb_code)
    filename_pred = os.path.join(pdb_code_dir, f'{receptor_str}_{holo_str}_{method}.p')
    filename_supervision = os.path.join(pdb_code_dir, f'{receptor_str}_b_patch.p')
    return pinet_name, filename_pred, filename_supervision


def compute_all_score_dbd5_pinet(method_suffix='prob_patch', outrec=None,
                                 compute_holo=True, test_set=TEST_SET, do_test_set=False):
    """
    :param outrec:
    :param compute_holo:
    :param filter_list: this is a list of PDBs to include/exclude.
     For now one has to hardcode modify the behavior by adding a "not"
    :return:
    """
    if outrec is not None:
        with open(outrec, 'w') as _:
            pass

    data_path = '../data/dbd5'
    all_score = list()
    done = 0
    for pdb_code in custom_list_dir(data_path):

        pinet_name, filename_pred, filename_supervision = get_names_dbd5_pinet(pdb_code=pdb_code,
                                                                               method=method_suffix,
                                                                               receptor=True,
                                                                               holo=compute_holo,
                                                                               data_path=data_path)
        in_test = pinet_name in test_set
        if (in_test and do_test_set) or (not in_test and not do_test_set):
            if os.path.exists(filename_pred) and os.path.exists(filename_supervision):
                done += 1
                line = f'pdbcode: {pdb_code}_receptor'
                print(line)
                print(f'done {done}')
                if outrec is not None:
                    with open(outrec, 'a') as f:
                        f.write(line + '\n')
                prediction, supervision, score = compute_one_score_pinet(filename_prediction=filename_pred,
                                                                         filename_supervision=filename_supervision,
                                                                         outrec=outrec)
                all_score.append(score)

        pinet_name, filename_pred, filename_supervision = get_names_dbd5_pinet(pdb_code=pdb_code,
                                                                               method=method_suffix,
                                                                               receptor=False,
                                                                               holo=compute_holo,
                                                                               data_path=data_path)
        in_test = pinet_name in test_set
        if (in_test and do_test_set) or (not in_test and not do_test_set):
            if os.path.exists(filename_pred) and os.path.exists(filename_supervision):
                done += 1
                line = f'pdbcode: {pdb_code}_ligand'
                print(line)
                print(f'done {done}')
                if outrec is not None:
                    with open(outrec, 'a') as f:
                        f.write(line + '\n')
                prediction, supervision, score = compute_one_score_pinet(filename_prediction=filename_pred,
                                                                         filename_supervision=filename_supervision,
                                                                         outrec=outrec)
                all_score.append(score)

    return np.nanmean(np.asarray(all_score))


def get_names_epipred_pinet(pdb_code, method='prob_patch', data_path='../data/epipred'):
    pdb_code_dir = os.path.join(data_path, pdb_code)
    filename_pred = os.path.join(pdb_code_dir, f'receptor_{method}.p')
    filename_supervision = os.path.join(pdb_code_dir, f'receptor_patch.p')
    return filename_pred, filename_supervision


def compute_all_score_epipred_pinet(method_suffix='prob_patch', outrec=None):
    """
    :param outrec:
    :return:
    """
    if outrec is not None:
        with open(outrec, 'w') as _:
            pass

    data_path = '../data/epipred'
    all_score = list()
    done = 0
    for pdb_code in custom_list_dir(data_path):

        filename_pred, filename_supervision = get_names_epipred_pinet(pdb_code=pdb_code,
                                                                      method=method_suffix,
                                                                      data_path=data_path)
        if os.path.exists(filename_pred) and os.path.exists(filename_supervision):
            done += 1
            line = f'pdbcode: {pdb_code}_receptor'
            print(line)
            print(f'done {done}')
            if outrec is not None:
                with open(outrec, 'a') as f:
                    f.write(line + '\n')
            prediction, supervision, score = compute_one_score_pinet(filename_prediction=filename_pred,
                                                                     filename_supervision=filename_supervision,
                                                                     outrec=outrec)
            all_score.append(score)

    return np.nanmean(np.asarray(all_score))


def rename_all(data_path='../data/dbd5/'):
    """
    We need to change our naming standards because in the train/test, there are some receptor or ligands together.
    :param data_path:
    :return:
    """

    def rename(a, b):
        try:
            os.rename(a, b)
        except FileNotFoundError:
            pass

    def remove(a):
        try:
            os.remove(a)
        except FileNotFoundError:
            pass

    for pdb_code in (custom_list_dir(data_path)):
        rename(os.path.join(data_path, pdb_code, 'apo_pred_hpo.npz'),
               os.path.join(data_path, pdb_code, 'receptor_apo_pred_hpo.npz'))
        rename(os.path.join(data_path, pdb_code, 'holo_pred_hpo.npz'),
               os.path.join(data_path, pdb_code, 'receptor_holo_pred_hpo.npz'))
        rename(os.path.join(data_path, pdb_code, 'apo_pred_gaussian_blur.npz'),
               os.path.join(data_path, pdb_code, 'receptor_apo_pred_gaussian_blur.npz'))
        rename(os.path.join(data_path, pdb_code, 'holo_pred_gaussian_blur.npz'),
               os.path.join(data_path, pdb_code, 'receptor_holo_pred_gaussian_blur.npz'))
        rename(os.path.join(data_path, pdb_code, 'apo_pred_hpo_double.npz'),
               os.path.join(data_path, pdb_code, 'receptor_apo_pred_hpo_double.npz'))
        rename(os.path.join(data_path, pdb_code, 'holo_pred_hpo_double.npz'),
               os.path.join(data_path, pdb_code, 'receptor_holo_pred_hpo_double.npz'))
        rename(os.path.join(data_path, pdb_code, 'ligand.pdb'), os.path.join(data_path, pdb_code, 'ligand_b.pdb'))
        rename(os.path.join(data_path, pdb_code, 'ligand_prob.seg'),
               os.path.join(data_path, pdb_code, 'ligand_b_prob.seg'))
        rename(os.path.join(data_path, pdb_code, 'ligand.pts'), os.path.join(data_path, pdb_code, 'ligand_b.pts'))
        rename(os.path.join(data_path, pdb_code, 'ligand.seg'), os.path.join(data_path, pdb_code, 'ligand_b.seg'))
        remove(os.path.join(data_path, pdb_code, 'apo_pred.npz'))
        remove(os.path.join(data_path, pdb_code, 'holo_pred.npz'))

        ############
        rename(os.path.join(data_path, pdb_code, 'receptor_apo_pred_hpo.npz'),
               os.path.join(data_path, pdb_code, 'receptor_u_pred_hpo.npz'))
        rename(os.path.join(data_path, pdb_code, 'receptor_apo_pred_hpo_double.npz'),
               os.path.join(data_path, pdb_code, 'receptor_u_pred_hpo_double.npz'))
        rename(os.path.join(data_path, pdb_code, 'receptor_apo_pred_gaussian_blur.npz'),
               os.path.join(data_path, pdb_code, 'receptor_u_pred_gaussian_blur.npz'))
        rename(os.path.join(data_path, pdb_code, 'receptor_holo_pred_hpo.npz'),
               os.path.join(data_path, pdb_code, 'receptor_b_pred_hpo.npz'))
        rename(os.path.join(data_path, pdb_code, 'receptor_holo_pred_hpo_double.npz'),
               os.path.join(data_path, pdb_code, 'receptor_b_pred_hpo_double.npz'))
        rename(os.path.join(data_path, pdb_code, 'receptor_holo_pred_gaussian_blur.npz'),
               os.path.join(data_path, pdb_code, 'receptor_b_pred_gaussian_blur.npz'))
        rename(os.path.join(data_path, pdb_code, 'ligand_holo_pred_hpo_double.npz'),
               os.path.join(data_path, pdb_code, 'ligand_b_pred_hpo_double.npz.npz'))


def project_to_b(mapping, pdb, outname):
    # mapping: dict() (chain, resid): val
    cmd.reinitialize()
    cmd.load(pdb, 'mypdb')
    for chain, resid in mapping.keys():
        val = mapping[(chain, resid)]
        cmd.alter(f'mypdb and resi {resid} and chain {chain}', f'b={val}')
    if outname is not None:
        cmd.save(outname, selection='mypdb')


"""
3D validation
"""


def get_3D_supervision(receptor_pdb, ligand_pdb, selection=None, distance_thresh=6):
    # First get the ligands and ligand_atoms
    cmd.reinitialize()
    cmd.load(ligand_pdb, 'ligand')
    if selection is None:
        pymolsel = 'ligand and polymer.protein'
    else:
        pymolsel = f'ligand and polymer.protein and {selection}'
    pymolspace = {'atoms_name_ligand': [], 'resname_ligand': []}
    cmd.iterate(pymolsel,
                'atoms_name_ligand.append(name); resname_ligand.append(resn)',
                space=pymolspace)
    atoms_name_ligand = np.asarray(pymolspace['atoms_name_ligand'])
    resname_ligand = np.asarray(pymolspace['resname_ligand'])
    ligand_coords = cmd.get_coords(selection=pymolsel)

    cmd.load(receptor_pdb, 'receptor')
    receptor_coords = cmd.get_coords(selection='receptor and polymer.protein')

    dm = distance.cdist(receptor_coords, ligand_coords)
    ligand_distances = np.min(dm, axis=0)
    distance_selection = ligand_distances < distance_thresh
    selected_coords = ligand_coords[distance_selection]
    selected_resn = resname_ligand[distance_selection]
    selected_atoms_names = atoms_name_ligand[distance_selection]
    return selected_coords, selected_resn, selected_atoms_names


def split_coords(coords, resn, atom_names, atomtypes=ATOMTYPES):
    """
    This is used to split coords by channel. We cannot use directly the data processing code as it only part of the
    Complex object
    :param coords:
    :param resn:
    :param atom_names:
    :param atomtypes:
    :return:
    """
    # First we need to build the correct atomtype dict, the one we have is {channel : list of (resn, atom)}
    atom_mapping = dict()
    for i, (channel, atom_list) in enumerate(atomtypes.items()):
        for atom in atom_list:
            atom_mapping[atom] = i
    channels_ids = list()
    for res, atom in zip(resn, atom_names):
        if (res, atom) in atom_mapping:
            channels_ids.append(atom_mapping[(res, atom)])
        else:
            channels_ids.append(-1)
    channels_ids = np.asarray(channels_ids)
    sel = channels_ids >= 0
    coords = coords[sel]
    channels_ids = channels_ids[sel]
    return coords, channels_ids


def compute_distributions(lig_coords, pocket_coords, pocket_probs):
    """
    :param lig_coords: coords of a ligand shape n,3
    :param pocket_coords: list of pockets coords of shape n_i, 3
    :param pocket_probs: list of probs vectors for each pocket n_i, 6
    :return:
    """
    all_distribs = list()
    for pocket_coord, pocket_prob in zip(pocket_coords, pocket_probs):
        # For each ligand atom, get the best assignment
        dm = distance.cdist(lig_coords, pocket_coord)
        best_assignments = np.argmin(dm, axis=1)
        shortest_distance = np.take_along_axis(dm, np.expand_dims(best_assignments, axis=1), axis=1).squeeze()
        shortest_distance = np.atleast_1d(shortest_distance)
        # Then if this atom is close enough, look at the predicted distribution for each ligand atom of this type,
        # in this pocket
        selected = best_assignments[shortest_distance < 2]
        distrib = pocket_prob[selected]
        # Finally remove the void prediction and renormalize
        cropped = distrib[:, :5]
        distrib = cropped / np.sum(cropped, axis=1)[:, None]
        all_distribs.append(distrib)
    # Then aggregate the resulting values across blobs.
    all_distribs = np.concatenate(all_distribs, axis=0)
    return all_distribs


def compute_one_score_3d(receptor_npz, receptor_pdb, ligand_pdb, n_top_blobs=1, return_channels=False, outrec=None,
                         vol=300):
    """
    :param scop_dir:
    :param holo:
    :param precomputed_lig_info: to provide precomputed lig centers, should unwrap into two iterables :
    the first one with ligand ids, the second one with their coords
    :param cavity: If we want to use Volsite cavity instead of ligands coordinates.
    We can also ask for ligands_cavity that are ligands who generated volsite cavities
    :return: iterable of names and results of the same length
    """

    # Get Ligand locations
    ligcoords, lig_resn, lig_atomsname = get_3D_supervision(receptor_pdb, ligand_pdb)
    if return_channels:
        ligcoords, lig_channels_ids = split_coords(ligcoords, lig_resn, lig_atomsname)

    # Get predictions and if empty return a placeholder
    try:
        pockets = np.load(receptor_npz)
        pocket_coords, pocket_probs = get_pocket_coords(pockets, hd=True, vol=vol, return_channels=return_channels)
    except FileNotFoundError:
        pocket_coords, pocket_probs = [], []
    pocket_coords, pocket_probs = pocket_coords[:n_top_blobs], pocket_probs[:n_top_blobs]

    if len(pocket_coords) == 0:
        return 100 * np.ones(shape=len(ligcoords)), np.zeros(shape=len(ligcoords))

    # Shape of dccs : (n_cavities, n_pockets)
    if not return_channels:
        dccs, dvos = compute_score([ligcoords], pocket_coords, pocket_probs)
        if outrec is not None:
            with open(outrec, 'a') as f:
                f.write(f'dcc: {dccs[0]}\n')
                f.write(f'dvo: {dvos[0]}\n')
                f.write('\n')
        return dccs, dvos

    else:
        # Compute DCC score by channel
        for i, channel in enumerate(ATOMTYPES):
            if i in lig_channels_ids:
                # Do same computation for the selected channels
                ligcoords_channel = ligcoords[lig_channels_ids == i]
                pocket_probs_channel = [pocket[:, i] for pocket in pocket_probs]
                dccs_channel, dvos_channel = compute_score([ligcoords_channel], pocket_coords,
                                                           pocket_probs_channel)

            else:
                dccs_channel, dvos_channel = [0], [0]
            if outrec is not None:
                with open(outrec, 'a') as f:
                    f.write(f'dcc_{channel}: {dccs_channel[0]}\n')
                    f.write(f'dvo_{channel}: {dvos_channel[0]}\n')
        # Compute overall DCC
        pocket_probs_overall = [1 - pocket[:, -1] for pocket in pocket_probs]
        dccs_overall, dvos_overall = compute_score([ligcoords], pocket_coords, pocket_probs_overall)
        if outrec is not None:
            with open(outrec, 'a') as f:
                f.write(f'dcc_overall: {dccs_overall[0]}\n')
                f.write(f'dvo_overall: {dvos_overall[0]}\n')
                f.write('\n')

        # Get distributions values, we recompute it with a larger volume and maybe more top?
        try:
            pockets = np.load(receptor_npz)
            pocket_coords, pocket_probs = get_pocket_coords(pockets, hd=True, vol=1200, return_channels=return_channels)
        except FileNotFoundError:
            pocket_coords, pocket_probs = [], []
        pocket_coords, pocket_probs = pocket_coords[:n_top_blobs], pocket_probs[:n_top_blobs]

        for i, channel in enumerate(ATOMTYPES):
            if i in lig_channels_ids:
                ligcoords_channel = ligcoords[lig_channels_ids == i]
                distrib_channel = compute_distributions(ligcoords_channel, pocket_coords,
                                                        pocket_probs)
                ALL_DISTRIB_RESULTS[channel].append(distrib_channel)
        return dccs_overall, dvos_overall


def get_names_dbd5_3d(pdb_code, method_suffix, receptor=True, holo=True, data_path='../data/dbd5'):
    holo_str = 'b' if holo else 'u'
    # other_holo_str = 'u' if holo else 'b'
    receptor_str = 'receptor' if receptor else 'ligand'
    receptor_code = 'r' if receptor else 'l'
    ligand_str = 'ligand' if receptor else 'receptor'
    pinet_name = f"{pdb_code}-{receptor_code}"
    pdb_code_dir = os.path.join(data_path, pdb_code)
    filename_npz = os.path.join(pdb_code_dir, f'{receptor_str}_{holo_str}_pred_{method_suffix}.npz')
    filename_rec_pdb = os.path.join(pdb_code_dir, f'{receptor_str}_{holo_str}.pdb')
    filename_lig_pdb = os.path.join(pdb_code_dir, f'{ligand_str}_{holo_str}.pdb')
    return pinet_name, filename_npz, filename_rec_pdb, filename_lig_pdb


def compute_all_score_dbd5_3d(method_suffix='hpo_double', outrec=None,
                              compute_holo=True, test_set=TEST_SET, do_test_set=False, return_channels=False, vol=300):
    """
    :param radius:
    :param n_top_blob:
    :param vol:
    :param grid_aggregation_function:
    :param outrec:
    :param compute_holo:
    :param do_test_set: if we want inference on the test set
    :return:
    """
    # create a clean one
    if outrec is not None:
        with open(outrec, 'w') as _:
            pass

    data_path = '../data/dbd5'
    all_score = list()
    done = 0
    for pdb_code in custom_list_dir(data_path):
        # Get the prediction for the receptor and ligand files.
        # We only get it if it is present in the filter set (train/test)
        pinet_name, filename_npz, filename_rec_pdb, filename_lig_pdb = get_names_dbd5_3d(pdb_code=pdb_code,
                                                                                         method_suffix=method_suffix,
                                                                                         receptor=True,
                                                                                         holo=compute_holo,
                                                                                         data_path=data_path)
        in_test = pinet_name in test_set
        if (in_test and do_test_set) or (not in_test and not do_test_set):
            if os.path.exists(filename_npz):
                done += 1
                line = f'pdbcode: {pdb_code}_receptor'
                print(line)
                if outrec is not None:
                    with open(outrec, 'a') as f:
                        f.write(line + '\n')
                dccs, dvos = compute_one_score_3d(receptor_pdb=filename_rec_pdb,
                                                  receptor_npz=filename_npz,
                                                  ligand_pdb=filename_lig_pdb,
                                                  return_channels=return_channels,
                                                  outrec=outrec, vol=vol)
                print(f'done {done}')
                all_score.append(dccs[0])

        pinet_name, filename_npz, filename_rec_pdb, filename_lig_pdb = get_names_dbd5_3d(pdb_code=pdb_code,
                                                                                         method_suffix=method_suffix,
                                                                                         receptor=False,
                                                                                         holo=compute_holo,
                                                                                         data_path=data_path)
        in_test = pinet_name in test_set
        if (in_test and do_test_set) or (not in_test and not do_test_set):
            if os.path.exists(filename_npz):
                done += 1
                line = f'pdbcode: {pdb_code}_ligand'
                print(line)
                if outrec is not None:
                    with open(outrec, 'a') as f:
                        f.write(line + '\n')
                dccs, dvos = compute_one_score_3d(receptor_pdb=filename_rec_pdb,
                                                  receptor_npz=filename_npz,
                                                  ligand_pdb=filename_lig_pdb,
                                                  return_channels=return_channels,
                                                  outrec=outrec,
                                                  vol=vol)
                print(f'done {done}')
                all_score.append(dccs[0])

    return np.nanmean(np.asarray(all_score))


def get_names_epipred_3d(pdb_code, method_suffix, data_path='../data/epipred'):
    pdb_code_dir = os.path.join(data_path, pdb_code)
    filename_npz = os.path.join(pdb_code_dir, f'holo_pred_{method_suffix}.npz')
    filename_rec_pdb = os.path.join(pdb_code_dir, f'receptor.pdb')
    filename_lig_pdb = os.path.join(pdb_code_dir, f'ligand.pdb')
    return filename_npz, filename_rec_pdb, filename_lig_pdb


def compute_all_score_epipred_3d(method_suffix='hpo_double', outrec=None, return_channels=False, vol=300):
    """
    :param radius:
    :param n_top_blob:
    :param vol:
    :param grid_aggregation_function:
    :param outrec:
    :return:
    """
    # create a clean one
    if outrec is not None:
        with open(outrec, 'w') as _:
            pass

    data_path = '../data/epipred'
    all_score = list()
    done = 0
    for pdb_code in custom_list_dir(data_path):
        filename_npz, filename_rec_pdb, filename_lig_pdb = get_names_epipred_3d(pdb_code=pdb_code,
                                                                                method_suffix=method_suffix,
                                                                                data_path=data_path)
        if os.path.exists(filename_npz):
            done += 1
            line = f'pdbcode: {pdb_code}_receptor'
            print(line)
            if outrec is not None:
                with open(outrec, 'a') as f:
                    f.write(line + '\n')
            dccs, dvos = compute_one_score_3d(receptor_pdb=filename_rec_pdb,
                                              receptor_npz=filename_npz,
                                              ligand_pdb=filename_lig_pdb,
                                              return_channels=return_channels,
                                              outrec=outrec,
                                              vol=vol)
            print(f'done {done}')
            all_score.append(dccs[0])

    return np.nanmean(np.asarray(all_score))


if __name__ == '__main__':
    pass
    # build_database()
    # compute_all_npz_epipred(exp_path=os.path.join(script_dir, '../results/experiments/HPO.exp'),
    #                         overwrite=True)
    # compute_all_npz_dbd5(exp_path=os.path.join(script_dir, '../results/experiments/HPO.exp'),
    #                      method_suffix='hpo_double',
    #                      overwrite=False)

    # rename_all()

    # This is pre-hpo, to get first values
    # compute_all_score_dbd5(radius=8., n_top_blob=3, vol=None,
    #                        grid_aggregation_function='onion', method_suffix='hpo_double',
    #                        outrec='first_dbd5.rec', compute_holo=True, do_test_set=True)
    simple_dbd5 = compute_all_score_dbd5_pinet(method_suffix='prob_final_double_patch',
                                               outrec='first_dbd5_pinet_final.rec', compute_holo=True, do_test_set=True)
    double_dbd5 = compute_all_score_dbd5_pinet(method_suffix='prob_final_patch',
                                               outrec='honest_dbd5_pinet_final.rec', compute_holo=True,
                                               do_test_set=True)

    # compute_all_score_epipred(radius=8., n_top_blob=3, vol=None, grid_aggregation_function='onion',
    #                           method_suffix='hpo_double', outrec='first_epipred.rec')
    simple_epipred = compute_all_score_epipred_pinet(method_suffix='prob_final_double_patch',
                                                     outrec='first_epipred_pinet_final.rec')
    double_epipred = compute_all_score_epipred_pinet(method_suffix='prob_final_patch',
                                                     outrec='honest_epipred_pinet_final.rec')
    print(simple_dbd5, double_dbd5, simple_epipred, double_epipred)
    # grid_search(method_suffix='gaussian_blur', overwrite_rec=True, parallel=True, outfile='grid_searc h.csv')

    # Post-HPO, we get the best settings : 1 blob, no vol, larger radius for onion and hpo/hpo_double
    # compute_all_score_dbd5(radius=8., n_top_blob=1, vol=None,
    #                        grid_aggregation_function='onion', method_suffix='hpo_double',
    #                        outrec='best_dbd5.rec', compute_holo=True, do_test_set=True)
    # compute_all_score_epipred(radius=8., n_top_blob=1, vol=None, grid_aggregation_function='onion',
    #                           method_suffix='hpo_double', outrec='best_epipred.rec')

    # print(agg_pred)
    # agg_pred = pickle.load(open('misc/1A2K/mine.p', 'rb'))
    # project_to_b(agg_pred, 'misc/1A2K/receptor_b.pdb', 'misc/1A2K/mine.pdb')
    # agg_pred = pickle.load(open('misc/1A2K/theirs.p', 'rb'))
    # project_to_b(agg_pred, 'misc/1A2K/receptor_b.pdb', 'misc/1A2K/theirs.pdb')
    # agg_pred = pickle.load(open('misc/1A2K/theirs_shifted.p', 'rb'))
    # project_to_b(agg_pred, 'misc/1A2K/receptor_b.pdb', 'misc/1A2K/theirs_shifted.pdb')

    # agg_pred = pickle.load(open('misc/1ML0/receptor_b_dropbox_prob_patch.p', 'rb'))
    # print(agg_pred)
    # project_to_b(agg_pred, 'misc/1ML0/receptor_b.pdb', 'misc/1ML0/receptor_honest.pdb')
    #
    # agg_pred = pickle.load(open('misc/1ML0/receptor_b_dropbox_prob_double_patch.p', 'rb'))
    # project_to_b(agg_pred, 'misc/1ML0/receptor_b.pdb', 'misc/1ML0/receptor_first.pdb')
    #

    # agg_pred = pickle.load(open('misc/epipred/1ahw/receptor_dropbox_prob_patch.p', 'rb'))
    # print(agg_pred)
    # project_to_b(agg_pred, 'misc/epipred/1ahw/receptor.pdb', 'misc/epipred/1ahw/receptor_honest.pdb')

    # agg_pred = pickle.load(open('misc/epipred/1ahw/receptor_dropbox_prob_double_patch.p', 'rb'))
    # project_to_b(agg_pred, 'misc/epipred/1ahw/receptor.pdb', 'misc/epipred/1ahw/receptor_first.pdb')

    # agg_pred = pickle.load(open('misc/3vlb/ligand_b_dropbox_prob_patch.p', 'rb'))
    # project_to_b(agg_pred, 'misc/3vlb/ligand_b.pdb', 'misc/3vlb/ligand_honest.pdb')
    # get_3D_supervision(receptor_pdb='../data/dbd5/CP57/receptor_b.pdb', ligand_pdb='../data/dbd5/CP57/ligand_b.pdb')
    # dccs, dvos = compute_one_score_3d(receptor_pdb='../data/dbd5/CP57/receptor_b.pdb',
    #                                   ligand_pdb='../data/dbd5/CP57/ligand_b.pdb',
    #                                   receptor_npz='../data/dbd5/CP57/receptor_b_pred_hpo_double.npz')
    # print(dccs, dvos)

    # compute_all_score_dbd5_3d(outrec='dbd5_dcc_hpo_double.rec', method_suffix='hpo_double', do_test_set=True)
    # compute_all_score_epipred_3d(outrec='epipred_dcc_hpo_double.rec', method_suffix='hpo_double')

    # compute_all_score_dbd5_3d(outrec='dbd5_channels.rec', method_suffix='hpo_double',
    #                           do_test_set=True, return_channels=True, vol=300)
    # compute_all_score_dbd5_3d(outrec='dbd5_channels.rec', method_suffix='hpo_double',
    #                           do_test_set=False, return_channels=True, vol=300)
    # print(ALL_DISTRIB_RESULTS)
    """
    pairwise_matrix = np.zeros(shape=(5, 5))
    reverse_atomtype_map = {channel: i for i, channel in enumerate(ATOMTYPES)}
    # all_distribs is a list of arrays of shape n_i, 6. We want to make it a single 6, array
    for channel, all_distribs in ALL_DISTRIB_RESULTS.items():
        all_distribs = np.concatenate(all_distribs, axis=0)
        # How many atoms for each type
        print(all_distribs.shape)
        avg_distrib = np.mean(all_distribs, axis=0)
        ALL_DISTRIB_RESULTS[channel] = avg_distrib
        pairwise_matrix[reverse_atomtype_map[channel]] = avg_distrib

    print(ATOMTYPES.keys())
    print(pairwise_matrix)
    """
#     dict_keys(['HAD', 'CA', 'COB', 'POS', 'NEG'])
# A = [[0.25710052, 0.18294287, 0.38374564, 0.11778748, 0.05842343],
#      [0.259195, 0.18456776, 0.38391948, 0.11616282, 0.05615518],
#      [0.22867979, 0.16669032, 0.44129226, 0.10426994, 0.05906767],
#      [0.25935495, 0.18196583, 0.39336604, 0.11515831, 0.05015484],
#      [0.23769444, 0.17400837, 0.41974851, 0.11021471, 0.05833374]]
# A = np.asarray(A)
# C = (A - A.mean(axis=0)) / (A.std(axis=0))
