#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2021-05-03 15:57:58 (UTC+0200)

# To run here: /c7/scratch/bougui/sc-PDB

import os
import sys
import glob
import time

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from learning.predict import Predicter


def get_chains():
    seldict = dict()
    with open('binding_chains.rec', 'r') as chainfile:
        for line in chainfile.readlines():
            if len(line.split()) > 0:
                if line.split()[0] == 'system:':
                    system = line.split()[1]
                if line.split()[0] == 'chains:':
                    chains = line.split()[1:]
                    if len(chains) > 1:
                        selection = f'chain {"+".join(chains)}'
                    else:
                        selection = f'chain {chains[0]}'
                    seldict[system] = selection
    return seldict


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--exp', type=str, help='experiment file for neural network architecture', default=os.path.join(script_dir, '../results/experiments/HPO.exp'))
    args = parser.parse_args()
    seldict = get_chains()
    predicter = Predicter(expfilename=args.exp)
    # predicter.pred_pdb('scPDB_prot_pdb/1a26_1.pdb',
    #                    outname='pockets_indeep/1a26_1',
    #                    print_blobs=False,
    #                    save_mrcs=False,
    #                    save_npz=True)
    pdbs = glob.glob('scPDB_prot_pdb/*.pdb')
    done = set(glob.glob('pockets_indeep/*.npz'))
    f = open('scPDB_log.rec', 'w')
    f.close()
    for i, pdb in enumerate(pdbs):
        sys.stdout.write(f'{i+1}/{len(pdbs)} {pdb}         \r')
        system = os.path.splitext(os.path.split(pdb)[1])[0]
        outpred = f'pockets_indeep/{system}'
        test = f"{outpred}.npz"
        if test not in done:
            try:
                t0 = time.time()
                out_hd, out_pl, origin = predicter.pred_pdb(pdb, outname=outpred, print_blobs=False,
                                                            no_save_mrcs=True, no_save_npz=False)
                delta_t = time.time() - t0
                nx, ny, nz = out_pl.squeeze().shape
                with open('scPDB_log.rec', 'a') as timelog:
                    timelog.write(f'pdb: {pdb}\n')
                    timelog.write(f'chains: {seldict[system]}\n')
                    timelog.write(f'runtime: {delta_t}\n')
                    timelog.write(f'shape: {nx} {ny} {nz}\n')
                    timelog.write(f'size: {nx*ny*nz}\n')
                    timelog.write('\n')
            except RuntimeError:
                print(f"Inference failed for {pdb}")
                with open('scPDB_log.rec', 'a') as timelog:
                    timelog.write(f'pdb: {pdb}\n')
                    timelog.write(f'chains: {seldict[system]}\n')
                    timelog.write('status: failed\n')
                    timelog.write('\n')
        sys.stdout.flush()
