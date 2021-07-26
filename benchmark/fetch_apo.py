#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2021-04-30 11:09:18 (UTC+0200)

import os
import glob
import numpy as np
import scipy.spatial.distance as distance
from pymol import cmd
scriptdir = os.path.dirname(os.path.realpath(__file__))
cmd.set('fetch_path', cmd.exp_path('~/pdb'), quiet=1)


def get_chain(mobile, target):
    coords = cmd.get_coords(f'{mobile} and polymer.protein')
    coords_ref = cmd.get_coords(f'{target} and polymer.protein')
    cdistmat = distance.cdist(coords, coords_ref)
    myspace = {'chains': []}
    # cmd.iterate(f'{mobile} and polymer.protein', 'chains.append(chain)', space=myspace)
    cmd.iterate_state(1, f'{mobile} and polymer.protein', 'chains.append(chain)', space=myspace)
    chains_out = np.asarray(myspace['chains'])
    sel = cdistmat.min(axis=1) <= 2.
    # print(sel.shape, chains_out.shape, coords.shape)
    chains_unique, count = np.unique(chains_out[sel], return_counts=True)
    chain_aligned = chains_unique[count.argmax()]
    return chain_aligned


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-l', '--list', type=str,
                        default=f'{scriptdir}/data/missing_pdbs_testset.txt')
    args = parser.parse_args()

    with open(args.list, 'r') as apofile:
        lines = apofile.readlines()
        n = len(lines)
        visited = set()
        for i, line in enumerate(lines):
            line = line.strip()
            line = os.path.splitext(line)[0]
            uniprot, apo, holo = line.split('_')
            if holo in visited:
                continue
            if apo == holo:
                continue
            visited.add(holo)
            print(f'id: {i+1}/{n}')
            print(f'uniprot: {uniprot}')
            print(f'holo: {holo}')
            print(f'apo: {apo}')
            holopdb = f'PL-database/{holo[0]}/{holo[1]}/{holo[2]}/{holo[3]}/{holo}-?-{uniprot}.pdb'
            holopdb = glob.glob(holopdb)
            assert len(holopdb) == 1
            holopdb = holopdb[0]
            cmd.load(holopdb, 'holo')
            cmd.fetch(apo, name='apo', type='pdb')
            # cmd.remove("not alt ''+A")
            # cmd.alter("all", "alt=''")
            rmsd = cmd.align(mobile='apo and polymer.protein',
                             target='holo and polymer.protein')[0]
            print(f'rmsd: {rmsd:.3f}')
            chain_aligned = get_chain('apo', 'holo')
            print(f'chain_aligned: {chain_aligned}')
            outfilename = f'{uniprot}_{apo}_{holo}.pdb'
            cmd.save(f'apo/{outfilename}', selection=f'apo and chain {chain_aligned}')
            print(f'apofile: {outfilename}')
            print()
            cmd.reinitialize()
            cmd.set('fetch_path', cmd.exp_path('~/pdb'), quiet=1)
