#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2021-04-13 10:17:47 (UTC+0200)

from pymol import cmd
import numpy as np
import scipy.spatial.distance as distance

cmd.set('fetch_path', '/home/bougui/pdb')


def get_ligands(pymol_sel='ref'):
    myspace = {'resids': [], 'resnames': [], 'chains': []}
    cmd.iterate(f'{pymol_sel} and not polymer.protein',
                'resids.append(resi); resnames.append(resn); chains.append(chain)', space=myspace)
    resids = np.int_(myspace['resids'])
    _, ind = np.unique(resids, return_index=True)
    resids = resids[ind]
    resnames = np.asarray(myspace['resnames'])[ind]
    chains = np.asarray(myspace['chains'])[ind]
    return list(zip(resnames, resids, chains))


def get_aligned_chain():
    pass


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-p', '--pdb', help='PDB file to search for PPI', type=str)
    parser.add_argument('--homo', help='List of homologous PDB codes', type=str, nargs='+')
    args = parser.parse_args()

    cmd.load(args.pdb, 'ref')
    coords_ref = cmd.get_coords('ref and polymer.protein')
    ligands = get_ligands()
    print(f"pdb: {args.pdb}")
    # print(ligands)
    for i, PDBCODE in enumerate(args.homo):
        homo = f'homo_{i}'
        cmd.fetch(PDBCODE, homo, type='pdb')
        rmsd = cmd.align(homo, 'ref and polymer.protein', cycles=0)[0]
        print(f"homo: {PDBCODE}")
        print(f"rmsd: {rmsd}")
        coords_homo = cmd.get_coords(f'{homo} and polymer.protein')
        cdistmat = distance.cdist(coords_homo, coords_ref)
        myspace = {'chains': []}
        cmd.iterate(f'{homo} and polymer.protein', 'chains.append(chain)', space=myspace)
        chains_homo = np.asarray(myspace['chains'])
        sel = cdistmat.min(axis=1) <= rmsd
        chains_unique, count = np.unique(chains_homo[sel], return_counts=True)
        chain_aligned = chains_unique[count.argmax()]
        print(f'chain: {chain_aligned}')
        coords_binders = cmd.get_coords(f'{homo} and polymer.protein and not chain {chain_aligned}')
        for ligand in ligands:
            resname, resid, chain = ligand
            print(f"ligand: {resname} {resid} {chain}")
            if chain == "":
                coords_lig = cmd.get_coords(selection=f'ref and resname {resname} and resid {resid}')
            else:
                coords_lig = cmd.get_coords(selection=f'ref and resname {resname} and resid {resid} and chain {chain}')
            if coords_binders is not None:
                ligclash = distance.cdist(coords_lig, coords_binders).min()
            else:
                ligclash = 9999.99
            print(f"ligclash: {ligclash}")
    print("")
