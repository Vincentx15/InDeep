import collections
import os
import shutil
import sys
from tqdm import tqdm
import subprocess
import glob

from pymol import cmd

import numpy as np
from scipy.spatial import distance

try:
    ICHEM_CMD_STR = f"{os.environ['ICHEM_DIR']}/IChem"
except KeyError:
    ICHEM_CMD_STR = None


def get_ligands(pymol_sel='inpdb'):
    myspace = {'resids': [], 'resnames': [], 'chains': []}
    cmd.iterate(f'{pymol_sel} and not polymer.protein',
                'resids.append(resi); resnames.append(resn); chains.append(chain)', space=myspace)
    resids = np.array(myspace['resids'])
    _, ind = np.unique(resids, return_index=True)
    resids = resids[ind]
    resnames = np.asarray(myspace['resnames'])[ind]
    chains = np.asarray(myspace['chains'])[ind]
    return list(zip(resnames, resids, chains))


def protein_pymolsel_to_mol2(pymol_name, dump_filename, temp_filename='tmp_pdb.pdb'):
    if not os.path.exists(dump_filename):
        cmd.save(filename=temp_filename, selection=f'{pymol_name} and polymer.protein')
        chimera_call = 'chimera --nogui --script'.split(' ')
        args = [f'chimera_pdbtomol2.py {temp_filename} {dump_filename}']
        result = subprocess.run(chimera_call + args, check=True)
        os.remove(temp_filename)


def ligandpdb_to_mol2(ligand_path, dump_filename):
    chimera_call = 'chimera --nogui --script'.split(' ')
    args = [f'chimera_ligpdbtomol2.py {ligand_path} {dump_filename}']
    subprocess.run(chimera_call + args)


def deal_with_volsite_output(dump_path, name_ligand='apo_cavity_0'):
    """

    :param dump_path:
    :param name_ligand:
    :return: 1 if volsite found a ligand
    """
    # Filter volsite outputs to keep the ones at 6A cutoff and discard others.
    # We move the 6A ones to dump dir
    glob_list = glob.glob('CAVITY*.mol2')
    glob_list = sorted(glob_list)
    glob_6 = list()
    glob_others = list()
    for cavity in glob_list:
        if cavity.split('_')[2].split('.')[0] == '6':
            glob_6.append(cavity)
        else:
            glob_others.append(cavity)
    for j, cavity in enumerate(glob_6):
        ligand_dump = os.path.join(dump_path, f'{name_ligand}_{j}.mol2')
        volsite_dump = os.path.join(os.getcwd(), cavity)
        shutil.move(volsite_dump, ligand_dump)
    # Clean other results :
    for j, cavity in enumerate(glob_others):
        volsite_dump = os.path.join(os.getcwd(), cavity)
        os.remove(volsite_dump)

    if glob_6:
        return 1
    return 0


def get_cavities(dump_dir, num_ligs, holo=True, recompute=False):
    """
    Iterate over the ligands in dump_dir and compute the corresponding volsite cavities on the apo/holo structure.
    :param dump_dir: 
    :param num_ligs: 
    :param holo: 
    :param recompute: 
    :return: 
    """
    n_apo = 0
    holo_str = 'holo' if holo else 'apo'
    filename_apo = os.path.join(dump_dir, f'{holo_str}_dump.mol2')
    if os.path.exists(filename_apo):
        path_to_search = os.path.join(dump_dir, f'{holo_str}_cavity_*_0.mol2')
        existing_cavities = set(glob.glob(path_to_search))
        n_existing_apo = len(existing_cavities)
        if not recompute and n_existing_apo > 0:
            return n_existing_apo
        for i in range(num_ligs):
            filename_ligand = os.path.join(dump_dir, f'lig_{i}.mol2')
            cmd_call = f'{ICHEM_CMD_STR} --hydrogen volsite {filename_apo} {filename_ligand}'
            subprocess.run(cmd_call, shell=True)
            success = deal_with_volsite_output(dump_path=dump_dir, name_ligand=f'{holo_str}_cavity_{i}')
            n_apo += success
    return n_apo


def process_one_chen(holo_path, dump_path, apo_path=None, recompute=False):
    """

    :param holo_path:
    :param apo_path:
    :param dump_path:
    :return:
    """
    holo_pymol_name = 'holo_pdb'
    cmd.load(holo_path, holo_pymol_name)

    # Dump the holo protein with chimera
    # We use Chimera for pdb to mol2 on the proteins to deal with charges
    dump_filename_holo = os.path.join(dump_path, 'holo_dump.mol2')
    if recompute or not os.path.exists(dump_filename_holo):
        protein_pymolsel_to_mol2(pymol_name=holo_pymol_name,
                                 dump_filename=dump_filename_holo)

    # Get ligands and dump in mol2 format
    ligs = get_ligands(pymol_sel=holo_pymol_name)
    for i, (resname, resid, chain) in enumerate(ligs):
        filename_ligand = os.path.join(dump_path, f'lig_{i}.mol2')
        if recompute or not os.path.exists(filename_ligand):
            lig_selection = f'{holo_pymol_name} and resname {resname} and resid {resid}'
            if chain != '':
                lig_selection = lig_selection + f' and chain {chain}'
            # Small computations to get an idea of the distance.
            # lig_coords = cmd.get_coords(selection=lig_selection)
            # prot_coords = cmd.get_coords(selection=f'{holo_pymol_name} and polymer.protein')
            # distances = distance.cdist(lig_coords, prot_coords)
            # min_dist = np.min(distances, axis=1)
            # min_dist = np.mean(min_dist)
            # print(min_dist)
            cmd.save(filename=filename_ligand, selection=lig_selection)

    if apo_path is not None:
        # Align apo onto holo pdb and dump with chimera
        dump_filename_apo = os.path.join(dump_path, 'apo_dump.mol2')
        if recompute or not os.path.exists(dump_filename_holo):
            apo_pymol_name = 'apo_pdb'
            cmd.load(apo_path, apo_pymol_name)
            cmd.align(mobile=f'{apo_pymol_name} and polymer.protein',
                      target=f'{holo_pymol_name} and polymer.protein')
            protein_pymolsel_to_mol2(pymol_name=apo_pymol_name,
                                     dump_filename=dump_filename_apo)
    cmd.reinitialize()
    return len(ligs)


def process_chen(holo_path, dump_path, apo_path=None, recompute=False):
    # create the rec file
    out_rec = os.path.join(dump_path, 'chen_dev.rec')
    with open(out_rec, 'w') as outfile:
        pass

    # Iterate through the db and log in rec file.
    # Dump holo and apo mol2 as well as ligands.mol2.
    # Then turn the ligands into volsite cavities mol2 files
    for holo_pdb in tqdm(os.listdir(holo_path)):
        scop = holo_pdb.split('_')[0]
        for apo_pdb in os.listdir(apo_path):
            if apo_pdb.startswith(scop):
                dump_scop = os.path.join(dump_path, scop)
                holo = os.path.join(holo_path, holo_pdb)
                apo = os.path.join(apo_path, apo_pdb)

                if not os.path.exists(dump_scop):
                    os.mkdir(dump_scop)
                # For the pair, dump the holo, aligned apo and ligands
                num_ligs = process_one_chen(holo_path=holo,
                                            apo_path=apo,
                                            dump_path=dump_scop,
                                            recompute=recompute)
                with open(out_rec, 'a') as outfile:
                    outfile.write(f'scop: {scop}\n')
                    outfile.write(f'holo: {holo}\n')
                    outfile.write(f'apo: {apo}\n')
                    outfile.write(f'num_ligs: {num_ligs}\n')

                # Run Volsite on Holo
                n_holo = get_cavities(dump_dir=dump_scop, num_ligs=num_ligs, holo=True, recompute=recompute)

                # Run Volsite on Apo
                n_apo = get_cavities(dump_dir=dump_scop, num_ligs=num_ligs, holo=False, recompute=recompute)

                with open(out_rec, 'a') as outfile:
                    outfile.write(f'holo_lig: {n_holo}\n')
                    outfile.write(f'apo_lig: {n_apo}\n')
                    outfile.write('\n')
                break


def process_one_testset(holo_path, ligands_path, dump_path, apo_path=None, recompute=False):
    """

    :param holo_path:
    :param apo_path:
    :param dump_path:
    :return:
    """
    holo_pymol_name = 'holo_pdb'
    cmd.load(holo_path, holo_pymol_name)

    # Dump the holo protein with chimera
    # We use Chimera for pdb to mol2 on the proteins to deal with charges
    dump_filename_holo = os.path.join(dump_path, 'holo_dump.mol2')
    if recompute or not os.path.exists(dump_filename_holo):
        try:
            protein_pymolsel_to_mol2(pymol_name=holo_pymol_name,
                                 dump_filename=dump_filename_holo)
        except:
            with open('failed.txt','w+') as f:
                f.write(f'{dump_filename_holo}\n')
                return
    # Get ligands and dump in mol2 format
    for i, lig_path in enumerate(ligands_path):
        filename_ligand = os.path.join(dump_path, f'lig_{i}.mol2')
        if recompute or not os.path.exists(filename_ligand):
            ligandpdb_to_mol2(ligand_path=lig_path, dump_filename=filename_ligand)

    if apo_path is not None:
        # Align apo onto holo pdb and dump with chimera
        dump_filename_apo = os.path.join(dump_path, 'apo_dump.mol2')
        if recompute or not os.path.exists(dump_filename_apo):
            apo_pymol_name = 'apo_pdb'
            cmd.load(apo_path, apo_pymol_name)
            cmd.align(mobile=f'{apo_pymol_name} and polymer.protein',
                      target=f'{holo_pymol_name} and polymer.protein')
            try:
                protein_pymolsel_to_mol2(pymol_name=apo_pymol_name,
                                 dump_filename=dump_filename_apo)
            except:
                with open('failed_apo.txt','a') as f:
                    f.write(f'{dump_filename_apo}\n')
    cmd.reinitialize()
    return len(ligands_path)


def process_testset(list_of_files='../data/PL-database_test_set.txt',
                    pl_path='../data/PL-database',
                    all_apo_path=None,
                    dump_path='../data/testset_res',
                    recompute=False):
    """

    :param list_of_files:
    :param pl_path:
    :param all_apo_path: The path to a db of pdb files that contain the aligned chain in an unbound chain.
    These files are named : {Uniprot}_{apopartner}_{pdboftheholo}.pdb
    Attention dans le .txt de karen, c'est dans l'ordre différent des noms de pdb : apo_holo_partenaire
    :param dump_path:
    :return:
    """
    # create the rec file
    out_rec = os.path.join(dump_path, 'test.rec')
    with open(out_rec, 'w') as outfile:
        pass

    # Iterate through the db and log in rec file.
    # Dump holo and apo mol2 as well as ligands.mol2.
    # Then turn the ligands into volsite cavities mol2 files
    with open(list_of_files, 'r') as f:
        files_list = f.readlines()

    grouped_test_set = collections.defaultdict(list)
    for file in files_list:
        # Remove \n and P:, L:, and group same ligands together
        P, L = file[:-1].split()
        P, L = P[2:], L[2:]
        grouped_test_set[P].append(L)

    # Apo_dict is a dictionnary holo pdb code : apo pdb code
    apo_dict = dict()
    if all_apo_path is not None:
        all_apos = os.listdir(all_apo_path)
        for apo_pdb in all_apos:
            basename, extension = apo_pdb.split('.')
            if extension != 'pdb':
                continue
            uniprot, apo_code, holo_code = basename.split('_')
            if apo_code != holo_code:
                apo_dict[holo_code] = apo_pdb
    
    # then get the database format
    for protein, ligands in grouped_test_set.items():
        def get_path(pdb_name, pl_path):
            return os.path.join(pl_path, pdb_name[0], pdb_name[1], pdb_name[2], pdb_name[3], pdb_name)
        protein_path = get_path(protein, pl_path=pl_path)
        ligands_path = [get_path(ligand, pl_path=pl_path) for ligand in ligands]
        pdb_code = protein.split('-')[0]
        if pdb_code in apo_dict:
            apo_pdb = apo_dict[pdb_code]
            apo_path = os.path.join(all_apo_path, apo_pdb)
        else:
            apo_path = None
        p_name = protein[:-4]
        dump_dir = os.path.join(dump_path, p_name)
        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)
            pass

        # For the pair, dump the holo and ligands
        num_ligs = process_one_testset(holo_path=protein_path,
                                       ligands_path=ligands_path,
                                       apo_path=apo_path,
                                       dump_path=dump_dir,
                                       recompute=recompute)

        # Run Volsite on Holo
        n_holo = get_cavities(dump_dir=dump_dir, num_ligs=num_ligs, holo=True, recompute=recompute)

        # Run Volsite on Apo
        n_apo = get_cavities(dump_dir=dump_dir, num_ligs=num_ligs, holo=False, recompute=recompute)
    return


if __name__ == '__main__':
    # pdbfile = sys.argv[1]
    # try:
    #     apo = sys.argv[2]
    # except IndexError:
    #     apo = None
    # process_one(holo_path=pdbfile, dump_path='.', apo_path=apo)
    # subprocess.run(f'{ICHEM_CMD_STR} --hydrogen volsite toto.pdb'.split(' '), shell=True)

    process_chen(holo_path='../data/holo', apo_path='../data/apo', dump_path='../data/chen_res')
    # process_testset(dump_path='../data/test_set',
    #                all_apo_path='/c7/scratch2/vmallet/indeep_data/PL_align')
    # process_testset(list_of_files='../data/PL-database_validation_set.txt', dump_path='../data/test_set')
