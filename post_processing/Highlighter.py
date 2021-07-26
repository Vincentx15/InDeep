import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

import hashlib
import numpy as np
import scipy
from pymol import cmd

from post_processing import blobber
from learning import utils as lutils, model_factory, predict
from data_processing import utils


class Highlighter:
    def __init__(self, pdb_path=None, experiments_name='large', hd=True, normalize=False, alanine=True):
        """
        This class is intended to detect the important residues for a prediction. It re-does the computations with
        residues mutated in alanine or masked (as in usual ML) and computes the correlation with the result

        :param pdb_path: the one on which to compute prediction. By default the last prediction done with the GUI
        :param experiments_name: The same one as previously
        :param hd: do the prediction on hd or pl predicted blob ?
        :param normalize:
        """
        self.experiments_name = experiments_name

        self.exp_file_path = os.path.join(script_dir, '../results/experiments', f'{experiments_name}.exp')
        hparams = lutils.ConfParser(path_to_exp=self.exp_file_path)
        model = model_factory.model_from_hparams(hparams, load_weights=True)

        self.model = model

        self.hd = hd
        self.normalize = normalize
        self.alanine = alanine
        self.spacing = 1
        self.padding = 8
        self.saving_path = os.path.join(script_dir, '../results/pymol_preds')
        self.temp_path = os.path.join(script_dir, '../deepPPI_pymol_script/deepPPI/temp_dir')
        self.pdb_path = os.path.join(self.temp_path, 'current_prot.pdb') if pdb_path is None else pdb_path
        self.dump_path = 'highlighted_protein.pdb'
        self.load_pdb()

        self.atomtypes = utils.read_atomtypes()

    def __del__(self):
        cmd.delete('protein_highlighted')
        cmd.delete('blob')
        cmd.delete('pocket')
        cmd.delete('select_trunc0')
        cmd.delete('select_trunc')
        cmd.delete('prot')

    @property
    def npz_path(self):
        return os.path.join(self.saving_path, f'{self.pdb_hash}_{self.experiments_name}_norm{int(self.normalize)}.npz')

    @property
    def pdb_hash(self):
        # Remove the .exp from the name to only get the name
        string_name = open(self.pdb_path).read()
        string_name = string_name.encode('utf-8')
        return hashlib.md5(string_name).hexdigest()

    def load_pdb(self):
        cmd.set('retain_order', 1)
        cmd.load(self.pdb_path, 'raw_protein')
        cmd.create(name='protein_highlighted', selection='raw_protein and polymer.protein')
        cmd.remove('hydrogens')
        cmd.delete('raw_protein')
        self.coords = cmd.get_coords(selection='protein_highlighted')
        self.xyz_min, self.xyz_max = self.coords.min(axis=0), self.coords.max(axis=0)

    def predict(self):
        try:
            self.npzfile = np.load(self.npz_path)
        except FileNotFoundError:
            out_hd, out_pl, origin, _ = predict.predict_pdb(self.model,
                                                            self.pdb_path,
                                                            spacing=self.spacing,
                                                            padding=self.padding)

            # Watershed on HD
            grid = np.squeeze(out_hd)
            grid = grid[-1]
            grid = 1. - grid
            # utils.save_density(grid, 'raw.mrc', spacing, origin, padding)
            hd_coords, hd_distribs, hd_ids = blobber.to_blobs(grid)
            # Watershed on PL
            grid = np.squeeze(out_pl)
            pl_coords, pl_distribs, pl_ids = blobber.to_blobs(grid)
            np.savez_compressed(self.npz_path,
                                hd=np.squeeze(out_hd),
                                pl=np.squeeze(out_pl),
                                origin=origin,
                                hd_coords=hd_coords,
                                hd_distribs=hd_distribs,
                                hd_ids=hd_ids,
                                pl_coords=pl_coords,
                                pl_distribs=pl_distribs,
                                pl_ids=pl_ids)
            self.npzfile = np.load(self.npz_path)

    @property
    def origin(self):
        return self.npzfile['origin']

    @property
    def grid(self):
        if self.hd:
            grid = self.npzfile['hd']
            grid = grid[-1]
            grid = 1. - grid
            return grid
        grid = self.npzfile['pl']
        return grid

    @property
    def all_coords(self):
        if self.hd:
            return self.npzfile['hd_coords']
        return self.npzfile['pl_coords']

    @property
    def all_distribs(self):
        if self.hd:
            return self.npzfile['hd_distribs']
        return self.npzfile['pl_distribs']

    @property
    def all_ids(self):
        if self.hd:
            return self.npzfile['hd_ids']
        return self.npzfile['pl_ids']

    def extract_around(self, radius=3, blob_id=1):
        """
        given the current predicted blobs and an id, extract all residues closer to this blob than a given radius
        and dump them in the temp dir in 'extracted_around.pdb'
        :param radius:
        :return: the list of resid extracted
        """
        coords_blob = np.float_(self.all_coords[self.all_ids == blob_id])
        true_coords_blob = coords_blob + self.origin - 7.5

        # Get the blob in the pdb system
        blob_path = os.path.join(self.temp_path, 'blob_coords.xyz')
        extracted_path = os.path.join(self.temp_path, 'extracted_around.pdb')
        with open(blob_path, 'w') as f:
            f.write(f'{len(true_coords_blob)} \n')
            f.writelines([f"C {x} {y} {z}\n" for x, y, z in true_coords_blob])

        # Extract by radius
        cmd.load(blob_path, 'blob')
        cmd.select(selection=f'byres (blob around {radius} and protein_highlighted)', name='pocket')
        cmd.save(extracted_path, selection='pocket')
        # get list of results and return it
        myspace = {'resid_pocket': []}
        cmd.iterate('pocket', 'resid_pocket.append(resi)', space=myspace)
        resid_ids = np.unique(np.int_(myspace['resid_pocket']))
        print(resid_ids)
        return resid_ids

    def extract_around_np(self, radius=3, blob_id=1):
        """
        Same but with numpy instead of the pymol implementation
        :param radius:
        :param blob_id:
        :return:
        """
        coords_blob = np.float_(self.all_coords[self.all_ids == blob_id])
        true_coords_blob = coords_blob + self.origin - 7.5

        coords_pdb = cmd.get_coords('protein_highlighted')
        all_dists = scipy.spatial.distance.cdist(coords_pdb, true_coords_blob)
        i_min = np.where(all_dists < radius)[0]
        for i in i_min:
            cmd.select(name='pocket', selection='by res rank ' + str(i), merge=1)

        extracted_path = os.path.join(self.temp_path, 'extracted_around.pdb')
        cmd.save(extracted_path, selection='pocket')

        myspace = {'resid_pocket': []}
        cmd.iterate('pocket', 'resid_pocket.append(resi)', space=myspace)
        resid_ids = np.unique(np.int_(myspace['resid_pocket']))
        return resid_ids

    def mutate_experiment(self, blob_id, radius):
        # get the first prediction
        self.predict()
        res_list = self.extract_around(blob_id=blob_id, radius=radius)

        _, id_grid = lutils.dense(self.grid,
                                  self.all_coords,
                                  self.all_distribs,
                                  self.all_ids)
        flat_in = self.grid[id_grid == blob_id]
        correlations = list()
        cmd.alter(selection='protein_highlighted', expression='b=1')

        #  Iterate over list_id of residues close to the blob and do the prediction
        for i, resid in enumerate(res_list):
            cmd.select(name='select_trunc0', selection=f'(protein_highlighted and not (sidechain and resid {resid}))')

            # Add the CB to mutate in alanine, otherwise simply mask id
            if self.alanine:
                cmd.select(name='select_trunc', selection=f'select_trunc0 or (name CB and protein_highlighted)')
            cmd.create(name='select_trunc_altered', selection='select_trunc')
            if self.alanine:
                cmd.alter(selection=f'resi {resid} and select_trunc_altered', expression='resn="ALA"')
            temp_save_path = os.path.join(self.temp_path, f'test_truncated_{i}.pdb')
            cmd.save(filename=temp_save_path, selection='select_trunc_altered')
            cmd.delete('select_trunc_altered')

            out_hd, out_pl, origin, _ = predict.predict_pdb(self.model,
                                                            temp_save_path,
                                                            xyz_max=self.xyz_max,
                                                            xyz_min=self.xyz_min)

            if self.hd:
                grid = np.squeeze(out_hd)
                grid = grid[-1]
                grid = 1. - grid
            else:
                grid = np.squeeze(out_pl)

            # print('debug : ', self.hd, grid.shape)
            # print((id_grid == blob_id).shape)
            flat_out = grid[id_grid == blob_id]
            correlation = 1 - scipy.spatial.distance.correlation(flat_in, flat_out)
            correlations.append(correlation)
            cmd.alter(selection=f'protein_highlighted and resi {resid}', expression=f'b={correlation:5f}')
            # continue
            print(correlation)
        print(correlations)
        cmd.save(filename=self.dump_path, selection='protein_highlighted')
        # cmd.save(filename=os.path.join(self.temp_path, 'highlighted_protein.pdb'), selection='protein_highlighted')


if __name__ == '__main__':
    highlight = Highlighter(experiments_name='large.exp')
    highlight.mutate_experiment(blob_id=1, radius=6)
