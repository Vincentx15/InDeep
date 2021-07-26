import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Subset, DataLoader

import h5py
import numpy as np

from data_processing.Complex import Complex

"""
This script takes an hdf5 and organize it into a Database object that yields Complexes
Then this Database is fed to Pytorch Dataloaders
"""


class Database(object):
    """
    Create a generator from the hdf5 database file
    """

    def __init__(self, h5file):
        """
        - h5file: h5py File object or its path
        """
        if isinstance(h5file, str):
            h5file = h5py.File(h5file, 'r')
        self.h5file = h5file
        # self.proteins are all the keys for the 'receptor' part of the complex
        self.proteins = list(self.h5file.keys())
        self.protligs = self.get_protlig_keys()

    def get_protlig_keys(self):
        protlig_keys = []
        for prot in self.proteins:
            liglist = list(set(self.h5file[prot].keys()) - {'coords'})
            liglist.sort()
            protlig_keys.extend([(prot, lig) for lig in liglist])
        return protlig_keys

    def get_complex(self, protein, ligand, rotate=True):
        """
        Return the protein-ligand complex object
        >>> cp = db.get_complex('1a29-A-P62157.pdb', '1a29-A-TFP-153.pdb')

        Test if the complex is a PL system:
        >>> cp.is_PL
        True

        or a HD system:
        >>> cp.is_HD
        False
        """
        return Complex(self.h5file, protein, ligand, rotate=rotate)


class HDPLDataset(Dataset):
    """
        Uses a HDF5 file as defined above and turn it into a Pytorch data set
        """

    def __init__(self, data_file, rotate=True, return_enveloppe=False):
        self.data_file = data_file

        # For pytorch loading, we need the file reading in the get_item
        self.database = Database(data_file)
        self.keys = self.database.get_protlig_keys()
        self.database = None

        self.noload_hd = False
        self.noload_pl = False
        self.rotate = rotate
        self.return_enveloppe = return_enveloppe

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        """
        Returns the desired complex.
        If noload_hd is activated, only returns PL complexes otherwise the is_pl is negative
        :param item:
        :return:
        """
        if self.database is None:
            self.database = Database(self.data_file)

        prot, lig = self.keys[item]

        # Small test based only on the key to avoid building useless objects
        is_pl = True if len(lig) < 33 else False

        # If we get an hd and don't want to load these :
        if (self.noload_hd and not is_pl) or (self.noload_pl and is_pl):
            return 0, 0, -1, self.keys[item]

        cp = self.database.get_complex(prot, lig, rotate=self.rotate)

        gprot = cp.grid_prot.astype(np.float32)
        glig = cp.grid_lig.astype(np.float32)

        gprot = torch.from_numpy(gprot)
        glig = torch.from_numpy(glig)

        if self.return_enveloppe:
            enveloppe = cp.mask_grid_prot
            enveloppe = torch.from_numpy(enveloppe)
        else:
            enveloppe = 0

        # cp.save_mrc_lig()
        # cp.save_mrc_prot()

        return gprot, glig, cp.is_PL, self.keys[item], enveloppe


class InferenceDataset(Dataset):
    """
        Almost the same with less options and different returns, with the name of the ligand.
        """

    def __init__(self, data_file, return_enveloppe=False):
        self.data_file = data_file

        # For pytorch loading, we need the file reading in the get_item
        self.database = Database(data_file)
        self.keys = self.database.get_protlig_keys()
        self.database = None
        self.return_enveloppe = return_enveloppe

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        """
        Returns the desired complex.
        :param item:
        :return:
        """

        if self.database is None:
            self.database = Database(self.data_file)

        prot, lig = self.keys[item]

        cp = self.database.get_complex(prot, lig, rotate=False)
        gprot = cp.grid_prot.astype(np.float32)
        gprot = torch.from_numpy(gprot)

        if self.return_enveloppe:
            enveloppe = cp.mask_grid_prot
            enveloppe = torch.from_numpy(enveloppe)
        else:
            enveloppe = 0

        return gprot, cp.is_PL, self.keys[item], enveloppe


class Loader:
    def __init__(self, df,
                 batch_size=1,
                 num_workers=10,
                 rotate=True,
                 return_enveloppe=False,
                 splits=(0.7, 0.85)):
        """
        :param df: hdf5 file to load
        :param batch_size:
        :param num_workers:
        :param rotate:
        :param splits: If we want to split our dataset, we should specify the proportion, else call with None
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = HDPLDataset(data_file=df, rotate=rotate, return_enveloppe=return_enveloppe)
        self.splits = splits

    def get_data(self):
        if self.splits is None:
            train_loader = DataLoader(dataset=self.dataset, shuffle=True, batch_size=self.batch_size,
                                      num_workers=self.num_workers, worker_init_fn=np.random.seed)
            return train_loader, None, None

        split_train, split_valid = self.splits
        n = len(self.dataset)
        train_index, valid_index = int(split_train * n), int(split_valid * n)
        indices = list(range(n))

        train_indices = indices[:train_index]
        valid_indices = indices[train_index:valid_index]
        test_indices = indices[valid_index:]

        train_set = Subset(self.dataset, train_indices)
        valid_set = Subset(self.dataset, valid_indices)
        test_set = Subset(self.dataset, test_indices)

        train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                  num_workers=self.num_workers, worker_init_fn=np.random.seed)
        valid_loader = DataLoader(dataset=valid_set, shuffle=True, batch_size=self.batch_size,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=self.batch_size,
                                 num_workers=self.num_workers)

        return train_loader, valid_loader, test_loader
