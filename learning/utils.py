import os
import sys
import torch
import numpy as np
from scipy import ndimage

import configparser
from ast import literal_eval

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from post_processing import blobber
from data_processing import utils


def soft_mkdir(name, permissive=True):
    try:
        os.mkdir(name)
    except FileExistsError:
        if not permissive:
            raise FileExistsError("Name already exists")


def check_and_create(name, permissive=True):
    """
    Setup the directories used for the learning pipeline
    :param name:
    :param permissive: if set to True, will overwrite existing logs and models
    :return:
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    resdir = os.path.join(script_dir, '../results')
    log_dir = os.path.join(script_dir, '../results/logs/')
    weights_dir = os.path.join(script_dir, '../results/weights/')
    experiment_dir = os.path.join(script_dir, '../results/experiments/')
    log_dir_name = os.path.join(script_dir, f'../results/logs/{name}')
    weights_dir_name = os.path.join(script_dir, f'../results/weights/{name}')
    soft_mkdir(resdir)
    soft_mkdir(log_dir)
    soft_mkdir(weights_dir)
    soft_mkdir(experiment_dir)
    soft_mkdir(log_dir_name, permissive=permissive)
    soft_mkdir(weights_dir_name, permissive=permissive)
    return log_dir_name, weights_dir_name


def remove(name):
    """
    delete an experiment results
    :param name:
    :return:
    """
    import shutil

    script_dir = os.path.dirname(__file__)
    logdir = os.path.join(script_dir, f'../results/logs/{name}')
    weights_dir = os.path.join(script_dir, f'../results/weights/{name}')
    experiment = os.path.join(script_dir, f'../results/experiments/{name}.exp')
    shutil.rmtree(logdir)
    shutil.rmtree(weights_dir)
    os.remove(experiment)
    return True


class ConfParser:
    def __init__(self,
                 default_path=os.path.join(os.path.dirname(__file__), 'inis/default.ini'),
                 path_to_ini=None,
                 name_exp=None,
                 path_to_exp=None,
                 argparse_dict=None,
                 dump_path=None):
        self.dump_path = dump_path
        self.default_path = default_path

        # Build the hparam object
        self.hparams = configparser.ConfigParser()

        # Add the default configurations, optionaly another .conf and an argparse object
        self.hparams.read(self.default_path)

        if path_to_ini is not None:
            self.add_ini(path_to_ini)
        if argparse_dict is not None:
            self.add_argparse(argparse_dict)

        if name_exp is not None:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            path_to_exp = os.path.join(script_dir, '../results/experiments/', name_exp)
        if path_to_exp is not None:
            self.add_exp(path_to_exp)

    @staticmethod
    def merge_ini_into_default(default, new):
        for section in new.sections():
            for keys in new[section]:
                try:
                    default[section][keys]
                except KeyError:
                    raise KeyError(f'The provided value {section, keys} in the .ini are not present in the default, '
                                   f'thus not acceptable values, for retro-compatibility issues')
                # print(section, keys)
                default[section][keys] = new[section][keys]

    @staticmethod
    def merge_exp_into_default(default, new):
        for section in new.sections():
            try:
                default[section]
            except KeyError:
                default[section] = {}
            finally:
                for keys in new[section]:
                    try:
                        default[section][keys] = new[section][keys]
                    except KeyError:
                        pass
                    # print(section, keys)

    def add_ini(self, path_to_new):
        """
        Merge another conf parsing into self.hparams
        :param path_to_new:
        :return:
        """
        conf = configparser.ConfigParser()
        assert os.path.exists(path=path_to_new)
        conf.read(path_to_new)
        print(f'confing using {path_to_new}')
        return self.merge_ini_into_default(self.hparams, conf)

    def add_exp(self, path_to_exp):
        """
        Merge an exp parsing into self.hparams
        :param path_to_exp:
        :return:
        """
        conf = configparser.ConfigParser()
        assert os.path.exists(path=path_to_exp)
        conf.read(path_to_exp)
        print(f'confing using {path_to_exp}')
        return self.merge_exp_into_default(self.hparams, conf)

    @staticmethod
    def merge_dict_into_default(default, new):
        """
        Same merge but for a dict of dicts
        :param default:
        :param new:
        :return:
        """
        for section in new.sections():
            for keys in new[section]:
                try:
                    default[section][keys]
                except KeyError:
                    raise KeyError(f'The provided value {section, keys} in the .conf are not present in the default, '
                                   f'thus not acceptable values')
                default[section][keys] = new[section][keys]
        return default

    def add_dict(self, section_name, dict_to_add):
        """
        Add a dictionnary as a section of the .conf. It needs to be turned into strings
        :param section_name: string to be the name of the section
        :param dict_to_add: any dictionnary
        :return:
        """

        new = {item: str(value) for item, value in dict_to_add.items()}

        try:
            self.hparams[section_name]
        # If it does not exist
        except KeyError:
            self.hparams[section_name] = new
            return

        for keys in new:
            self.hparams[section_name][keys] = new[keys]

    def add_value(self, section_name, key, value):
        """
        Add a dictionnary as a section of the .conf. It needs to be turned into strings
        :param section_name: string to be the name of the section
        :param key: the key inside the section name
        :param value: the value to insert
        :return:
        """

        value = str(value)

        try:
            self.hparams[section_name]
        # If it does not exist
        except KeyError:
            self.hparams[section_name] = {key: value}
            return

        self.hparams[section_name][key] = value

    def add_argparse(self, argparse_obj):
        """
        Add the argparse object as a section of the .conf.
        It can be either the argparse itself or its dict
        :param argparse_obj:
        :return:
        """
        try:
            self.add_dict('argparse', argparse_obj.__dict__)
        except AttributeError:
            self.add_dict('argparse', argparse_obj)

    def get(self, section, key):
        """
        A get function that also does the casting into what is useful for model results
        :param section:
        :param key:
        :return:
        """
        try:
            return literal_eval(self.hparams[section][key])
        except ValueError:
            return self.hparams[section][key]
        except SyntaxError:
            return self.hparams[section][key]

    def __str__(self):
        print(self.hparams.sections())
        for section in self.hparams.sections():
            print(section.upper())
            for keys in self.hparams[section]:
                print(keys)
            print('-' * 10)
        return ' '

    def dump(self, dump_path=None):
        """
        If no dump is given, use the default one if it exists. Otherwise, set the dumping path as the new default
        :param dump_path:
        :return:
        """
        if dump_path is None:
            if self.dump_path is None:
                raise ValueError('Please add a path')
            with open(self.dump_path, 'w') as save_path:
                self.hparams.write(save_path)
        else:
            self.dump_path = dump_path
            with open(dump_path, 'w') as save_path:
                self.hparams.write(save_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default='default', help="name of the experiment")
    parser.add_argument("-w", "--weights", default=None, help="If we retrain, the path to previous weights")
    parser.add_argument("-a", "--abort", default=None, type=int, help="abort each epoch after a few points")
    parser.add_argument("-es", "--early", default=20, type=int,
                        help="Successive degradation of performance before early stopping")

    args, _ = parser.parse_known_args()
    parser1 = ConfParser()
    parser1.add_argparse(args)

    print(parser1.get('argparse', 'name'))  # default
    print(type(parser1.get('argparse', 'name')))  # <class 'str'>
    print(parser1.get('argparse', 'early'))  # 20
    print(type(parser1.get('argparse', 'early')))  # <class 'int'>
    print(parser1.get('argparse', 'abort'))  # None
    print(type(parser1.get('argparse', 'abort')))  # <class 'NoneType'>

    parser1.dump('example.ini')


def dense(grid, all_coords, all_distribs, all_ids):
    """
    Same as sparse to dense but in 3D
    :param grid:
    :param all_coords:
    :param all_distribs:
    :param all_ids:
    :return:
    """

    # If there are no coords to densify
    try:
        iter(all_coords)
    except TypeError:
        return None, None

    value_grid, id_grid = np.zeros_like(grid), np.zeros_like(grid, dtype=int)

    for i, (x, y, z) in enumerate(all_coords):
        value_grid[x, y, z], id_grid[x, y, z] = all_distribs[i], all_ids[i]
    return value_grid, id_grid


def DVO(predicted, true):
    """
    Intersect over Union
    :param predicted:
    :param true:
    :return:
    """
    print(f'pred volume', np.sum(predicted))
    print(f'real volume', np.sum(true))
    intersect = np.sum(true * predicted)
    union = np.sum(true) + np.sum(predicted) - intersect
    print(f'intersect volume', intersect)
    print(f'union volume', union)
    return intersect / union


def min_distance_metric(set_of_points, ligand_coords, proba_threshold=None, center_only=False):
    """

    :param set_of_points: shape (n_predicted, 3)
    :param ligand_coords: shape (n_ligands, 3)
    :param  proba_threshold: if we want to restrict the blob to be above something
    :param center_only: if we only want the distance to the center
    :return:
    """
    from scipy.spatial.distance import cdist
    if proba_threshold is not None:
        set_of_points = set_of_points[np.nonzero(set_of_points > proba_threshold)]
    if center_only:
        ligand_coords = np.mean(ligand_coords)
    return np.min(cdist(ligand_coords, set_of_points))


def mean_rank(labels, counts):
    """
    Get the weighted average label
    """
    return np.average(labels, weights=counts)


def median_rank(labels, count):
    return int(np.median(np.repeat(labels, count)))


def rankfactors(labels, counts):
    """
    Proportion of element in each label class
    """
    labels = np.asarray(labels)
    counts = np.asarray(counts, dtype=float)
    factors = counts / counts.sum()
    return factors


def rankfactor(labels, counts, rank=1):
    """
    Proportion of elements in the given rank
    """
    factors = rankfactors(labels, counts)
    selection = (labels == rank)
    if selection.sum() == 0:
        return 0.
    else:
        return factors[selection][0]


def maxrankfactors(labels, counts):
    """
    Rank of the class with the maximum proportion of element
    """
    factors = rankfactors(labels, counts)
    return labels[np.argmax(factors)]


def mcc(y_true, y_pred):
    from sklearn.metrics import matthews_corrcoef
    # num_bin = 10
    # y_true, y_pred = y_true * num_bin, y_pred * num_bin
    # y_true, y_pred = y_true.astype(dtype=np.int, casting='unsafe'), \
    #                  y_pred.astype(dtype=np.int, casting='unsafe')
    # coef = matthews_corrcoef(y_true, y_pred)

    y_true = y_true > 0.4

    if not y_true.any():
        return 0
    sorted_values = np.sort(y_pred)
    thresh = sorted_values[-np.sum(y_true)]
    y_pred = y_pred > thresh
    if np.sum(y_pred) == 0:
        return 0
    coef = matthews_corrcoef(y_true, y_pred)
    return coef


def rocauc(y_true, y_pred):
    from sklearn.metrics import roc_auc_score
    y_true, y_pred = y_true > 0.4, y_pred
    if y_true.sum() == 0 or y_true.sum() == y_true.size:
        return -1
    score = roc_auc_score(y_true, y_pred)
    return score


def sensitivity(y_true, y_pred):
    num = np.sum(y_true * y_pred)
    if y_true.sum() > 0:
        se = num / np.sum(y_true)
    else:
        return 0
    return se.item()


def specificity(y_true, y_pred):
    y_true_comp = 1. - y_true
    y_pred_comp = 1. - y_pred
    sp = np.sum(y_true_comp * y_pred_comp) / np.sum(y_true_comp)
    return sp.item()


def pmax(y_true, y_pred):
    pmax_val = np.max(y_true * y_pred)
    return pmax_val.item()


def to_flat(y_true, y_pred):
    """
    Go from the [1,6/1,x,y,z] format to a flattened array format
        with the probability density used for blobbing and metrics
    :param y_true:
    :param y_pred:
    :return:
    """
    if y_pred.shape[1] > 1:  # HD
        y_true = 1. - y_true[
            0, -1, ...]  # The last channel of y_true is the probability of the absence of protein
        y_pred = 1. - y_pred[0, -1, ...]
    else:  # PL
        y_true, y_pred = y_true, y_pred
    y_true, y_pred = y_true.squeeze(), y_pred.squeeze()
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    return y_true, y_pred


def select(y_true, y_pred, selection=None):
    """

    :param y_true: flat arrays
    :param y_pred: flat arrays
    :param selection: a boolean array of same shape as y_true/pred
    :return:
    """
    if selection is not None:
        y_true, y_pred = y_true[selection], y_pred[selection]
    return y_true, y_pred


def compute_metrics(y_true, y_pred):
    """

    :param y_true: 1D array
    :param y_pred: 1D array
    :return:
    """
    metrics = dict()
    metrics['se'] = sensitivity(y_true, y_pred)
    metrics['sp'] = specificity(y_true, y_pred)
    metrics['pmax'] = pmax(y_true, y_pred)
    metrics['mcc'] = mcc(y_true, y_pred)
    metrics['auc'] = rocauc(y_true, y_pred)
    return metrics


class Blob(object):
    def __init__(self, id_blob, id_grid, lig_grid_probas, value_grid, channel=None):
        selection = (id_grid == id_blob)
        # Get center of mass
        self.center_lig = self.compute_center(lig_grid_probas)
        self.center_blob = self.compute_center(selection * value_grid)

        # Organize the blobs as sorted lists with highest predicted values first
        self.y_true_label_local, self.y_pred_label_local = select(lig_grid_probas, value_grid, selection)
        sorter = np.argsort(self.y_pred_label_local)[::-1]
        self.y_pred_label_local = self.y_pred_label_local[sorter]
        self.y_true_label_local = self.y_true_label_local[sorter]
        all_probas = lig_grid_probas.sum()

        # We need this to penalize missing out on some part of the ligand :
        # append the missed ligand parts as zero predictions
        missing_true = int(all_probas - self.y_true_label_local.sum())
        true_missed, pred_missed = np.ones(missing_true), np.zeros(missing_true)
        self.y_true_label_global = np.concatenate((self.y_true_label_local, true_missed))
        self.y_pred_label_global = np.concatenate((self.y_pred_label_local, pred_missed))
        self.metrics = dict()
        self.cutblob = None
        self.channel = channel
        self.get_metrics()

    @property
    def y_true_label(self):
        if self.local:
            return self.y_true_label_local[:self.cutblob]
        else:
            return self.y_true_label_global[:self.cutblob]

    @property
    def y_pred_label(self):
        if self.local:
            return self.y_pred_label_local[:self.cutblob]
        else:
            return self.y_pred_label_global[:self.cutblob]

    def format_name_channel(self, name):
        if self.channel is not None:
            name = f'{name}_{self.channel}'
        return name

    def format_metric_name(self, metric_name):
        if self.local is False:
            out = f'{metric_name}_global'
        else:
            out = f'{metric_name}_local'
        if self.cutblob is not None:
            out = f'{out}_{self.cutblob}'
        out = self.format_name_channel(out)
        return out

    def __str__(self):
        outstr = ""
        for metric_name in self.metrics:
            outstr += f"{metric_name}: {self.metrics[metric_name]}\n"
        return outstr

    def get_metrics(self):
        self.distance = self.get_distance()
        self.distance_sim = self.get_distance_sim()
        self.overdist = self.get_overdist()
        self.volume = self.get_volume()
        for local in [False, True]:
            for cutblob in [None, 150, 300, 600]:
                if local is False and cutblob is not None:
                    continue
                self.local = local
                self.cutblob = cutblob
                self.score = self.get_score()
                self.overlap = self.get_overlap()
                self.jaccard = self.get_jaccard()
                self.se = self.get_se()
                self.sp = self.get_sp()
                self.pmax = self.get_pmax()
                self.mcc = self.get_mcc()
                self.auc = self.get_auc()

    def get_volume(self):
        val = len(self.y_true_label_local)
        metric_name = self.format_name_channel('volume')
        self.metrics[metric_name] = val
        return val

    def compute_center(self, grid):
        """
        Computes the center of mass of a grid of probabilities
        :param grid:
        :return:
        """
        if grid.sum() < 0.01:
            nan_array = np.empty(shape=(3,))
            nan_array[:] = np.NaN
            return nan_array
        return np.asarray(ndimage.measurements.center_of_mass(grid))

    def get_distance_sim(self):
        val = np.sqrt(np.sum(np.square(self.center_lig - self.center_blob)))
        val = np.exp(-val / 8)
        metric_name = self.format_name_channel('distance_sim')
        self.metrics[metric_name] = val
        return val

    def get_overdist(self):
        dist = np.sqrt(np.sum(np.square(self.center_lig - self.center_blob)))
        sim_dist = np.exp(-dist / 8)
        overlap_150 = self.y_true_label_local[:150].mean()
        val = (sim_dist + overlap_150) / 2
        metric_name = self.format_name_channel('overdist')

        self.metrics[metric_name] = val
        return val

    def get_distance(self):
        val = np.sqrt(np.sum(np.square(self.center_lig - self.center_blob)))
        metric_name = self.format_name_channel('distance')
        self.metrics[metric_name] = val
        return val

    def get_score(self):
        val = self.y_pred_label.mean()
        metric_name = self.format_metric_name('score')
        self.metrics[metric_name] = val
        return val

    def get_overlap(self):
        val = self.y_true_label.mean()
        metric_name = self.format_metric_name('overlap')
        self.metrics[metric_name] = val
        return val

    def get_jaccard(self):
        num = self.y_true_label * self.y_pred_label
        den = self.y_true_label + self.y_pred_label - num
        val = (num.sum() / den.sum())
        metric_name = self.format_metric_name('jaccard')
        self.metrics[metric_name] = val
        return val

    def get_se(self):
        val = sensitivity(self.y_true_label, self.y_pred_label)
        metric_name = self.format_metric_name('se')
        self.metrics[metric_name] = val
        return val

    def get_sp(self):
        val = specificity(self.y_true_label, self.y_pred_label)
        metric_name = self.format_metric_name('sp')
        self.metrics[metric_name] = val
        return val

    def get_pmax(self):
        val = pmax(self.y_true_label, self.y_pred_label)
        metric_name = self.format_metric_name('pmax')
        self.metrics[metric_name] = val
        return val

    def get_mcc(self):
        val = mcc(self.y_true_label, self.y_pred_label)
        metric_name = self.format_metric_name('mcc')
        self.metrics[metric_name] = val
        return val

    def get_auc(self):
        val = rocauc(self.y_true_label, self.y_pred_label)
        metric_name = self.format_metric_name('auc')
        self.metrics[metric_name] = val
        return val


def get_metrics_blob(lig_grid, out_grid, hetatm=False, blobber_params={}):
    """
    Perform blobbing on the input grids
    Need to use the full [1,5,x,y,z] or [5,x,y,z] format to perform channel-wise metrics
    :param lig_grid:
    :param out_grid:
    :param hetatm:if it is a pl system (ligand is a small molecule)
    :return:
    """

    out_grid = out_grid.squeeze()
    lig_grid = lig_grid.squeeze()
    # Compute the blobs on the relevant grids
    if not hetatm:
        out_grid_probas = 1 - out_grid[-1]
        lig_grid_probas = 1 - lig_grid[-1]
    else:
        out_grid_probas = out_grid
        lig_grid_probas = lig_grid
    coords, distribs, ids = blobber.to_blobs(out_grid_probas, hetatm=hetatm, **blobber_params)
    _, id_grid = dense(out_grid_probas, coords, distribs, ids)

    if coords is None:
        return [], []

    # Then score each of those blobs.
    blobs = []
    blobs_channel = []
    atomtypes = list(utils.read_atomtypes().keys())
    for id_blob in np.unique(ids):
        # Compute on the whole proba grids
        blob = Blob(id_blob, id_grid, lig_grid_probas, out_grid_probas)

        # Add a channels list, that is empty for PL and filled with blob-wise metrics for HD
        blobs_all_channel = []
        if not hetatm:
            for channel_id in range(out_grid.shape[0] - 1):
                channel = atomtypes[channel_id]
                blob_channel = Blob(id_blob, id_grid, lig_grid[channel_id], out_grid[channel_id], channel=channel)
                blobs_all_channel.append(blob_channel)
        blobs_channel.append(blobs_all_channel)
        blobs.append(blob)
    return blobs, blobs_channel


class MyOwnRunningMean:
    def __init__(self):
        """
        We just want to get rid of tf
        """
        self.n = 0
        self.val = 0

    def __call__(self, new_val):
        self.n += 1

        if self.n == 1:
            self.val = new_val
        else:
            self.val = (self.n - 1) / self.n * self.val + new_val / self.n

    def reset_states(self):
        self.n = 0
        self.val = 0

    def result(self):
        return self.val


class Metrics:
    def __init__(self, writer, print_each=100, message=None, mode="train"):
        """
        Encapsulate the metrics computations for pytorch looping.
        Main function is update that does this computation repeatedly and update running means.
        Dump and print running average every print each call.
        :param writer:
        :param print_each:
        :param message:
        :param mode:
        """
        self.mean_loss = MyOwnRunningMean()
        self.mean_se = MyOwnRunningMean()
        self.mean_sp = MyOwnRunningMean()
        self.mean_pmax = MyOwnRunningMean()
        self.mean_mcc = MyOwnRunningMean()
        self.mean_auc = MyOwnRunningMean()
        self.metrics_name = ["loss", 'Se', 'Sp', 'pmax', 'mcc', 'auc']
        self.metrics = [self.mean_loss, self.mean_se, self.mean_sp, self.mean_pmax, self.mean_mcc, self.mean_auc]
        self.writer = writer
        self.print_each = print_each
        self.message = message
        self.counter = 0
        self.mode = mode

    def update(self, loss, y_true, y_pred, enveloppe=torch.Tensor(0)):
        self.mean_loss(loss)

        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        if enveloppe.sum() > 0:
            enveloppe = enveloppe.detach().cpu().numpy().squeeze().astype('bool').flatten()
        else:
            enveloppe = None

        y_true, y_pred = to_flat(y_true, y_pred)
        y_true, y_pred = select(y_true, y_pred, enveloppe)
        metrics = compute_metrics(y_true, y_pred)
        self.mean_se(metrics['se'])
        self.mean_sp(metrics['sp'])
        self.mean_pmax(metrics['pmax'])
        self.mean_mcc(metrics['mcc'])
        self.mean_auc(metrics['auc'])

        if self.print_each is not None and self.counter % self.print_each == 0:
            self.print_and_log(self.counter)
        self.counter += 1

    def log(self, step):
        for metric, metric_name in zip(self.metrics, self.metrics_name):
            self.writer.add_scalar(f"{self.mode}_{metric_name}_" + self.message, metric.result(), step)

    def display(self):
        """
        Displays a message to std:out
        """
        outstr = f'{self.mode}ing Loss: {self.mean_loss.result():.5f}, Se: {self.mean_se.result():.3f},' + \
                 f' Sp: {self.mean_sp.result():.3f}, pmax: {self.mean_pmax.result():.3f}'
        if self.message is None:
            print(outstr)
        else:
            print(f"after {self.counter} steps for {self.message}, " + outstr)

    def reset(self):
        """
        Reset all running means
        :return:
        """
        for metric in self.metrics:
            metric.reset_states()

    def print_and_log(self, step):
        # We need the step here to log differently epochs and steps
        self.log(step)
        self.display()
        self.reset()


def debug_memory(tf=True):
    """
    Track the tensors in memory to debug memory leaks
    :param tf: True if you're looking for leaks in tf, false otherwise
    :return: Nothing, but does some printing
    """
    import collections
    import gc

    def try_tensor_tf(o, _tf=tf):
        """
        return True if the object is a Tensor (tf by default, torch if '_tf' is set to False
        :param o:
        :param _tf:
        :return:
        """
        if _tf:
            import tensorflow as tf
            try:
                tf.is_tensor(o)
                if isinstance(o, tf.Tensor):
                    return True
            except:
                pass
            return False
        else:
            try:
                return o.is_tensor()
            except:
                return False

    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if try_tensor_tf(o, tf))
    for line in sorted(tensors.items()):
        print('{}\t{}'.format(*line))


def read_dbfile(dbfilename, dbpath):
    """
    dbfilename: text file containing pdb names for each system as:
        P:1blx-AB-Q00534-Q60773-A.pdb L:1blx-AB-Q00534-Q60773-B.pdb
    dbpath: path to the database (HD-database or PL-database)
    """
    db = np.genfromtxt(dbfilename, dtype=str)
    PL_list = []
    for AB in db:  # ['P:1blx-AB-Q00534-Q60773-A.pdb' 'L:1blx-AB-Q00534-Q60773-B.pdb']
        pdbcode = AB[0][2:6]
        pdbcode2 = AB[1][2:6]
        assert pdbcode == pdbcode2
        for e in AB:  # 'P:1blx-AB-Q00534-Q60773-A.pdb'
            if e[0] == 'P':
                P = e[2:]
            if e[0] == 'L':
                if dbpath == 'HD-database':
                    L = f'{e[2:-4]}-short.pdb'
                else:
                    L = e[2:]
        path = [dbpath, ]
        path.extend(list(pdbcode))
        path = '/'.join(path)
        pathP = f"{path}/{P}"
        pathL = f"{path}/{L}"
        assert os.path.exists(pathP), f'{pathP} not found'
        assert os.path.exists(pathL), f'{pathL} not found'
        PL_list.append((pathP, pathL))
    return PL_list
