#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2019-10-23 14:59:28 (UTC+0200)

import h5py
import numpy as np
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
import os

from learning.utils import ConfParser
from data_processing import pytorch_loading
from learning.utils import Metrics, check_and_create
from learning import model_factory, predict


def learn(name,
          data='20201120_PLHD-database_chunks.hdf5',
          ini_file=None,
          argparse_dict=None,
          n_epochs=15,
          device=0,
          print_each=100,
          abort=None,
          weights=None,
          early_stop_threshold=20,
          blobber_params={},
          blob_metrics=False):
    """
    Run the training
    :param n_epochs: the number of epochs
    :param data: the database for results
    :param device:
    :param print_each: print results information each given step
    :param abort: If not None, abort after the given number of step (for testing purpose)
    :param weights: if some weights are given for resuming a training
    :param early_stop_threshold:
    :param blobber_params: useful for model optimization
    :param blob_metrics: Whether to use the usual random validation split or the CATH one along with blobs metrics
    :return:
    """
    np.random.seed(0)
    torch.manual_seed(0)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    logdir, weights_dir = check_and_create(name, permissive=True)
    checkpoint_path = "%s/cp-{epoch:04d}.ckpt" % f'../results/weights/{name}'
    data_dir = os.path.join(script_dir, '..', 'data')
    data_file = os.path.join(data_dir, data)

    if ini_file is not None:
        ini_file = os.path.join(script_dir, 'inis',
                                f'{ini_file}.ini')
    hparams = ConfParser(default_path=os.path.join(script_dir,
                                                   'inis/default.ini'),
                         path_to_ini=ini_file,
                         argparse_dict=argparse_dict)
    hparams.dump(dump_path=os.path.join(script_dir,
                                        f'../results/experiments/{name}.exp'))

    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    model = model_factory.model_from_hparams(hparams, load_weights=False)
    if weights is not None:
        model.load_state_dict(torch.load(weights))
    model.to(device)

    writer = SummaryWriter(logdir)
    metrics_hd = Metrics(writer, print_each=print_each, message='HD')
    metrics_pl = Metrics(writer, print_each=print_each, message='PL')
    train_metrics = [metrics_hd, metrics_pl]

    if not blob_metrics:
        train_dataset, val_dataset, test_dataset = pytorch_loading.Loader(data_file,
                                                                          num_workers=10,
                                                                          return_enveloppe=True).get_data()
        metrics_hd_test = Metrics(writer, print_each=None, message='HD', mode="test")
        metrics_pl_test = Metrics(writer, print_each=None, message='PL', mode="test")
        validation_metrics = [metrics_hd_test, metrics_pl_test]

    else:
        train_dataset, _, _ = pytorch_loading.Loader(data_file,
                                                     num_workers=10,
                                                     return_enveloppe=True, splits=None).get_data()

    best_loss = sys.maxsize
    epochs_from_best = 0

    # Add default best path and best weights
    hparams.add_value('best_weights', 'path', checkpoint_path.format(epoch=0))
    hparams.dump()
    torch.save(model.state_dict(), os.path.join(script_dir, checkpoint_path.format(epoch=0)))

    # This is really not clean, as the dataset is a shared object...
    # loadhd = input("Load HD y or n?")
    # if loadhd == "n":
    #     train_dataset.dataset.dataset.noload_hd = True
    #     test_dataset.dataset.dataset.noload_hd = True
    #
    # loadpl = input("Load PL y or n?")
    # if loadpl == "n":
    # train_dataset.dataset.dataset.noload_pl = True
    # test_dataset.dataset.dataset.noload_pl = True

    import time
    time_passed = time.perf_counter()

    for epoch in range(n_epochs):
        passed = 0
        for i, (grid_prot, grid_lig, branch, _, enveloppe) in enumerate(train_dataset):
            passed += 1

            # NO LOAD -> branch ==-1
            if int(branch) == -1:
                continue

            if abort is not None and passed == abort:
                print('train abort')
                break


            grid_prot, grid_lig, enveloppe = grid_prot.to(device), grid_lig.to(device), enveloppe.to(device)
            out, loss = model.persistent_training_call(grid_prot, grid_lig, branch, enveloppe=enveloppe)
            grid_lig, out = grid_lig.detach(), out.detach()
            metric = train_metrics[branch]
            metric.update(loss, grid_lig, out, enveloppe=enveloppe)
            if not i % print_each:
                print(f"Time elapsed since learning started : {time.perf_counter() - time_passed}")

        model.zero_grad()

        # Save the weights at the end of each epoch, the pointer is then updated based on the test
        # if early-stopping is on
        model.to('cpu')
        torch.save(model.state_dict(), os.path.join(script_dir, checkpoint_path.format(epoch=epoch)))
        model.to(device)
        # model.wpl += 0.005
        # model.whd += 0.005

        if not blob_metrics:
            # Do the testing
            for i, (grid_prot, grid_lig, branch, _, enveloppe) in enumerate(test_dataset):
                if abort is not None and i == abort:
                    print('test abort')
                    break

                if int(branch) == -1:
                    continue

                grid_prot, grid_lig, enveloppe = grid_prot.to(device), grid_lig.to(device), enveloppe.to(device)
                out, loss = model.testing_call(grid_prot, grid_lig, branch)
                metric = validation_metrics[branch]
                metric.update(loss, grid_lig, out, enveloppe=enveloppe)

            metrics_hd_test.print_and_log(epoch)
            metrics_pl_test.print_and_log(epoch)

            # Aggregate/log the test results and compute mean from pl and hd for early stopping
            validation_loss = np.mean([met.mean_loss.result() for met in validation_metrics])

        else:
            validation_score = predict.v_2(model,
                                           outmetricfilename=os.path.join(logdir, f"metrics_epoch_{epoch}"),
                                           blobber_params=blobber_params,
                                           hd_txt=os.path.join(data_dir, 'HD-database_validation_set.txt'),
                                           hd_dir=os.path.join(data_dir, 'HD-database'),
                                           pl_txt=os.path.join(data_dir, 'PL-database_validation_set.txt'),
                                           pl_dir=os.path.join(data_dir, 'PL-database'),
                                           device=device)
            writer.add_scalar(tag='Aggregated Metric', scalar_value=validation_score, global_step=epoch)

            # We then need to turn it into a 'loss-like' result
            validation_loss = -validation_score
        # Early stopping
        if early_stop_threshold is not None:
            if validation_loss < best_loss:
                best_loss = validation_loss
                epochs_from_best = 0
                best_path = checkpoint_path.format(epoch=epoch)
                hparams.add_value('best_weights', 'path', best_path)
                hparams.dump()
            else:
                epochs_from_best += 1
                if epochs_from_best > early_stop_threshold:
                    print('This model was early stopped')
                    break
        else:
            best_path = checkpoint_path.format(epoch=epoch)
            hparams.add_value('best_weights', 'path', best_path)
            hparams.dump()

    return best_loss


if __name__ == '__main__':
    experiments_path = 'test'
    MODEL = model_factory.model_from_exp(expfilename=experiments_path, load_weights=False)

    H5FILE = h5py.File('data/PLHD-database.hdf5', 'r')
    N_EPOCHS = 10
    DB = pytorch_loading.Database(H5FILE)
    LOGFILE = 'v3'

    # Check that the logfile name does not already exists as it breaks tensorboard...
    # Comment while developping
    logdir, weights_dir = check_and_create(LOGFILE)
    learn(MODEL, N_EPOCHS, DB, outlogs=logdir, print_each=1,
          checkpoint_path="results/weights/%s/cp-{epoch:04d}.ckpt" % LOGFILE)
