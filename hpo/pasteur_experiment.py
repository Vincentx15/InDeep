#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import re
# import pika
import subprocess
from random import getrandbits

script_dir = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(script_dir, '..')
INI_TEMPLATE = os.path.join(ROOT_DIR, "learning", "inis", "template.ini")

sys.path.append(ROOT_DIR)

from learning import learn


def log(s):
    print("[%d] %s" % (os.getpid(), s), file=sys.stderr)


def pasteur_experiment(args):
    ########## 0. Get a unique random hash to identify that experiment
    exp_id = "Pasteur-HPO-" + "%016x" % getrandbits(64)
    log("Experiment ID: %s" % exp_id)
    log("Params: %s" % args)

    common_filters = [2 ** (args["common_filters_start"] + i) for i in range(args["common_filters_size"])]
    pl_filters = [common_filters[-1]] + [2 ** (i + 3) for i in range(args["pl_filters_size"], 0, -1)] + [1]
    hd_filters = [common_filters[-1]] + [2 ** (i + 3) for i in range(args["hd_filters_size"], 0, -1)] + [6]

    with open(INI_TEMPLATE, "r") as f:
        ini = f.read()
    ini = ini.replace("{{pl_drate}}", str(args["pl_drate"]))
    ini = ini.replace("{{wenveloppe}}", str(args["wenveloppe"]))
    ini = ini.replace("{{common_filters}}", str(common_filters))
    ini = ini.replace("{{pl_filters}}", str(pl_filters))
    ini = ini.replace("{{hd_filters}}", str(hd_filters))
    ini_name = exp_id + ".ini"
    ini_file = os.path.join(ROOT_DIR, "learning", "inis", ini_name)
    with open(ini_file, "w") as f:
        f.write(ini)
    log("File '%s' created." % ini_file)

    # The HDF5_USE_FILE_LOCKING var is used to avoid reading issues (timeout maybe) at
    # first read of an HDF5 file. See https://groups.google.com/g/h5py/c/0kgiMVGSTBE?pli=1
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    num_epochs = 4
    batch_size = 20
    print_each = 1000
    name = exp_id
    data = '20210219_PLHD-database_train.hdf5'
    early = 5
    use_unet = False
    use_old_unet = False
    out_channels = 32
    model_depth = 2
    double_conv = True
    blob_metrics = True
    abort = 5000

    args_dict = {
        'num_epochs': num_epochs,
        'print_each': print_each,
        'batch_size': batch_size,
        'data': data,
        'name': name,
        'ini': exp_id,
        'weights': None,
        'abort': None,
        'device': args["gpu_id"],
        'early': early,
        'use_unet': use_unet,
        'use_old_unet': use_old_unet,
        'out_channels': out_channels,
        'model_depth': model_depth,
        'double_conv': double_conv}

    blobber_params = {
        "min_merging_value": args["blobber_min_merging_value"],
        "min_euclidean_distance": args["blobber_min_euclidean_distance"],
        "max_euclidean_distance": args["blobber_max_euclidean_distance"],
        "blob_score_min": 0.15}

    best_loss = learn.learn(name=name,
                            ini_file=exp_id,
                            device=args["gpu_id"],
                            n_epochs=num_epochs,
                            data=data,
                            print_each=print_each,
                            early_stop_threshold=early,
                            blob_metrics=blob_metrics,
                            blobber_params=blobber_params,
                            abort=abort,
                            argparse_dict=args_dict)

    return best_loss


def pasteur_experiment_unet(args):
    ########## 0. Get a unique random hash to identify that experiment
    exp_id = "Pasteur-HPO-" + "%016x" % getrandbits(64)
    log("Experiment ID: %s" % exp_id)
    log("Params: %s" % args)

    with open(INI_TEMPLATE, "r") as f:
        ini = f.read()
    ini = ini.replace("{{wenveloppe}}", str(args["wenveloppe"]))
    ini_name = exp_id + ".ini"
    ini_file = os.path.join(ROOT_DIR, "learning", "inis", ini_name)
    with open(ini_file, "w") as f:
        f.write(ini)
    log("File '%s' created." % ini_file)

    # The HDF5_USE_FILE_LOCKING var is used to avoid reading issues (timeout maybe) at
    # first read of an HDF5 file. See https://groups.google.com/g/h5py/c/0kgiMVGSTBE?pli=1
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    num_epochs = 4
    batch_size = 20
    print_each = 1000
    name = exp_id
    data = '20210219_PLHD-database_train.hdf5'
    early = 5
    use_unet = True
    use_old_unet = bool(args["use_old_unet"])
    out_channels = args["out_channels"]
    model_depth = args["model_depth"]
    num_feature_map = args["num_feature_map"]
    double_conv = True
    blob_metrics = True
    abort = 5000

    args_dict = {
        'num_epochs': num_epochs,
        'print_each': print_each,
        'batch_size': batch_size,
        'data': data,
        'name': name,
        'ini': exp_id,
        'weights': None,
        'abort': None,
        'device': args["gpu_id"],
        'early': early,
        'use_unet': use_unet,
        'use_old_unet': use_old_unet,
        'out_channels': out_channels,
        'model_depth': model_depth,
        'double_conv': double_conv,
        'num_feature_map': num_feature_map}

    blobber_params = {
        "min_merging_value": args["blobber_min_merging_value"],
        "min_euclidean_distance": args["blobber_min_euclidean_distance"],
        "max_euclidean_distance": args["blobber_max_euclidean_distance"],
        "blob_score_min": 0.15}

    best_loss = learn.learn(name=name,
                            ini_file=exp_id,
                            device=args["gpu_id"],
                            n_epochs=num_epochs,
                            data=data,
                            print_each=print_each,
                            early_stop_threshold=early,
                            blob_metrics=blob_metrics,
                            blobber_params=blobber_params,
                            abort=abort,
                            argparse_dict=args_dict)

    return best_loss
