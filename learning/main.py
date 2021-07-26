#!/usr/bin/env python
# -*- coding: UTF8 -*-

"""
I see this script as a way to call python function from the command line, in a
similar manner to what would be done
in a if __name__ ... but to avoid having a distinct python script for calling
predict...

This is to avoid having a lot of scripts that would be only executables,
similar to a makefile function
"""
import sys
import argparse
import os

FUNCTIONS = ['predict', 'train', 'rm', 'setup', 'evaluate']

if __name__ != '__main__':
    raise ImportError('Cannot import the main')

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

try:
    function = sys.argv[1]
except IndexError:
    function = 'train'
    # raise ValueError("Specify a valid function please")

if function not in FUNCTIONS:
    function = 'train'

if function == 'rm':
    from learning.utils import remove

    exp = sys.argv[2]
    remove(exp)
    print(f"removed {exp}")

if function == 'train':
    from learning.utils import check_and_create
    from learning import learn

    parser = argparse.ArgumentParser()
    parser.add_argument("-ep", "--num_epochs", type=int,
                        help="number of epochs to train", default=60)
    parser.add_argument("-bs", "--batch_size", type=int, default=20,
                        help="choose the batch size")
    parser.add_argument("-pe", "--print_each", type=int, default=20,
                        help="The display frequencies of the metrics")
    parser.add_argument("-d", "--data", default='20201120_PLHD-database_chunks.hdf5',
                        help="data to use")
    parser.add_argument("-n", "--name", default='default',
                        help="name of the experiment")
    parser.add_argument("-ini", "--ini", default=None,
                        help="name of the additional .ini to use")
    parser.add_argument("-w", "--weights", default=None,
                        help="If we retrain, the path to previous weights")
    parser.add_argument("-a", "--abort", default=None, type=int,
                        help="abort each epoch after a few points")
    parser.add_argument("-dev", "--device", default=0, type=int, help="gpu device to use")
    parser.add_argument("-es", "--early", default=20, type=int,
                        help="Successive degradation of performance before \
                              early stopping")
    parser.add_argument("--blob_metrics", default=True, action='store_false', help="")

    # UNET
    parser.add_argument("-unet", "--use_unet", default=True, action='store_false', help="")
    parser.add_argument("--use_old_unet", default=False, action='store_true',
                        help="Was an experiment to add activation inside the convolution blocks")
    parser.add_argument("-out", "--out_channels", default=128, type=int,
                        help="The number of channels after the 'U' of the UNet")
    parser.add_argument("-dep", "--model_depth", default=3, type=int,
                        help="The depth of the Unet model : number of convolution/deconvolution blocks")
    parser.add_argument("--num_feature_map", default=16, type=int,
                        help="number of feature_maps in the first and last layer of the Unet")
    parser.add_argument("-conv", "--double_conv", default=True, action='store_false', help="")
    args, _ = parser.parse_known_args()

    learn.learn(name=args.name,
                device=args.device,
                n_epochs=args.num_epochs,
                weights=args.weights,
                data=args.data,
                print_each=args.print_each,
                abort=args.abort,
                early_stop_threshold=args.early,
                blob_metrics=args.blob_metrics,
                argparse_dict=args,
                ini_file=args.ini)
