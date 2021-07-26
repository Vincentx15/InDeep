#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import skopt
import os, time
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor as Pool

from pasteur_experiment import pasteur_experiment

GPU_COUNT = 4

DIMENSIONS = [
    skopt.space.Real(0.3, 0.6, name="pl_drate"),
    skopt.space.Real(0.1, 2.0, name="wenveloppe", prior="log-uniform"),
    skopt.space.Integer(3, 5, name="common_filters_start"),
    # first element of common filters will be 2^common_filters_start (8, 16, 32 or 64)
    skopt.space.Integer(2, 4, name="common_filters_size"),  # total length of common filters (doubles every time)
    skopt.space.Integer(2, 4, name="pl_filters_size"),
    # total length of PL filters (excluding the first whose size will match output of 'common')
    skopt.space.Integer(2, 4, name="hd_filters_size"),
    # total length of HD filters (excluding the first whose size will match output of 'common')
    skopt.space.Real(0.0, 1.0, name="blobber_min_merging_value"),
    skopt.space.Integer(0, 10, name="blobber_min_euclidean_distance"),
    skopt.space.Integer(10, 20, name="blobber_max_euclidean_distance")
]

UNET_DIMENSIONS = [
    skopt.space.Integer(0, 1, name="use_old_unet"),
    skopt.space.Integer(32, 256, name="out_channels"),
    skopt.space.Integer(2, 5, name="model_depth"),
    skopt.space.Integer(8, 64, name="num_feature_map"),
    skopt.space.Real(0.1, 2.0, name="wenveloppe", prior="log-uniform"),
    skopt.space.Real(0.0, 1.0, name="blobber_min_merging_value"),
    skopt.space.Integer(0, 10, name="blobber_min_euclidean_distance"),
    skopt.space.Integer(10, 20, name="blobber_max_euclidean_distance")
]


# We can't set the constraint that min_euclidean_distance < max_euclidean_distance.
# A pull request is opened but not yet merged:
# https://github.com/scikit-optimize/scikit-optimize/pull/971


def camelback6(x):
    # Six-hump camelback function, useful for debugging (switch the Optimizer params to use it)
    print(x)
    x1 = x[0]
    x2 = x[1]
    return (4 - 2.1 * (x1 * x1) + (x1 * x1 * x1 * x1) / 3.0) * (x1 * x1) + x1 * x2 + (-4 + 4 * (x2 * x2)) * (x2 * x2)


# Use the 'use_named_args' function from skopt.utils to get a dictionnary of params with variables names as keys.
# Add a fake "gpu_id" dimension so that the GPU ID is passed as a named parameter in that dictionnary of params
# but not used in the dimensions proposed by the Optimizer below

@skopt.utils.use_named_args(dimensions=DIMENSIONS + [skopt.space.Integer(0, GPU_COUNT, name="gpu_id")])
def f(**kwargs):
    # skopt expects a function to minimize, reverse the output of pasteur_experiment
    # return camelback6([-1.0, 1.0])
    return pasteur_experiment(kwargs)


if __name__ == "__main__":
    start_time = datetime.now().strftime("%Y-%m-%dT%H:%M+00:00")

    # create log directory and checkpoints subdirectory if they do not exist
    os.makedirs("logs/%s" % start_time + "-checkpoints", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--steps", type=int, default=20,
                        help="Number of optimization steps (each step runs one experiment per GPU in parallel). Default: 20")
    parser.add_argument("-r", "--restore",
                        help="Optional path of a results checkpoint (.pkl or .pkl.gz file) to restart from. Default: None")
    args = parser.parse_args()

    opt = skopt.Optimizer(DIMENSIONS)

    previous_step = 0
    if args.restore:  # restore previous checkpoint and update the previous step from which to restart
        print("Restoring results from '%s'." % args.restore)
        previous_result = skopt.utils.load(args.restore)
        opt.tell(previous_result.x_iters, list(previous_result.func_vals))
        previous_step = len(previous_result.x_iters) // GPU_COUNT

    for i in range(previous_step, previous_step + args.steps):
        # get 'GPU_COUNT' new sets of params to test
        suggested = opt.ask(n_points=GPU_COUNT)
        print("[Optimizer - step %d] Suggested: %s" % (i, suggested))
        # run 'GPU_COUNT' experiments in parallel, one per GPU:
        with Pool(GPU_COUNT) as p:
            # add a gpu_id (from 0 to GPU_COUNT) to the list of params in each suggestion and pass it 
            # to the pasteur_experiment function.
            # a lambda would be cool to avoid f() but lambda can't be pickled by 'multiprocessing'
            y = p.map(f, [s + [gpu_id] for gpu_id, s in enumerate(suggested)])
            # 'y' is a generator and needs to be unpacked to a list
            y = [result for result in y]
        print("[Optimizer - step %d] Result: %s" % (i, y))
        # update the optimizer
        result = opt.tell(suggested, y)
        # save a checkpoint
        checkpoint_path = os.path.join("logs/", start_time + "-checkpoints",
                                       "pasteur-skopt-" + start_time + "-checkpoint-%02d" % i + ".pkl.gz")
        skopt.utils.dump(result, checkpoint_path)
        print("[Optimizer - step %d] Checkpoint saved to: %s" % (i, checkpoint_path))
