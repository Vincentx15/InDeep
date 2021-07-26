#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import skopt, skopt.plots
import pandas as pd
import matplotlib.pyplot as plt


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--checkpoint", required=True,
                        help="Path of the checkpoint file to load.")
    parser.add_argument("--csv", default=None,
                        help="When set, path of the CSV output file to create.")
    parser.add_argument("--plots", default=None,
                        help="When set, path (without extension) of the plots to generate. Suffix with plot type and '.png' will be appended.")
    args = parser.parse_args()


    res = skopt.load(args.checkpoint)
    data = [[i, result] + params for i, (result, params) in enumerate(zip(res.func_vals, res.x_iters))]
    df = pd.DataFrame(data, columns=["id", "result"] + [dim.name for dim in res.space]).set_index("id")

    if args.plots:
        output = args.plots + "_convergence.png"
        skopt.plots.plot_convergence(res)
        plt.savefig(output)
        print("'%s' generated." % output)

        output = args.plots + "_evaluations.png"
        skopt.plots.plot_evaluations(res)
        plt.savefig(output)
        print("'%s' generated." % output)

        output = args.plots + "_objective.png"
        skopt.plots.plot_objective(res)
        plt.savefig(output)
        print("'%s' generated." % output)

    if args.csv:
        df.to_csv(args.csv)
        print("'%s' generated." % args.csv)

