import skopt
import os, time, sys
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor as Pool
import torch

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from learning import model_factory
from learning import predict

DIMENSIONS_HD = [
    skopt.space.Real(0.5, 2.0, name="min_merging_value_hd"),
    skopt.space.Integer(15, 25, name="max_euclidean_distance_hd")
]

DIMENSIONS_PL = [
    skopt.space.Real(0.0, 1.0, name="min_merging_value_pl"),
    skopt.space.Integer(0, 10, name="min_euclidean_distance_pl"),
    skopt.space.Integer(10, 20, name="max_euclidean_distance_pl")
]

if __name__ == "__main__":
    start_time = datetime.now().strftime("%Y-%m-%dT%H:%M+00:00")
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--steps", type=int, default=20,
                        help="Number of optimization steps, Default: 20")
    parser.add_argument("-r", "--restore",
                        help="Optional path of a results checkpoint "
                             "(.pkl or .pkl.gz file) to restart from. Default: None")
    parser.add_argument('--do_pl', help='If we want to print the blobs', action='store_true')
    args = parser.parse_args()

    # this makes hd True by default, and the command to add is --do_pl not dont_hd...
    hd = not args.do_pl
    dimensions = DIMENSIONS_HD if hd else DIMENSIONS_PL
    opt = skopt.Optimizer(dimensions=dimensions)


    # This function declaration is really not elegant but decorator with dimensions makes it that we need it here
    # and that we cannot give hd as an argument.
    @skopt.utils.use_named_args(dimensions=dimensions)
    def f_hd(**kwargs):
        hetatm = False
        default_exp_path = os.path.join(script_dir, '../results/experiments/HPO.exp')
        model = model_factory.model_from_exp(expfilename=default_exp_path, load_weights=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        final_score = predict.v_2(model, device=device, no_HD=hetatm, no_PL=not hetatm,
                                  blobber_params=kwargs)
        print(final_score)
        # We need to make it a loss-like (the lower the better)
        final_loss = - final_score
        return final_loss


    @skopt.utils.use_named_args(dimensions=dimensions)
    def f_pl(**kwargs):
        hetatm = True
        default_exp_path = os.path.join(script_dir, '../results/experiments/HPO.exp')
        model = model_factory.model_from_exp(expfilename=default_exp_path, load_weights=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        final_score = predict.v_2(model, device=device, no_HD=hetatm, no_PL=not hetatm,
                                  blobber_params=kwargs)
        # We need to make it a loss-like (the lower the better)
        final_loss = - final_score
        return final_loss


    f = f_hd if hd else f_pl

    # create log directory and checkpoints subdirectory if they do not exist
    hd_str = "hd_2" if hd else "pl"
    log_dir = f"logs/{hd_str}"
    os.makedirs(log_dir, exist_ok=True)

    previous_step = 0
    if args.restore:  # restore previous checkpoint and update the previous step from which to restart
        print("Restoring results from '%s'." % args.restore)
        previous_result = skopt.utils.load(args.restore)
        opt.tell(previous_result.x_iters, list(previous_result.func_vals))
        previous_step = len(previous_result.x_iters)

    for i in range(previous_step, previous_step + args.steps):
        suggested = opt.ask(n_points=1)
        suggested = suggested[0]
        print("[Optimizer - step %d] Suggested: %s" % (i, suggested))
        y = f(suggested)
        print("[Optimizer - step %d] Result: %s" % (i, y))
        # update the optimizer
        result = opt.tell(suggested, y)
        # save a checkpoint
        checkpoint_path = os.path.join(log_dir, f"step_{i}.pkl.gz")
        skopt.utils.dump(result, checkpoint_path)
        print("[Optimizer - step %d] Checkpoint saved to: %s" % (i, checkpoint_path))
