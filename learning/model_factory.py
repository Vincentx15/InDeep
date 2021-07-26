from learning.Nnet import c3d_from_hparams
from learning.utils import ConfParser

def model_from_hparams(hparams, load_weights=False):
    # Old versions don't use Unets
    try:
        use_unet = hparams.get('argparse', 'use_unet')
    except KeyError:
        use_unet = False

    try:
        use_old_unet = hparams.get('argparse', 'use_old_unet')
    except KeyError:
        use_old_unet = True

    if use_unet:
        if use_old_unet:
            from learning.Unet import unet_from_hparams
        else:
            from learning.NewUnet import unet_from_hparams
        return unet_from_hparams(hparams=hparams, load_weights=load_weights)
    else:
        return c3d_from_hparams(hparams=hparams, load_weights=load_weights)


def model_from_exp(expfilename, load_weights=False):
    hparams = ConfParser(path_to_exp=expfilename)
    model = model_from_hparams(hparams, load_weights=load_weights)
    return model

