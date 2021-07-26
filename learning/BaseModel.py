import torch.nn as nn
import os
import torch


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def load_weights(self, hparams):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        try:
            print('Successful loading')
            rel_path = hparams.get('best_weights', 'path')
        except KeyError:
            print('Could not load best weights, loading epoch 38')
            rel_path = os.path.join("../results/weights/", f"{hparams.get('argparse', 'name')}", "cp-0038.ckpt")
        weights = os.path.join(script_dir, rel_path)
        self.load_state_dict(torch.load(weights))

    @staticmethod
    def compute_weight(target, scale_ones, enveloppe=None, scale_enveloppe=None):
        weight = target + scale_ones * torch.ones_like(target)
        if enveloppe.sum() > 0:
            weight += scale_enveloppe * enveloppe.squeeze()
        return weight

    def weighted_CE_loss(self, output, target, weight=None):
        """
        Return weighted CE for the HD branch
        :param output:
        :param target:
        :return:
        """
        # target_proba = 1 - target[0, -1, ...]
        # weight_proba = target_proba + self.whd * torch.ones_like(target_proba)
        grid_loss_values = target * torch.log(output + 1e-5)
        if weight is None:
            return -torch.mean(grid_loss_values)
        expanded = weight.expand_as(target)
        return -torch.mean(expanded * grid_loss_values)

    def weighted_BCE_loss(self, output, target, weight=None):
        """
        Return weighted BCE for the PL branch
        :param output:
        :param target:
        :return:
        """
        loss_obj = nn.BCELoss(weight=weight)
        return loss_obj(output, target)

    def weighted_dice_loss(self, output, target, weight=None):
        """
        Return weighted Dice loss for the HD branch
        :param output:
        :param target:
        :return:
        """
        grid_loss_values = (2 * target * output + 0.01) / (target + output + 0.01)
        if weight is None:
            return -torch.mean(grid_loss_values)
        expanded = weight.expand_as(target)
        return -torch.mean(expanded * grid_loss_values)

    def weighted_binary_dice_loss(self, output, target, weight=None):
        """
        Return weighted Dice loss for the PL branch
        :param output:
        :param target:
        :return:
        """
        grid_loss_values = (2 * target * output + 0.01) / (target + output + 0.01)
        if weight is None:
            return -torch.mean(grid_loss_values)
        return -torch.mean(weight * grid_loss_values)

    # def custom_loss_HD(self, output, target):
    #     """
    #     Tries to get the binary score and match the distributions only where it's non zero.
    #     Even better with sigmoid AND softmax instead of just softmax
    #     :param output:
    #     :param target:
    #     :return:
    #     """
    #     # This is the probability of the void, almost always 1
    #     probas_output, probas_target = output[0, -1, ...], target[0, -1, ...]
    #     binary = nn.BCELoss()(probas_output, probas_target)
    #
    #     distrib_output, distrib_target = output[0, :-1, ...], target[0, :-1, ...]
    #     match_distrib = self.CELoss(distrib_output, distrib_target)
    #     return match_distrib + binary

    # @staticmethod
    # def CELoss(output, target):
    #     """
    #     Return Cross Entropy loss summed over inputs and outputs.
    #     Tensors must have matching sizes
    #     :param output:
    #     :param target:
    #     :return:
    #     """
    #     return -torch.mean(target * torch.log(output + 1e-5))
