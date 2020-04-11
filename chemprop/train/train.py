from argparse import Namespace
import logging
from typing import Callable, List

# from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import trange

from chemprop.data import MolPairDataset, convert2contrast
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR


def train(model: nn.Module,
          data: MolPairDataset,
          loss_func: Callable,
          optimizer: Optimizer,
          args: Namespace,
          n_grad_step: int = 3) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data: A MolPairDataset (or a list of MolPairDatasets if using moe).
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param args: Arguments.
    :return: The total number of iterations (training examples) trained on so far.
    """
    model.train()

    data.shuffle()  # Very important this is done before conversion to maintain randomness in contrastive dataset.

    loss_sum, n_iter = 0, 0

    if args.loss_func == 'contrastive':
        data = convert2contrast(data)
    batch_size = len(data) // n_grad_step
    num_iters = n_grad_step * batch_size  # don't use the last batch if it's small, for stability

    for i in range(0, num_iters, batch_size):
        mol_batch = MolPairDataset(data[i:i + batch_size])
        smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()
        batch = smiles_batch
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])
        if args.loss_func == 'contrastive':
            mask = targets
        else:
            mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])

        if next(model.parameters()).is_cuda:
            mask, targets = mask.cuda(), targets.cuda()

        if args.dataset_type == 'regression':
            class_weights = torch.ones(targets.shape)
        else:
            class_weights = targets*(args.class_weights-1) + 1

        if args.cuda:
            class_weights = class_weights.cuda()

        # Run model
        model.zero_grad()
        preds = model(batch, features_batch)

        if args.dataset_type == 'multiclass':
            targets = targets.long()
            loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * class_weights * mask
        else:
            loss = loss_func(preds, targets) * class_weights * mask
        loss = loss.sum() / mask.sum()

        loss_sum += loss.item()

        loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        n_iter += args.batch_size

    return n_iter
