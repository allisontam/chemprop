from typing import List
import csv
import os

import torch
import torch.nn as nn
from tqdm import trange

from chemprop.data import MolPairDataset, StandardScaler
from chemprop.data.utils import flip_data


def predict(model: nn.Module,
            data: MolPairDataset,
            batch_size: int,
            scaler: StandardScaler = None,
            avg_reverse: bool = True) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MolPairDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    preds = []

    num_iters, iter_step = len(data), batch_size

    for i in trange(0, num_iters, iter_step):
        # Prepare batch
        batches = [MolPairDataset(data[i:i + batch_size])]
        ret = []
        if avg_reverse:
            batches.append( flip_data(batches[-1]) )

        for mol_batch in batches:
            smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

            # Run model
            batch = smiles_batch

            with torch.no_grad():
                batch_preds, _ = model(batch, features_batch)

            batch_preds = batch_preds.data.cpu().numpy()

            # Inverse scale if regression
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)
            ret.append(batch_preds)

        if avg_reverse:
            batch_preds = (ret[0] + ret[1])/2
        else:
            batch_preds = ret[0]

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds

def save_predictions(save_dir: str,
                     train_data: MolPairDataset,
                     val_data: MolPairDataset,
                     test_data: MolPairDataset,
                     train_preds: List[List[float]],
                     val_preds: List[List[float]],
                     test_preds: List[List[float]],
                     task_names: List[str],
                     scaler: StandardScaler = None) -> None:
    """
    Saves predictions to csv file for entire model.

    Any of the datasets can be absent. They will not be saved in that case.
    """
    with open(os.path.join(save_dir, 'preds.csv'), 'w') as f:
        writer = csv.writer(f)
        header = ['SMILE1', 'SMILE2', 'SPLIT'] + task_names + ['PRED_' + task for task in task_names]
        writer.writerow(header)

        splits = ['train', 'val', 'test']
        dataSplits = [train_data, val_data, test_data]
        predSplits = [train_preds, val_preds, test_preds]
        for k, split in enumerate(splits):
            if dataSplits[k] is None:
                continue
            smiles = dataSplits[k].smiles()
            targets = dataSplits[k].targets()
            # Inverse scale if regression and only for training data
            if k == 0 and scaler is not None:
                targets = scaler.inverse_transform(targets)

            preds = predSplits[k]
            for i in range(len(smiles)):
                row = [smiles[i][0], smiles[i][1], split] + targets[i] + preds[i]
                writer.writerow(row)
