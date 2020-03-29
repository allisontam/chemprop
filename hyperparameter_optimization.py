"""Optimizes hyperparameters using Bayesian optimization."""

from argparse import ArgumentParser, Namespace
from copy import deepcopy
import json
from typing import Dict, Union
import os
import time

from hyperopt import fmin, hp, tpe
import numpy as np

from chemprop.parsing import add_train_args, modify_train_args
from chemprop.train import cross_validate
from chemprop.utils import create_logger, makedirs


SPACE = {
    'hidden_size': hp.quniform('hidden_size', low=100, high=1500, q=100),
    'dropout': hp.quniform('dropout', low=0.0, high=0.4, q=0.05),
    'ffn_num_layers': hp.quniform('ffn_num_layers', low=1, high=3, q=1),
    'sample_ratio': hp.quniform('sample_ratio', low=10, high=100, q=20)
}
INT_KEYS = ['hidden_size', 'ffn_num_layers', 'sample_ratio']


def grid_search(args: Namespace):
    # Create loggers
    logger = create_logger(name='hyperparameter_optimization', save_dir=args.log_dir, quiet=True)
    train_logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)

    # Run grid search
    results = []

    # Define hyperparameter optimization
    def objective(hyperparams: Dict[str, Union[int, float]]) -> float:
        # Convert hyperparams from float to int when necessary
        for key in INT_KEYS:
            hyperparams[key] = int(hyperparams[key])

        # Update args with hyperparams
        hyper_args = deepcopy(args)
        if args.save_dir is not None:
            folder_name = '_'.join(f'{key}_{value}' for key, value in hyperparams.items())
            hyper_args.save_dir = os.path.join(hyper_args.save_dir, folder_name)
        for key, value in hyperparams.items():
            setattr(hyper_args, key, value)

        # Record hyperparameters
        logger.info(hyperparams)

        # Cross validate
        mean_score, std_score = cross_validate(hyper_args, train_logger)

        # Record results
        logger.info(f'{mean_score} +/- {std_score} {hyper_args.metric}')

        results.append({
            'mean_score': mean_score,
            'std_score': std_score,
            'hyperparams': hyperparams,
        })

        # Deal with nan
        if np.isnan(mean_score):
            if hyper_args.dataset_type == 'classification':
                mean_score = 0
            else:
                raise ValueError('Can\'t handle nan score for non-classification dataset.')

        return (1 if hyper_args.minimize_score else -1) * mean_score

    fmin(objective, SPACE, algo=tpe.suggest, max_evals=args.num_iters)

    # Report best result
    results = [result for result in results if not np.isnan(result['mean_score'])]
    best_result = min(results, key=lambda result: (1 if args.minimize_score else -1) * result['mean_score'])
    logger.info('best')
    logger.info(best_result['hyperparams'])
    logger.info(f'{best_result["mean_score"]} +/- {best_result["std_score"]} {args.metric}')

    # Save best hyperparameter settings as JSON config file
    makedirs(args.config_save_path, isfile=True)

    with open(args.config_save_path, 'w') as f:
        json.dump(best_result['hyperparams'], f, indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    add_train_args(parser)
    parser.add_argument('--num_iters', type=int, default=20,
                        help='Number of hyperparameter choices to try')
    args = parser.parse_args()
    modify_train_args(args)

    args.config_save_path = os.path.join(args.save_dir, 'hyper.json')
    args.log_dir = args.save_dir
    assert args.config_save_path  # check if supplied

    if args.loss_func == 'default':
        del SPACE['sample_ratio']
        INT_KEYS.remove('sample_ratio')

    start = time.time()
    grid_search(args)
    print('Execution time:', (time.time()-start)/3600, 'hrs')
