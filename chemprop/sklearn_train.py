from argparse import Namespace
from logging import Logger
import os
import pickle
from pprint import pformat
from typing import Callable, List, Tuple
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from tqdm import trange, tqdm

from chemprop.data import MolPairDataset
from chemprop.data.utils import get_data, get_task_names, flip_data, split_data
from chemprop.features import get_features_generator
from chemprop.train.evaluate import evaluate_predictions
from chemprop.train.predict import save_predictions
from chemprop.utils import get_metric_func, makedirs


def predict(model,
            model_type: str,
            dataset_type: str,
            features: List[np.ndarray]) -> List[List[float]]:
    if dataset_type == 'regression':
        preds = model.predict(features)

        if len(preds.shape) == 1:
            preds = [[pred] for pred in preds]
    elif dataset_type == 'classification':
        if model_type == 'random_forest':
            preds = model.predict_proba(features)

            if type(preds) == list:
                # Multiple tasks
                num_tasks, num_preds = len(preds), len(preds[0])
                preds = [[preds[i][j, 1] for i in range(num_tasks)] for j in range(num_preds)]
            else:
                # One task
                preds = [[preds[i, 1]] for i in range(len(preds))]
        elif model_type == 'svm':
            preds = model.decision_function(features)
            preds = [[pred] for pred in preds]
        else:
            raise ValueError(f'Model type "{model_type}" not supported')
    else:
        raise ValueError(f'Dataset type "{dataset_type}" not supported')

    return preds


def _compose(feat_tuple):
    ret = [feat for feat in feat_tuple if feat is not None]
    return np.concatenate(ret)

def compose_feats(data):
    return [_compose(feat) for feat in data.features()]

def multi_task_sklearn(model,
                       train_data: MolPairDataset,
                       val_data: MolPairDataset,
                       test_data: MolPairDataset,
                       metric_func: Callable,
                       args: Namespace,
                       logger: Logger = None) -> List[float]:
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print
    num_tasks = train_data.num_tasks()

    train_targets = train_data.targets()
    if train_data.num_tasks() == 1:
        train_targets = [targets[0] for targets in train_targets]

    # Train
    feats = compose_feats(train_data)
    debug(f'Using {feats[0].shape} features')
    model.fit(feats, train_targets)

    # Save model
    with open(os.path.join(args.save_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    debug('Predicting')
    test_preds = predict(
        model=model,
        model_type=args.model_type,
        dataset_type=args.dataset_type,
        features=compose_feats(test_data)
    )

    if args.save_preds:
        val_preds = predict(
            model=model,
            model_type=args.model_type,
            dataset_type=args.dataset_type,
            features=compose_feats(val_data)
        )
        val_scores = evaluate_predictions(
            preds=val_preds,
            targets=val_data.targets(),
            num_tasks=num_tasks,
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            logger=logger
        )
        info(f'Val {args.metric} = {np.nanmean(val_scores)}')
        save_predictions(args.save_dir, None, val_data, test_data, None, val_preds, test_preds, args.task_names)

    scores = evaluate_predictions(
        preds=test_preds,
        targets=test_data.targets(),
        num_tasks=num_tasks,
        metric_func=metric_func,
        dataset_type=args.dataset_type,
        logger=logger
    )
    return scores


def run_sklearn(args: Namespace, logger: Logger = None) -> List[float]:
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    debug(pformat(vars(args)))

    metric_func = get_metric_func(args.metric)

    debug('Loading data')
    data = get_data(path=args.data_path, args=args)

    if args.model_type == 'svm' and data.num_tasks() != 1:
        raise ValueError(f'SVM can only handle single-task data but found {data.num_tasks()} tasks')

    debug(f'Splitting data with seed {args.seed}')
    # Need to have val set so that train and test sets are the same as when doing MPN
    train_data, val_data, test_data = split_data(
        data=data,
        split_type=args.split_type,
        seed=args.seed,
        sizes=args.split_sizes,
        args=args
    )
    if args.symmetric:
        train_data = flip_data(train_data)

    debug(f'Total size = {len(data):,} | train size = {len(train_data):,} | test size = {len(test_data):,}')

    if args.features_path is None or len(args.features_path) == 0:
        debug('Computing morgan fingerprints')
        morgan_fingerprint = get_features_generator('morgan')
        for dataset in [train_data, val_data, test_data]:
            for datapoint in tqdm(dataset, total=len(dataset)):
                datapoint.set_features(morgan_fingerprint(mol=datapoint.drug_smiles, radius=args.radius, num_bits=args.num_bits), 0)
                datapoint.set_features(morgan_fingerprint(mol=datapoint.cmpd_smiles, radius=args.radius, num_bits=args.num_bits), 1)

    debug('Building model')
    if args.dataset_type == 'regression':
        if args.model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=args.num_trees, n_jobs=-1)
        elif args.model_type == 'svm':
            model = SVR()
        else:
            raise ValueError(f'Model type "{args.model_type}" not supported')
    elif args.dataset_type == 'classification':
        if args.model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=args.num_trees, n_jobs=-1, class_weight=args.class_weight)
        elif args.model_type == 'svm':
            model = SVC()
        else:
            raise ValueError(f'Model type "{args.model_type}" not supported')
    else:
        raise ValueError(f'Dataset type "{args.dataset_type}" not supported')

    debug(model)

    debug('Training')
    scores = multi_task_sklearn(
        model=model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        metric_func=metric_func,
        args=args,
        logger=logger
    )

    info(f'Test {args.metric} = {np.nanmean(scores)}')

    return scores


def cross_validate_sklearn(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    # TODO: Multi-task might not work with RF.
    info = logger.info if logger is not None else print
    args.task_names = get_task_names(args.data_path, args.data_format)
    init_seed = args.seed
    save_dir = args.save_dir

    # Run training on different random seeds for each fold
    all_scores = []
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        model_scores = run_sklearn(args, logger)
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)

    # Report scores for each fold
    for fold_num, scores in enumerate(all_scores):
        info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

    # Report scores across folds
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')

    return mean_score, std_score
