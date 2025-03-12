#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from tap import Tap
from typing import List, Optional, Literal
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from mgktools.data.data import CachedDict, Dataset
from mgktools.evaluators.cross_validation import Evaluator, Metric
from mgktools.exe.args import CommonArgs


class ModelArgs(Tap):
    model: Literal['rf', 'xgboost'] = 'rf'
    """The machine learning model to be applied."""
    device: Optional[str] = None
    """The device used for XGBoost"""


class TrainArgs(CommonArgs, ModelArgs):
    task_type: Literal['regression', 'binary', 'multi-class'] = None
    """Type of task. This determines the loss function used during training. """
    cross_validation: Literal["kFold", "leave-one-out", "Monte-Carlo", "no"] = "no"
    """The way to split data for cross-validation."""
    n_splits: int = None
    """The number of fold for kFold CV."""
    split_type: Literal["random", "scaffold_order", "scaffold_random", "stratified"] = None
    """Method of splitting the data into train/test sets."""
    split_sizes: List[float] = None
    """Split proportions for train/test sets."""
    num_folds: int = 1
    """Number of folds when performing cross validation."""
    seed: int = 0
    """Random seed."""
    metric: Metric = None
    """metric"""
    extra_metrics: List[Metric] = []
    """Metrics"""
    separate_test_path: str = None
    """Path to separate test set, optional."""

    @property
    def metrics(self) -> List[Metric]:
        return [self.metric] + self.extra_metrics


class RFClassifier(RandomForestClassifier):
    def predict_proba(self, X):
        return super().predict_proba(X)[:, 1]


class XGBC(XGBClassifier):
    def predict_proba(self, X):
        return super().predict_proba(X)[:, 1]


def main(args: TrainArgs) -> None:
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    # read data set.
    dataset = Dataset.from_df(
        df=pd.read_csv(args.data_path),
        smiles_columns=args.smiles_columns,
        features_columns=args.features_columns,
        targets_columns=args.targets_columns,
        n_jobs=1,
    )
    dataset.set_status(graph_kernel_type="no",
                       features_generators=args.features_generators,
                       features_combination=args.features_combination)
    if args.cache_path is not None:
        print(f"**\tLoading cache file: {args.cache_path}.\t**")
        cache = CachedDict.load(filename=args.cache_path)
        dataset.set_cache(cache)
    # set model
    if args.task_type == 'regression':
        if args.model == 'rf':
            model = RandomForestRegressor(random_state=args.seed, n_jobs=args.n_jobs)
        elif args.model == 'xgboost':
            model = XGBRegressor(random_state=args.seed, n_jobs=args.n_jobs, device=args.device)
    else:
        if args.model == 'rf':
            model = RFClassifier(random_state=args.seed, n_jobs=args.n_jobs)
        elif args.model == 'xgboost':
            model = XGBC(random_state=args.seed, n_jobs=args.n_jobs, device=args.device)
    # cross validation
    evaluator = Evaluator(
        save_dir=args.save_dir,
        dataset=dataset,
        model=model,
        task_type=args.task_type,
        metrics=args.metrics,
        cross_validation=args.cross_validation,
        n_splits=args.n_splits,
        split_type=args.split_type,
        split_sizes=args.split_sizes,
        num_folds=args.num_folds,
        kernel=None,
        seed=args.seed,
        verbose=True,
    )
    if args.separate_test_path is not None:
        df = pd.read_csv(args.separate_test_path)
        dataset_test = Dataset.from_df(
            df=df,
            smiles_columns=args.smiles_columns,
            features_columns=args.features_columns,
            targets_columns=args.targets_columns,
            n_jobs=1,
        )
        dataset_test.set_status(graph_kernel_type="no",
                                features_generators=args.features_generators, 
                                features_combination=args.features_combination)
        if args.cache_path is not None:
            dataset_test.set_cache(cache)
        evaluator.run_external(dataset_test)
    else:
        evaluator.run_cross_validation()


if __name__ == '__main__':
    main(args=TrainArgs().parse_args())
