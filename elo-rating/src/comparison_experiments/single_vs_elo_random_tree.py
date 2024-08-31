import csv
import os
import random
import uuid
import numpy as np
from river.datasets import synth
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

from sacred import Experiment
from src.train_experiments.hfd_ensemble_train import ex as elo_training_exp
from src.train_experiments.hfd_single_train import ex as single_training_exp
from pymongo import MongoClient

ex = Experiment(name="single_vs_elo_random_tree")
ex.observers.append(MongoObserver(url='mongodb://mongo_user:mongo_password@127.0.0.1:27017/sacred?authSource=admin',
                                  db_name='sacred'))
ex.observers.append(FileStorageObserver('./runs'))


@ex.config
def cfg():
    meta_experiment = str(uuid.uuid4())


def write_artifact(_run, data, meta_uuid, filename):
    filename = f"{meta_uuid}_{filename}"
    with open(filename, 'w') as f:
        write = csv.writer(f)
        write.writerows(data)
    _run.add_artifact(filename=filename, name=filename)
    os.remove(filename)


# not very efficient for start, but is OK for small datasets
def generate_data_sets(_rnd, _seed, n_num_features: int, n_cat_features: int, n_categories_per_feature: int,
                       max_tree_depth: int, first_leaf_level: int, fraction_leaves_per_level: float,
                       nr_samples_train: int, nr_samples_test: int):
    generator = synth.RandomTree(seed_tree=_seed, seed_sample=_seed, n_classes=2, n_num_features=n_num_features,
                                 n_cat_features=n_cat_features, n_categories_per_feature=n_categories_per_feature,
                                 max_tree_depth=max_tree_depth, first_leaf_level=first_leaf_level,
                                 fraction_leaves_per_level=fraction_leaves_per_level)
    generated = list(generator.take(nr_samples_train + nr_samples_test))
    return generated[:nr_samples_train], generated[nr_samples_train:]


def collect_metrics(meta_uuid, _run):
    uri = "mongodb://mongo_user:mongo_password@127.0.0.1:27017/?authSource=admin"
    try:
        with MongoClient(uri) as client:
            database = client["sacred"]
            runs = database["runs"]
            metrics = database["metrics"]
            sub_runs = runs.find({"config.meta_experiment": meta_uuid})
            majority_accuracies = []
            best_rated_accuracies = []

            for sub_run in sub_runs:
                sub_run_id = sub_run.get("_id")
                sub_run_name = sub_run.get("experiment").get("name")
                if sub_run_name == "single_hfd_train":
                    single_accuracy = metrics.find({"run_id": sub_run_id, "name": "test_accuracy"})[0].get(
                        "values")
                    for index, value in enumerate(single_accuracy):
                        _run.log_scalar("single.test_accuracy", value, index + 1)
                if sub_run_name in ["elo_ensemble_hfd_train", "elo_ensemble_hfd_train_all_labels"]:
                    majority_accuracy = metrics.find({"run_id": sub_run_id, "name": "ensemble.majority_accuracy"})[
                        0].get("values")
                    best_rated_accuracy = metrics.find({"run_id": sub_run_id, "name": "ensemble.best_rated_accuracy"})[
                        0].get("values")
                    majority_accuracies.append(majority_accuracy)
                    best_rated_accuracies.append(best_rated_accuracy)

            average_majority_accuracy = np.mean(np.array(majority_accuracies), axis=0)
            average_best_rated_accuracy = np.mean(np.array(best_rated_accuracies), axis=0)
            for index, value in enumerate(average_majority_accuracy.tolist()):
                _run.log_scalar("ensemble.average_majority_accuracy", value, index + 1)
            for index, value in enumerate(average_best_rated_accuracy.tolist()):
                _run.log_scalar("ensemble.average_best_rated_accuracy", value, index + 1)

    except Exception as e:
        raise Exception("Error collecting data from Mongo: ", e)


@ex.automain
def run(_run, _seed, meta_experiment, n_num_features, n_cat_features, n_categories_per_feature, max_tree_depth, first_leaf_level,
        fraction_leaves_per_level, nr_runs_per_config, nr_samples_train, mask_probability, nr_samples_test,
        test_step, nr_learners, pick_train_pairs_strategy, pick_play_pairs_strategy, number_of_pairs):
    random.seed(_seed)
    train_set, test_set = generate_data_sets(random, _seed, n_num_features, n_cat_features, n_categories_per_feature,
                                             max_tree_depth, first_leaf_level, fraction_leaves_per_level,
                                             nr_samples_train, nr_samples_test)
    # apply mask to train set to remove labels
    train_set_mask = [(x, None) if random.random() < mask_probability else (x, True if y == 1 else False) for (x, y) in
                      train_set]
    test_set_adapted = [(x, True if y == 1 else False) for (x, y) in test_set]
    write_artifact(_run, train_set_mask, meta_experiment, 'train_data_set.txt')
    write_artifact(_run, test_set_adapted, meta_experiment, 'test_data_set.txt')

    single_training_exp.add_config(
        meta_experiment=meta_experiment,
        train_data_set=train_set_mask,
        test_data_set=test_set_adapted,
        nr_samples_train=nr_samples_train,
        nr_samples_test=nr_samples_test,
        test_step=test_step,
        mask_probability=mask_probability,
    )
    single_training_exp.run()

    run_count = 0
    while run_count < nr_runs_per_config:
        elo_training_exp.add_config(
            meta_experiment=meta_experiment,
            train_data_set=train_set_mask,
            test_data_set=test_set_adapted,
            nr_samples_train=nr_samples_train,
            nr_samples_test=nr_samples_test,
            test_step=test_step,
            mask_probability=mask_probability,
            rating_width=400,
            k_factor=64,
            nr_learners=nr_learners,
            pick_train_pairs_strategy=pick_train_pairs_strategy,
            pick_play_pairs_strategy=pick_play_pairs_strategy,
            number_of_pairs=number_of_pairs)
        elo_training_exp.run()
        run_count += 1
    collect_metrics(meta_experiment, _run)
