import csv
import os
import random
import uuid
import numpy as np
from river.datasets import synth
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

from sacred import Experiment
from elo_ensemble_hf_trees import ex as training_exp
from pymongo import MongoClient

ex = Experiment(name="elo_random_tree_dataset")
ex.observers.append(MongoObserver(url='mongodb://mongo_user:mongo_password@127.0.0.1:27017/sacred?authSource=admin',
                                  db_name='sacred'))
ex.observers.append(FileStorageObserver('../runs'))


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
                majority_accuracy = metrics.find({"run_id": sub_run_id, "name": "ensemble.majority_accuracy"})[0].get(
                    "values")
                best_rated_accuracy = metrics.find({"run_id": sub_run_id, "name": "ensemble.best_rated_accuracy"})[
                    0].get("values")
                majority_accuracies.append(majority_accuracy)
                best_rated_accuracies.append(best_rated_accuracy)

            average_majority_accuracy = np.mean(np.array(majority_accuracies), axis=0)
            average_best_rated_accuracy = np.mean(np.array(best_rated_accuracies), axis=0)
            for index, value in enumerate(average_majority_accuracy.tolist()):
                _run.log_scalar("average_majority_accuracy", value, index + 1)
            for index, value in enumerate(average_best_rated_accuracy.tolist()):
                _run.log_scalar("average_best_rated_accuracy", value, index + 1)

    except Exception as e:
        raise Exception("Error collecting data from Mongo: ", e)


@ex.config
def cfg():
    n_num_features = 1
    n_cat_features = 1
    n_categories_per_feature = 2
    max_tree_depth = 5
    first_leaf_level = 3
    fraction_leaves_per_level = 0.15
    nr_of_runs_per_config = 5
    nr_samples_train = 600
    mask_probability = 0.5
    nr_samples_test = 50
    test_step = 20
    nr_learners = 3
    pick_pairs_strategy = 'all'


@ex.automain
def run(_run, _seed, n_num_features, n_cat_features, n_categories_per_feature, max_tree_depth, first_leaf_level,
        fraction_leaves_per_level, nr_of_runs_per_config, nr_samples_train, mask_probability, nr_samples_test,
        test_step, nr_learners, pick_pairs_strategy):
    meta_uuid = str(uuid.uuid4())
    random.seed(_seed)
    train_set, test_set = generate_data_sets(random, _seed, n_num_features, n_cat_features, n_categories_per_feature,
                                             max_tree_depth, first_leaf_level, fraction_leaves_per_level,
                                             nr_samples_train, nr_samples_test)
    # apply mask to train set to remove labels
    train_set_mask = [(x, None) if random.random() < mask_probability else (x, True if y == 1 else False) for (x, y) in
                      train_set]
    test_set_adapted = [(x, True if y == 1 else False) for (x, y) in test_set]
    write_artifact(_run, train_set_mask, meta_uuid, 'train_data_set.txt')
    write_artifact(_run, test_set_adapted, meta_uuid, 'test_data_set.txt')
    run_count = 0
    while run_count < nr_of_runs_per_config:
        training_exp.add_config(
            meta_experiment=meta_uuid,
            train_data_set=train_set_mask,
            test_data_set=test_set_adapted,
            nr_samples_train=nr_samples_train,
            nr_samples_test=nr_samples_test,
            test_step=test_step,
            mask_probability=mask_probability,
            rating_width=400,
            k_factor=64,
            nr_learners=nr_learners,
            pick_pairs_strategy=pick_pairs_strategy)
        training_exp.run()
        run_count += 1
    collect_metrics(meta_uuid, _run)
