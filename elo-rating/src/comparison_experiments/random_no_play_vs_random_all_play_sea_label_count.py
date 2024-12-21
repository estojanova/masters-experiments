import csv
import os
import random
import uuid
import numpy as np
from river.datasets import synth
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

from sacred import Experiment
from src.train_experiments.random_ensemble_hfd_train import ex as random_ensemble_training_exp
from pymongo import MongoClient

ex = Experiment(name="random_no_play_vs_random_all_play_sea_label_count")
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


def generate_data_sets(_rnd, _seed, nr_samples_train: int, nr_samples_test: int, nr_samples_validation: int):
    generator = synth.SEA(variant=0, seed=_seed)
    generated = list(generator.take(nr_samples_test + nr_samples_validation + nr_samples_train))
    return (generated[:nr_samples_test], generated[nr_samples_test:nr_samples_test + nr_samples_validation],
            generated[nr_samples_test + nr_samples_validation:])


def collect_metrics(meta_uuid, play_pairs_strategy, _run):
    uri = "mongodb://mongo_user:mongo_password@127.0.0.1:27017/?authSource=admin"
    try:
        with MongoClient(uri) as client:
            database = client["sacred"]
            runs = database["runs"]
            metrics = database["metrics"]
            sub_runs = runs.find(
                {"config.meta_experiment": meta_uuid, "config.pick_play_pairs_strategy": play_pairs_strategy})
            majority_accuracies = []

            for sub_run in sub_runs:
                sub_run_id = sub_run.get("_id")
                majority_accuracy = metrics.find({"run_id": sub_run_id, "name": "ensemble.majority_accuracy"})[
                    0].get("values")
                majority_accuracies.append(majority_accuracy)

            average_majority_accuracy = np.mean(np.array(majority_accuracies), axis=0)
            for index, value in enumerate(average_majority_accuracy.tolist()):
                _run.log_scalar("ran_ensemble.{}_play.avg_majority_accuracy".format(play_pairs_strategy), value, index + 1)

    except Exception as e:
        raise Exception("Error collecting data from Mongo: ", e)


@ex.automain
def run(_run, _seed, meta_experiment, nr_runs_per_config, nr_samples_train, label_count, nr_samples_test,
        nr_samples_validation, test_step, nr_learners, nr_repeats):
    random.seed(_seed)
    test_set, validation_set, train_set = generate_data_sets(random, _seed, nr_samples_train, nr_samples_test,
                                                             nr_samples_validation)
    train_set_mask = train_set[:label_count] + [(x, None) for (x, y) in train_set[label_count:]]
    write_artifact(_run, train_set_mask, meta_experiment, 'train_data_set.txt')
    write_artifact(_run, test_set, meta_experiment, 'test_data_set.txt')
    write_artifact(_run, test_set, meta_experiment, 'validation_data_set.txt')

    # run multiple training sessions as per configuration of elo ensemble with no play on unlabeled data points
    for count in range(0, nr_runs_per_config):
        random_ensemble_training_exp.add_config(
            meta_experiment=meta_experiment,
            train_data_set=train_set_mask,
            test_data_set=test_set,
            nr_samples_train=nr_samples_train,
            nr_samples_test=nr_samples_test,
            test_step=test_step,
            mask_info='label-' + str(label_count),
            nr_learners=nr_learners,
            pick_train_pairs_strategy='random_subset',
            pick_play_pairs_strategy='none',
            nr_pairs=0,
            nr_repeats=nr_repeats)
        random_ensemble_training_exp.run()
    collect_metrics(meta_experiment, "none", _run)

    # run multiple training sessions as per configuration of elo ensemble with all play on unlabeled data points
    for count in range(0, nr_runs_per_config):
        random_ensemble_training_exp.add_config(
            meta_experiment=meta_experiment,
            train_data_set=train_set_mask,
            test_data_set=test_set,
            nr_samples_train=nr_samples_train,
            nr_samples_test=nr_samples_test,
            test_step=test_step,
            mask_info='label-' + str(label_count),
            nr_learners=nr_learners,
            pick_train_pairs_strategy='random_subset',
            pick_play_pairs_strategy='all',
            nr_pairs=0,
            nr_repeats=nr_repeats)
        random_ensemble_training_exp.run()
    collect_metrics(meta_experiment, "all", _run)
