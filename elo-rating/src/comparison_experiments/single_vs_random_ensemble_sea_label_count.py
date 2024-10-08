import csv
import os
import random
import uuid
import numpy as np
from river.datasets import synth
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

from sacred import Experiment
from src.train_experiments.random_ensemble_hfd_train import ex as ensemble_training_exp
from src.train_experiments.single_hfd_train import ex as single_training_exp
from pymongo import MongoClient

ex = Experiment(name="single_vs_random_ensemble_sea_label_count")
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


# not very efficient
def generate_data_sets(_rnd, _seed, nr_samples_train: int, nr_samples_test: int):
    generator = synth.SEA(variant=0, seed=_seed)
    generated = list(generator.take(nr_samples_train + nr_samples_test))
    return generated[nr_samples_test:], generated[:nr_samples_test]


def collect_metrics(meta_uuid, _run):
    uri = "mongodb://mongo_user:mongo_password@127.0.0.1:27017/?authSource=admin"
    try:
        with MongoClient(uri) as client:
            database = client["sacred"]
            runs = database["runs"]
            metrics = database["metrics"]
            sub_runs = runs.find({"config.meta_experiment": meta_uuid})
            majority_accuracies = []

            for sub_run in sub_runs:
                sub_run_id = sub_run.get("_id")
                sub_run_name = sub_run.get("experiment").get("name")
                if sub_run_name == "single_hfd_train":
                    single_accuracy = metrics.find({"run_id": sub_run_id, "name": "test_accuracy"})[0].get(
                        "values")
                    for index, value in enumerate(single_accuracy):
                        _run.log_scalar("single.test_accuracy", value, index + 1)
                if sub_run_name == "random_ensemble_hfd_train":
                    majority_accuracy = metrics.find({"run_id": sub_run_id, "name": "ensemble.majority_accuracy"})[
                        0].get("values")
                    majority_accuracies.append(majority_accuracy)

            average_majority_accuracy = np.mean(np.array(majority_accuracies), axis=0)
            for index, value in enumerate(average_majority_accuracy.tolist()):
                _run.log_scalar("ensemble.average_majority_accuracy", value, index + 1)

    except Exception as e:
        raise Exception("Error collecting data from Mongo: ", e)


@ex.automain
def run(_run, _seed, meta_experiment, nr_runs_per_config, nr_samples_train, label_count, nr_samples_test,
        test_step, nr_learners, pick_train_pairs_strategy, pick_play_pairs_strategy, number_of_pairs):
    random.seed(_seed)
    # generate train & test sets
    train_set, test_set = generate_data_sets(random, _seed, nr_samples_train, nr_samples_test)
    train_set_mask = train_set[:label_count] + [(x, None) for (x,y) in train_set[label_count:]]
    write_artifact(_run, train_set_mask, meta_experiment, 'train_data_set.txt')
    write_artifact(_run, test_set, meta_experiment, 'test_data_set.txt')

    # run single learner for benchmark
    single_training_exp.add_config(
        meta_experiment=meta_experiment,
        train_data_set=train_set_mask,
        test_data_set=test_set,
        nr_samples_train=nr_samples_train,
        nr_samples_test=nr_samples_test,
        test_step=test_step,
        mask_info='label-' + str(label_count),
    )
    single_training_exp.run()

    # run multiple training sessions as per configuration of elo ensemble
    run_count = 0
    while run_count < nr_runs_per_config:
        ensemble_training_exp.add_config(
            meta_experiment=meta_experiment,
            train_data_set=train_set_mask,
            test_data_set=test_set,
            nr_samples_train=nr_samples_train,
            nr_samples_test=nr_samples_test,
            test_step=test_step,
            mask_info='label-' + str(label_count),
            nr_learners=nr_learners,
            pick_train_pairs_strategy=pick_train_pairs_strategy,
            pick_play_pairs_strategy = pick_play_pairs_strategy,
            number_of_pairs = number_of_pairs)
        ensemble_training_exp.run()
        run_count += 1
    collect_metrics(meta_experiment, _run)
