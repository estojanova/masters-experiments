
import csv
import os
import random
import uuid
from river.datasets import synth
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

from sacred import Experiment
from elo_rating_hf_trees import ex as sub_experiment
from pymongo import MongoClient

ex = Experiment(name="meta_elo_rating")
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


# not very efficient
def generate_data_sets(_rnd, _seed, nr_samples_train: int, nr_samples_test: int, mask_prob: float):
    generator = synth.SEA(variant=0, seed=_seed)
    # take first nr_samples_train from the generator for training
    train_set_orig = generator.take(nr_samples_train)
    # take additional nr_samples_test from the generator for testing
    test_set_orig = generator.take(nr_samples_test)
    return train_set_orig, test_set_orig


def collect_metrics(meta_uuid):
    try:
        uri = "mongodb://mongo_user:mongo_password@127.0.0.1:27017/?authSource=admin"
        client = MongoClient(uri)
        database = client["sacred"]
        collection = database["runs"]
        sub_runs = collection.find({"config.meta_experiment": meta_uuid})
        for sub_run in sub_runs:
            print(sub_run.get("_id"))
        client.close()
    except Exception as e:
        raise Exception("Error collecting data from Mongo: ", e)


@ex.config
def cfg():
    nr_of_runs_per_config = 5
    nr_samples_train = 600
    mask_probability = 0.5
    nr_samples_test = 50
    test_step = 20


@ex.automain
def run(_run, _seed, nr_samples_train, mask_probability, nr_samples_test, test_step):
    meta_uuid = str(uuid.uuid4())
    random.seed(_seed)
    train_set, test_set = generate_data_sets(random, _seed, nr_samples_train, nr_samples_test, mask_probability)
    # apply mask to train set to remove labels
    train_set_mask = [(x, None) if random.random() < mask_probability else (x, y) for (x, y) in train_set]
    write_artifact(_run, train_set, meta_uuid, 'train_data_set.txt')
    write_artifact(_run, test_set, meta_uuid, 'test_data_set.txt')

    sub_experiment.add_config(
        meta_experiment=meta_uuid,
        train_data_set=train_set_mask,
        test_data_set=test_set,
        nr_samples_train=nr_samples_train,
        nr_samples_test=nr_samples_test,
        test_step=test_step,
        mask_probability = mask_probability,
        rating_width=400,
        k_factor=64,
        nr_learners=3,
        pick_pairs_strategy='random_subset')
    sub_experiment.run()
    collect_metrics(meta_uuid)
