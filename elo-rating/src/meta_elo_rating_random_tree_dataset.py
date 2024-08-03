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
    n_num_features = 1
    n_cat_features = 1
    n_categories_per_feature = 2
    max_tree_depth = 5
    first_leaf_level = 3
    fraction_leaves_per_level = 0.15
    nr_samples_train = 600
    mask_probability = 0.5
    nr_samples_test = 50
    test_step = 20


@ex.automain
def run(_run, _seed, nr_samples_train, mask_probability, nr_samples_test, test_step, n_num_features, n_cat_features,
        n_categories_per_feature, max_tree_depth, first_leaf_level, fraction_leaves_per_level):
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

    sub_experiment.add_config(
        meta_experiment=meta_uuid,
        train_data_set=train_set_mask,
        test_data_set=test_set_adapted,
        nr_samples_train=nr_samples_train,
        nr_samples_test=nr_samples_test,
        test_step=test_step,
        mask_probability=mask_probability,
        rating_width=400,
        k_factor=64,
        nr_learners=3,
        pick_pairs_strategy='random_subset')
    sub_experiment.run()
    collect_metrics(meta_uuid)
