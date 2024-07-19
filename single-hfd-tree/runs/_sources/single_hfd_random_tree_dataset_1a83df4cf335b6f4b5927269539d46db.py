import csv
import os
import random
from river import tree
from river.datasets import synth
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

from sacred import Experiment

ex = Experiment(name="single_hfd_random_tree_dataset")
ex.observers.append(MongoObserver(url='mongodb://mongo_user:mongo_password@127.0.0.1:27017/sacred?authSource=admin',
                                  db_name='sacred'))
ex.observers.append(FileStorageObserver('../runs'))


@ex.config
def cfg():
    nr_samples_train = 2000
    mask_probability = 0.3
    nr_samples_test = 50
    test_step = 20
    n_num_features = 1
    n_cat_features = 1
    n_categories_per_feature = 2
    max_tree_depth = 5
    first_leaf_level = 3
    fraction_leaves_per_level = 0.15


# not very efficient for start, but is OK for small datasets
def generate_dataset_with_mask(_rnd, _seed, n_num_features: int, n_cat_features: int, n_categories_per_feature: int,
                               max_tree_depth: int, first_leaf_level: int, fraction_leaves_per_level: float,
                               nr_samples: int, mask_prob: float):
    orig = synth.RandomTree(seed_tree=_seed, seed_sample=_seed, n_classes=2, n_num_features=n_num_features,
                            n_cat_features=n_cat_features, n_categories_per_feature=n_categories_per_feature,
                            max_tree_depth=max_tree_depth, first_leaf_level=first_leaf_level,
                            fraction_leaves_per_level=fraction_leaves_per_level).take(nr_samples)
    masked = []
    for x, y in orig:
        if _rnd.random() < mask_prob:
            masked.append((x, None))
        else:
            masked.append((x, True if y == 1 else False))
    return masked


def generate_test_set(_rnd, _seed, n_num_features: int, n_cat_features: int, n_categories_per_feature: int,
                      max_tree_depth: int, first_leaf_level: int, fraction_leaves_per_level: float, nr_samples: int):

    orig = synth.RandomTree(seed_tree=_seed, seed_sample=_seed - 7, n_classes=2, n_num_features=n_num_features,
                     n_cat_features=n_cat_features, n_categories_per_feature=n_categories_per_feature,
                     max_tree_depth=max_tree_depth, first_leaf_level=first_leaf_level,
                     fraction_leaves_per_level=fraction_leaves_per_level).take(nr_samples)
    with_replaced_classes = []
    for x, y in orig:
        with_replaced_classes.append((x, True if y == 1 else False))
    return with_replaced_classes

def train(_run, _rnd, data_set, test_step, learner, test_set, nr_samples_test):
    step = 0
    train_observed = 0
    train_correct = 0
    for x, y in data_set:
        if y is not None:
            train_observed += 1
            y_pred = learner.predict_one(x)
            if y_pred == y:
                train_correct += 1
            learner.learn_one(x, y)
        step += 1
        if step % test_step == 0:
            test(_run, test_set, learner, step, nr_samples_test)
        _run.log_scalar("train_observed", train_observed, step)
        _run.log_scalar("train_correct", train_correct, step)


def test(_run, test_set, learner, step_nr, nr_samples_test):
    nr_correct = 0
    for x, y in test_set:
        y_pred = learner.predict_one(x)
        if y == y_pred:
            nr_correct += 1
    learner_accuracy = nr_correct / nr_samples_test
    _run.log_scalar("test_accuracy", learner_accuracy, step_nr)


def write_artifact(_run, data, filename):
    with open(filename, 'w') as f:
        write = csv.writer(f)
        write.writerows(data)
    _run.add_artifact(filename=filename, name=filename)
    os.remove(filename)


@ex.automain
def run(_run, _seed, n_num_features: int, n_cat_features, n_categories_per_feature,
        max_tree_depth, first_leaf_level, fraction_leaves_per_level, nr_samples_train, mask_probability,
        nr_samples_test, test_step):
    random.seed(_seed)
    learner = tree.HoeffdingTreeClassifier()
    train_set = generate_dataset_with_mask(random, _seed, n_num_features, n_cat_features, n_categories_per_feature,
                                           max_tree_depth, first_leaf_level, fraction_leaves_per_level,
                                           nr_samples_train, mask_probability)
    test_set = generate_test_set(random, _seed, n_num_features, n_cat_features, n_categories_per_feature,
                                 max_tree_depth, first_leaf_level, fraction_leaves_per_level, nr_samples_test)
    write_artifact(_run, train_set, 'train_data_set.txt')
    write_artifact(_run, test_set, 'test_data_set.txt')
    train(_run, random, train_set, test_step, learner, test_set, nr_samples_test)
