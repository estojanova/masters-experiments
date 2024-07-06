import csv
import os
import random
from river import tree
from river.datasets import synth
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

from sacred import Experiment

ex = Experiment(name="single_hfd_random_tree_stream")
ex.observers.append(MongoObserver(url='mongodb://mongo_user:mongo_password@127.0.0.1:27017/sacred?authSource=admin',
                                  db_name='sacred'))
ex.observers.append(FileStorageObserver('../runs'))


@ex.config
def cfg():
    nr_samples_train = 600
    mask_probability = 0.3
    nr_samples_test = 50
    test_step = 20


# not very efficient for start, but is OK for small datasets
def generate_dataset_with_mask(_rnd, _seed, nr_samples: int, mask_prob: float):
    orig = synth.SEA(variant=0, seed=_seed).take(nr_samples)
    masked = []
    for x, y in orig:
        if _rnd.random() < mask_prob:
            masked.append((x, None))
        else:
            masked.append((x, y))
    return masked


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
def run(_run, _seed, nr_samples_train, mask_probability, nr_samples_test, test_step):
    random.seed(_seed)
    learner = tree.HoeffdingTreeClassifier()
    train_set = generate_dataset_with_mask(random, _seed, nr_samples_train, mask_probability)
    test_set = list(synth.SEA(variant=0, seed=_seed).take(nr_samples_test))
    write_artifact(_run, train_set, 'train_data_set.txt')
    write_artifact(_run, test_set, 'test_data_set.txt')
    train(_run, random, train_set, test_step, learner, test_set, nr_samples_test)
