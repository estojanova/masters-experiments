import random
from river import tree
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

from sacred import Experiment

ex = Experiment(name="single_hfd_train")
ex.observers.append(MongoObserver(url='mongodb://mongo_user:mongo_password@127.0.0.1:27017/sacred?authSource=admin',
                                  db_name='sacred'))
ex.observers.append(FileStorageObserver('./runs'))


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


@ex.automain
def run(_run, _seed, meta_experiment, train_data_set, test_data_set, nr_samples_train, mask_info,
        nr_samples_test, test_step):
    learner = tree.HoeffdingTreeClassifier()
    train(_run, random, train_data_set, test_step, learner, test_data_set, nr_samples_test)
