import random
from river import tree
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

from dataclasses import dataclass
from sacred import Experiment

ex = Experiment(name="single_hfd_train")
ex.observers.append(MongoObserver(url='mongodb://mongo_user:mongo_password@127.0.0.1:27017/sacred?authSource=admin',
                                  db_name='sacred'))
ex.observers.append(FileStorageObserver('./runs'))


@dataclass
class SingleLearner:
    model: tree.HoeffdingTreeClassifier
    train_correct: int = 0
    train_total: int = 0


def train_single(_run, _rnd, data_set, test_step, learner: SingleLearner, test_set, nr_samples_train, nr_samples_test,
                 repeat):
    step = nr_samples_train * repeat
    for x, y in data_set:
        if y is not None:
            learner.train_total += 1
            y_pred = learner.model.predict_one(x)
            learner.train_correct += (1 if y_pred == y else 0)
            learner.model.learn_one(x, y)
        step += 1
        if step % test_step == 0:
            test_single(_run, test_set, learner, step // test_step, nr_samples_test)


def test_single(_run, test_set, learner, step_nr, nr_samples_test):
    nr_correct = 0
    for x, y in test_set:
        y_pred = learner.model.predict_one(x)
        nr_correct += (1 if y_pred == y else 0)
    _run.log_scalar("test_accuracy", nr_correct / nr_samples_test, step_nr)
    _run.log_scalar("train_total", learner.train_total, step_nr)
    _run.log_scalar("train_correct", learner.train_correct, step_nr)


@ex.automain
def run(_run, _seed, meta_experiment, train_data_set, test_data_set, nr_samples_train, mask_info,
        nr_samples_test, test_step, nr_repeats):
    learner = SingleLearner(model=tree.HoeffdingTreeClassifier())
    for repeat in range(0, nr_repeats):
        train_single(_run, random, train_data_set, test_step, learner, test_data_set, nr_samples_train, nr_samples_test,
                     repeat)
