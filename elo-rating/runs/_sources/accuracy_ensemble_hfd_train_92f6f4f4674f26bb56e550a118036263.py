from dataclasses import dataclass

import random
from river import tree
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

from sacred import Experiment

ex = Experiment(name="accuracy_ensemble_hfd_train")
ex.observers.append(MongoObserver(url='mongodb://mongo_user:mongo_password@127.0.0.1:27017/sacred?authSource=admin',
                                  db_name='sacred'))
ex.observers.append(FileStorageObserver('./runs'))


@dataclass
class Model:
    id: int
    model: tree.HoeffdingTreeClassifier
    train_correct_prediction: int = 0
    train_total_observed: int = 0
    games_played: int = 0
    games_lost: int = 0
    games_won: int = 0
    games_tie: int = 0
    test_accuracy: int = 0


def generate_pairs_random(_rnd, ensemble):
    ensemble_length = len(ensemble)
    all_ids = list(range(0, ensemble_length))
    _rnd.shuffle(all_ids)
    # randomly remove 1 element from tail if odd number
    if ensemble_length % 2 == 1:
        all_ids.pop()
    id_iterator = iter(all_ids)
    pairs = []
    for id1, id2 in zip(id_iterator, id_iterator):
        pairs.append((ensemble[id1], ensemble[id2]))
    return pairs


def pick_pairs(_rnd, pairs, pick_pairs_strategy, fixed_nr):
    if pick_pairs_strategy == 'none':
        return []
    if pick_pairs_strategy == 'fixed_nr':
        return _rnd.sample(pairs, fixed_nr)
    if pick_pairs_strategy == 'random_subset':
        return _rnd.sample(pairs, _rnd.randrange(len(pairs) + 1))
    if pick_pairs_strategy == 'all':
        return pairs
    raise Exception("No pick_pairs_strategy provided")


def play_pair( _rnd, x, learner1: Model, learner2: Model):
    learner1.games_played += 1
    learner2.games_played += 1
    prediction1 = learner1.model.predict_one(x)
    prediction2 = learner2.model.predict_one(x)
    if prediction1 is None or prediction2 is None:
        return
    if prediction1 == prediction2:
        learner1.games_tie += 1
        learner2.games_tie += 1
        return
    if learner1.test_accuracy > learner2.test_accuracy:
        learner2.model.learn_one(x, prediction1)
        learner1.games_won += 1
        learner2.games_lost += 1
        return
    if learner2.test_accuracy > learner1.test_accuracy:
        learner1.model.learn_one(x, prediction2)
        learner1.games_lost += 1
        learner2.games_won += 1
        return


def train_pair(x, y, learner1: Model, learner2: Model):
    prediction1 = learner1.model.predict_one(x)
    prediction2 = learner2.model.predict_one(x)
    learner_1_is_correct = (prediction1 is not None) and (prediction1 == y)
    learner_2_is_correct = (prediction2 is not None) and (prediction2 == y)
    learner1.model.learn_one(x, y)
    learner2.model.learn_one(x, y)
    learner1.train_correct_prediction += (1 if learner_1_is_correct else 0)
    learner2.train_correct_prediction += (1 if learner_2_is_correct else 0)
    learner1.train_total_observed += 1
    learner2.train_total_observed += 1


def train(_run, _rnd, train_set, ensemble, test_step, test_set, nr_samples_test,
          pick_train_pairs_strategy, pick_play_pairs_strategy, number_of_pairs):
    nr_learners = len(ensemble)
    step = 0
    for x, y in train_set:
        all_pairs = generate_pairs_random(_rnd, ensemble)
        if y is None:
            play_pairs = pick_pairs(_rnd, all_pairs, pick_play_pairs_strategy, number_of_pairs)
            for (learner1, learner2) in play_pairs:
                play_pair(_rnd, x, learner1, learner2)
                if nr_learners < 15:
                    log_train_metrics(_run, learner1, step)
                    log_train_metrics(_run, learner2, step)
        else:
            train_pairs = pick_pairs(_rnd, all_pairs, pick_train_pairs_strategy, number_of_pairs)
            for (learner1, learner2) in train_pairs:
                train_pair(x, y, learner1, learner2)
                if nr_learners < 15:
                    log_train_metrics(_run, learner1, step)
                    log_train_metrics(_run, learner2, step)
        step += 1
        if step % test_step == 0:
            test(_run, test_set, ensemble, step, nr_samples_test)


def test(_run, test_set, ensemble, step_nr, nr_samples_test):
    nr_learners = len(ensemble)
    individual_correct_counts = list(0 for i in range(nr_learners))
    majority_correct_count = 0
    for x, y in test_set:
        predictions = list((learner.id, learner.model.predict_one(x)) for learner in ensemble)
        true_count = 0
        for (learner_id, prediction) in predictions:
            individual_correct_counts[learner_id] += (1 if prediction is not None and prediction == y else 0)
            true_count += (1 if prediction is True else 0)
        majority_prediction = True if true_count >= (nr_learners / 2) else False
        majority_correct_count += (1 if majority_prediction == y else 0)
    for learner in ensemble:
        learner.test_accuracy = individual_correct_counts[learner.id] / nr_samples_test
        _run.log_scalar("learner{}.test_accuracy".format(learner.id), learner.test_accuracy, step_nr)
    _run.log_scalar("ensemble.majority_accuracy", majority_correct_count / nr_samples_test, step_nr)


def log_train_metrics(_run, learner: Model, step_nr: int):
    _run.log_scalar("learner{}.train_correct".format(learner.id), learner.train_correct_prediction, step_nr)
    _run.log_scalar("learner{}.train_observed".format(learner.id), learner.train_total_observed, step_nr)
    _run.log_scalar("learner{}.games_played".format(learner.id), learner.games_played, step_nr)
    _run.log_scalar("learner{}.games_won".format(learner.id), learner.games_won, step_nr)
    _run.log_scalar("learner{}.games_lost".format(learner.id), learner.games_lost, step_nr)
    _run.log_scalar("learner{}.games_tie".format(learner.id), learner.games_tie, step_nr)


@ex.automain
def run(_run, _seed, meta_experiment, train_data_set, test_data_set, nr_samples_train, nr_samples_test, test_step,
        mask_info, nr_learners, pick_train_pairs_strategy, pick_play_pairs_strategy, number_of_pairs):
    random.seed(_seed)
    ensemble = list(Model(i, tree.HoeffdingTreeClassifier()) for i in range(nr_learners))
    train(_run, random, train_data_set, ensemble, test_step, test_data_set,
          nr_samples_test, pick_train_pairs_strategy, pick_play_pairs_strategy, number_of_pairs)
