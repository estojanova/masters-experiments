from dataclasses import dataclass

import random
import numpy as np
from river import tree
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

from sacred import Experiment

ex = Experiment(name="elo_ensemble_hfd_train")
ex.observers.append(MongoObserver(url='mongodb://mongo_user:mongo_password@127.0.0.1:27017/sacred?authSource=admin',
                                  db_name='sacred'))
ex.observers.append(FileStorageObserver('./runs'))


@dataclass
class ModelWithElo:
    id: int
    model: tree.HoeffdingTreeClassifier
    rating: float
    train_correct: int = 0
    train_total: int = 0
    games_played: int = 0
    games_lost: int = 0
    games_won: int = 0
    games_tie: int = 0


def adjust_rating(k_factor, rating_width, model1: ModelWithElo, model2: ModelWithElo, result_model_1, result_model_2):
    expected_win_model_1 = np.round(1.0 / (1 + 10 ** ((model1.rating - model2.rating) / rating_width)), 3)
    expected_win_model_2 = 1 - expected_win_model_1
    model1.rating += k_factor * (result_model_1 - expected_win_model_1)
    model2.rating -= k_factor * (result_model_2 - expected_win_model_2)


def generate_pairs_random(_rnd, ensemble):
    ensemble_length = len(ensemble)
    shuffled_ids = list(range(0, ensemble_length))
    _rnd.shuffle(shuffled_ids)
    if ensemble_length % 2 == 1:
        shuffled_ids.pop()
    id_iterator = iter(shuffled_ids)
    return [(ensemble[id1], ensemble[id2]) for id1, id2 in zip(id_iterator, id_iterator)]


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


def play_pair(x, learner1: ModelWithElo, learner2: ModelWithElo):
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
    if learner1.rating > learner2.rating:
        learner2.model.learn_one(x, prediction1)
        learner1.games_won += 1
        learner2.games_lost += 1
        return
    if learner2.rating > learner1.rating:
        learner1.model.learn_one(x, prediction2)
        learner1.games_lost += 1
        learner2.games_won += 1
        return


def train_pair(x, y, learner1: ModelWithElo, learner2: ModelWithElo, k_factor, rating_width):
    prediction1 = learner1.model.predict_one(x)
    prediction2 = learner2.model.predict_one(x)
    learner_1_is_correct = (prediction1 is not None) and (prediction1 == y)
    learner_2_is_correct = (prediction2 is not None) and (prediction2 == y)
    if learner_1_is_correct and learner_2_is_correct:
        adjust_rating(k_factor, rating_width, learner1, learner2, 0.5, 0.5)
    if learner_1_is_correct and (not learner_2_is_correct):
        adjust_rating(k_factor, rating_width, learner1, learner2, 1, 0)
    if (not learner_1_is_correct) and learner_2_is_correct:
        adjust_rating(k_factor, rating_width, learner1, learner2, 0, 1)
    learner1.model.learn_one(x, y)
    learner2.model.learn_one(x, y)
    learner1.train_correct += (1 if learner_1_is_correct else 0)
    learner2.train_correct += (1 if learner_2_is_correct else 0)
    learner1.train_total += 1
    learner2.train_total += 1


def test_ensemble(_run, test_set, ensemble: list[ModelWithElo], step_nr, nr_samples_test):
    nr_learners = len(ensemble)
    individual_correct_counts = [0 for _ in range(nr_learners)]
    majority_correct_count = 0
    best_rated_correct_count = 0
    best_rated_learner = sorted(ensemble, key=lambda l: l.rating, reverse=True)[0]
    for x, y in test_set:
        predictions = list((learner.id, learner.model.predict_one(x)) for learner in ensemble)
        true_count = 0
        for (learner_id, prediction) in predictions:
            individual_correct_counts[learner_id] += (1 if prediction is not None and prediction == y else 0)
            true_count += (1 if prediction is True else 0)
        majority_prediction = True if true_count >= (nr_learners / 2) else False
        majority_correct_count += (1 if majority_prediction == y else 0)
        best_rated_correct_count += (1 if best_rated_learner.model.predict_one(x) == y else 0)
    for learner in ensemble:
        if nr_learners < 10:
            _run.log_scalar("learner{}.train_correct".format(learner.id), learner.train_correct, step_nr)
            _run.log_scalar("learner{}.train_total".format(learner.id), learner.train_total, step_nr)
            _run.log_scalar("learner{}.games_played".format(learner.id), learner.games_played, step_nr)
            _run.log_scalar("learner{}.games_won".format(learner.id), learner.games_won, step_nr)
            _run.log_scalar("learner{}.games_lost".format(learner.id), learner.games_lost, step_nr)
            _run.log_scalar("learner{}.games_tie".format(learner.id), learner.games_tie, step_nr)
            _run.log_scalar("learner{}.rating".format(learner.id), learner.rating, step_nr)
            _run.log_scalar("learner{}.test_accuracy".format(learner.id),
                            individual_correct_counts[learner.id] / nr_samples_test, step_nr)
            _run.log_scalar("learner{}.relative_rating".format(learner.id), learner.rating / best_rated_learner.rating,
                            step_nr)
    _run.log_scalar("ensemble.best_rated_accuracy", best_rated_correct_count / nr_samples_test, step_nr)
    _run.log_scalar("ensemble.majority_accuracy", majority_correct_count / nr_samples_test, step_nr)


def train_ensemble(_run, _rnd, ensemble: list[ModelWithElo], train_set, test_set, nr_samples_train, nr_samples_test,
                   test_step, k_factor, rating_width, pick_train_pairs_strategy, pick_play_pairs_strategy, nr_pairs,
                   repeat):
    step = nr_samples_train * repeat
    for x, y in train_set:
        all_pairs = generate_pairs_random(_rnd, ensemble)
        if y is None:
            play_pairs = pick_pairs(_rnd, all_pairs, pick_play_pairs_strategy, nr_pairs)
            for (learner1, learner2) in play_pairs:
                play_pair(x, learner1, learner2)
        else:
            train_pairs = pick_pairs(_rnd, all_pairs, pick_train_pairs_strategy, nr_pairs)
            for (learner1, learner2) in train_pairs:
                train_pair(x, y, learner1, learner2, k_factor, rating_width)
        step += 1
        if step % test_step == 0:
            test_ensemble(_run, test_set, ensemble, step // test_step, nr_samples_test)


@ex.automain
def run(_run, _seed, meta_experiment, train_data_set, test_data_set, nr_samples_train, nr_samples_test, test_step,
        mask_info, rating_width, k_factor, nr_learners, pick_train_pairs_strategy, pick_play_pairs_strategy,
        nr_pairs, nr_repeats):
    random.seed(_seed)
    ensemble = list(ModelWithElo(i, tree.HoeffdingTreeClassifier(), 800) for i in range(nr_learners))
    for repeat in range(0, nr_repeats):
        train_ensemble(_run, random, ensemble, train_data_set, test_data_set, nr_samples_train, nr_samples_test,
                       test_step, k_factor, rating_width, pick_train_pairs_strategy, pick_play_pairs_strategy, nr_pairs,
                       repeat)
