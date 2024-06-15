from dataclasses import dataclass

import random
import numpy as np
from river import tree
from river.datasets import synth
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

from sacred import Experiment

ex = Experiment()
ex.observers.append(MongoObserver(url='mongodb://mongo_user:mongo_password@127.0.0.1:27017/sacred?authSource=admin',
                                  db_name='sacred'))
ex.observers.append(FileStorageObserver('../runs'))


@ex.config
def cfg():
    mean_rating = 1500
    rating_width = 400
    k_factor = 64
    nr_learners = 4
    nr_samples_train = 200
    mask_probability = 0.1
    nr_samples_test = 50
    test_step = 20


@dataclass
class ModelWithElo:
    id: int
    model: tree.HoeffdingTreeClassifier
    rating: float
    train_correct_prediction: int = 0
    train_total_observed: int = 0
    games_played: int = 0
    games_lost: int = 0
    games_won: int = 0


def adjust_rating(k_factor, rating_width, model1: ModelWithElo, model2: ModelWithElo, result_model_1, result_model_2):
    expected_win_model_1 = np.round(1.0 / (1 + 10 ** ((model1.rating - model2.rating) / rating_width)), 3)
    expected_win_model_2 = 1 - expected_win_model_1
    model1.rating += k_factor * (result_model_1 - expected_win_model_1)
    model2.rating -= k_factor * (result_model_2 - expected_win_model_2)


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


def generate_pairs(_rnd, ensemble):
    # works only for even nr of elements !
    all_ids = list(range(0, len(ensemble)))
    _rnd.shuffle(all_ids)
    id_iterator = iter(all_ids)
    pairs = []
    for id1, id2 in zip(id_iterator, id_iterator):
        # print(id1, id2)
        pairs.append((ensemble[id1], ensemble[id2]))
    return _rnd.sample(pairs, _rnd.randrange(len(pairs)))


def train_ensemble(_run, _rnd, k_factor, rating_width, data_set, ensemble, test_step, test_set, nr_samples_test):
    step = 0

    for x, y in data_set:
        pairs = generate_pairs(_rnd, ensemble)
        for (learner1, learner2) in pairs:
            prediction1 = learner1.model.predict_one(x)
            prediction2 = learner2.model.predict_one(x)
            if y is None:
                learner1.games_played += 1
                learner2.games_played += 1
                if prediction1 is None or prediction2 is None:
                    continue
                if prediction1 == prediction2:
                    continue
                else:
                    if learner1.rating > learner2.rating:
                        learner2.model.learn_one(x, prediction1)
                        learner1.games_won += 1
                    else:
                        learner1.model.learn_one(x, prediction2)
                        learner2.games_won += 1
            else:
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
                learner1.train_correct_prediction += (1 if learner_1_is_correct else 0)
                learner2.train_correct_prediction += (1 if learner_2_is_correct else 0)
                learner1.train_total_observed += 1
                learner2.train_total_observed += 1
                log_train_metrics(_run, learner1, step)

        step += 1
        if step % test_step == 0:
            test_ensemble(_run, test_set, ensemble, step, nr_samples_test)


def log_train_metrics(_run, learner: ModelWithElo, step_nr: int):
    train_accuracy_template = "learner{}.train_correct"
    train_observed_template = "learner{}.train_observed"
    games_played_template = "learner{}.games_played"
    games_won_template = "learner{}.games_won"
    games_lost_template = "learner{}.games_lost"
    rating_template = "learner{}.rating"
    _run.log_scalar(train_accuracy_template.format(learner.id), learner.train_correct_prediction, step_nr)
    _run.log_scalar(train_observed_template.format(learner.id), learner.train_total_observed, step_nr)
    _run.log_scalar(games_played_template.format(learner.id), learner.games_played, step_nr)
    _run.log_scalar(games_won_template.format(learner.id), learner.games_won, step_nr)
    _run.log_scalar(games_lost_template.format(learner.id), learner.games_lost, step_nr)
    _run.log_scalar(rating_template.format(learner.id), learner.rating, step_nr)


def test_ensemble(_run, test_set, ensemble, step_nr, nr_samples_test):
    accuracy_template = "learner{}.test_accuracy"
    rating_template = "learner{}.rating"
    accuracy = [0 for i in range(len(ensemble))]
    for learner in ensemble:
        nr_correct = 0
        for x, y in test_set:
            y_pred = learner.model.predict_one(x)
            if y == y_pred:
                nr_correct += 1
        learner_accuracy = nr_correct / nr_samples_test
        _run.log_scalar(accuracy_template.format(learner.id), learner_accuracy, step_nr)
        _run.log_scalar(rating_template.format(learner.id), learner.rating, step_nr)
        accuracy[learner.id] = nr_correct / nr_samples_test
    print('Accuracy at ', step_nr, ': ', accuracy)
    print('Rating at ', step_nr, ': ', list(l.rating for l in ensemble))


def log_initial_state(_run, ensemble):
    for learner in ensemble:
        log_train_metrics(_run, learner, 0)


@ex.automain
def run(_run, _seed, mean_rating, rating_width, k_factor, nr_learners, nr_samples_train, mask_probability,
        nr_samples_test, test_step):
    random.seed(_seed)
    ensemble = list(
        ModelWithElo(i, tree.HoeffdingTreeClassifier(), random.randint(mean_rating - 200, mean_rating + 200)) for i in
        range(nr_learners))
    data_set = generate_dataset_with_mask(random, _seed, nr_samples_train, mask_probability)
    test_set = list(synth.SEA(variant=0, seed=_seed).take(nr_samples_test))
    log_initial_state(_run, ensemble)
    train_ensemble(_run, random, k_factor, rating_width, data_set, ensemble, test_step, test_set, nr_samples_test)
