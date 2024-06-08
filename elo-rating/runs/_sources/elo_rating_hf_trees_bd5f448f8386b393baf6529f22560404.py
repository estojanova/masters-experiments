from dataclasses import dataclass

import random
import numpy as np
from river import tree
from river.datasets import synth
from sacred.observers import FileStorageObserver

from sacred import Experiment

ex = Experiment()
ex.observers.append(FileStorageObserver('../runs'))


@ex.config
def cfg():
    mean_rating = 1500
    rating_width = 400
    k_factor = 64
    nr_learners = 20
    nr_samples_train = 200
    mask_probability = 0.3
    nr_samples_test = 50
    test_step = 20


@dataclass
class ModelWithElo:
    id: int
    model: tree.HoeffdingTreeClassifier
    rating: float
    train_correct_prediction: int = 0
    train_total_observed: int = 0


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


def train_ensemble(_rnd, k_factor, rating_width, data_set, ensemble, test_step, test_set, nr_samples_test):
    steps = 0
    for learner in ensemble:
        learner.train_correct_prediction = 0
        learner.train_total_observed = 0
    for x, y in data_set:
        pairs = generate_pairs(_rnd, ensemble)
        for (learner1, learner2) in pairs:
            prediction1 = learner1.model.predict_one(x)
            prediction2 = learner2.model.predict_one(x)
            if y is None:
                if prediction1 is None or prediction2 is None:
                    continue
                if prediction1 == prediction2:
                    continue
                else:
                    if learner1.rating > learner2.rating:
                        learner2.model.learn_one(x, prediction1)
                    else:
                        learner1.model.learn_one(x, prediction2)
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
        steps += 1
        if steps % test_step == 0:
            test_ensemble(test_set, ensemble, steps // test_step, nr_samples_test)


def test_ensemble(test_set, ensemble, step_nr, nr_samples_test):
    accuracy = [0 for i in range(len(ensemble))]
    for learner in ensemble:
        nr_correct = 0
        for x, y in test_set:
            y_pred = learner.model.predict_one(x)
            if y == y_pred:
                nr_correct+=1
        accuracy[learner.id] = nr_correct/nr_samples_test
    print('Accuracy at ', step_nr, ': ', accuracy)
    print('Rating at ', step_nr, ': ',  list(l.rating for l in ensemble))


def print_initial_state(ensemble):
    print('Rating at 0: ', list(l.rating for l in ensemble))


@ex.automain
def run(_seed, mean_rating, rating_width, k_factor, nr_learners, nr_samples_train, mask_probability,
        nr_samples_test, test_step):
    random.seed(_seed)
    ensemble = list(
        ModelWithElo(i, tree.HoeffdingTreeClassifier(), random.randint(mean_rating - 200, mean_rating + 200)) for i in
        range(nr_learners))
    data_set = generate_dataset_with_mask(random, _seed, nr_samples_train, mask_probability)
    test_set = list(synth.SEA(variant=0, seed=_seed).take(nr_samples_test))
    print_initial_state(ensemble)
    train_ensemble(random, k_factor, rating_width, data_set, ensemble, test_step, test_set, nr_samples_test)
