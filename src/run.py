#!/usr/bin/env python -O
# -*- coding: ascii -*-

import argparse
import math
import numpy as np
import json

from domains import RandomWalk
from learners import TD
from typing import Any, Dict


def main(args: Dict[str, Any]) -> None:

    # load seeds
    with open('src/seeds.json', 'r') as infile:
        seeds = json.load(infile)['seeds'][:args['seed_count']]

    # build arrays to track learning
    first_predictions = np.ones((
        args['seed_count'],
        math.ceil(args['step_count'] / args['step_interval']),
        RandomWalk.NUM_STATES)) * np.nan
    second_predictions = np.copy(first_predictions)
    direct_predictions = np.copy(first_predictions)

    # build phi vectors
    val_phi = np.ones((RandomWalk.NUM_STATES, 2))
    val_phi[:, 1] *= (np.arange(RandomWalk.NUM_STATES) + 1) / RandomWalk.NUM_STATES
    var_phi = np.hstack((val_phi, val_phi[:, 1, np.newaxis]))
    var_phi[:, 2] *= (np.arange(RandomWalk.NUM_STATES) + 1) / RandomWalk.NUM_STATES

    try:
        for seed_index, seed in enumerate(seeds):

            # build domain
            domain = RandomWalk(random_generator=np.random.RandomState(seed))

            # build learners
            first_learner = TD(val_phi[domain.INITIAL_STATE, :])
            second_learner = TD(var_phi[domain.INITIAL_STATE, :])
            direct_learner = TD(var_phi[domain.INITIAL_STATE, :])

            for step in range(args['step_count']):

                transition = domain.step()
                reward = transition['reward']
                gamma = transition['gamma']
                state = transition['state']

                # update learners
                first_delta = first_learner.update(
                    reward,
                    gamma,
                    val_phi[state, :],
                    args['val_alpha'],
                    args['val_kappa'])
                first_prediction = first_learner.predict(val_phi[state, :])
                second_learner.update(
                    reward ** 2 + 2 * gamma * reward * first_prediction,
                    gamma ** 2,
                    var_phi[state, :],
                    args['var_alpha'],
                    args['var_kappa'])
                direct_learner.update(
                    first_delta ** 2,
                    gamma ** 2,
                    var_phi[state, :],
                    args['var_alpha'],
                    args['var_kappa'])

                # save predictions at each interval steps
                if not (step % args['step_interval']):
                    for state in range(domain.NUM_STATES):
                        step_index = int(step // args['step_interval'])
                        first_predictions[seed_index, step_index, state] = \
                            first_learner.predict(val_phi[state, :])
                        second_predictions[seed_index, step_index, state] = \
                            second_learner.predict(var_phi[state, :])
                        direct_predictions[seed_index, step_index, state] = \
                            direct_learner.predict(var_phi[state, :])

    finally:
        np.save('{}/first_predictions.npy'.format(args['directory']),
                np.squeeze(first_predictions))
        np.save('{}/second_predictions.npy'.format(args['directory']),
                np.squeeze(second_predictions))
        np.save('{}/direct_predictions.npy'.format(args['directory']),
                np.squeeze(direct_predictions))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('val_alpha', type=float)
    parser.add_argument('val_kappa', type=float)
    parser.add_argument('var_alpha', type=float)
    parser.add_argument('var_kappa', type=float)
    parser.add_argument('--directory', type=str, default='.')
    parser.add_argument('--seed-count', type=int, default=1)
    parser.add_argument('--step-count', type=int, default=1)
    parser.add_argument('--step-interval', type=int, default=1)
    return vars(parser.parse_args())


if __name__ == '__main__':
    np.seterr(divide='raise', over='warn', under='ignore', invalid='raise')
    main(parse_args())
