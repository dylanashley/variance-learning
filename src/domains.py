# -*- coding: ascii -*-

import numpy as np

from typing import Dict, List, Union

__all__ = ('RandomWalk',)


class RandomWalk:

    INITIAL_STATE = 0
    NUM_STATES = 30

    def __init__(self, random_generator=np.random) -> None:
        self.random_generator = random_generator
        self.current_state = 0

    def step(self) -> Dict[str, Union[int, float]]:
        action = 0 if self.random_generator.rand() > 0.7 else 1
        if self.current_state + (2 * action - 1) == self.NUM_STATES:
            reward = 0
            gamma = 0
            state = 0
        else:
            reward = -1
            gamma = 1
            state = max(0, self.current_state + (2 * action - 1))
        self.current_state = state
        return {'reward': reward, 'gamma': gamma, 'state': state}
