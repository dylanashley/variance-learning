# -*- coding: ascii -*-

import numpy as np

from typing import List, Union

__all__ = ('TD',)


class TD:

    def __init__(self, initial_x: Union[List[float], np.ndarray]) -> None:
        self.e = np.array(initial_x, dtype=float)
        self.w = np.zeros(self.e.shape, dtype=float)
        self.last_prediction = 0

    def predict(self, x: Union[List[float], np.ndarray]) -> float:
        return float(np.dot(self.w, x))

    def reset(self, initial_x: Union[List[float], np.ndarray]) -> None:
        self.e.fill(0)
        self.e += initial_x
        self.last_prediction = self.predict(initial_x)

    def update(self,
               reward: float,
               gamma: float,
               x: Union[List[float], np.ndarray],
               alpha: float,
               lamda: float,
               replacing: bool=False,
               rho: float=1) -> float:
        delta = reward + gamma * self.predict(x) - self.last_prediction
        self.e *= rho
        self.w += alpha * delta * self.e
        self.e *= lamda * gamma
        self.e += x
        if replacing:
            self.e = np.clip(self.e, 0, 1)
        self.last_prediction = self.predict(x)
        return delta
