import numpy as np


class SQRTImpact:
    def __init__(self, coef: float, max_size: int):
        self.coef = coef
        self.max_size = max_size
        self.ts = np.full(max_size, -np.inf)
        self.sides = np.zeros(max_size, dtype=int)
        self.curr_index = 0
        self.value = 0

    def compute_impact(self, t: float) -> None:
        self.value = np.sum(self.coef * self.sides / np.sqrt(t - self.ts))

    def update(self, t: float, side: int) -> None:
        self.ts[self.curr_index] = t
        self.sides[self.curr_index] = side
        self.curr_index = (self.curr_index + 1) % self.max_size