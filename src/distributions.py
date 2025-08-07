from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Distribution(ABC):
    @abstractmethod
    def sample(self) -> Any:
        raise NotImplementedError


class ConstantDistribution(Distribution):
    def __init__(self, value: Any) -> None:
        self.value = value

    def sample(self, n: int = 1) -> Any:
        return np.asarray([self.value] * n)


class DiscreteDistribution(Distribution):
    def __init__(
        self,
        values: np.ndarray,
        probabilities: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        self.values = values
        self.probabilities = probabilities
        self.rng = rng

    def sample(self, n: int = 1) -> np.ndarray:
        return self.rng.choice(self.values, p=self.probabilities, size=n)

class UniformDistribution(Distribution):
    def __init__(self, min: float, max: float, rng: np.random.Generator) -> None:
        self.min = min
        self.max = max
        self.rng = rng
    
    def sample(self, n: int = 1) -> np.ndarray:
        return self.rng.uniform(self.min, self.max, size=n)

class ExpDistribution(Distribution):
    def __init__(self, lamda: float, rng: np.random.Generator) -> None:
        """Lambda here is the mean of the exponential distribution"""
        self.lamda = lamda
        self.rng = rng
    
    def sample(self, n: int = 1) -> np.ndarray:
        return np.ceil((-np.log(self.rng.uniform(size=n)) * self.lamda).astype(int))
    
class GammaDistribution(Distribution):
    def __init__(self, shape: float, scale: float, rng: np.random.Generator) -> None:
        """Shape and scale parameters for the gamma distribution"""
        self.shape = shape
        self.scale = scale
        self.rng = rng
    
    def sample(self, n: int = 1) -> np.ndarray:
        return np.ceil(self.rng.gamma(self.shape, self.scale, size=n).astype(int))
    
