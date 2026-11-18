import math
import random
from typing import List, Optional, Sequence, TypeVar

T = TypeVar("T")


class Randomizer:

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def uniform01(self) -> float:
        return 1.0 - self._rng.random()

    def exponential(self, mean: float) -> float:
        return -mean * math.log(self.uniform01())

    def chance(self, probability: float) -> bool:
        return self.uniform01() < probability

    def shuffled(self, items: Sequence[T]) -> List[T]:
        result = list(items)
        for i in range(len(result) - 1, 0, -1):
            j = int(self._rng.random() * (i + 1))
            result[i], result[j] = result[j], result[i]
        return result