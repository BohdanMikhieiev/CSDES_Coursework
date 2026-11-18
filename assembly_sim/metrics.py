import math
from collections import defaultdict
from typing import Dict

from .network import Snapshot


class TimeWeightedStat:
    def __init__(self) -> None:
        self._time_at_value: Dict[int, float] = defaultdict(float)
        self.total_time = 0.0

    def add(self, value: int, duration: float) -> None:
        self._time_at_value[value] += duration
        self.total_time += duration

    def mean(self) -> float:
        return sum(v * t for v, t in self._time_at_value.items()) / self.total_time

    def std(self) -> float:
        m = self.mean()
        return math.sqrt(sum((v - m) ** 2 * t for v, t in self._time_at_value.items())
                         / self.total_time)

    def histogram(self) -> Dict[int, float]:
        return {v: t / self.total_time for v, t in sorted(self._time_at_value.items())}


class Statistics:

    NAMES = ("L_Q1", "L_Q21", "L_Q22", "L_Q4", "N", "rho_D1", "rho_D3", "rho_D4")

    def __init__(self) -> None:
        self.stats = {name: TimeWeightedStat() for name in self.NAMES}

    def observe(self, s: Snapshot, duration: float) -> None:
        values = {
            "L_Q1": s.q1, "L_Q21": s.q21, "L_Q22": s.q22, "L_Q4": s.q4,
            "N": s.parts_on_site,
            "rho_D1": int(s.d1_busy), "rho_D3": int(s.d3_busy), "rho_D4": int(s.d4_busy),
        }
        for name, value in values.items():
            self.stats[name].add(value, duration)

    def __getitem__(self, name: str) -> TimeWeightedStat:
        return self.stats[name]