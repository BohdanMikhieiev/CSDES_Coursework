import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from scipy import stats as scipy_stats

from assembly_sim import ModelConfig, SimulationResult, run_simulation

OUTPUTS = ("L_Q1", "L_Q21", "L_Q22", "L_Q4", "N", "rho_D1", "rho_D3", "rho_D4")

def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def std(values: Sequence[float]) -> float:
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def run_pilots(config: ModelConfig, seeds: Sequence[int], t_max: float,
               checkpoints: Sequence[float]) -> List[Dict[float, Dict[str, float]]]:
    return [run_simulation(config, t_max, seed=s, checkpoints=checkpoints).trajectory
            for s in seeds]


def spread(pilots: List[Dict[float, Dict[str, float]]], name: str, t: float) -> float:
    ys = [p[t][name] for p in pilots]
    return max(ys) - min(ys)


def transient_time(pilots: List[Dict[float, Dict[str, float]]], name: str, eps: float,
                   checkpoints: Sequence[float]) -> Optional[float]:
    found = None
    for t in sorted(checkpoints, reverse=True):
        if spread(pilots, name, t) < eps:
            found = t
        else:
            break
    return found


def required_replications(sigma: float, eps: float, beta: float = 0.95) -> int:
    return math.ceil(sigma ** 2 / (eps ** 2 * (1.0 - beta)))


def achieved_accuracy(sigma: float, n: int, beta: float = 0.95) -> float:
    return sigma / math.sqrt(n * (1.0 - beta))


def replicate(config: ModelConfig, n: int, warmup: float, t_obs: float,
              seed0: int) -> List[SimulationResult]:
    return [run_simulation(config, warmup + t_obs, warmup=warmup, seed=seed0 + k)
            for k in range(n)]


def run_means(results: Sequence[SimulationResult], name: str) -> List[float]:
    return [r.statistics[name].mean() for r in results]


@dataclass
class AnovaResult:
    d_factor: float
    d_residual: float
    f_value: float
    f_critical: float
    df_factor: int
    df_residual: int
    significant: bool


def anova_one_way(groups: Sequence[Sequence[float]], alpha: float = 0.05) -> AnovaResult:
    q, p = len(groups), len(groups[0])
    group_means = [mean(g) for g in groups]
    grand_mean = mean(group_means)
    df_factor, df_residual = q - 1, q * (p - 1)
    d_factor = p * sum((m - grand_mean) ** 2 for m in group_means) / df_factor
    d_residual = sum((y - m) ** 2 for g, m in zip(groups, group_means) for y in g) / df_residual
    f_value = d_factor / d_residual
    f_critical = scipy_stats.f.ppf(1.0 - alpha, df_factor, df_residual)
    return AnovaResult(d_factor, d_residual, f_value, f_critical,
                       df_factor, df_residual, f_value > f_critical)


def sample_states(config: ModelConfig, attribute: str, n_samples: int, warmup: float,
                  spacing: float, seed: int) -> List[int]:
    times = [warmup + spacing * (k + 1) for k in range(n_samples)]
    result = run_simulation(config, times[-1], seed=seed, sample_times=times)
    return [getattr(result.samples[t], attribute) for t in times]


def zero_modified_geometric(p0: float, tail_mean: float) -> Callable[[int], float]:
    q = (tail_mean - 1.0) / tail_mean
    return lambda k: p0 if k == 0 else (1.0 - p0) * (1.0 - q) * q ** (k - 1)


def fit_zero_modified_geometric(sample: Sequence[int]) -> Callable[[int], float]:
    p0 = sum(1 for v in sample if v == 0) / len(sample)
    positive = [v for v in sample if v > 0]
    return zero_modified_geometric(p0, mean(positive))


@dataclass
class ChiSquareResult:
    rows: List[tuple]
    chi2: float
    df: int
    critical: float
    accepted: bool


def chi_square_test(sample: Sequence[int], pmf: Callable[[int], float], n_params: int,
                    alpha: float = 0.05, min_expected: float = 5.0) -> ChiSquareResult:
    n = len(sample)
    top = max(sample)
    observed = [sum(1 for v in sample if v == k) for k in range(top + 1)]
    expected = [n * pmf(k) for k in range(top + 1)]
    expected[-1] += n * (1.0 - sum(pmf(k) for k in range(top + 1)))
    groups = []
    o_acc, e_acc, first = 0, 0.0, 0
    for k in range(top + 1):
        o_acc += observed[k]
        e_acc += expected[k]
        if e_acc >= min_expected:
            groups.append([first, k, o_acc, e_acc])
            o_acc, e_acc, first = 0, 0.0, k + 1
    if o_acc or e_acc > 0:
        if groups:
            groups[-1][1] = top
            groups[-1][2] += o_acc
            groups[-1][3] += e_acc
        else:
            groups.append([0, top, o_acc, e_acc])
    rows = []
    for i, (lo, hi, o, e) in enumerate(groups):
        last = i == len(groups) - 1
        label = f"{lo}+" if last else (str(lo) if lo == hi else f"{lo}-{hi}")
        rows.append((label, o, e))
    chi2 = sum((o - e) ** 2 / e for _, o, e in rows)
    df = len(rows) - 1 - n_params
    critical = scipy_stats.chi2.ppf(1.0 - alpha, df)
    return ChiSquareResult(rows, chi2, df, critical, chi2 < critical)