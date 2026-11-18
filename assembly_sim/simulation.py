from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .metrics import Statistics
from .network import AssemblyNetwork, ModelConfig, Snapshot
from .rng import Randomizer


@dataclass
class SimulationResult:
    t_mod: float
    warmup: float
    statistics: Statistics
    n_input: int
    parts_arrived: int
    assemblies: int
    defects: int
    n_processed: int
    n_blocked: int
    parts_on_site_end: int
    end_state: Optional[Snapshot] = None
    trajectory: Dict[float, Dict[str, float]] = field(default_factory=dict)
    samples: Dict[float, Snapshot] = field(default_factory=dict)


def run_simulation(config: ModelConfig, t_mod: float, warmup: float = 0.0,
                   seed: Optional[int] = None,
                   trace: Optional[List[str]] = None,
                   checkpoints: Optional[Sequence[float]] = None,
                   sample_times: Optional[Sequence[float]] = None) -> SimulationResult:
    network = AssemblyNetwork(config, Randomizer(seed), trace)
    statistics = Statistics()
    pending = sorted(c for c in (checkpoints or ()) if 0 < c <= t_mod)
    trajectory: Dict[float, Dict[str, float]] = {}
    pending_samples = sorted(c for c in (sample_times or ()) if 0 < c <= t_mod)
    samples: Dict[float, Snapshot] = {}

    while network.now < t_mod:
        t_next = network.next_event_time()
        t_end = min(t_next, t_mod)
        snapshot = network.snapshot()
        seg_start = max(network.now, warmup)
        while pending_samples and pending_samples[0] <= t_end:
            samples[pending_samples.pop(0)] = snapshot
        while pending and pending[0] <= t_end:
            c = pending.pop(0)
            if c > seg_start:
                statistics.observe(snapshot, c - seg_start)
                seg_start = c
            if statistics["N"].total_time > 0:
                trajectory[c] = {name: statistics[name].mean() for name in statistics.NAMES}
        if t_end > seg_start:
            statistics.observe(snapshot, t_end - seg_start)
        if t_next >= t_mod:
            network.now = t_mod
            break
        network.now = t_next
        network.transform_state()

    end_state = network.snapshot()
    return SimulationResult(
        t_mod=t_mod, warmup=warmup, statistics=statistics,
        n_input=network.n_input, parts_arrived=network.parts_arrived,
        assemblies=network.assemblies, defects=network.defects,
        n_processed=network.n_processed, n_blocked=network.n_blocked,
        parts_on_site_end=end_state.parts_on_site, end_state=end_state,
        trajectory=trajectory, samples=samples)