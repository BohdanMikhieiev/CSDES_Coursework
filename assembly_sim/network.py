from dataclasses import dataclass
from typing import List, Optional

from .elements import Device, DeviceState, Part, PartKind, RouteBlocking, Station
from .rng import Randomizer


@dataclass(frozen=True)
class ModelConfig:
    mean_interarrival: float = 30.0
    batch_size: int = 3
    p_processed: float = 0.5
    t_preprocess: float = 7.0
    t_assembly: float = 6.0
    mean_regulation: float = 8.0
    p_defect: float = 0.04
    balanced_types: bool = False


@dataclass(frozen=True)
class Snapshot:
    q1: int
    q21: int
    q22: int
    q4: int
    d1_busy: bool
    d3_busy: bool
    d4_busy: bool
    parts_on_site: int


class AssemblyNetwork:
    def __init__(self, config: ModelConfig, rng: Randomizer,
                 trace: Optional[List[str]] = None) -> None:
        self.config = config
        self.rng = rng
        self.trace = trace
        self.now = 0.0

        self.s1 = Station("1", lambda: config.t_preprocess)
        self.s21 = Station("21", lambda: 0.0)
        self.s22 = Station("22", lambda: 0.0)
        self.s4 = Station("4", lambda: rng.exponential(config.mean_regulation))
        self.d3 = Device("D3")

        self.blockings = [RouteBlocking(self.s21.device, self._route_predicate),
                          RouteBlocking(self.s22.device, self._route_predicate)]
        self._blocking_of = {b.source: b for b in self.blockings}

        self.devices = [self.s1.device, self.s21.device, self.s22.device,
                        self.d3, self.s4.device]
        self._handlers = {
            self.s1.device: self._finish_preprocessing,
            self.s21.device: self._finish_waiting_stub,
            self.s22.device: self._finish_waiting_stub,
            self.d3: self._finish_assembly,
            self.s4.device: self._finish_regulation,
        }

        self.next_arrival = rng.exponential(config.mean_interarrival)   # t0
        self._next_uid = 0
        self._credit = 0.0
        self.n_input = 0
        self.n_processed = 0
        self.parts_arrived = 0
        self.assemblies = 0
        self.defects = 0

    @property
    def n_blocked(self) -> int:
        return sum(b.blocked_attempts for b in self.blockings)

    def next_event_time(self) -> float:
        return min([self.next_arrival] + [d.finish_time for d in self.devices])

    def transform_state(self) -> None:
        for device in self.rng.shuffled(self.devices):       # foreach randomly
            if device.finish_time == self.now:
                self._handlers[device](device)
        if self.next_arrival == self.now:
            self._on_arrival()
        self._update_blocking()

    def _route_predicate(self) -> bool:
        return (self.d3.state is DeviceState.BUSY
                or self.s21.device.is_free
                or self.s22.device.is_free)

    def _update_blocking(self) -> None:
        for blocking in self.blockings:
            if not blocking.update() and blocking.source.state is DeviceState.BLOCKED:
                blocking.source.unblock(self.now)
                self._log(f"маршрут до D3 відкрито, {blocking.source.name} розблоковано")

    def _on_arrival(self) -> None:
        cfg = self.config
        self.next_arrival = self.now + self.rng.exponential(cfg.mean_interarrival)
        self.n_input += 1
        parts = []
        for _ in range(cfg.batch_size):
            kind = self._choose_kind()
            part = self._new_part(kind)
            parts.append(part)
            self.parts_arrived += 1
            (self.s1 if kind is PartKind.PROCESSED else self.s21).accept(part, self.now)
        self._log("надходження партії: " + ", ".join(map(str, parts)))

    def _choose_kind(self) -> PartKind:
        cfg = self.config
        if not cfg.balanced_types:
            return PartKind.PROCESSED if self.rng.chance(cfg.p_processed) else PartKind.RAW
        self._credit += cfg.p_processed
        if self._credit >= 1.0:
            self._credit -= 1.0
            return PartKind.PROCESSED
        return PartKind.RAW

    def _finish_preprocessing(self, device: Device) -> None:
        part = device.release()
        self.s1.pull_next(self.now)
        self.s22.accept(part, self.now)
        self._log(f"D1 завершив обробку {part}")

    def _finish_waiting_stub(self, device: Device) -> None:
        blocking = self._blocking_of[device]
        if blocking.update():                       # b := B(S,t)
            blocking.blocked_attempts += 1          # N_blocked++
            device.block()                          # K := 2, t := inf, запит лишається в пристрої
            self._log(f"{device.name} заблоковано (очікує пару або вільний D3)")
        else:
            self._start_assembly()

    def _start_assembly(self) -> None:
        raw = self.s21.device.release()
        processed = self.s22.device.release()
        self.d3.start((raw, processed), self.now, self.config.t_assembly)
        self.s21.pull_next(self.now)
        self.s22.pull_next(self.now)
        self._log(f"початок зборки пари {raw} + {processed}")

    def _finish_assembly(self, device: Device) -> None:
        raw, processed = device.release()
        if self.rng.chance(self.config.p_defect):
            self.defects += 1
            self.s1.accept(processed, self.now)
            self.s21.accept(raw, self.now)
            self._log(f"брак при зборці: {processed} -> Q1, {raw} -> Q21")
        else:
            product = self._new_part(PartKind.PRODUCT)
            self.assemblies += 1
            self.s4.accept(product, self.now)
            self._log(f"зборка завершена, {product} -> Q4")

    def _finish_regulation(self, device: Device) -> None:
        product = device.release()
        self.s4.pull_next(self.now)
        self.n_processed += 1
        self._log(f"D4 завершив регулювання {product}, виріб залишає ділянку")

    def snapshot(self) -> Snapshot:
        s1, s21, s22, s4 = self.s1, self.s21, self.s22, self.s4
        occupied = lambda dev: 0 if dev.is_free else 1
        parts = (len(s1.queue) + len(s21.queue) + len(s22.queue) + len(s4.queue)
                 + occupied(s1.device) + occupied(s21.device) + occupied(s22.device)
                 + 2 * occupied(self.d3)
                 + occupied(s4.device))
        return Snapshot(
            q1=len(s1.queue), q21=len(s21.queue), q22=len(s22.queue), q4=len(s4.queue),
            d1_busy=s1.device.state is DeviceState.BUSY,
            d3_busy=self.d3.state is DeviceState.BUSY,
            d4_busy=s4.device.state is DeviceState.BUSY,
            parts_on_site=parts,
        )

    def _new_part(self, kind: PartKind) -> Part:
        self._next_uid += 1
        return Part(self._next_uid, kind)

    def _log(self, message: str) -> None:
        if self.trace is None:
            return
        s = self.snapshot()
        k = lambda dev: int(dev.state)
        self.trace.append(
            f"t={self.now:9.3f} | {message:<52} | "
            f"Q1={s.q1} Q21={s.q21} Q22={s.q22} Q4={s.q4} | "
            f"K1={k(self.s1.device)} K21={k(self.s21.device)} K22={k(self.s22.device)} "
            f"K3={k(self.d3)} K4={k(self.s4.device)}")