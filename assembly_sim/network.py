"""Мережа масового обслуговування складальної ділянки (рисунок 2.1).

Структура мережі:
    надходження exp(30) -> дуга 1/3 -> розгалуження 0,5 / 0,5
    T1: Q21 -> D21 (t=0) --\\
                             B(S,t) -> D3 (t=6, без черги) -> 0,96 -> Q4 -> D4 exp(8) -> вихід
    T2: Q1 -> D1 (t=7) -> Q22 -> D22 (t=0) --/             -> 0,04 -> дуга 1/2:
                                                              T2 -> Q1, T1 -> Q21

Зміна стану мережі в момент події виконана за алгоритмом 3.2 та таблицею 3.5 методички.
"""
from dataclasses import dataclass
from typing import List, Optional

from .elements import Device, DeviceState, Part, PartKind, RouteBlocking, Station
from .rng import Randomizer


@dataclass(frozen=True)
class ModelConfig:
    """Вхідні змінні моделі (таблиця 1.1)."""
    mean_interarrival: float = 30.0   # середній інтервал між партіями, exp(d)
    batch_size: int = 3               # кількість деталей у партії
    p_processed: float = 0.5          # частка деталей, що проходять попередню обробку (T2)
    t_preprocess: float = 7.0         # попередня обробка на D1
    t_assembly: float = 6.0           # зборка на D3
    mean_regulation: float = 8.0      # регулювання на D4, exp(d)
    p_defect: float = 0.04            # ймовірність браку при зборці
    balanced_types: bool = False      # модифікація: чергування типів деталей замість випадкового вибору


@dataclass(frozen=True)
class Snapshot:
    """Стан мережі, що потрібен для збору статистики."""
    q1: int
    q21: int
    q22: int
    q4: int
    d1_busy: bool
    d3_busy: bool
    d4_busy: bool
    parts_on_site: int   # кількість деталей на ділянці, формула (2.4)


class AssemblyNetwork:
    """Стан мережі та правила його зміни."""

    def __init__(self, config: ModelConfig, rng: Randomizer,
                 trace: Optional[List[str]] = None) -> None:
        self.config = config
        self.rng = rng
        self.trace = trace          # якщо задано, сюди пишеться протокол подій
        self.now = 0.0

        self.s1 = Station("1", lambda: config.t_preprocess)                      # Q1,  D1
        self.s21 = Station("21", lambda: 0.0)                                    # Q21, D21 (заглушка)
        self.s22 = Station("22", lambda: 0.0)                                    # Q22, D22 (заглушка)
        self.s4 = Station("4", lambda: rng.exponential(config.mean_regulation))  # Q4,  D4
        self.d3 = Device("D3")                                                   # зборка, без черги

        # Блокування маршрутів D21 -> D3 і D22 -> D3 (два трикутники на рисунку 2.1)
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
        self._credit = 0.0       # лічильник для збалансованого вибору типу (модифікація)
        # лічильники (позначення методички)
        self.n_input = 0         # N_input: надійшло партій
        self.n_processed = 0     # N_processed: виробів залишило ділянку
        self.parts_arrived = 0   # надійшло деталей
        self.assemblies = 0      # успішні зборки (виробів утворено)
        self.defects = 0         # браковані пари

    @property
    def n_blocked(self) -> int:
        """N_blocked: невдалі спроби пройти заблокованим маршрутом."""
        return sum(b.blocked_attempts for b in self.blockings)

    # ---------------------------------------------------------------- час
    def next_event_time(self) -> float:
        """Момент найближчої події: надходження або завершення обробки в пристрої."""
        return min([self.next_arrival] + [d.finish_time for d in self.devices])

    def transform_state(self) -> None:
        """Зміна стану E(S,t) у поточний момент (алгоритм 3.2).

        Спочатку завершення обробки в пристроях (у випадковому порядку),
        потім надходження, наприкінці оновлення умов блокування.
        Події, що виникли в цей самий момент, виконуються наступним кроком
        алгоритму 3.1 (інтервал нульової довжини).
        """
        for device in self.rng.shuffled(self.devices):       # foreach randomly
            if device.finish_time == self.now:
                self._handlers[device](device)
        if self.next_arrival == self.now:
            self._on_arrival()
        self._update_blocking()

    # ---------------------------------------------------------- блокування
    def _route_predicate(self) -> bool:
        """Предикат B(S,t) = (K3 = 1) v (K21 = 0) v (K22 = 0), формула (2.1)."""
        return (self.d3.state is DeviceState.BUSY
                or self.s21.device.is_free
                or self.s22.device.is_free)

    def _update_blocking(self) -> None:
        """foreach blocking: b := B(S,t); якщо маршрут відкрито, знімає блокування
        з пристрою і планує для нього завершення обробки на поточний момент."""
        for blocking in self.blockings:
            if not blocking.update() and blocking.source.state is DeviceState.BLOCKED:
                blocking.source.unblock(self.now)
                self._log(f"маршрут до D3 відкрито, {blocking.source.name} розблоковано")

    # ------------------------------------------------------------- події
    def _on_arrival(self) -> None:
        """Надходження партії: дуга 1/3 розгруповує її на деталі, кожна з яких обирає тип."""
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
        """Тип нової деталі: T2 (з обробкою) з ймовірністю p_processed або, у модифікованій
        системі, за правилом чергування, що точно дотримується частки p_processed."""
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
        """Завершення на D21/D22: спроба пройти маршрутом до D3."""
        blocking = self._blocking_of[device]
        if blocking.update():                       # b := B(S,t)
            blocking.blocked_attempts += 1          # N_blocked++
            device.block()                          # K := 2, t := inf, запит лишається в пристрої
            self._log(f"{device.name} заблоковано (очікує пару або вільний D3)")
        else:
            self._start_assembly()

    def _start_assembly(self) -> None:
        """Маршрут відкрито: обидві деталі пари одночасно переходять у D3."""
        raw = self.s21.device.release()
        processed = self.s22.device.release()
        self.d3.start((raw, processed), self.now, self.config.t_assembly)
        self.s21.pull_next(self.now)
        self.s22.pull_next(self.now)
        self._log(f"початок зборки пари {raw} + {processed}")

    def _finish_assembly(self, device: Device) -> None:
        raw, processed = device.release()
        if self.rng.chance(self.config.p_defect):
            # дуга 1/2: бракована пара розгруповується, кожна деталь повертається за своїм типом
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

    # -------------------------------------------------------------- стан
    def snapshot(self) -> Snapshot:
        s1, s21, s22, s4 = self.s1, self.s21, self.s22, self.s4
        occupied = lambda dev: 0 if dev.is_free else 1
        parts = (len(s1.queue) + len(s21.queue) + len(s22.queue) + len(s4.queue)
                 + occupied(s1.device) + occupied(s21.device) + occupied(s22.device)
                 + 2 * occupied(self.d3)          # під час зборки в D3 дві деталі
                 + occupied(s4.device))
        return Snapshot(
            q1=len(s1.queue), q21=len(s21.queue), q22=len(s22.queue), q4=len(s4.queue),
            d1_busy=s1.device.state is DeviceState.BUSY,
            d3_busy=self.d3.state is DeviceState.BUSY,
            d4_busy=s4.device.state is DeviceState.BUSY,
            parts_on_site=parts,
        )

    # ---------------------------------------------------------- допоміжне
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