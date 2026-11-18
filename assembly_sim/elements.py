import math
from collections import deque
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Callable, Deque, Optional

INF = math.inf


class PartKind(Enum):
    RAW = "T1"
    PROCESSED = "T2"
    PRODUCT = "виріб"

@dataclass(frozen=True)
class Part:
    uid: int
    kind: PartKind

    def __str__(self) -> str:
        return f"{self.kind.value}#{self.uid}"


class DeviceState(IntEnum):
    FREE = 0
    BUSY = 1
    BLOCKED = 2


class Queue:
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Deque[Part] = deque()

    def __len__(self) -> int:
        return len(self._items)

    def push(self, part: Part) -> None:
        self._items.append(part)

    def pop(self) -> Part:
        return self._items.popleft()


class Device:

    def __init__(self, name: str) -> None:
        self.name = name
        self.state = DeviceState.FREE
        self.finish_time = INF
        self.content: Any = None

    @property
    def is_free(self) -> bool:
        return self.state is DeviceState.FREE

    def start(self, content: Any, now: float, duration: float) -> None:
        assert self.state is DeviceState.FREE, f"{self.name}: запуск обробки на неготовому пристрої"
        self.state = DeviceState.BUSY
        self.content = content
        self.finish_time = now + duration

    def block(self) -> None:
        self.state = DeviceState.BLOCKED
        self.finish_time = INF

    def unblock(self, now: float) -> None:
        self.state = DeviceState.BUSY
        self.finish_time = now

    def release(self) -> Any:
        content = self.content
        self.state = DeviceState.FREE
        self.finish_time = INF
        self.content = None
        return content


class RouteBlocking:
    def __init__(self, source: Device, predicate: Callable[[], bool]) -> None:
        self.source = source
        self._predicate = predicate
        self.b = False
        self.blocked_attempts = 0

    def update(self) -> bool:
        self.b = self._predicate()
        return self.b


class Station:
    def __init__(self, name: str, service_time: Callable[[], float]) -> None:
        self.name = name
        self.queue = Queue(f"Q{name}")
        self.device = Device(f"D{name}")
        self._service_time = service_time

    def accept(self, part: Part, now: float) -> None:
        if self.device.is_free:
            self.device.start(part, now, self._service_time())
        else:
            self.queue.push(part)

    def pull_next(self, now: float) -> Optional[Part]:
        if self.device.is_free and len(self.queue) > 0:
            part = self.queue.pop()
            self.device.start(part, now, self._service_time())
            return part
        return None