from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SceneObject:
    id: str
    shape: str
    color: str
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    size: str
    z_order: int
    rotation: float = 0.0

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def label(self) -> str:
        return f"{self.color} {self.shape}"


@dataclass(slots=True)
class Scene:
    scene_id: str
    width: int
    height: int
    background: str
    objects: list[SceneObject]
    split: str
    ood_type: str | None = None
    style: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QASample:
    sample_id: str
    scene_id: str
    image: str
    question: str
    answer: str
    task: str
    difficulty: int
    split: str
    ood_type: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

