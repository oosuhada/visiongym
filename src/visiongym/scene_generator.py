from __future__ import annotations

import random
from dataclasses import dataclass

from visiongym.geometry import overlaps
from visiongym.schema import Scene, SceneObject


ID_SHAPES = ("circle", "rectangle", "triangle")
OOD_SHAPES = ("star", "hexagon", "diamond")
ID_COLORS = ("red", "blue", "yellow", "green")
OOD_COLORS = ("purple", "orange", "cyan", "pink")
SIZE_RANGES = {
    "small": (44, 62),
    "medium": (68, 92),
    "large": (100, 132),
}


@dataclass(slots=True)
class SceneGenerator:
    width: int = 512
    height: int = 512
    seed: int = 42

    def generate(self, scene_index: int, split: str, object_count: int, ood_type: str | None = None) -> Scene:
        rng = random.Random(self.seed + scene_index * 7919 + sum(ord(char) for char in split))
        shapes = list(OOD_SHAPES if ood_type == "shape" else ID_SHAPES)
        colors = list(OOD_COLORS if ood_type == "palette" else ID_COLORS)
        background_style = "solid"
        background = "white"
        if ood_type == "background":
            background_style = rng.choice(["dark", "noise"])

        objects: list[SceneObject] = []
        used_labels: set[tuple[str, str]] = set()
        allow_overlap = object_count >= 6 or ood_type in {"count", "background", "occlusion"}

        for obj_index in range(object_count):
            candidates = [(color, shape) for color in colors for shape in shapes if (color, shape) not in used_labels]
            if not candidates:
                candidates = [(color, shape) for color in colors for shape in shapes]
            color, shape = rng.choice(candidates)
            used_labels.add((color, shape))
            size_name = rng.choice(["small", "medium", "large"])
            if ood_type == "count":
                size_name = rng.choice(["small", "small", "medium"])
            low, high = SIZE_RANGES[size_name]
            width = rng.randint(low, high)
            height = width if shape != "rectangle" else rng.randint(max(38, int(width * 0.65)), int(width * 1.25))

            placed: SceneObject | None = None
            for _ in range(120):
                margin = 18
                cx = rng.randint(margin + width // 2, self.width - margin - width // 2)
                cy = rng.randint(margin + height // 2, self.height - margin - height // 2)
                bbox = (cx - width // 2, cy - height // 2, cx + width // 2, cy + height // 2)
                candidate = SceneObject(
                    id=f"obj_{obj_index + 1}",
                    shape=shape,
                    color=color,
                    bbox=bbox,
                    center=(float(cx), float(cy)),
                    size=size_name,
                    z_order=obj_index,
                )
                collisions = sum(1 for existing in objects if overlaps(candidate, existing))
                if collisions == 0 or (allow_overlap and collisions <= 1 and rng.random() < 0.3):
                    placed = candidate
                    break
            if placed is None:
                placed = candidate
            objects.append(placed)

        if ood_type == "occlusion" and len(objects) >= 2:
            anchor = objects[0]
            occluder = objects[-1]
            cx = int(anchor.center[0] + rng.randint(-18, 18))
            cy = int(anchor.center[1] + rng.randint(-18, 18))
            cx = max(occluder.width // 2 + 4, min(self.width - occluder.width // 2 - 4, cx))
            cy = max(occluder.height // 2 + 4, min(self.height - occluder.height // 2 - 4, cy))
            occluder.center = (float(cx), float(cy))
            occluder.bbox = (
                cx - occluder.width // 2,
                cy - occluder.height // 2,
                cx + occluder.width // 2,
                cy + occluder.height // 2,
            )

        scene_id = f"{split}_{scene_index:06d}"
        return Scene(
            scene_id=scene_id,
            width=self.width,
            height=self.height,
            background=background,
            objects=objects,
            split=split,
            ood_type=ood_type,
            style={"background_style": background_style},
        )
