from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from visiongym.schema import Scene, SceneObject


COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "red": (220, 55, 55),
    "blue": (52, 105, 210),
    "yellow": (238, 196, 52),
    "green": (58, 158, 93),
    "purple": (135, 78, 181),
    "orange": (235, 128, 48),
    "cyan": (44, 178, 190),
    "pink": (225, 105, 150),
}


def _regular_polygon(cx: float, cy: float, radius: float, sides: int, rotation: float = -90.0) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for i in range(sides):
        angle = math.radians(rotation + i * 360.0 / sides)
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def _star(cx: float, cy: float, outer: float, inner: float) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for i in range(10):
        radius = outer if i % 2 == 0 else inner
        angle = math.radians(-90.0 + i * 36.0)
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def _background(scene: Scene, rng: random.Random) -> Image.Image:
    style = scene.style.get("background_style", "solid")
    if style == "dark":
        return Image.new("RGB", (scene.width, scene.height), (30, 34, 43))
    if style == "noise":
        np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
        base = np_rng.normal(232, 18, (scene.height, scene.width, 3)).clip(185, 255).astype(np.uint8)
        return Image.fromarray(base, mode="RGB")
    return Image.new("RGB", (scene.width, scene.height), scene.background)


def _draw_object(draw: ImageDraw.ImageDraw, obj: SceneObject) -> None:
    fill = COLOR_RGB[obj.color]
    x1, y1, x2, y2 = obj.bbox
    cx, cy = obj.center
    radius = min(obj.width, obj.height) / 2.0
    if obj.shape == "circle":
        draw.ellipse(obj.bbox, fill=fill)
    elif obj.shape in {"rectangle", "square"}:
        draw.rectangle(obj.bbox, fill=fill)
    elif obj.shape == "triangle":
        draw.polygon(_regular_polygon(cx, cy, radius, 3), fill=fill)
    elif obj.shape == "hexagon":
        draw.polygon(_regular_polygon(cx, cy, radius, 6), fill=fill)
    elif obj.shape == "star":
        draw.polygon(_star(cx, cy, radius, radius * 0.45), fill=fill)
    elif obj.shape == "diamond":
        draw.polygon([(cx, y1), (x2, cy), (cx, y2), (x1, cy)], fill=fill)
    else:
        draw.rectangle(obj.bbox, fill=fill)


def render_scene(scene: Scene, output_path: str | Path, seed: int = 0) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    image = _background(scene, rng)
    draw = ImageDraw.Draw(image, "RGBA")
    for obj in sorted(scene.objects, key=lambda item: item.z_order):
        _draw_object(draw, obj)
    image.save(output, format="PNG", optimize=True)
    return output

