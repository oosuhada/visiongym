from __future__ import annotations

import math

from visiongym.schema import SceneObject


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def is_left(a: SceneObject, b: SceneObject, margin: float = 6.0) -> bool:
    return a.center[0] < b.center[0] - margin


def is_right(a: SceneObject, b: SceneObject, margin: float = 6.0) -> bool:
    return a.center[0] > b.center[0] + margin


def is_above(a: SceneObject, b: SceneObject, margin: float = 6.0) -> bool:
    return a.center[1] < b.center[1] - margin


def is_below(a: SceneObject, b: SceneObject, margin: float = 6.0) -> bool:
    return a.center[1] > b.center[1] + margin


def overlaps(a: SceneObject, b: SceneObject) -> bool:
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def contains(outer: SceneObject, inner: SceneObject) -> bool:
    ox1, oy1, ox2, oy2 = outer.bbox
    ix1, iy1, ix2, iy2 = inner.bbox
    return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2


def center_distance(obj: SceneObject, width: int, height: int) -> float:
    return euclidean(obj.center, (width / 2.0, height / 2.0))


def between(candidate: SceneObject, a: SceneObject, b: SceneObject) -> bool:
    ax, ay = a.center
    bx, by = b.center
    px, py = candidate.center
    ab = euclidean(a.center, b.center)
    if ab < 1e-6:
        return False
    ap = euclidean(a.center, candidate.center)
    pb = euclidean(candidate.center, b.center)
    line_error = abs((ap + pb) - ab) / ab
    min_x, max_x = sorted((ax, bx))
    min_y, max_y = sorted((ay, by))
    in_segment_box = min_x <= px <= max_x and min_y <= py <= max_y
    return line_error < 0.18 and in_segment_box

