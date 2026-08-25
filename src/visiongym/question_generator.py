from __future__ import annotations

import random
from collections.abc import Callable

from visiongym.geometry import between, center_distance, contains, euclidean, is_above, is_below, is_left, is_right, overlaps
from visiongym.schema import QASample, Scene, SceneObject


QuestionBuilder = Callable[[Scene, random.Random], tuple[str, str, int, dict] | None]


def _choice_pair(objects: list[SceneObject], rng: random.Random) -> tuple[SceneObject, SceneObject]:
    first, second = rng.sample(objects, 2)
    return first, second


def _counting(scene: Scene, rng: random.Random) -> tuple[str, str, int, dict] | None:
    mode = rng.choice(["color", "shape", "above", "left"])
    if mode == "color":
        color = rng.choice(sorted({obj.color for obj in scene.objects}))
        answer = sum(obj.color == color for obj in scene.objects)
        return f"How many {color} objects are in the image?", str(answer), 1, {"filter": {"color": color}}
    if mode == "shape":
        shape = rng.choice(sorted({obj.shape for obj in scene.objects}))
        answer = sum(obj.shape == shape for obj in scene.objects)
        return f"How many {shape} objects are in the image?", str(answer), 1, {"filter": {"shape": shape}}
    anchor = rng.choice(scene.objects)
    if mode == "above":
        answer = sum(obj.id != anchor.id and is_above(obj, anchor) for obj in scene.objects)
        return f"How many objects are above the {anchor.label}?", str(answer), 2, {"anchor": anchor.id, "relation": "above"}
    answer = sum(obj.id != anchor.id and is_left(obj, anchor) for obj in scene.objects)
    return f"How many objects are left of the {anchor.label}?", str(answer), 2, {"anchor": anchor.id, "relation": "left"}


def _left_right(scene: Scene, rng: random.Random) -> tuple[str, str, int, dict] | None:
    anchor = rng.choice(scene.objects)
    relation = rng.choice(["left", "right"])
    candidates = [obj for obj in scene.objects if obj.id != anchor.id and (is_left(obj, anchor) if relation == "left" else is_right(obj, anchor))]
    if not candidates:
        return None
    target = min(candidates, key=lambda obj: abs(obj.center[0] - anchor.center[0]))
    question = f"Which object is immediately {relation} of the {anchor.label}?"
    return question, target.label, 2, {"anchor": anchor.id, "target": target.id, "relation": relation}


def _above_below(scene: Scene, rng: random.Random) -> tuple[str, str, int, dict] | None:
    anchor = rng.choice(scene.objects)
    relation = rng.choice(["above", "below"])
    candidates = [obj for obj in scene.objects if obj.id != anchor.id and (is_above(obj, anchor) if relation == "above" else is_below(obj, anchor))]
    if not candidates:
        return None
    target = min(candidates, key=lambda obj: abs(obj.center[1] - anchor.center[1]))
    question = f"Which object is immediately {relation} the {anchor.label}?"
    return question, target.label, 2, {"anchor": anchor.id, "target": target.id, "relation": relation}


def _nearest_farthest(scene: Scene, rng: random.Random) -> tuple[str, str, int, dict] | None:
    anchor = rng.choice(scene.objects)
    others = [obj for obj in scene.objects if obj.id != anchor.id]
    if len(others) < 2:
        return None
    mode = rng.choice(["nearest", "farthest"])
    key = lambda obj: euclidean(obj.center, anchor.center)
    target = min(others, key=key) if mode == "nearest" else max(others, key=key)
    return f"Which object is {mode} from the {anchor.label}?", target.label, 2, {"anchor": anchor.id, "target": target.id, "relation": mode}


def _relative_size(scene: Scene, rng: random.Random) -> tuple[str, str, int, dict] | None:
    first, second = _choice_pair(scene.objects, rng)
    if abs(first.area - second.area) < 250:
        return None
    mode = rng.choice(["larger", "smaller"])
    if mode == "larger":
        target = first if first.area > second.area else second
    else:
        target = first if first.area < second.area else second
    question = f"Which object is {mode}, the {first.label} or the {second.label}?"
    return question, target.label, 2, {"objects": [first.id, second.id], "target": target.id, "relation": mode}


def _center_proximity(scene: Scene, rng: random.Random) -> tuple[str, str, int, dict] | None:
    first, second = _choice_pair(scene.objects, rng)
    first_distance = center_distance(first, scene.width, scene.height)
    second_distance = center_distance(second, scene.width, scene.height)
    if abs(first_distance - second_distance) < 8:
        return None
    target = first if first_distance < second_distance else second
    question = f"Which object is closer to the center of the image, the {first.label} or the {second.label}?"
    return question, target.label, 2, {"objects": [first.id, second.id], "target": target.id, "relation": "center_proximity"}


def _relative_ordering(scene: Scene, rng: random.Random) -> tuple[str, str, int, dict] | None:
    mode = rng.choice(["leftmost", "rightmost", "topmost", "bottommost"])
    if mode == "leftmost":
        target = min(scene.objects, key=lambda obj: obj.center[0])
    elif mode == "rightmost":
        target = max(scene.objects, key=lambda obj: obj.center[0])
    elif mode == "topmost":
        target = min(scene.objects, key=lambda obj: obj.center[1])
    else:
        target = max(scene.objects, key=lambda obj: obj.center[1])
    return f"Which object is the {mode} in the image?", target.label, 2, {"target": target.id, "relation": mode}


def _between(scene: Scene, rng: random.Random) -> tuple[str, str, int, dict] | None:
    if len(scene.objects) < 3:
        return None
    for candidate in scene.objects:
        others = [obj for obj in scene.objects if obj.id != candidate.id]
        for first, second in [tuple(rng.sample(others, 2)) for _ in range(min(5, len(others) * 2))]:
            if between(candidate, first, second):
                question = f"Which object is between the {first.label} and the {second.label}?"
                return question, candidate.label, 3, {"objects": [first.id, second.id], "target": candidate.id, "relation": "between"}
    return None


def _overlap(scene: Scene, rng: random.Random) -> tuple[str, str, int, dict] | None:
    first, second = _choice_pair(scene.objects, rng)
    answer = "yes" if overlaps(first, second) else "no"
    question = f"Do the {first.label} and the {second.label} overlap? Answer yes or no."
    return question, answer, 2, {"objects": [first.id, second.id], "relation": "overlap"}


def _inside_outside(scene: Scene, rng: random.Random) -> tuple[str, str, int, dict] | None:
    first, second = _choice_pair(scene.objects, rng)
    answer = "yes" if contains(second, first) else "no"
    question = f"Is the {first.label} completely inside the {second.label}? Answer yes or no."
    return question, answer, 2, {"objects": [first.id, second.id], "relation": "inside"}


def _multi_hop(scene: Scene, rng: random.Random) -> tuple[str, str, int, dict] | None:
    if len(scene.objects) < 3:
        return None
    relations: list[tuple[str, Callable[[SceneObject, SceneObject], bool]]] = [
        ("left of", is_left),
        ("right of", is_right),
        ("above", is_above),
        ("below", is_below),
    ]
    for _ in range(30):
        first_anchor, second_anchor = rng.sample(scene.objects, 2)
        relation_a, predicate_a = rng.choice(relations)
        relation_b, predicate_b = rng.choice(relations)
        if relation_a == relation_b:
            continue
        candidates = [
            obj
            for obj in scene.objects
            if obj.id not in {first_anchor.id, second_anchor.id}
            and predicate_a(obj, first_anchor)
            and predicate_b(obj, second_anchor)
        ]
        if len(candidates) == 1:
            target = candidates[0]
            question = (
                f"Which object is both {relation_a} the {first_anchor.label} "
                f"and {relation_b} the {second_anchor.label}?"
            )
            return question, target.label, 4, {
                "anchors": [first_anchor.id, second_anchor.id],
                "target": target.id,
                "relations": [relation_a, relation_b],
            }
    return None


TASK_BUILDERS: dict[str, QuestionBuilder] = {
    "counting": _counting,
    "left_right": _left_right,
    "above_below": _above_below,
    "nearest_farthest": _nearest_farthest,
    "relative_size": _relative_size,
    "center_proximity": _center_proximity,
    "relative_ordering": _relative_ordering,
    "between": _between,
    "overlap": _overlap,
    "inside_outside": _inside_outside,
    "multi_hop": _multi_hop,
}


def generate_questions(
    scene: Scene,
    image_path: str,
    count: int,
    seed: int,
    enabled_tasks: list[str] | None = None,
) -> list[QASample]:
    rng = random.Random(seed + sum(ord(char) for char in scene.scene_id))
    task_names = enabled_tasks or list(TASK_BUILDERS)
    task_names = [name for name in task_names if name in TASK_BUILDERS]
    rng.shuffle(task_names)
    samples: list[QASample] = []
    attempted: set[str] = set()

    for task_name in task_names * 3:
        if len(samples) >= count:
            break
        builder = TASK_BUILDERS[task_name]
        result = builder(scene, rng)
        if result is None:
            continue
        question, answer, difficulty, metadata = result
        signature = f"{task_name}:{question}"
        if signature in attempted:
            continue
        attempted.add(signature)
        samples.append(
            QASample(
                sample_id=f"{scene.scene_id}_q{len(samples) + 1:02d}",
                scene_id=scene.scene_id,
                image=image_path,
                question=question,
                answer=answer,
                task=task_name,
                difficulty=difficulty,
                split=scene.split,
                ood_type=scene.ood_type,
                metadata=metadata,
            )
        )
    return samples

