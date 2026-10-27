"""Offline arithmetic for chapter 23; no browser, agent imports or input files."""

import math
import random
from datetime import datetime


def wilson_interval(flagged: int, defects: int, z: float = 1.96) -> tuple[float, float]:
    """Invert the binomial score test for sensitivity on one labeled slice."""
    if defects <= 0 or not 0 <= flagged <= defects or not math.isfinite(z) or z <= 0:
        raise ValueError("valid counts and a positive finite normal quantile required")
    estimate = flagged / defects
    denominator = 1 + z * z / defects
    center = (estimate + z * z / (2 * defects)) / denominator
    half = z * math.sqrt(estimate * (1 - estimate) / defects + z * z / (4 * defects**2))
    half /= denominator
    return max(0.0, center - half), min(1.0, center + half)


def parts_per_slice(defects: int, prevalence: float) -> float:
    """Expected independent parts to observe this many defects at the given rate."""
    if defects <= 0 or not 0 < prevalence <= 1:
        raise ValueError("positive defect count and prevalence in (0, 1] required")
    return defects / prevalence


def box_overlap_and_union(prediction: tuple, reference: tuple) -> tuple[float, float]:
    """Areas of continuous (left, top, right, bottom) rectangles, without +1 edges."""
    for left, top, right, bottom in (prediction, reference):
        if not all(math.isfinite(v) for v in (left, top, right, bottom)):
            raise ValueError("finite box coordinates required")
        if right <= left or bottom <= top:
            raise ValueError("positive box area required")
    overlap = max(0, min(prediction[2], reference[2]) - max(prediction[0], reference[0]))
    overlap *= max(0, min(prediction[3], reference[3]) - max(prediction[1], reference[1]))
    predicted_area = (prediction[2] - prediction[0]) * (prediction[3] - prediction[1])
    reference_area = (reference[2] - reference[0]) * (reference[3] - reference[1])
    return overlap, predicted_area + reference_area - overlap


def intersection_over_union(prediction: tuple, reference: tuple) -> float:
    overlap, union = box_overlap_and_union(prediction, reference)
    return overlap / union


def page_age_seconds(page: dict, used_at: datetime) -> float:
    """Age of the displayed update, irrespective of a page capture's time."""
    updated = next((node for node in page["children"] if node["id"] == "updated"), None)
    if updated is None:
        raise ValueError("missing_page_update")
    observed_at = datetime.fromisoformat(updated["text"])
    if observed_at.utcoffset() is None or used_at.utcoffset() is None:
        raise ValueError("timezone_required")
    age = (used_at - observed_at).total_seconds()
    if age < 0:
        raise ValueError("future_page_update")
    return age


def locate_element(page: dict, element_id: str, prediction: tuple, threshold: float = 0.5) -> dict:
    """Refuse a proposed target unless its named element's reference box matches."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    node = next((node for node in page["children"] if node["id"] == element_id), None)
    if node is None:
        raise ValueError("element_not_found")
    if intersection_over_union(prediction, node["box"]) < threshold:
        raise ValueError("localization_refused")
    return node


def _check_action_model(actions: int, p_act: float, p_catch: float) -> None:
    if actions < 1 or not 0 <= p_act <= 1 or not 0 <= p_catch <= 1:
        raise ValueError("positive action count and probabilities in [0, 1] required")


def action_probabilities(actions: int, p_act: float, p_catch: float) -> dict[str, float]:
    """One retry, checked too; a detected wrong retry stops the run.

    A retry replaces a reversible effect; 'all right' describes final effects.
    Correct effects are always accepted. A silent error permits continuation.
    Completed silent runs and stopped runs are disjoint; a stopped run may
    already have had a silent error. Every attempt and detection is independent.
    """
    _check_action_model(actions, p_act, p_catch)
    good = p_act + (1 - p_act) * p_catch * p_act
    stop = ((1 - p_act) * p_catch) ** 2
    silent = (1 - p_act) * (1 - p_catch) * (1 + (1 - p_act) * p_catch)
    complete = (1 - stop) ** actions
    return {
        "unchecked_all_right": p_act**actions,
        "checked_all_right": good**actions,
        "checked_silent_completed": complete - good**actions,
        "checked_stopped": 1 - complete,
        # The first silent error follows zero or more correct final effects.
        "checked_any_silent": silent * sum(good**step for step in range(actions)),
    }


def simulate_actions(
    actions: int, p_act: float, p_catch: float, *, runs: int, seed: int
) -> dict[str, int]:
    """Sample effects, check each effect and stop on an exhausted detected failure."""
    _check_action_model(actions, p_act, p_catch)
    if runs < 1:
        raise ValueError("positive run count required")
    rng = random.Random(seed)
    counts = dict.fromkeys(action_probabilities(actions, p_act, p_catch), 0)
    for _ in range(runs):
        # Consume one first-attempt draw for each unchecked action.
        counts["unchecked_all_right"] += all([rng.random() < p_act for _ in range(actions)])
        all_right, stopped = True, False
        for _ in range(actions):
            for attempt in range(2):
                if rng.random() < p_act:
                    break
                if rng.random() >= p_catch:
                    all_right = False
                    break
                if attempt == 1:
                    stopped = True
            if stopped:
                break
        outcome = "checked_stopped" if stopped else (
            "checked_all_right" if all_right else "checked_silent_completed"
        )
        counts[outcome] += 1
        counts["checked_any_silent"] += not all_right
    return counts
