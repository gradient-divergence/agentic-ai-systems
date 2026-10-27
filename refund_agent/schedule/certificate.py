"""Weak-duality arithmetic for maximization with rows <= and variables >= 0."""

import numpy as np

from .inputs import TOL


def gaps(value, bound):
    absolute = float(bound - value)
    if value != 0:
        return absolute, absolute / abs(value), None
    if bound == 0:
        return absolute, 0.0, None
    return absolute, "undefined", "zero incumbent under a nonzero bound"


def _arrays(rows, rhs, objective, plan, multipliers, tol):
    rows = np.asarray(rows, dtype=float)
    rhs, objective, plan, multipliers = (
        np.asarray(a, dtype=float) for a in (rhs, objective, plan, multipliers)
    )
    if (
        rows.shape != (len(rhs), len(objective))
        or plan.shape != objective.shape
        or multipliers.shape != rhs.shape
    ):
        raise ValueError("incompatible certificate dimensions")
    if (
        tol < 0
        or not np.isfinite(tol)
        or any(not np.isfinite(a).all() for a in (rows, rhs, objective, plan, multipliers))
    ):
        raise ValueError("certificate inputs and tolerance must be finite")
    return rows, rhs, objective, plan, multipliers


def check_certificate(rows, rhs, objective, plan, multipliers, tol=TOL):
    rows, rhs, objective, plan, multipliers = _arrays(rows, rhs, objective, plan, multipliers, tol)
    weighted = multipliers @ rows
    violation = max(
        0.0, float(np.max(rows @ plan - rhs, initial=0)), float(np.max(-plan, initial=0))
    )
    nonnegative = bool(np.all(multipliers >= -tol))
    covering = bool(np.all(weighted >= objective - tol))
    permitted, certificate = violation <= tol, nonnegative and covering
    value, bound = float(objective @ plan), float(multipliers @ rhs)
    absolute, relative, reason = gaps(value, bound)
    return dict(
        permitted=permitted,
        largest_violation=violation,
        nonnegative=nonnegative,
        covering=covering,
        certificate=certificate,
        weighted_rows=weighted.tolist(),
        value=value,
        bound=bound,
        absolute_gap=absolute,
        relative_gap=relative,
        relative_gap_reason=reason,
        optimal=bool(permitted and certificate and abs(absolute) <= tol),
        tolerance=tol,
    )
