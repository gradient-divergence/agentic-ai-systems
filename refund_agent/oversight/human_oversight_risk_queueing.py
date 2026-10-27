"""One-server stationary formulas; units are hours and arrivals per hour."""

from dataclasses import dataclass
from fractions import Fraction
from math import exp, isfinite


@dataclass(frozen=True)
class MM1:
    status: str
    u: Fraction | None = None
    p0: Fraction | None = None
    L: Fraction | None = None
    W: Fraction | None = None
    W_wait: Fraction | None = None
    spare_rate: Fraction | None = None

    def occupancy(self, n: int) -> Fraction:
        if type(n) is not int or n < 0 or self.u is None or self.p0 is None:
            raise ValueError("occupancy needs a stable queue and a nonnegative integer")
        return self.p0 * self.u ** n

    def tail(self, t: float) -> float:
        if not isfinite(t) or t < 0 or self.spare_rate is None:
            raise ValueError("tail needs a stable queue and a nonnegative finite time")
        return exp(-float(self.spare_rate) * t)


def little_check(L, lam, W) -> float:
    if any(not isfinite(x) or x < 0 for x in (L, lam, W)):
        raise ValueError("Little check needs nonnegative finite values")
    return float(abs(L - lam * W) / max(abs(L), abs(lam * W), 1e-15))


def offered_load(classes, staffing) -> tuple[float, float]:
    """Weekly hours and utilization; classes supply daily rate and review minutes."""
    staffing = tuple(staffing)
    hours = sum(row.rate_per_day * 7 * row.minutes / 60 for row in classes)
    capacity = sum(staffing)
    if capacity <= 0 or any(not isfinite(h) or h < 0 for h in staffing):
        raise ValueError("staffing must have positive finite capacity")
    return hours, hours / capacity


def recovery(backlog, mu, lam):
    if any(not isfinite(x) or x < 0 for x in (backlog, mu, lam)):
        raise ValueError("recovery needs nonnegative finite values")
    return None if mu <= lam else backlog / (mu - lam)


def print_mm1():
    print("Constructed M/M/1; hours; homogeneous Poisson arrivals,")
    print("independent exponential service, continuous FCFS, unlimited queue.")
    for label, lam, mu, deadline in (
        ("bank", 12, 15, 1), ("reviewer", 1, 2, 4),
        ("reviewer", Fraction(3, 2), 2, 4),
        ("reviewer", Fraction(9, 5), 2, 4),
    ):
        row = mm1(lam, mu)
        assert row.u is not None and row.p0 is not None and row.L is not None
        assert row.W is not None and row.W_wait is not None
        print(f"{label} lambda={float(lam):g} mu={mu}: u={float(row.u):.3f}")
        print(f"  p0={float(row.p0):.3f} L={float(row.L):.3f}")
        print(f"  W={float(row.W):.3f}h wait={float(row.W_wait):.3f}h")
        print(f"  P(T>{deadline}h)={row.tail(deadline):.6f}")
        print(f"  Little relative gap={little_check(row.L, lam, row.W):.3g}")
        print(f"  P(wait>{deadline}h)={float(row.u) * row.tail(deadline):.6f}")
    unstable = mm1(18, 15)
    print("bank lambda=18 mu=15:", unstable.status)
    print("  stationary u,p0,L,W,W_wait:", unstable.u, unstable.p0,
          unstable.L, unstable.W, unstable.W_wait)
    print(f"At 4.4 reviews/h the available minutes each are {60 / 4.4:.3f}")


# region book:fnd.human-oversight-risk.mm1-01 output
def mm1(lam, mu) -> MM1:
    """Exact stationary means for rational rates; no means at capacity."""
    if (
        isinstance(lam, bool) or isinstance(mu, bool)
        or not isfinite(lam) or not isfinite(mu) or lam < 0 or mu <= 0
    ):
        raise ValueError("rates must be finite: lambda >= 0, mu > 0")
    lam, mu = Fraction(str(lam)), Fraction(str(mu))
    if lam >= mu:
        return MM1("unstable")
    u, spare = lam / mu, mu - lam
    return MM1(
        status="stable", u=u, p0=1 - u, L=u / (1 - u),
        W=1 / spare, W_wait=u / spare, spare_rate=spare,
    )


if __name__ == "__main__":
    print_mm1()
# endregion book:fnd.human-oversight-risk.mm1-01
