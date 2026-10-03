"""Lagrange interpolating polynomial construction and evaluation.

Given distinct nodes ``(x_0, y_0), ..., (x_n, y_n)`` the Lagrange form is

    P(x) = sum_{i=0}^{n} y_i * L_i(x)

where each cardinal (basis) polynomial is

    L_i(x) = prod_{j != i} (x - x_j) / (x_i - x_j)

We evaluate this directly rather than converting to monomial form. The direct
form is numerically better conditioned for small node sets and keeps the code
short and dependency-free. For very large node sets the well-known instability
of high-degree interpolation on equispaced nodes applies regardless of form;
we do not try to hide that, we just report it.
"""

from __future__ import annotations

from itertools import islice
from typing import Iterable, Sequence, Tuple


Point = Tuple[float, float]


class LagrangeInterpolator:
    """Lagrange interpolating polynomial through a fixed set of points.

    The nodes are copied at construction time so later mutation of the source
    sequence does not change the interpolator. Duplicate x-coordinates are
    rejected, because every L_i divides by (x_i - x_j) for j != i, which would
    be a division by zero.
    """

    __slots__ = ("_xs", "_ys", "_denoms")

    def __init__(self, points: Iterable[Point]) -> None:
        xs: list = []
        ys: list = []
        seen: set = set()
        for x, y in points:
            # Track duplicates by exact value. A float that compares equal to
            # an earlier node is a genuine duplicate for our purposes; we do
            # not attempt fuzzy deduplication because the right tolerance is
            # problem-dependent and guessing would hide user errors.
            if x in seen:
                raise ValueError(f"duplicate node x-coordinate: {x!r}")
            seen.add(x)
            xs.append(x)
            ys.append(y)

        if not xs:
            raise ValueError("at least one point is required")

        self._xs = xs
        self._ys = ys

        # Precompute the constant denominator of each L_i:
        #   prod_{j != i} (x_i - x_j)
        # Doing this once avoids recomputing O(n) work on every evaluate() call
        # and also lets us detect a singular node set up front.
        denoms = []
        for i, xi in enumerate(xs):
            d = 1.0
            for j, xj in enumerate(xs):
                if i == j:
                    continue
                d *= xi - xj
            if d == 0.0:
                # With the duplicate check above this should be unreachable
                # for finite floats, but NaN nodes can still trigger it.
                raise ValueError("singular node set (coincident x-values)")
            denoms.append(d)
        self._denoms = denoms

    @property
    def nodes(self) -> Tuple[Point, ...]:
        """The interpolation nodes as a tuple of (x, y) pairs."""
        return tuple(zip(self._xs, self._ys))

    def evaluate(self, x: float) -> float:
        """Evaluate the interpolating polynomial at ``x``.

        Uses the standard direct product form. At a node ``x_i`` the result is
        exactly ``y_i`` regardless of floating-point error in the other terms,
        because L_i(x_i) collapses to 1 and every other L_j(x_i) contains a
        zero factor (x_i - x_i).
        """
        xs = self._xs
        ys = self._ys
        denoms = self._denoms
        n = len(xs)

        total = 0.0
        for i in range(n):
            xi = xs[i]
            num = 1.0
            # Short-circuit: if x is exactly node i, L_i is 1 and the other
            # terms are 0, so the answer is just ys[i]. This keeps node
            # evaluations exact in floating point.
            if x == xi:
                return float(ys[i])
            for j in range(n):
                if i == j:
                    continue
                num *= x - xs[j]
            total += ys[i] * (num / denoms[i])
        return total

    def __call__(self, x: float) -> float:
        return self.evaluate(x)


def lagrange_interpolate(points: Sequence[Point], x: float) -> float:
    """Convenience wrapper: build an interpolator over ``points`` and evaluate at ``x``.

    If you need to evaluate at many x-values, construct a
    :class:`LagrangeInterpolator` once and call its ``evaluate`` method
    repeatedly instead — this avoids rebuilding the denominators each time.
    """
    return LagrangeInterpolator(points).evaluate(x)
