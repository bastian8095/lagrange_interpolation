# lagrange-interpolation

A small, dependency-free Python library that constructs the Lagrange interpolating polynomial through a given set of points and evaluates it.

## Usage

```python
from lagrange_interpolation import LagrangeInterpolator, lagrange_interpolate

points = [(0.0, 1.0), (1.0, 3.0), (2.0, 9.0)]

# One-off evaluation at a single x.
print(lagrange_interpolate(points, 0.5))   # -> 1.75

# Repeated evaluation: build once, call many times.
interp = LagrangeInterpolator(points)
print(interp.evaluate(0.0))   # -> 1.0  (exact at nodes)
print(interp(1.0))            # -> 3.0  (__call__ is an alias of evaluate)
print(interp.nodes)           # -> ((0.0, 1.0), (1.0, 3.0), (2.0, 9.0))
```

The library exports exactly two names from `lagrange_interpolation`:

- `LagrangeInterpolator` — the class. Construct with an iterable of `(x, y)` pairs; call `.evaluate(x)` or just `interp(x)`.
- `lagrange_interpolate(points, x)` — convenience function for a single evaluation.

## Why this exists

It answers the narrow question "what value does the Lagrange polynomial through these points take at this x?" without pulling in NumPy or SciPy. The implementation evaluates the polynomial in its direct product form rather than expanding to monomial coefficients. That keeps the code short, avoids a linear-algebra dependency, and is numerically reasonable for modest node counts. The trade-off is that building and evaluating has O(n) cost per evaluation rather than O(n) after a one-time O(n²) setup that expansion would allow; we judged simplicity and zero dependencies to matter more here.

## The awkward edge

Duplicate x-coordinates are rejected with a `ValueError`, because every Lagrange basis polynomial divides by `(x_i - x_j)`. Coincident nodes are a user error, not something to silently merge. Additionally, like every polynomial interpolation scheme, high-degree interpolation over many equispaced nodes is numerically unstable near the ends of the interval (Runge's phenomenon); this library does nothing to mitigate that beyond evaluating the textbook formula faithfully.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

