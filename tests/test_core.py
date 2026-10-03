import math
import unittest

from lagrange_interpolation import LagrangeInterpolator, lagrange_interpolate


class TestConstruction(unittest.TestCase):
    def test_single_point_returns_constant(self):
        interp = LagrangeInterpolator([(3.0, 7.5)])
        self.assertEqual(interp.evaluate(3.0), 7.5)
        self.assertEqual(interp.evaluate(100.0), 7.5)
        self.assertEqual(interp.evaluate(-2.0), 7.5)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            LagrangeInterpolator([])

    def test_duplicate_nodes_raise(self):
        with self.assertRaises(ValueError):
            LagrangeInterpolator([(1.0, 2.0), (1.0, 9.0)])

    def test_nodes_property_is_snapshot(self):
        src = [(0.0, 1.0), (2.0, 5.0)]
        interp = LagrangeInterpolator(src)
        src.append((3.0, 99.0))
        self.assertEqual(interp.nodes, ((0.0, 1.0), (2.0, 5.0)))


class TestExactness(unittest.TestCase):
    """Lagrange interpolation of n+1 points reproduces any polynomial of degree n."""

    def test_linear(self):
        # y = 2x + 3
        pts = [(-1.0, 1.0), (2.0, 7.0)]
        interp = LagrangeInterpolator(pts)
        for x in [-3.0, -1.0, 0.0, 0.5, 2.0, 10.0]:
            self.assertAlmostEqual(interp.evaluate(x), 2.0 * x + 3.0)

    def test_quadratic(self):
        # y = x^2 - 2x + 1
        f = lambda x: x * x - 2.0 * x + 1.0
        pts = [(0.0, f(0.0)), (1.0, f(1.0)), (3.0, f(3.0))]
        interp = LagrangeInterpolator(pts)
        for x in [-2.0, -0.5, 0.0, 1.0, 2.0, 3.0, 5.0]:
            self.assertAlmostEqual(interp.evaluate(x), f(x))

    def test_cubic(self):
        # y = x^3 - 4x
        f = lambda x: x * x * x - 4.0 * x
        xs = [-2.0, -1.0, 1.0, 2.0]
        pts = [(x, f(x)) for x in xs]
        interp = LagrangeInterpolator(pts)
        for x in [-3.0, -2.0, -1.0, 0.0, 0.7, 1.0, 2.0, 3.5]:
            self.assertAlmostEqual(interp.evaluate(x), f(x), places=10)

    def test_constant_function_multiple_nodes(self):
        pts = [(0.0, 5.0), (1.0, 5.0), (2.0, 5.0), (4.0, 5.0)]
        interp = LagrangeInterpolator(pts)
        for x in [-1.0, 0.0, 0.3, 1.0, 2.5, 4.0, 7.0]:
            self.assertAlmostEqual(interp.evaluate(x), 5.0)


class TestNodeRecovery(unittest.TestCase):
    def test_evaluating_at_each_node_returns_its_y(self):
        pts = [(0.0, 1.0), (1.0, -3.0), (2.0, 4.0), (4.0, 0.0)]
        interp = LagrangeInterpolator(pts)
        for x, y in pts:
            self.assertEqual(interp.evaluate(x), y)

    def test_call_alias_matches_evaluate(self):
        interp = LagrangeInterpolator([(0.0, 1.0), (2.0, 9.0)])
        self.assertEqual(interp(1.0), interp.evaluate(1.0))


class TestWrapper(unittest.TestCase):
    def test_wrapper_matches_class(self):
        pts = [(0.0, 1.0), (1.0, 2.0), (2.0, 5.0)]
        interp = LagrangeInterpolator(pts)
        for x in [-1.0, 0.0, 0.5, 1.0, 2.0, 3.0]:
            self.assertEqual(lagrange_interpolate(pts, x), interp.evaluate(x))

    def test_wrapper_single_point(self):
        self.assertEqual(lagrange_interpolate([(4.0, 8.0)], 4.0), 8.0)
        self.assertEqual(lagrange_interpolate([(4.0, 8.0)], 99.0), 8.0)


class TestEdgeCases(unittest.TestCase):
    def test_two_nodes_same_y(self):
        # Horizontal line through two points.
        interp = LagrangeInterpolator([(0.0, 3.0), (5.0, 3.0)])
        self.assertAlmostEqual(interp.evaluate(2.5), 3.0)

    def test_large_x_ratio(self):
        # Nodes spanning very different magnitudes. We only assert that the
        # value at the midpoint of two nodes lies between the endpoint ys,
        # which must hold for a degree-1 interpolant.
        interp = LagrangeInterpolator([(1e-6, -1.0), (1e6, 1.0)])
        # Geometric midpoint on log scale; linear in x is (1e6+1e-6)/2 ~= 5e5.
        mid = (1e-6 + 1e6) / 2.0
        val = interp.evaluate(mid)
        self.assertGreater(val, -1.0)
        self.assertLess(val, 1.0)

    def test_negative_and_zero_nodes(self):
        f = lambda x: x * x + 1.0
        pts = [(-2.0, f(-2.0)), (0.0, f(0.0)), (2.0, f(2.0))]
        interp = LagrangeInterpolator(pts)
        self.assertAlmostEqual(interp.evaluate(-1.0), f(-1.0))
        self.assertAlmostEqual(interp.evaluate(0.0), f(0.0))
        self.assertAlmostEqual(interp.evaluate(1.0), f(1.0))

    def test_iterable_consumed_once(self):
        # Points can come from a generator; construction must materialise them.
        def gen():
            yield (0.0, 1.0)
            yield (1.0, 3.0)
            yield (2.0, 7.0)

        interp = LagrangeInterpolator(gen())
        self.assertAlmostEqual(interp.evaluate(0.5), 1.75)

    def test_runs_test_value(self):
        # Sanity value: line through (0,1) and (2,5) is y = 2x+1; at x=1 -> 3.
        interp = LagrangeInterpolator([(0.0, 1.0), (2.0, 5.0)])
        self.assertAlmostEqual(interp.evaluate(1.0), 3.0)


if __name__ == "__main__":
    unittest.main()
