import math
import unittest

from py_spherical_bessel_functions import spherical_jn, spherical_yn


class TestSphericalJn(unittest.TestCase):
    def _close(self, a, b, tol=1e-9):
        self.assertTrue(abs(a - b) < tol, f"{a} != {b} (tol={tol})")

    def test_j0_at_pi(self):
        # j0(pi) = sin(pi)/pi ~ 0
        self._close(spherical_jn(0, math.pi), math.sin(math.pi) / math.pi, tol=1e-12)

    def test_j0_at_1(self):
        self._close(spherical_jn(0, 1.0), math.sin(1.0) / 1.0, tol=1e-14)

    def test_j1_at_1(self):
        expected = math.sin(1.0) / (1.0 ** 2) - math.cos(1.0) / 1.0
        self._close(spherical_jn(1, 1.0), expected, tol=1e-14)

    def test_j0_at_zero(self):
        self._close(spherical_jn(0, 0.0), 1.0)

    def test_jn_at_zero_n_gt_0(self):
        self._close(spherical_jn(1, 0.0), 0.0)
        self._close(spherical_jn(5, 0.0), 0.0)

    def test_small_x_series(self):
        # For small x, j_n should match the series. Cross-check against
        # the known relation j_n(x) ~ x^n / (2n+1)!! for very small x.
        x = 0.05
        for n in range(6):
            # (2n+1)!!
            df = 1
            for k in range(n):
                df *= (2 * k + 1)
            expected_leading = x ** n / df
            # The full value should be close to the leading term for tiny x
            val = spherical_jn(n, x)
            self.assertTrue(
                abs(val - expected_leading) < 0.01 * abs(expected_leading) + 1e-15,
                f"j_{n}({x}) = {val}, leading term {expected_leading}",
            )

    def test_j2_at_2(self):
        # Known: j2(2) = (3/x^3 - 1/x) sin(x) - 3 cos(x)/x^2
        x = 2.0
        expected = ((3.0 / x ** 3 - 1.0 / x) * math.sin(x)
                    - (3.0 * math.cos(x) / x ** 2))
        self._close(spherical_jn(2, x), expected, tol=1e-12)

    def test_recurrence_relation(self):
        # (2n+1)/x * j_n - j_{n+1} should equal j_{n-1}
        x = 3.0
        for n in range(1, 5):
            jn = spherical_jn(n, x)
            jnp1 = spherical_jn(n + 1, x)
            jnm1 = spherical_jn(n - 1, x)
            lhs = (2 * n + 1) / x * jn - jnp1
            self._close(lhs, jnm1, tol=1e-10)

    def test_negative_n_raises(self):
        with self.assertRaises(ValueError):
            spherical_jn(-1, 1.0)

    def test_non_integer_n_raises(self):
        with self.assertRaises(TypeError):
            spherical_jn(1.5, 1.0)

    def test_odd_symmetry(self):
        # j_n(-x) = (-1)^n j_n(x)
        x = 1.5
        for n in range(4):
            self._close(spherical_jn(n, -x), (-1) ** n * spherical_jn(n, x), tol=1e-10)


class TestSphericalYn(unittest.TestCase):
    def _close(self, a, b, tol=1e-9):
        self.assertTrue(abs(a - b) < tol, f"{a} != {b} (tol={tol})")

    def test_y0_at_1(self):
        expected = -math.cos(1.0) / 1.0
        self._close(spherical_yn(0, 1.0), expected, tol=1e-14)

    def test_y1_at_1(self):
        expected = -math.cos(1.0) / (1.0 ** 2) - math.sin(1.0) / 1.0
        self._close(spherical_yn(1, 1.0), expected, tol=1e-14)

    def test_y2_at_2(self):
        # y2(x) = (-3/x^3 + 1/x) cos(x) - 3 sin(x)/x^2
        x = 2.0
        expected = ((-3.0 / x ** 3 + 1.0 / x) * math.cos(x)
                    - (3.0 * math.sin(x) / x ** 2))
        self._close(spherical_yn(2, x), expected, tol=1e-12)

    def test_y_recurrence(self):
        x = 3.0
        for n in range(1, 5):
            yn = spherical_yn(n, x)
            ynp1 = spherical_yn(n + 1, x)
            ynm1 = spherical_yn(n - 1, x)
            lhs = (2 * n + 1) / x * yn - ynp1
            self._close(lhs, ynm1, tol=1e-10)

    def test_y_at_zero_raises(self):
        with self.assertRaises(ValueError):
            spherical_yn(0, 0.0)

    def test_negative_n_raises(self):
        with self.assertRaises(ValueError):
            spherical_yn(-1, 1.0)

    def test_non_integer_n_raises(self):
        with self.assertRaises(TypeError):
            spherical_yn(1.5, 1.0)


if __name__ == "__main__":
    unittest.main()
