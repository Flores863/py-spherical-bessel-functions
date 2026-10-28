# py-spherical-bessel-functions

Evaluates spherical Bessel functions of the first kind (`j_n`) and second kind (`y_n`) for non-negative integer orders `n` and real `x`, using recurrences on plain floats.

```python
from py_spherical_bessel_functions import spherical_jn, spherical_yn

# j_0(2.0) == sin(2.0)/2.0
val = spherical_jn(0, 2.0)

# y_0(2.0) == -cos(2.0)/2.0
val = spherical_yn(0, 2.0)

# Higher orders via stable recurrence
j5 = spherical_jn(5, 3.7)
y3 = spherical_yn(3, 1.1)
```

## Why

The standard library's `math` module does not include spherical Bessel functions, and pulling in SciPy for two functions is heavy when you only need `float` results. This library fills that gap with a single-file, zero-dependency implementation.

`spherical_yn` uses forward recurrence throughout, which is stable because `|y_n|` grows with `n`. `spherical_jn` also uses forward recurrence for `|x| >= 0.5`, but switches to a Maclaurin series below that threshold — forward recurrence on `j_n` has a `1/x` factor that causes catastrophic cancellation when `x` is small. The series handles the small-`x` regime cleanly.

## Edge cases

- `spherical_jn(n, 0.0)` returns `1.0` for `n == 0` and `0.0` for `n > 0` (the removable singularity).
- `spherical_yn(n, 0.0)` raises `ValueError`; `y_n` is genuinely singular at the origin.
- `n` must be a non-negative `int`; a float or negative value raises.
- `j_n(-x) = (-1)^n j_n(x)` is respected; `y_n` is handled through the recurrence which inherits the sign naturally.
