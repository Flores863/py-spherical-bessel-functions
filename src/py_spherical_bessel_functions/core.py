import math


def _spherical_jn_small_x(n, x):
    """Small-x series for j_n(x).

    For x << 1 the forward recurrence j_{n+1} = (2n+1)/x * j_n - j_{n-1}
    is violently unstable because of the 1/x term. The Maclaurin series is
    accurate and stable here, so we switch to it below a threshold that is
    chosen empirically to stay well inside the stable regime for the
    forward recurrence used above it.
    """
    # Maclaurin series: j_n(x) = x^n / (1*3*...(2n+1)) * [1 - x^2/(2(2n+3)) + ...]
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 1.0 - x * x / 6.0
    # Compute leading coefficient x^n / (2n+1)!!
    x_n = 1.0
    double_factorial_odd = 1.0
    for _ in range(n):
        x_n *= x
    for k in range(n):
        double_factorial_odd *= (2 * k + 1)
    leading = x_n / double_factorial_odd
    # Series in brackets: 1 - x^2/(2(2n+3)) + x^4/(8(2n+3)(2n+5)) - ...
    term = 1.0
    total = 1.0
    x2 = x * x
    m = 0
    while True:
        m += 1
        denom = 2.0 * m
        for k in range(m):
            denom *= (2 * n + 3 + 2 * k)
        # The sign alternates: (-1)^m
        term *= -x2 / denom
        if abs(term) < 1e-18 * abs(total) and m > 1:
            break
        total += term
        if m > 50:
            break
    return leading * total


def _spherical_jn_forward(n_max, x):
    """Forward recurrence for j_0 .. j_{n_max}.

    j_{n+1} = ((2n+1)/x) * j_n - j_{n-1}

    Forward recurrence is stable for j_n when x is not too small relative
    to n, which is the complement of the regime where we use the series.
    """
    j0 = math.sin(x) / x
    if n_max == 0:
        return [j0]
    j1 = (math.sin(x) / (x * x)) - (math.cos(x) / x)
    vals = [j0, j1]
    for n in range(1, n_max):
        jn1 = ((2 * n + 1) / x) * vals[-1] - vals[-2]
        vals.append(jn1)
    return vals


def _spherical_yn_forward(n_max, x):
    """Forward recurrence for y_0 .. y_{n_max}.

    y_{n+1} = ((2n+1)/x) * y_n - y_{n-1}

    Unlike j_n, forward recurrence is stable for y_n across the whole
    domain, because |y_n| grows with n. No special-casing needed.
    """
    y0 = -math.cos(x) / x
    if n_max == 0:
        return [y0]
    y1 = (-math.cos(x) / (x * x)) - (math.sin(x) / x)
    vals = [y0, y1]
    for n in range(1, n_max):
        yn1 = ((2 * n + 1) / x) * vals[-1] - vals[-2]
        vals.append(yn1)
    return vals


def spherical_jn(n, x):
    """Spherical Bessel function of the first kind, j_n(x).

    n: non-negative integer order.
    x: float, must be non-zero (j_n has a removable singularity at 0
       but this library does not special-case x=0).

    For |x| < 0.5 a Maclaurin series is used, because the forward
    recurrence j_{n+1} = (2n+1)/x * j_n - j_{n-1} suffers catastrophic
    cancellation when x is small. For |x| >= 0.5 the forward recurrence
    starting from j_0 = sin(x)/x and j_1 = sin(x)/x^2 - cos(x)/x is
    stable and accurate.
    """
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non-negative")
    if x == 0.0:
        # j_n(0) = 1 for n=0, 0 for n>0, but the closed forms we use
        # divide by x. We define the limit explicitly.
        return 1.0 if n == 0 else 0.0
    ax = abs(x)
    if ax < 0.5:
        val = _spherical_jn_small_x(n, x)
        # j_n(-x) = (-1)^n * j_n(x)
        if x < 0 and n % 2 == 1:
            return -val
        return val
    vals = _spherical_jn_forward(n, x)
    return vals[n]


def spherical_yn(n, x):
    """Spherical Bessel function of the second kind, y_n(x).

    n: non-negative integer order.
    x: float, must be non-zero. y_n is singular at the origin.

    Forward recurrence is used throughout: y_{n+1} = (2n+1)/x * y_n - y_{n-1},
    starting from y_0 = -cos(x)/x and y_1 = -cos(x)/x^2 - sin(x)/x.
    This is stable because |y_n| grows with n, unlike j_n.
    """
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non-negative")
    if x == 0.0:
        raise ValueError("spherical_yn is singular at x = 0")
    vals = _spherical_yn_forward(n, x)
    return vals[n]
