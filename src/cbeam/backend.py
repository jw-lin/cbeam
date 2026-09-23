import os

backend_choice = os.environ.get("CBEAM_BACKEND", "numpy").lower()
using_jax = backend_choice == "jax"

if using_jax:
    # Grow-on-demand + return-to-driver GPU allocation.  Without this JAX
    # preallocates ~75% of the device on first use and never gives it back, so
    # a workload that builds/discards big arrays per iteration (e.g. the
    # multi-wavelength lantern pipeline: characterize -> propagate -> release,
    # one wavelength at a time) OOMs even though its steady-state footprint is
    # small.  Respect an explicit user setting if one is already present.
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as xp
    print("Using JAX backend.")
    try:
        import diffrax
        print("Using diffrax for ODE integration.")
    except ImportError:
        raise ImportError("Please install diffrax: pip install diffrax")
else:
    import numpy as xp
    import scipy
    import scipy.integrate
    import scipy.interpolate


def get_xp():
    return xp


def get_backend():
    return "jax" if using_jax else "numpy"


def get_jax_device():
    """Return the JAX device selected by CBEAM_JAX_DEVICE_INDEX (default 0).

    Falls back gracefully when the requested index does not exist (e.g. no GPU
    or only one GPU is available).
    """
    if not using_jax:
        return None
    import jax
    idx = int(os.environ.get("CBEAM_JAX_DEVICE_INDEX", "0"))
    devices = jax.devices()
    if idx >= len(devices):
        idx = 0
    return devices[idx]


# ---------------------------------------------------------------------------
# solve_ivp shim
# ---------------------------------------------------------------------------

def solve_ivp(fun, t_span, y0, **kwargs):
    """Unified solve_ivp wrapper.

    For the numpy backend this is a thin pass-through to scipy.integrate.solve_ivp.
    For the JAX backend it wraps diffrax.diffeqsolve.

    Memory note: by default only the final state is saved (SaveAt(t1=True)).
    Pass ``t_eval`` to request specific save-points (still no full-step buffer).
    The old ``SaveAt(t1=True, steps=True)`` was the primary cause of OOM errors
    because it allocated a buffer for every adaptive micro-step.
    """
    if using_jax:
        import diffrax

        t0, t1 = float(t_span[0]), float(t_span[1])
        dt0 = kwargs.get("dt0", abs(t1 - t0) * 0.01)

        diffrax_fun = lambda t, y, args=None: fun(t, y)
        term = diffrax.ODETerm(diffrax_fun)

        t_eval = kwargs.get("t_eval", None)
        if t_eval is not None:
            saveat = diffrax.SaveAt(ts=xp.asarray(t_eval, dtype=xp.float64))
        else:
            # Save only the endpoint — avoids allocating a per-step buffer.
            saveat = diffrax.SaveAt(t1=True)
            # saveat=diffrax.SaveAt(steps=True) # for fair comparison with numpy path


        rtol = kwargs.get("rtol", 1e-12)
        atol = kwargs.get("atol", 1e-10)
        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

        out = diffrax.diffeqsolve(
            term,
            diffrax.Dopri5(),
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=200_000,
        )
        return out
    else:
        return scipy.integrate.solve_ivp(fun, t_span, y0, **kwargs)


# ---------------------------------------------------------------------------
# _JAXCubicSpline
# ---------------------------------------------------------------------------

class _JAXCubicSpline:
    """
    Cubic spline for the JAX backend that mirrors the parts of
    ``scipy.interpolate.CubicSpline`` the propagator relies on: evaluation,
    ``.derivative()``, ``.antiderivative()``, N-d and complex output arrays.

    To stay numerically aligned with the numpy / upstream path the knot
    coefficients are fitted on the host with scipy (``not-a-knot`` end
    conditions by default, ``natural`` on request); only the *evaluation* runs
    in ``xp`` so it stays traceable/differentiable in ``z``.  Queries outside
    ``[xs[0], xs[-1]]`` are extrapolated with the boundary-segment polynomial
    (``extrapolate=True``, scipy's default), unless ``extrapolate=False``.

    ``y`` shape: (n_points, *trailing_dims)
    """

    def __init__(self, xs, ys, axis=0, bc_type="not-a-knot",
                 extrapolate=True, **kwargs):
        ys = xp.moveaxis(xp.asarray(ys), axis, 0)
        xs = xp.asarray(xs, dtype=xp.float64)
        self.xs = xs
        self.ys = ys
        self.axis = 0
        self.bc_type = "natural" if bc_type == "natural" else "not-a-knot"
        self.extrapolate = bool(extrapolate)
        self._coeffs = self._fit(xs, ys, self.bc_type)

    @staticmethod
    def _fit(xs, ys, bc_type="not-a-knot"):
        import numpy as np
        import scipy.interpolate as _si

        xs_np = np.asarray(xs, dtype=np.float64)
        ys_np = np.asarray(ys)
        if xs_np.shape[0] < 3 and bc_type == "not-a-knot":
            bc_type = "natural"

        if np.iscomplexobj(ys_np):
            cs_r = _si.CubicSpline(xs_np, ys_np.real, axis=0, bc_type=bc_type)
            cs_i = _si.CubicSpline(xs_np, ys_np.imag, axis=0, bc_type=bc_type)
            cc = cs_r.c + 1j * cs_i.c
        else:
            cc = _si.CubicSpline(xs_np, ys_np, axis=0, bc_type=bc_type).c

        # scipy stores cc[k] as the coefficient of (x - x_i)**(3 - k); the
        # evaluator below expects a + b*dx + c*dx**2 + d*dx**3.
        return (xp.asarray(cc[3]), xp.asarray(cc[2]),
                xp.asarray(cc[1]), xp.asarray(cc[0]))

    def _eval(self, z, coeffs):
        a, b, c, d = coeffs
        xs = self.xs
        if not self.extrapolate:
            z = xp.clip(z, xs[0], xs[-1])
        idx = xp.searchsorted(xs, z, side="right") - 1
        idx = xp.clip(idx, 0, xs.shape[0] - 2)
        dx  = z - xs[idx]
        return a[idx] + dx * (b[idx] + dx * (c[idx] + dx * d[idx]))

    def __call__(self, z):
        return self._eval(z, self._coeffs)

    def derivative(self):
        a, b, c, d = self._coeffs
        deriv_coeffs = (b, 2 * c, 3 * d, xp.zeros_like(d))
        parent = self

        class _Deriv:
            def __call__(self_, z):
                b2, c2, d2, _ = deriv_coeffs
                xs  = parent.xs
                z_  = z if parent.extrapolate else xp.clip(z, xs[0], xs[-1])
                idx = xp.searchsorted(xs, z_, side="right") - 1
                idx = xp.clip(idx, 0, xs.shape[0] - 2)
                dx  = z_ - xs[idx]
                return b2[idx] + dx * (c2[idx] + dx * d2[idx])

        return _Deriv()

    def antiderivative(self):
        a, b, c, d = self._coeffs
        xs         = self.xs
        h          = xp.diff(xs)
        shape_tail = a.shape[1:]
        sl         = tuple([slice(None)] + [None] * len(shape_tail))

        seg_integrals = (
            a * h[sl]
            + b * h[sl] ** 2 / 2
            + c * h[sl] ** 3 / 3
            + d * h[sl] ** 4 / 4
        )
        cumulative = xp.concatenate(
            [xp.zeros((1,) + shape_tail, dtype=seg_integrals.dtype),
             xp.cumsum(seg_integrals, axis=0)], axis=0
        )
        parent = self

        class _Antideriv:
            def __call__(self_, z):
                a2, b2, c2, d2 = parent._coeffs
                xs2 = parent.xs
                z_  = z if parent.extrapolate else xp.clip(z, xs2[0], xs2[-1])
                idx = xp.searchsorted(xs2, z_, side="right") - 1
                idx = xp.clip(idx, 0, xs2.shape[0] - 2)
                dx  = z_ - xs2[idx]
                seg = cumulative[idx]
                return seg + dx * (a2[idx] + dx * (b2[idx] / 2 + dx * (c2[idx] / 3 + dx * d2[idx] / 4)))

        return _Antideriv()


# ---------------------------------------------------------------------------
# Public interpolation helpers
# ---------------------------------------------------------------------------

def myCubicSpline(x, y, axis=0, bc_type="not-a-knot", extrapolate=True, **kwargs):
    if using_jax:
        return _JAXCubicSpline(x, y, axis=axis, bc_type=bc_type,
                               extrapolate=extrapolate)
    else:
        return scipy.interpolate.CubicSpline(
            x, y, axis=axis, bc_type=bc_type, extrapolate=extrapolate, **kwargs)


def UnivariateSpline(x, y, **kwargs):
    if using_jax:
        # scipy's UnivariateSpline(s=0) is an interpolating cubic; the closest
        # xp-evaluable match is a not-a-knot cubic spline.  ext=0/"extrapolate"
        # -> polynomial extrapolation (the mode used at every call site here).
        ext = kwargs.get("ext", 0)
        return _JAXCubicSpline(x, y, axis=0, bc_type="not-a-knot",
                               extrapolate=ext in (0, "extrapolate"))
    else:
        return scipy.interpolate.UnivariateSpline(x, y, **kwargs)


def interp1d(x, y, kind="linear", axis=0, **kwargs):
    if using_jax:
        # cubic / quadratic -> scipy-fitted not-a-knot spline.  With exactly 3
        # samples not-a-knot is the exact parabola, matching
        # interp1d(kind='quadratic', fill_value='extrapolate').
        if kind in ("cubic", 3, "quadratic", 2):
            return _JAXCubicSpline(x, y, axis=axis, bc_type="not-a-knot",
                                   extrapolate=True)
        x = xp.asarray(x, dtype=xp.float64)
        y = xp.asarray(y)

        def interp_func(x_new):
            x_new = xp.asarray(x_new, dtype=xp.float64)
            idx   = xp.searchsorted(x, x_new, side="right") - 1
            idx   = xp.clip(idx, 0, x.shape[0] - 2)
            x0, x1 = x[idx], x[idx + 1]
            y0 = y[idx]
            y1 = y[idx + 1]
            t  = (x_new - x0) / (x1 - x0)
            return y0 + t[..., *([None] * (y.ndim - 1))] * (y1 - y0)

        return interp_func
    else:
        return scipy.interpolate.interp1d(x, y, kind=kind, axis=axis, **kwargs)