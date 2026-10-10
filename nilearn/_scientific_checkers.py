"""Opt-in scientific runtime checks used by the SciBench pilot.

The module is inert unless ``SCIBENCH_TRIGGER_LOG`` names a log file. Checks
append stable IDs and never raise into nilearn, never change a return value and
never touch global random state. Re-calling checkers work on copies and run with
checking suppressed (re-entrancy guard).

Tolerance convention (SANITIZER.md 5.8): ``_C * eps(dtype) * size * scale``
with each factor justified at the call site.

All imports of nilearn itself are lazy (inside functions) so that this module
sits outside the package import layering.
"""

from __future__ import annotations

import functools
import json
import os
import warnings

import numpy as np

_ACTIVE = False
_EPS = float(np.finfo(np.float64).eps)
_C = 64.0
_ENV = "SCIBENCH_TRIGGER_LOG"
_MAX_ELEMS = 20_000_000
_CALLS: dict[str, int] = {}


def enabled():
    return bool(os.environ.get(_ENV)) and not _ACTIVE


def trigger(checker_id):
    path = os.environ.get(_ENV)
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"checker_id": checker_id}) + "\n")
    except Exception:
        _swallowed()


def trigger_if(condition, checker_id):
    _reach(checker_id)
    if bool(condition):
        trigger(checker_id)


SWALLOWED = []
REACHED = {}


def _reach(checker_id):
    """Curator-only: count predicate evaluations (observation reachability)."""
    if os.environ.get("SCIBENCH_CHECKER_DEBUG"):
        REACHED[checker_id] = REACHED.get(checker_id, 0) + 1


def _swallowed():
    """Curator-only: record exceptions swallowed inside a checker."""
    if os.environ.get("SCIBENCH_CHECKER_DEBUG"):
        import traceback

        SWALLOWED.append(traceback.format_exc())


def _guarded(fn):
    """Run ``fn`` once, never nested, never raising into nilearn."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _ACTIVE
        if _ACTIVE or not os.environ.get(_ENV):
            return None
        _ACTIVE = True
        try:
            # the host test suite may turn warnings into errors; a checker must
            # never be aborted by one
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with np.errstate(all="ignore"):
                    return fn(*args, **kwargs)
        except Exception:
            _swallowed()
            return None
        finally:
            _ACTIVE = False

    return wrapper


def _safe(fn):
    """Like ``_guarded`` for snapshot helpers called before the work."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            with np.errstate(all="ignore"):
                return fn(*args, **kwargs)
        except Exception:
            _swallowed()
            return None

    return wrapper


def sampled(name):
    """Deterministic call schedule: calls 1, 2, 4, 8, ... are checked."""
    n = _CALLS.get(name, 0) + 1
    _CALLS[name] = n
    return (n & (n - 1)) == 0


# ----------------------------------------------------------------- helpers
def _eps_of(x):
    try:
        dt = np.dtype(getattr(x, "dtype", np.float64))
        if dt.kind == "f":
            return float(np.finfo(dt).eps)
    except Exception:
        _swallowed()
    return _EPS


def _f64(x):
    return np.asarray(x, dtype=np.float64)


def _eps_in(*arrays):
    """Largest unit round-off among the (floating) input arrays, >= float64 eps."""
    eps = _EPS
    for a in arrays:
        if a is not None:
            eps = max(eps, _eps_of(np.asarray(a)))
    return eps


def _scale_ok(*arrays, lo=1e-100, hi=1e100):
    """P: every non-zero magnitude lies where float64 products/squares neither
    underflow into subnormals nor overflow (the checkers form such products)."""
    for a in arrays:
        if a is None:
            continue
        a = np.asarray(a)
        if a.dtype.kind == "f":
            # the data's own dtype: below ~1e4 tiny its products are subnormal
            lo_a = max(lo, 1e4 * float(np.finfo(a.dtype).tiny))
        else:
            lo_a = lo
        m = np.abs(a.astype(np.float64))
        if m.size == 0:
            continue
        mx = float(m.max())
        if mx != 0.0 and not (lo_a <= mx <= hi):
            return False
    return True


_STATE = {"nonconverged": False}


def note_nonconvergence():
    """Curator hook: a Frechet-mean iteration stopped at max_iter (it warns)."""
    _STATE["nonconverged"] = True


def reset_nonconvergence():
    _STATE["nonconverged"] = False


def _finite(x):
    return bool(np.all(np.isfinite(np.asarray(x))))


# =================================================================== signal
@_safe
def l1_cols(x):
    a = np.asarray(x)
    if a.ndim != 2 or a.size > _MAX_ELEMS:
        return None
    return np.abs(a.astype(np.float64)).sum(axis=0)


@_guarded
def check_detrend(out, l1):
    """NL-SIG-001: detrended columns are orthogonal to 1 and the ramp."""
    if l1 is None:
        return
    out = np.asarray(out)
    if out.ndim != 2 or out.shape[0] < 2 or not _finite(out):
        return
    n = out.shape[0]
    if not _scale_ok(l1 / n):
        return
    # float32 input stays float32 inside _detrend; each output element carries a
    # few ulps of the *input* magnitude, hence C * eps_d * sum|x_in|.
    # the column mean / slope are sequential reductions over n samples: their
    # round-off grows like n * eps_d (P: beyond n * eps_d ~ 1e-2 no statement)
    eps_d = _eps_of(out)
    if (_C + n) * eps_d > 1e-2:
        return
    tol = (_C + n) * eps_d * l1
    r = np.arange(n, dtype=np.float64)
    r -= r.mean()
    r /= np.sqrt((r**2).sum())
    o = out.astype(np.float64)
    bad = np.any(np.abs(o.sum(axis=0)) > tol) or np.any(np.abs(r @ o) > tol)
    trigger_if(bad, "NL-SIG-001")


@_safe
def snap_standardize(signals):
    a = np.asarray(signals)
    if a.ndim != 2 or a.size > _MAX_ELEMS:
        return None
    return a.astype(np.float64)


@_safe
def col_absmax(x):
    a = np.asarray(x)
    if a.ndim != 2 or a.size > _MAX_ELEMS:
        return None
    return np.abs(a.astype(np.float64)).max(axis=0)


_CLEAN_MX = {"mx": None}


@_safe
def note_clean_mx(signals):
    """Called inside ``clean``: remember the column magnitudes of the raw
    input so that the final standardize_signal call (which only sees the
    detrended / filtered / regressed residual) can scale its tolerance by the
    input magnitude. Cleared by ``end_clean`` at the end of ``clean``."""
    a = np.asarray(signals)
    _CLEAN_MX["mx"] = (
        np.abs(a.astype(np.float64)).max(axis=0)
        if a.ndim == 2 and a.size <= _MAX_ELEMS
        else None
    )


@_guarded
def check_standardize(x_in, out, standardize, detrend, mx_in=None):
    """NL-SIG-003 (psc scaling)."""
    if x_in is None:
        return
    x = np.asarray(x_in, dtype=np.float64)
    o = np.asarray(out)
    if x.ndim != 2 or x.shape[0] < 2 or o.shape != x.shape:
        return
    if not (_finite(x) and _finite(o)):
        return
    # P: magnitudes where the variance (a sum of squares) is representable
    if not _scale_ok(x, lo=0.0):
        return
    eps = _eps_of(o)
    o64 = o.astype(np.float64)
    # P: the library forms sum(x^2) in the data's own dtype; beyond
    # sqrt(finfo.max / n) that sum overflows (float32: ~1e18), not a defect
    if o.dtype.kind == "f" and np.abs(x).max() * np.sqrt(x.shape[0]) > 0.5 * np.sqrt(np.finfo(o.dtype).max):
        return
    # error scale is the magnitude of the *input* (before detrending), not of
    # the possibly detrended-to-rounding-noise column
    mx = np.abs(x).max(axis=0) if mx_in is None else np.asarray(mx_in)
    raw = _CLEAN_MX["mx"]
    if raw is not None and raw.shape == mx.shape:
        mx = np.maximum(mx, raw)  # P: spread is judged against the raw input
    std = x.std(axis=0, ddof=1)
    mean = x.mean(axis=0)
    if standardize == "psc" and not detrend:
        ok = (
            (std > 0)
            & (np.abs(mean) > _EPS)
            & (np.abs(mean) >= 1e6 * eps * std)
        )
        if not ok.any():
            return
        oc = o64[:, ok]
        am = np.abs(mean[ok])
        # output elements carry eps_d * 100 * max|x| / |mean| of error
        # the library's column mean is a sequential reduction: relative error n*eps_d
        nn = x.shape[0]
        if (_C + nn) * eps > 1e-2:
            return  # P: n * eps_d small enough for a meaningful statement
        tol_mean = (_C + nn) * eps * 100.0 * mx[ok] / am
        want = 100.0 * std[ok] / am
        rel = (_C * mx[ok] / std[ok] + nn) * eps
        got = oc.std(axis=0, ddof=1)
        bad = np.any(np.abs(oc.mean(axis=0)) > tol_mean) or np.any(
            np.abs(got - want) > rel * want
        )
        trigger_if(bad, "NL-SIG-003")


@_guarded
def check_butterworth_cutoff(sos, critical_freq, sampling_rate, order):
    """NL-SIG-004: |H(f_c)| = 1/sqrt(2) at each critical frequency."""
    from scipy.signal import sosfreqz

    if order > 10:
        return
    nyq = 0.5 * float(sampling_rate)
    freqs = np.atleast_1d(np.asarray(critical_freq, dtype=np.float64))
    # P: away from 0 and Nyquist so that the float64 sos stays well-conditioned
    if np.any(freqs / nyq < 1e-3) or np.any(freqs / nyq > 1 - 1e-3):
        return
    # P: a band-pass whose edges are closer than 0.1 % of the cut-off is a
    # numerically degenerate design (sos coefficients lose the edge separation)
    if freqs.size > 1 and np.ptp(freqs) < 1e-3 * freqs.max():
        return
    _, h = sosfreqz(sos, worN=freqs, fs=sampling_rate)
    gain = np.abs(h)
    trigger_if(np.any(np.abs(gain - 1.0 / np.sqrt(2.0)) > 1e-6), "NL-SIG-004")


@_safe
def end_clean():
    """Called at the end of ``clean``: drop the remembered raw magnitudes."""
    _CLEAN_MX["mx"] = None


@_guarded
def check_cosine_drift(drift, frame_times, high_pass):
    """NL-SIG-006 (orthonormal basis) and NL-SIG-007 (cut-off frequency)."""
    d = np.asarray(drift, dtype=np.float64)
    ft = np.asarray(frame_times, dtype=np.float64)
    n = len(ft)
    if d.ndim != 2 or d.shape[0] != n or n < 2 or not _finite(d):
        return
    m = d.shape[1] - 1  # non-constant columns
    if m >= 1:
        b = d[:, :m]
        gram = b.T @ b
        tol = _C * _EPS * (2 * np.pi * m + 1)
        bad = np.max(np.abs(gram - np.eye(m))) > tol
        bad = bad or np.max(np.abs(b.sum(axis=0))) > tol
        trigger_if(bad, "NL-SIG-006")
    dts = np.diff(ft)
    dt = float(np.median(dts))
    if dt <= 0 or np.max(np.abs(dts - dt)) > 1e-6 * dt or high_pass is None:
        return
    if high_pass <= 0:
        return
    # the relative margin absorbs the floor() boundary at integer 2 n dt high_pass
    # and the ambiguity of dt on a slightly jittered grid (the library uses the
    # mean step, the median is used here)
    margin = 1e-9 + 4.0 * float(np.max(np.abs(dts - dt))) / dt
    f = lambda k: k / (2.0 * n * dt)  # noqa: E731
    too_high = m >= 1 and f(m) > high_pass * (1 + margin)
    too_low = m < n - 1 and f(m + 1) <= high_pass * (1 - margin)
    trigger_if(too_high or too_low, "NL-SIG-007")


@_guarded
def check_compcor(u, s, ix_, n_confounds, n_samples, mx_in=None, n_kept=1):
    """NL-SIG-008: components lie in the span of detrended series."""
    u = np.asarray(u)
    s = np.asarray(s, dtype=np.float64)
    if u.ndim != 2 or not _finite(u) or n_samples < 3:
        return
    ss = s[ix_]
    smax = ss[0]
    if smax <= 0 or not _scale_ok(np.sqrt(smax)):
        return
    k = min(n_confounds, u.shape[1])
    r = np.arange(n_samples, dtype=np.float64)
    r -= r.mean()
    r /= np.sqrt((r**2).sum())
    eps = _eps_of(u)
    # the detrended series carry eps * max|x_in| of rounding noise per entry, which
    # leaks into the constant/ramp modes with weight ||E||_F / ||S||_F, where
    # ||S||_F^2 = n * trace(S S^T / n) = n * sum(eigenvalues)
    fro = np.sqrt(n_samples * max(float(np.sum(np.maximum(s, 0.0))), 1e-300))
    leak = 1.0 + (0.0 if mx_in is None else mx_in * np.sqrt(n_samples * n_kept) / fro)
    bad = False
    for j in range(k):
        if ss[j] < 1e-8 * smax:
            continue  # beyond the numerical rank: eigenvector arbitrary
        gaps = []
        if j > 0:
            gaps.append(ss[j - 1] - ss[j])
        gaps.append(ss[j] - ss[j + 1] if j + 1 < len(ss) else ss[j])
        gap = min(gaps)
        if gap < 1e-6 * smax:
            continue  # eigenvector conditioning (P)
        tol = _C * eps * np.sqrt(n_samples) * (smax / gap) * leak
        uj = u[:, j].astype(np.float64)
        if abs(uj.sum()) > tol or abs(r @ uj) > tol:
            bad = True
    trigger_if(bad, "NL-SIG-008")


# ===================================================================== GLM
def _svd_condition(x, max_kappa=1e8):
    """(kappa_eff, ok) of a design, over singular values above the pinv cut."""
    n, p = x.shape
    s = np.linalg.svd(x, compute_uv=False)
    if s.size == 0 or s[0] <= 0:
        return None
    cutoff = max(n, p) * _EPS * s[0]
    # P: no singular value in the numerical-rank ambiguity band
    if np.any((s > cutoff * 1e-2) & (s < cutoff * 1e2)):
        return None
    kappa = s[0] / s[s > cutoff].min()
    if kappa > max_kappa:
        return None
    return kappa


@_guarded
def check_ols_normal_equations(xw, wy, wresid):
    """NL-GLM-001: X_w^T r_w = 0."""
    eps = _eps_in(xw, wy, wresid)
    x, y, r = _f64(xw), _f64(wy), _f64(wresid)
    if x.ndim != 2 or not (_finite(x) and _finite(y) and _finite(r)):
        return
    if not _scale_ok(x, y):
        return
    n, p = x.shape
    kappa = _svd_condition(x)
    if kappa is None:
        return
    y2, r2 = y.reshape(n, -1), r.reshape(n, -1)
    g = np.abs(x.T @ r2).max(axis=0)
    tol = _C * eps * (n + p) * kappa * np.linalg.norm(x) * np.linalg.norm(y2, axis=0)
    trigger_if(np.any(g > tol), "NL-GLM-001")


@_guarded
def check_r_square(model, xw, wy, wresid, r2):
    """NL-GLM-002: R^2 = 1 - SSE / (n var(Y)) when the constant is modelled."""
    rho = getattr(model, "rho", None)
    if rho is not None and np.any(np.asarray(rho) != 0):
        return
    x, y, r = _f64(xw), _f64(wy), _f64(wresid)
    r2 = np.atleast_1d(_f64(r2)).ravel()
    if x.ndim != 2 or not (_finite(x) and _finite(y) and _finite(r)):
        return
    if not _scale_ok(x, y):
        return
    n = x.shape[0]
    kappa = _svd_condition(x)
    if kappa is None:
        return
    ones = np.ones(n)
    beta = np.linalg.lstsq(x, ones, rcond=None)[0]
    if np.linalg.norm(x @ beta - ones) > _C * _EPS * kappa * np.sqrt(n):
        return  # P: no constant in the column space
    y2, r2d = y.reshape(n, -1), r.reshape(n, -1)
    idx = np.unique(np.linspace(0, y2.shape[1] - 1, min(50, y2.shape[1])).astype(int))
    var = y2[:, idx].var(axis=0)
    ok = var > 0
    if not ok.any():
        return
    sse = (r2d[:, idx] ** 2).sum(axis=0)
    mu = y2[:, idx].mean(axis=0)
    ref = 1.0 - sse / (n * np.where(ok, var, 1.0))
    # prediction error is eps * kappa * ||y|| (backward-stable LS), i.e. relative to
    # the spread eps * kappa * sqrt(1 + mean^2/var): the design conditioning and the
    # data offset multiply. Factor 4: SSE and var each accumulate that error.
    # the pinv is formed in the design's dtype: its round-off enters the same way
    eps = _eps_in(xw, wy, wresid)
    tol = 4 * _C * eps * (1.0 + kappa * np.sqrt(1.0 + mu**2 / np.where(ok, var, 1.0)))
    # P: the tolerance is smaller than the range of R^2 (a spread that is rounding
    # noise of a huge offset leaves R^2 undetermined)
    ok = ok & (tol < 0.5)
    if not ok.any():
        return
    got = r2[idx]
    bad = np.any(ok & (np.abs(got - ref) > tol))
    bad = bad or np.any(ok & ((got < -tol) | (got > 1 + tol)))
    trigger_if(bad, "NL-GLM-002")


@_guarded
def check_zscore_tails(pvalue, one_minus_pvalue, z):
    """NL-GLM-003: the z-score inverts the tail probabilities it came from."""
    from scipy.stats import norm

    eps = _eps_in(pvalue, one_minus_pvalue, z)
    p = np.atleast_1d(_f64(pvalue)).ravel()
    z = np.atleast_1d(_f64(z)).ravel()
    if p.shape != z.shape:
        return
    ok = (p > 1e-300) & (p < 1 - 1e-16) & np.isfinite(z)
    omp = None
    if one_minus_pvalue is not None:
        omp = np.atleast_1d(_f64(one_minus_pvalue)).ravel()
        if omp.shape != p.shape:
            return
        ok &= (omp > 1e-300) & (omp < 1 - 1e-16)
    if not ok.any():
        return
    zz, pp = z[ok], p[ok]
    f = _C * eps * (1.0 + zz**2)
    pos = zz >= 0
    bad = np.any(np.abs(norm.sf(zz[pos]) - pp[pos]) > f[pos] * pp[pos])
    neg = ~pos
    if neg.any():
        if omp is not None:
            oo = omp[ok][neg]
            bad = bad or np.any(np.abs(norm.cdf(zz[neg]) - oo) > f[neg] * oo)
        else:
            bad = bad or np.any(np.abs(norm.sf(zz[neg]) - pp[neg]) > _C * eps)
    trigger_if(bad, "NL-GLM-003")


def _clone_contrast(con, effect=None, stat_type=None, dim=None):
    from nilearn.glm.contrasts import Contrast

    return Contrast(
        effect=con.effect if effect is None else effect,
        variance=con.variance,
        dim=con.dim if dim is None else dim,
        dof=con.dof,
        stat_type=con.stat_type if stat_type is None else stat_type,
        tiny=con.tiny,
        dofmax=con.dofmax,
    )


@_guarded
def check_t_f_equivalence(con, baseline, p_t):
    """NL-GLM-004: a 1-D F contrast is t^2; p_F is the two-sided t p-value."""
    if con.stat_type != "t" or con.dim != 1:
        return
    e = np.asarray(con.effect)
    if e.ndim == 2 and e.shape[0] != 1:
        return
    eps = _eps_in(e, con.variance, p_t)
    e2 = np.atleast_2d(_f64(e))
    cf = _clone_contrast(con, effect=e2, stat_type="F", dim=1)
    pf = np.asarray(cf.p_value(baseline), dtype=np.float64).ravel()
    t = np.sqrt(np.asarray(cf.stat_, dtype=np.float64).ravel())
    pt = np.asarray(p_t, dtype=np.float64).ravel()
    if pf.shape != pt.shape:
        return
    # two-sided p = 2 sf(|t|). The stored one-sided p = sf(t) is only resolved
    # on the right tail: for t < 0 it rounds to 1 - eps-level, so use it only
    # where 1 - p keeps three digits (cancellation, P).
    two = 2.0 * np.minimum(pt, 1.0 - pt)
    ok = np.isfinite(pf) & np.isfinite(two) & (pf > 1e-290)
    ok &= (pt <= 0.5) | ((1.0 - pt) >= 1e-3)
    if not ok.any():
        return
    # F and t tails go through different incomplete-beta paths; their relative
    # error grows like eps * t^2 in the far tail (conditioning of the tail
    # probability w.r.t. the statistic), plus the eps-level absolute error of 1-p
    tol = _C * eps * (1.0 + t**2) * pf + 2 * eps
    trigger_if(np.any(ok & (np.abs(pf - two) > tol)), "NL-GLM-004")


@_guarded
def check_contrast_invariance(labels, results, con_val, stat_type, con):
    """NL-GLM-005 (t) and NL-GLM-006 (F) invariances of compute_contrast."""
    from nilearn.glm._utils import pad_contrast
    from nilearn.glm.contrasts import compute_contrast

    cv0 = np.asarray(con_val)
    cv = np.asarray(con_val, dtype=np.float64)
    # round-off of the fit: its design / result dtype (the contrast keeps it)
    eps = _eps_in(con.effect, con.variance, cv0)
    if stat_type == "t":
        if cv.ndim != 1:
            return
        base = _clone_contrast(con).stat()
        v0 = np.asarray(con.variance, dtype=np.float64)
        e0 = np.asarray(con.effect, dtype=np.float64)
        ok = np.isfinite(base) & (v0.ravel() > 1e-40)
        if not ok.any():
            return
        bad = False
        for a in (2.0, -1.0):
            c2 = compute_contrast(labels, results, a * cv0, "t")
            s2 = _clone_contrast(c2).stat()
            scale = np.maximum(np.abs(base), 1.0)
            bad = bad or np.any(
                ok & (np.abs(s2 - np.sign(a) * base) > 8 * eps * scale)
            )
            e2 = np.asarray(c2.effect, dtype=np.float64)
            bad = bad or np.any(
                ok & (np.abs(e2 - a * e0).ravel() > 8 * eps * np.abs(a * e0).ravel())
            )
            v2 = np.asarray(c2.variance, dtype=np.float64)
            bad = bad or np.any(
                ok & (np.abs(v2 - a * a * v0).ravel() > 8 * eps * a * a * v0.ravel())
            )
        trigger_if(bad, "NL-GLM-005")
    elif stat_type == "F":
        if cv.ndim != 2 or cv.shape[0] < 2 or np.iscomplexobj(con.effect):
            return
        kappa = 1.0
        for lab in list(results)[:10]:
            reg = results[lab]
            cc = pad_contrast(con_val=cv, theta=reg.theta, stat_type="F")
            m = np.atleast_2d(reg.vcov(matrix=cc, dispersion=1.0))
            k = np.linalg.cond(m)
            if not np.isfinite(k):
                return
            kappa = max(kappa, k)
        if kappa > 1e8 or _C * eps * kappa > 0.1:
            return  # P: kappa where the F statistic still has digits
        base = _clone_contrast(con).stat()
        v0 = np.asarray(con.variance, dtype=np.float64).ravel()
        ok = np.isfinite(base) & (v0 > 1e-40)
        cv2 = cv.copy()
        cv2[0] *= 2.0
        bad = False
        for cm in (cv[::-1].copy(), cv2):
            c2 = compute_contrast(labels, results, cm, "F")
            if np.iscomplexobj(c2.effect):
                return
            s2 = _clone_contrast(c2).stat()
            tol = _C * eps * kappa * np.maximum(np.abs(base), 1.0)
            bad = bad or np.any(ok & (np.abs(s2 - base) > tol))
        trigger_if(bad, "NL-GLM-006")


@_guarded
def check_fixed_effects(contrasts, variances, precision_weighted, fx_con, fx_var):
    """NL-GLM-007: pooled effect/variance stay inside the physical bounds."""
    eps = _eps_in(contrasts, variances, fx_con, fx_var)
    c = _f64(contrasts)
    v = np.maximum(_f64(variances), 1e-16)
    fc, fv = _f64(fx_con), _f64(fx_var)
    if not (_finite(c) and _finite(v) and _finite(fc) and _finite(fv)):
        return
    n = c.shape[0]
    # P: voxels whose contrasts are not (sub)normal-scale noise
    cmax = np.abs(c).max(axis=0)
    keep = (cmax == 0) | ((cmax >= 1e-100) & (cmax <= 1e100))
    if not keep.any():
        return
    c, fc, fv, v = c[:, keep], fc[keep], fv[keep], v[:, keep]
    tol_e = _C * eps * n * np.abs(c).max(axis=0)
    bad = np.any(fc < c.min(axis=0) - tol_e) or np.any(fc > c.max(axis=0) + tol_e)
    vmin, vmax = v.min(axis=0), v.max(axis=0)
    upper = vmin if precision_weighted else vmax / n
    lower = vmin / n
    rel = _C * eps * n
    bad = bad or np.any(fv > upper * (1 + rel)) or np.any(fv < lower * (1 - rel))
    trigger_if(bad, "NL-GLM-007")


@_guarded
def check_zscore_odd(con, baseline, z):
    """NL-GLM-008: z(-effect) = -z(effect) for a t contrast."""
    if con.stat_type != "t":
        return
    eps = _eps_in(con.effect, con.variance, z)
    z = np.asarray(z, dtype=np.float64).ravel()
    c2 = _clone_contrast(con, effect=-np.asarray(con.effect, dtype=np.float64))
    z2 = np.asarray(c2.z_score(-baseline), dtype=np.float64).ravel()
    if z.shape != z2.shape:
        return
    ok = np.isfinite(z) & np.isfinite(z2)
    tol = 1024 * eps * (1.0 + z**2)
    trigger_if(np.any(ok & (np.abs(z + z2) > tol)), "NL-GLM-008")


# ===================================================================== HRF
@_safe
def snap_copy(x):
    a = np.asarray(x)
    if a.size > _MAX_ELEMS:
        return None
    return a.astype(np.float64)


@_guarded
def check_orthogonalize(x0, out):
    """NL-HRF-001: orthogonalised columns are mutually orthogonal."""
    if x0 is None:
        return
    eps = _eps_in(out)
    x = _f64(out)
    x0 = _f64(x0)
    if x.ndim != 2 or x.shape != x0.shape or not _finite(x):
        return
    if not _scale_ok(x0):
        return
    n, k = x.shape
    norms = np.linalg.norm(x, axis=0)
    if norms.max() <= 0:
        return
    keep = norms > 1e-10 * norms.max()
    gram = x.T @ x
    n0 = np.linalg.norm(x0, axis=0)
    bad = False
    for i in range(1, k):
        prev = [j for j in range(i) if keep[j]]
        if not keep[i] or not prev:
            continue
        nj = norms[prev]
        kappa = nj.max() / nj.min()
        tol = _C * eps * n * n0[i] * nj * kappa
        bad = bad or np.any(np.abs(gram[i, prev]) > tol)
    trigger_if(bad, "NL-HRF-001")


@_guarded
def check_bold_linearity(
    exp_condition,
    hrf_model,
    frame_times,
    con_id,
    oversampling,
    fir_delays,
    min_onset,
    reg,
):
    """NL-HRF-002 (superposition) and NL-HRF-003 (homogeneity)."""
    from nilearn.glm.first_level import hemodynamic_models as hm

    reg = _f64(reg)
    if reg.ndim != 2 or not _finite(reg):
        return
    onsets, durations, values = (np.asarray(a) for a in exp_condition)
    vals = _f64(values)
    ft = np.asarray(frame_times)
    if not _scale_ok(vals):
        return

    def call(on, du, va):
        return _f64(
            hm.compute_regressor(
                (on, du, va),
                hrf_model,
                ft,
                con_id=con_id,
                oversampling=oversampling,
                fir_delays=fir_delays,
                min_onset=min_onset,
            )[0]
        )

    mx = np.abs(reg).max()
    s = np.linalg.svd(reg, compute_uv=False) if mx > 0 else None
    if s is not None and s[-1] > 0 and s[0] / s[-1] <= 1e8:
        kappa = s[0] / s[-1]
        tol = _C * _EPS * (1.0 + kappa) * mx
        bad = False
        for a in (2.0, -1.0):
            bad = bad or np.max(np.abs(call(onsets, durations, a * vals) - a * reg)) > tol
        trigger_if(bad, "NL-HRF-003")
    if (reg.shape[1] == 1 or hrf_model == "fir") and len(vals) >= 2:
        t_r = hm._calculate_tr(ft)
        kernels = hm._hrf_kernel(hrf_model, t_r, oversampling, fir_delays)
        sumh = max(float(np.abs(k).sum()) for k in kernels)
        len_h = max(len(k) for k in kernels)
        n_hr = int(hm._compute_n_frames_high_res(ft, float(min_onset), oversampling))
        even = np.arange(len(vals)) % 2 == 0
        ra = call(onsets[even], durations[even], vals[even])
        rb = call(onsets[~even], durations[~even], vals[~even])
        # cumulative sum + convolution lengths, scaled by the summed amplitudes
        tol = _C * _EPS * (n_hr + len_h) * np.abs(vals).sum() * sumh
        trigger_if(np.max(np.abs(ra + rb - reg)) > tol, "NL-HRF-002")


@_guarded
def check_hrf_derivative(func, t_r, oversampling, time_length, onset, dt, d):
    """NL-HRF-004: the derivative kernel crosses zero at the HRF peak."""
    h = _f64(func(t_r, oversampling, time_length, onset))
    d = _f64(d)
    n = h.size
    if n < 4 or d.shape != h.shape or not (_finite(h) and _finite(d)):
        return
    if h.max() <= 0:
        return
    # P: the kernel holds the whole response. The kernel is sum-normalised, so a
    # kernel truncated inside the response is renormalised differently once it is
    # shifted and the finite difference no longer is the derivative of h.
    if abs(h[-1]) > 1e-2 * h.max() or abs(h[0]) > 1e-2 * h.max():
        return
    # P: the kernel resolves the response peak (at least three samples above half
    # maximum). On an undersampled kernel the per-sample sum normalisation
    # depends on the onset and adds a term proportional to h to the difference.
    if np.count_nonzero(h >= 0.5 * h.max()) < 3:
        return
    delta_grid = time_length / (n - 1)
    kp = int(np.argmax(h))
    if kp >= n - 2:
        return
    pos = np.where(d > 0)[0]
    if pos.size == 0:
        trigger_if(True, "NL-HRF-004")
        return
    k0 = pos[0]
    neg = np.where(d[k0:] <= 0)[0]
    if neg.size == 0:
        trigger_if(True, "NL-HRF-004")
        return
    kc = k0 + neg[0]
    trigger_if(abs(kc - kp) * delta_grid > dt + 2 * delta_grid, "NL-HRF-004")


# ===================================================================== THR
@_guarded
def check_fdr_control(stats, threshold, alpha, two_sided):
    """NL-THR-001: the estimated FDR of the selected set is at most alpha."""
    from scipy.stats import norm

    if not np.isfinite(threshold):
        return
    st = _f64(stats)
    if st.size < 1 or not _finite(st):
        return
    r = int(np.sum(st >= threshold))
    if r == 0:
        return
    f = 2.0 if two_sided else 1.0
    est = f * st.size * norm.sf(threshold) / r
    # tolerance: the -1e-12 shift of the threshold times |u| <= 40, plus rounding
    trigger_if(est > alpha * (1 + 1e-9), "NL-THR-001")


@_guarded
def check_multiple_comparison(
    stat_img, mask_img, alpha, two_sided, stats, method, threshold
):
    """NL-THR-002: discoveries are ordered bonferroni <= fdr <= fpr."""
    from nilearn.glm.thresholding import threshold_stats_img

    thr = {method: float(threshold)}
    for m in ("fpr", "fdr", "bonferroni"):
        if m in thr:
            continue
        _, t = threshold_stats_img(
            stat_img,
            mask_img=mask_img,
            alpha=alpha,
            height_control=m,
            cluster_threshold=0,
            two_sided=two_sided,
        )
        thr[m] = float(t)
    st = _f64(stats)
    if not _finite(st):
        return

    def delta(u):
        return 1e-9 * (1.0 + abs(u)) if np.isfinite(u) else 0.0

    def strict(u):
        return int(np.sum(st > u + delta(u)))

    def loose(u):
        return int(np.sum(st >= u - delta(u)))

    bad = strict(thr["bonferroni"]) > loose(thr["fdr"])
    bad = bad or strict(thr["fdr"]) > loose(thr["fpr"])
    trigger_if(bad, "NL-THR-002")


@_guarded
def check_cluster_extent(pre, out, k):
    """NL-THR-003: cluster-extent thresholding removes exactly small clusters."""
    from scipy.ndimage import generate_binary_structure, label

    pre, out = np.asarray(pre), np.asarray(out)
    if pre.ndim != 4 or pre.shape != out.shape or k <= 0:
        return
    st = generate_binary_structure(3, 1)  # 6-connectivity, as documented
    bad = False
    for v in range(pre.shape[3]):
        p, o = pre[..., v], out[..., v]
        # (the library filters in the output's dtype, the snapshot may be float)
        if o.dtype.kind == "i" and bool((p == np.iinfo(o.dtype).min).any()):
            continue  # P: -x is not representable at the signed-integer minimum
        keep = np.zeros(p.shape, dtype=bool)
        for sign in (1, -1):
            lab, nl = label((p * sign) > 0, st)
            if nl == 0:
                continue
            sizes = np.bincount(lab.ravel())[1:]
            big = np.where(sizes >= k)[0] + 1
            keep |= np.isin(lab, big)
        bad = bad or not np.array_equal(o, np.where(keep, p, 0))
    trigger_if(bad, "NL-THR-003")


# ================================================================ connectome
def _pow2_scale(p):
    """Exact (power-of-two) positive diagonal scaling used as transform."""
    return 2.0 ** ((np.arange(p) % 5) - 2)


@_guarded
def check_corr_scale_invariance(cov, corr):
    """NL-CON-001: correlation is invariant to rescaling the variables."""
    from nilearn.connectome.connectivity_matrices import cov_to_corr

    c = np.asarray(cov)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not _finite(c):
        return
    if np.any(np.diag(c) <= 0):
        return
    if not _scale_ok(np.diag(c), lo=1e-150, hi=1e150):
        return
    dt = c.dtype if c.dtype.kind == "f" else np.float64
    # P: the 2^k transform (factor 16) neither overflows nor reaches the
    # subnormal range of the matrix dtype
    ac = np.abs(c.astype(np.float64))
    nz = ac[ac > 0]
    if nz.size and (
        ac.max() * 16.0 > 0.5 * float(np.finfo(dt).max)
        or nz.min() / 16.0 < 1e4 * float(np.finfo(dt).tiny)
    ):
        return
    d = _pow2_scale(c.shape[0]).astype(dt)
    scaled = (c.astype(dt) * d[:, None]) * d[None, :]
    corr2 = np.asarray(cov_to_corr(scaled))
    diff = np.max(np.abs(corr2.astype(np.float64) - np.asarray(corr, dtype=np.float64)))
    trigger_if(diff > _C * _eps_of(corr2), "NL-CON-001")


@_guarded
def check_connectivity_psd(conns, kind):
    """NL-CON-002: covariance/correlation PSD, precision SPD, all symmetric."""
    arr = np.asarray(conns)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2] or not _finite(arr):
        return
    eps = _eps_of(arr)
    bad = False
    for m in arr[:5]:
        m64 = _f64(m)
        p = m64.shape[0]
        scale = np.max(np.abs(m64))
        if scale == 0:
            continue
        kappa = 1.0
        if kind == "precision":
            # an inverse carries eps * cond(cov) of rounding (P: cond <= 1e10)
            kappa = np.linalg.cond(m64)
            if not np.isfinite(kappa) or kappa > 1e10:
                continue
        tol = _C * eps * p * kappa * scale
        bad = bad or np.max(np.abs(m64 - m64.T)) > tol
        w = np.linalg.eigvalsh((m64 + m64.T) / 2.0)
        bad = bad or w[0] < -tol
    trigger_if(bad, "NL-CON-002")


@_guarded
def check_partial_correlation(covs, pcs):
    """NL-CON-003: partial correlation = correlation of regression residuals."""
    rng = np.random.default_rng(12345)  # local RNG: global state untouched
    bad = False
    for s, pc in list(zip(covs, pcs, strict=False))[:3]:
        eps = _eps_in(s, pc)
        s = _f64(s)
        pc = _f64(pc)
        if not _scale_ok(np.diag(s), lo=1e-150, hi=1e150):
            continue
        p = s.shape[0]
        if p < 2 or not (_finite(s) and _finite(pc)):
            continue
        kappa = np.linalg.cond(s)
        if not np.isfinite(kappa) or kappa > 1e8:
            continue
        if p <= 5:
            pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
        else:
            pairs = []
            while len(pairs) < 10:
                i, j = sorted(rng.choice(p, size=2, replace=False))
                pairs.append((int(i), int(j)))
        tol = _C * eps * kappa * np.sqrt(p)
        for i, j in pairs:
            rest = [k for k in range(p) if k not in (i, j)]
            sub = s[np.ix_([i, j], [i, j])]
            if rest:
                a = s[np.ix_([i, j], rest)]
                b = s[np.ix_(rest, rest)]
                sub = sub - a @ np.linalg.solve(b, a.T)
            rho = sub[0, 1] / np.sqrt(sub[0, 0] * sub[1, 1])
            bad = bad or abs(rho - pc[i, j]) > tol
    trigger_if(bad, "NL-CON-003")


@_guarded
def check_vec_isometry_to_vec(sym, vec, fn):
    """NL-CON-004: sym_matrix_to_vec is a scaled isometry."""
    s, v = _f64(sym), _f64(vec)
    if s.ndim < 2 or s.shape[-1] != s.shape[-2] or not (_finite(s) and _finite(v)):
        return
    p = s.shape[-1]
    if not _scale_ok(s):
        return
    if np.max(np.abs(s - np.swapaxes(s, -1, -2))) > _C * _EPS * max(np.abs(s).max(), 1e-300):
        return  # P: symmetric input only
    pp = max(p, 2)
    e0 = np.zeros((pp, pp))
    e0[0, 0] = 1.0
    e1 = np.zeros((pp, pp))
    e1[0, 1] = e1[1, 0] = 1.0
    r0 = np.sum(_f64(fn(e0)) ** 2) / np.sum(e0**2)
    r1 = np.sum(_f64(fn(e1)) ** 2) / np.sum(e1**2)
    bad = abs(r0 - r1) > _C * _EPS * max(r0, r1)
    sv = s.reshape(-1, p, p)
    vv = v.reshape(sv.shape[0], -1)
    den = (sv**2).sum(axis=(1, 2))
    nz = den > 0
    if nz.any():
        r = (vv[nz] ** 2).sum(axis=1) / den[nz]
        bad = bad or np.any(np.abs(r - r0) > _C * _EPS * p * p * r0)
    trigger_if(bad, "NL-CON-004")


@_guarded
def check_vec_isometry_to_sym(vec, sym, fn):
    """NL-CON-005: vec_to_sym_matrix is a scaled isometry."""
    v, s = _f64(vec), _f64(sym)
    if s.ndim < 2 or not (_finite(s) and _finite(v)):
        return
    p = s.shape[-1]
    n = v.shape[-1]
    if p < 2 or not _scale_ok(v):
        return
    ed = np.zeros(n)
    ed[0] = 1.0  # entry (0, 0)
    eo = np.zeros(n)
    eo[1] = 1.0  # entry (1, 0)
    r0 = np.sum(ed**2) / np.sum(_f64(fn(ed)) ** 2)
    r1 = np.sum(eo**2) / np.sum(_f64(fn(eo)) ** 2)
    bad = abs(r0 - r1) > _C * _EPS * max(r0, r1)
    sv = s.reshape(-1, p, p)
    vv = v.reshape(sv.shape[0], -1)
    den = (sv**2).sum(axis=(1, 2))
    nz = den > 0
    if nz.any():
        r = (vv[nz] ** 2).sum(axis=1) / den[nz]
        bad = bad or np.any(np.abs(r - r0) > _C * _EPS * p * p * r0)
    trigger_if(bad, "NL-CON-005")


@_guarded
def check_frechet_equivariance(matrices, init, max_iter, tol, gmean, converged):
    """NL-CON-006: the Frechet mean is congruence-equivariant."""
    from nilearn.connectome.connectivity_matrices import _geometric_mean

    if tol is None or not converged:
        return
    mats = np.asarray(matrices, dtype=np.float64)
    if mats.ndim != 3 or not _finite(mats) or mats.shape[0] > 20:
        return
    kappa = max(np.linalg.cond(m) for m in mats)
    if not np.isfinite(kappa) or kappa > 1e6:
        return
    p = mats.shape[1]
    d = _pow2_scale(p)
    dd = d[:, None] * d[None, :]
    scaled = mats * dd[None, :, :]
    init2 = None if init is None else np.asarray(init, dtype=np.float64) * dd
    g2 = _geometric_mean(scaled, init=init2, max_iter=max_iter, tol=tol)
    g = _f64(gmean)
    back = g2 / dd
    rel = np.linalg.norm(back - g) / max(np.linalg.norm(g), 1e-300)
    # both runs stop within the algorithm's own tolerance; plus rounding
    tol_rel = 4 * tol * g.size + _C * _EPS * kappa * p * max_iter
    trigger_if(rel > tol_rel, "NL-CON-006")


@_guarded
def check_tangent_centering(tangent):
    """NL-CON-007: tangent vectors average to zero at the Frechet mean."""
    if _STATE["nonconverged"]:
        # P: the Frechet mean iteration reached max_iter (nilearn warns): the
        # tangent vectors are centred only at a converged mean
        _STATE["nonconverged"] = False
        return
    t = np.asarray(tangent, dtype=np.float64)
    if t.ndim != 3 or t.shape[0] < 2 or not _finite(t):
        return
    m = np.linalg.norm(t.mean(axis=0))
    # ten times the convergence tolerance used to fit the mean
    trigger_if(m / t.shape[1] ** 2 > 1e-6, "NL-CON-007")


@_guarded
def check_group_sparse(omega, emp_covs, n_samples, alpha, precisions_init):
    """NL-CON-008 (empty graph above alpha_max) and NL-CON-009 (SPD)."""
    from nilearn.connectome.group_sparse_cov import compute_alpha_max

    om = _f64(omega)
    if om.ndim != 3 or not _finite(om):
        return
    p, _, k = om.shape
    bad = False
    for i in range(k):
        m = om[..., i]
        scale = max(np.abs(m).max(), 1e-300)
        bad = bad or np.max(np.abs(m - m.T)) > _C * _EPS * scale
        # definiteness is scale free: test the unit-diagonal rescaling D^-1/2 M D^-1/2
        # (P: legitimate precisions of variables in very different units have a
        # condition number beyond 1/(C eps p), though they are exactly SPD)
        d = np.diagonal(m)
        if not np.all(d > 0):
            bad = True
            continue
        dm = 1.0 / np.sqrt(d)
        wn = np.linalg.eigvalsh((m + m.T) / 2.0 * np.outer(dm, dm))
        # P: when the unit-diagonal empirical covariance is itself numerically
        # singular, the smallest eigenvalue of its inverse is unresolved:
        # only genuine indefiniteness is then an alarm
        thr = _C * _EPS * p * wn[-1]
        ec = _f64(emp_covs)
        singular_in = False
        if ec.ndim == 3 and ec.shape[:2] == (p, p) and ec.shape[2] == k:
            de = np.diagonal(ec[..., i])
            if not (np.all(de > 0) and _finite(ec[..., i])):
                singular_in = True
            else:
                de = 1.0 / np.sqrt(de)
                we = np.linalg.eigvalsh(ec[..., i] * np.outer(de, de))
                singular_in = we[0] <= _C * _EPS * p * we[-1]
        bad = bad or (wn[0] < -thr if singular_in else wn[0] <= thr)
    trigger_if(bad, "NL-CON-009")
    if precisions_init is None:
        amax = compute_alpha_max(_f64(emp_covs), np.asarray(n_samples, dtype=np.float64))[0]
        if alpha >= 1.001 * amax:
            off = ~np.eye(p, dtype=bool)
            diag = max(np.abs(np.diagonal(om)).max(), 1e-300)
            trigger_if(np.abs(om[off]).max() > _C * _EPS * diag, "NL-CON-008")


# ================================================================ mass-univ.
@_guarded
def check_tfce(arr4d, bin_struct, E, H, dh, two_sided, out):
    """NL-MU-002 (sign symmetry) and NL-MU-003 (homogeneity of degree H)."""
    from nilearn.mass_univariate._utils import calculate_tfce

    if not sampled("tfce"):
        return
    a, o = np.asarray(arr4d, dtype=np.float64), np.asarray(out, dtype=np.float64)
    if a.size > _MAX_ELEMS or not _finite(a):
        return
    mx = np.abs(o).max()
    # each voxel sums up to 2 * 1000 terms: accumulation-order rounding
    slack = _C * _EPS * 2000
    if two_sided:
        neg = calculate_tfce(-a, bin_struct, E=E, H=H, dh=dh, two_sided_test=True)
        trigger_if(np.max(np.abs(neg + o)) > slack * mx, "NL-MU-002")
    if dh == "auto" and mx > 0:
        dbl = calculate_tfce(2.0 * a, bin_struct, E=E, H=H, dh=dh, two_sided_test=two_sided)
        want = (2.0**H) * o
        trigger_if(np.max(np.abs(dbl - want)) > slack * np.abs(want).max(), "NL-MU-003")


@_guarded
def check_null_to_p(test_values, p, alternative):
    """NL-MU-004: p-values never increase with the evidence."""
    if test_values is None:
        return
    t = np.atleast_1d(np.asarray(test_values, dtype=np.float64)).ravel()
    p = np.atleast_1d(np.asarray(p, dtype=np.float64)).ravel()
    if t.shape != p.shape or t.size < 2:
        return
    ev = np.abs(t) if alternative == "two-sided" else (t if alternative == "larger" else -t)
    ok = np.isfinite(ev) & np.isfinite(p)
    order = np.argsort(ev[ok], kind="stable")
    ps, es = p[ok][order], ev[ok][order]
    bad = np.any(ps[1:] > ps[:-1]) or np.any((es[1:] == es[:-1]) & (ps[1:] != ps[:-1]))
    trigger_if(bad, "NL-MU-004")


@_guarded
def check_cluster_measures(arr4d, threshold, bin_struct, two_sided, sizes, masses):
    """NL-MU-005: cluster measures are symmetric in the sign of a 2-sided map."""
    from nilearn.mass_univariate._utils import calculate_cluster_measures

    if not two_sided or not sampled("cluster_measures"):
        return
    a = np.asarray(arr4d, dtype=np.float64)
    if a.size > _MAX_ELEMS or not _finite(a):
        return
    s2, m2 = calculate_cluster_measures(-a, threshold, bin_struct, two_sided_test=True)
    m0 = np.asarray(masses, dtype=np.float64)
    bad = np.any(np.asarray(sizes) != np.asarray(s2))
    bad = bad or np.any(np.abs(np.asarray(m2) - m0) > _C * _EPS * np.maximum(np.abs(m0), 1e-300) * a.size)
    trigger_if(bad, "NL-MU-005")


# ===================================================================== image
@_guarded
def check_smoothing(x_in, out, affine, fwhm, smooth_fn, copy_flag):
    """NL-IMG-001 (conservation) and NL-IMG-002 (FWHM adds in quadrature)."""
    if x_in is None or fwhm is None or isinstance(fwhm, str):
        return
    a = np.asarray(x_in)
    o = np.asarray(out)
    if a.dtype.kind == "b" or o.dtype.kind == "b":
        return  # P: numeric intensities (bool is not an image dtype)
    if a.shape != o.shape or a.ndim not in (3, 4) or not (_finite(a) and _finite(o)):
        return
    fw = np.asarray([fwhm]).ravel()
    fw = np.asarray([0.0 if e is None else e for e in fw], dtype=np.float64)
    aff = np.asarray(affine, dtype=np.float64)[:3, :3]
    vox = np.sqrt(np.sum(aff**2, axis=0))
    sigma = fw / (np.sqrt(8 * np.log(2)) * vox)
    sigma = np.broadcast_to(sigma, (3,))
    if not np.any(sigma > 0):
        return
    eps = _eps_of(o)
    a64, o64 = a.astype(np.float64), o.astype(np.float64)
    if not _scale_ok(a64):
        return
    ksum = sum(2 * int(4.0 * s + 0.5) + 1 for s in sigma if s > 0)
    # sum: every output element carries kernel_len * eps of its input magnitude
    tol_sum = _C * eps * (ksum + np.log2(max(a.size, 2))) * np.abs(a64).sum()
    rng = float(a64.max() - a64.min())
    tol_ext = _C * eps * max(rng, np.abs(a64).max())
    bad = abs(o64.sum() - a64.sum()) > tol_sum
    bad = bad or o64.min() < a64.min() - tol_ext or o64.max() > a64.max() + tol_ext
    trigger_if(bad, "NL-IMG-001")
    # NL-IMG-002: P: every smoothed axis keeps sigma_vox / sqrt(2) >= 1
    if np.all(sigma[sigma > 0] / np.sqrt(2.0) >= 1.0) and a.size <= 2_000_000:
        half = fw / np.sqrt(2.0)
        once = smooth_fn(a, affine, half, ensure_finite=False, copy=True)
        twice = smooth_fn(once, affine, half, ensure_finite=False, copy=True)
        # kernels truncated at 4 sigma lose 6.3e-5 per pass; sampling aliasing < 3e-9
        # truncation scales with the range, rounding with the magnitude (offset)
        tol = 4 * 3 * 6.3e-5 * rng + _C * eps * ksum * np.abs(a64).max()
        diff = np.max(np.abs(_f64(twice) - o64))
        trigger_if(diff > tol, "NL-IMG-002")


@_guarded
def check_resample_geometry(data, affine, out, out_affine, interpolation, fill_value, clip):
    """NL-IMG-003: output voxels hold the input value at the same world point."""
    from scipy.ndimage import map_coordinates

    d, o = np.asarray(data), np.asarray(out)
    if d.ndim not in (3, 4) or o.ndim != d.ndim:
        return
    vin = d[..., 0] if d.ndim == 4 else d
    vout = o[..., 0] if o.ndim == 4 else o
    if min(vin.shape) < 4 or vin.size > _MAX_ELEMS or vout.size > _MAX_ELEMS:
        return
    vin = vin.astype(np.float64)
    if not _finite(vin):
        return
    order = {"continuous": 3, "linear": 1, "nearest": 0}[interpolation]
    rng = np.random.default_rng(0)
    ns = min(200, vout.size)
    ijk = np.stack([rng.integers(0, s, ns) for s in vout.shape])
    homog = np.vstack([ijk, np.ones(ns)])
    src = np.linalg.solve(
        np.asarray(affine, dtype=np.float64),
        np.asarray(out_affine, dtype=np.float64) @ homog,
    )[:3]
    shp = np.array(vin.shape)[:, None]
    inside = np.all((src >= 1) & (src <= shp - 2), axis=0)
    # float64 round-off of inv(A_in) @ A_out @ v grows with the world-coordinate
    # magnitude (translations and the extent of the output grid)
    ai = np.asarray(affine, dtype=np.float64)
    ao = np.asarray(out_affine, dtype=np.float64)
    world = (
        np.abs(ai[:3, 3]).max()
        + np.abs(ao[:3, 3]).max()
        + np.abs(ao[:3, :3]).sum(1).max() * max(vout.shape)
    )
    dx = _C * _EPS * np.linalg.norm(np.linalg.inv(ai[:3, :3]), np.inf) * world
    if order == 0:
        inside &= np.all(np.abs(src - np.floor(src) - 0.5) > max(1e-6, dx), axis=0)
    if not inside.any():
        return
    exp = map_coordinates(vin, src[:, inside], order=order, mode="constant", cval=fill_value)
    if clip:
        exp = np.clip(exp, min(np.nanmin(d), 0), max(np.nanmax(d), 0))
    if o.dtype.kind in "iu":
        info = np.iinfo(o.dtype)  # P: the cast saturates cubic over/undershoot
        exp = np.clip(exp, info.min, info.max)
    got = vout[tuple(ijk[:, inside])].astype(np.float64)
    eps = max(_eps_of(o), _EPS)
    # cubic-spline conditioning: a few ulps amplified by ~8
    tol = _C * eps * 8 * max(np.abs(vin).max(), 1e-300)
    if o.dtype.kind in "iu":
        tol += 1.0  # P: integer output is quantised (rounded/truncated) by the cast
    # source-coordinate error dx (voxels) times the largest per-voxel change
    grad = max(np.abs(np.diff(vin, axis=a)).max() for a in range(3))
    tol += dx * grad * 3 * (2 if order == 3 else 1)
    trigger_if(np.any(np.abs(got - exp) > tol), "NL-IMG-003")


@_guarded
def check_labels_signal_roundtrip(signals, result_img, labels_img, mask_img, background_label):
    """NL-IMG-004: region means of an image built from signals are the signals."""
    from nilearn.regions.signal_extraction import img_to_signals_labels

    s = np.asarray(signals)
    if s.ndim != 2 or not _finite(s):
        return
    s2, _labels, _ = img_to_signals_labels(
        result_img, labels_img, mask_img=mask_img, background_label=background_label
    )
    if s2.shape != s.shape:
        return  # P: every label survives the mask
    eps = max(_eps_of(s2), _eps_of(s)) if s.dtype.kind == "f" else _eps_of(s2)
    diff = np.max(np.abs(_f64(s2) - _f64(s)))
    # ndimage.mean accumulates the region sum sequentially: error ~ n_voxels * eps
    from nilearn import image as nl_image

    lab = np.asarray(nl_image.get_data(nl_image.check_niimg_3d(labels_img)))
    n_max = int(np.bincount(np.abs(lab.astype(np.int64)).ravel()).max())
    trigger_if(
        diff > _C * eps * n_max * max(np.abs(s).max(), 1e-300), "NL-IMG-004"
    )


@_guarded
def check_maps_signal_consistency(region_signals, result_img, maps_img, mask_img):
    """NL-IMG-005: least-squares region signals recover the generating signals."""
    from nilearn import image as nl_image
    from nilearn.regions.signal_extraction import _trim_maps, img_to_signals_maps

    s = _f64(region_signals)
    if s.ndim != 2 or not _finite(s) or not _scale_ok(s):
        return
    s2, _ = img_to_signals_maps(result_img, maps_img, mask_img=mask_img)
    if s2.shape != s.shape:
        return
    maps_raw = nl_image.get_data(nl_image.check_niimg_4d(maps_img))
    # the least-squares products run in the promoted dtype of signals and maps
    eps = _eps_in(region_signals, maps_raw, nl_image.get_data(result_img))
    rd = np.asarray(nl_image.get_data(result_img)).dtype
    if rd.kind in "iu":
        mabs = np.abs(_f64(maps_raw)).reshape(-1, s.shape[1])
        if (np.abs(s) @ mabs.T).max() > np.iinfo(rd).max:
            return  # P: the integer product fits its dtype (no wraparound)
    maps = _f64(maps_raw)
    if mask_img is not None:
        mk = np.asarray(nl_image.get_data(nl_image.check_niimg_3d(mask_img)))
        maps, mk2, _ = _trim_maps(maps, mk, keep_empty=True)
        m = maps[np.asarray(mk2, dtype=bool), :]
    else:
        m = maps.reshape(-1, maps.shape[-1])
    if m.size > 5_000_000:
        return
    sv = np.linalg.svd(m, compute_uv=False)
    if sv[-1] <= 0 or sv[0] / sv[-1] > 1e6:
        return  # P: full column rank, kappa <= 1e6
    kappa = sv[0] / sv[-1]
    diff = np.max(np.abs(_f64(s2) - s))
    trigger_if(diff > _C * eps * kappa * np.linalg.norm(s), "NL-IMG-005")
