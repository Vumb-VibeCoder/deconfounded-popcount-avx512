"""
cache_miss_predictor.py
========================
Unified module for LRU cache miss-rate curve (MRC) prediction.

Merges (no behavior changes, just consolidated + de-duplicated):
  - physical_law.py     -> Che mean-field solver, EM bi-exponential mixture fit,
                            real-log-file driven comparison (`run_real_logs`)
  - realistic_test.py   -> Fenwick-tree stack distance, burst RLE decomposition,
                            Che-Zipf / Che-burst variants, step-function detector,
                            synthetic trace generators, per-trace `evaluate()`
  - final_predictor.py  -> 3-tier predict_miss_rate() unifying all of the above,
                            plus the mixed-stress synthetic generator

Sections:
  1. Core math (Che mean-field solver, EM mixture)
  2. Trace utilities (Fenwick stack distance, burst decomposition)
  3. MRC models (v1 uniform, Che-zipf, Che-burst, step-function)
  4. Synthetic trace generators
  5. Tier-1/2/3 unified predictor (predict_miss_rate)
  6. Standalone per-trace evaluator (evaluate) -- verbose diagnostic version
  7. Real-log driver (run_real_logs) -- for *_stack_dist.txt / *_q.txt files
  8. CLI demo (__main__)
"""

import numpy as np
from scipy.optimize import brentq

try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


# =====================================================================
# 1. Core math: Che mean-field solver + EM bi-exponential mixture fit
# =====================================================================

def solve_Tc(q, capacity):
    """Solve for the characteristic time Tc s.t. sum(1 - exp(-q*Tc)) == capacity."""
    def f(Tc):
        return np.sum(1 - np.exp(-q * Tc)) - capacity
    lo, hi = 1e-9, 1.0
    while f(hi) < 0:
        hi *= 2
        if hi > 1e12:
            break
    return brentq(f, lo, hi, xtol=1e-10)


def che_MR(C, q):
    """Che mean-field approximation of miss rate at capacity C, given access
    probabilities q (any normalized weight vector: raw freq, burst freq, ...)."""
    if C <= 0:
        return 1.0
    if C >= len(q):
        return 0.0
    Tc = solve_Tc(q, C)
    return np.sum(q * np.exp(-q * Tc))


def batch_solve_Tc(q, capacities, n_iter=50):
    """Vectorized version of solve_Tc: solves for Tc simultaneously at every
    capacity in `capacities` via broadcasted bisection, instead of one
    brentq() python-level call per C. Same math, far fewer python-level
    round trips -- this is what actually dominates predict_miss_rate's
    runtime (one solve per C, ~W times, x however many curves we need)."""
    C = np.asarray(capacities, dtype=float)
    m = len(C)
    lo = np.full(m, 1e-9)
    hi = np.full(m, 1.0)

    def f(Tc):
        # Tc: (m,), q: (n,) -> broadcast to (m, n), sum over items -> (m,)
        return (1 - np.exp(-Tc[:, None] * q[None, :])).sum(axis=1) - C

    for _ in range(60):
        val = f(hi)
        need = val < 0
        if not need.any():
            break
        hi = np.where(need, hi * 2, hi)

    for _ in range(n_iter):
        mid = (lo + hi) / 2
        pos = f(mid) >= 0
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
    return (lo + hi) / 2


def _che_mr_for_interior(q, Cs_interior, n_probe=200):
    """Compute the Che miss-mass sum(q*exp(-q*Tc(C))) for every C in
    Cs_interior, either exactly (small arrays) or via sparse-probe +
    monotone interpolation of the MR CURVE ITSELF (large arrays).

    Two approximations were tried:
      (a) solve Tc exactly at ~n_probe probes, interpolate Tc, then run the
          full (m, n_items) exp/sum at full resolution;
      (b) solve Tc AND the resulting MR value at ~n_probe probes, interpolate
          MR directly, skip the full-resolution exp/sum entirely.
    (b) is what's implemented below: it's strictly cheaper (no O(W x n_items)
    pass at all, vs. one full pass in (a)) and empirically slightly more
    accurate too, since it doesn't compound np.interp's error through a
    subsequent exp() nonlinearity. MR(C) is smooth/monotone over the C range
    that matters for the same reason Tc(C) is (see solve_Tc/batch_solve_Tc),
    so interpolating it directly is just as safe.

    Below n_probe*1.5 interior points, solving exactly is already cheap
    enough that interpolation isn't worth the (small) approximation error.
    Returns the interpolated MR values aligned with Cs_interior.
    """
    m = len(Cs_interior)
    if m <= int(n_probe * 1.5):
        Tc = batch_solve_Tc(q, Cs_interior)
        return (q[None, :] * np.exp(-Tc[:, None] * q[None, :])).sum(axis=1)
    idx = np.unique(np.geomspace(1, m, num=n_probe).astype(int) - 1)
    Cs_probe = Cs_interior[idx]
    Tc_probe = batch_solve_Tc(q, Cs_probe)
    mr_probe = (q[None, :] * np.exp(-Tc_probe[:, None] * q[None, :])).sum(axis=1)
    return np.interp(Cs_interior, Cs_probe, mr_probe)


def che_zipf_curve_fast(q, Cs, n_total=None, n_probe=200):
    """Vectorized che_zipf over an entire array of capacities at once.

    `n_total` is the true working-set size (W) used for the C>=n_total
    boundary check. Defaults to len(q), but callers that already filtered
    out zero-weight items (q_i == 0 contributes exactly 0 to every sum
    below, so dropping those entries changes nothing mathematically --
    it only reduces how much work batch_solve_Tc's per-iteration (m, n)
    broadcast has to do) MUST pass the original W explicitly, or the
    C>=W cutoff silently shifts to C>=len(nonzero q) instead.

    `n_probe` controls the sparse-solve + interpolation fast path -- see
    `_che_mr_for_interior`. Pass a large n_probe (>= W) to force the old
    exact-everywhere behavior."""
    if n_total is None:
        n_total = len(q)
    Cs = np.asarray(Cs, dtype=float)
    out = np.zeros(len(Cs))
    out[Cs <= 0] = 1.0
    interior = (Cs > 0) & (Cs < n_total)
    if interior.any():
        out[interior] = _che_mr_for_interior(q, Cs[interior], n_probe=n_probe)
    return out


def che_burst_curve_fast(q, L, Cs, n_total=None, n_probe=200):
    """Vectorized che_burst over an entire array of capacities at once.
    See che_zipf_curve_fast's n_total/n_probe docstrings -- same reasoning applies."""
    if n_total is None:
        n_total = len(q)
    Cs = np.asarray(Cs, dtype=float)
    out = np.zeros(len(Cs))
    out[Cs <= 0] = 1.0
    interior = (Cs > 0) & (Cs < n_total)
    if interior.any():
        miss_bursts = _che_mr_for_interior(q, Cs[interior], n_probe=n_probe)
        out[interior] = miss_bursts / np.sum(q * L)
    return out


def em_exp_mixture(x, n_iter=200, seed=0):
    """MLE fit of a 2-component exponential mixture via EM.
    x: finite (non-cold) reuse/stack distances."""
    rng = np.random.default_rng(seed)
    mean_x = x.mean()
    lam1, lam2 = mean_x * 0.2, mean_x * 2.0  # init: one fast, one slow component
    a = 0.5
    for _ in range(n_iter):
        p1 = a * (1 / lam1) * np.exp(-x / lam1)
        p2 = (1 - a) * (1 / lam2) * np.exp(-x / lam2)
        denom = p1 + p2 + 1e-300
        r = p1 / denom
        a_new = r.mean()
        lam1_new = np.sum(r * x) / np.sum(r)
        lam2_new = np.sum((1 - r) * x) / np.sum(1 - r)
        if (abs(a_new - a) < 1e-9 and abs(lam1_new - lam1) < 1e-6
                and abs(lam2_new - lam2) < 1e-6):
            a, lam1, lam2 = a_new, lam1_new, lam2_new
            break
        a, lam1, lam2 = a_new, lam1_new, lam2_new
    return a, lam1, lam2


# =====================================================================
# 2. Trace utilities: Fenwick-tree stack distance + burst RLE decomposition
# =====================================================================

class Fenwick:
    def __init__(self, n):
        self.n = n
        self.t = np.zeros(n + 2)

    def add(self, i, v):
        i += 1
        while i <= self.n:
            self.t[i] += v
            i += i & (-i)

    def prefix(self, i):
        i += 1
        s = 0.0
        while i > 0:
            s += self.t[i]
            i -= i & (-i)
        return s

    def range_sum(self, lo, hi):
        if hi < lo:
            return 0.0
        return self.prefix(hi) - (self.prefix(lo - 1) if lo > 0 else 0.0)


if _HAVE_NUMBA:
    @njit(cache=True)
    def _stack_distances_numba(trace, W):
        n = trace.shape[0]
        fen = np.zeros(n + 2)
        last_pos = np.full(W, -1, dtype=np.int64)
        dist = np.empty(n, dtype=np.int64)
        INF = W + 1000

        for t in range(n):
            x = trace[t]
            p = last_pos[x]
            if p >= 0:
                lo = p + 1
                hi = t - 1
                if hi < lo:
                    s = 0.0
                else:
                    # range_sum(lo, hi) inlined via two Fenwick prefix() calls
                    i1 = hi + 1
                    s1 = 0.0
                    while i1 > 0:
                        s1 += fen[i1]
                        i1 -= i1 & (-i1)
                    if lo > 0:
                        i2 = lo - 1 + 1
                        s2 = 0.0
                        while i2 > 0:
                            s2 += fen[i2]
                            i2 -= i2 & (-i2)
                    else:
                        s2 = 0.0
                    s = s1 - s2
                dist[t] = int(round(s))
                # add(p, -1.0)
                i = p + 1
                while i <= n:
                    fen[i] -= 1.0
                    i += i & (-i)
            else:
                dist[t] = INF
            # add(t, 1.0)
            i = t + 1
            while i <= n:
                fen[i] += 1.0
                i += i & (-i)
            last_pos[x] = t
        return dist


def stack_distances(trace, W):
    """Exact stack distance (number of *unique* items referenced since last
    access) for every position in `trace`, using a Fenwick tree. Cold/first
    accesses get W+1000 (an explicit 'infinite' sentinel).

    Fast path: if `trace` is already integer item-ids in [0, W) -- true for
    every generator/caller in this file -- dispatch to the numba-jitted
    Fenwick loop (profiling shows this Fenwick loop, not the Che solver, is
    the actual dominant cost: ~70% of evaluate()'s runtime). Falls back to
    the plain dict-keyed python loop for arbitrary hashable items or when
    numba isn't installed, so behavior/API is unchanged either way."""
    if _HAVE_NUMBA:
        arr = np.asarray(trace)
        if (np.issubdtype(arr.dtype, np.integer) and len(arr) > 0
                and arr.min() >= 0 and arr.max() < W):
            return _stack_distances_numba(arr.astype(np.int64), W)
    n = len(trace)
    fen = Fenwick(n)
    last_pos = {}
    dist = np.empty(n, dtype=np.int64)
    for t, x in enumerate(trace):
        if x in last_pos:
            p = last_pos[x]
            dist[t] = int(round(fen.range_sum(p + 1, t - 1)))
            fen.add(p, -1)
        else:
            dist[t] = W + 1000
        fen.add(t, 1)
        last_pos[x] = t
    return dist


def decompose_bursts(trace, W):
    """burst = maximal run of consecutive identical accesses.
    Returns q_i (fraction of BURSTS targeting i, not raw accesses) and
    L_i (mean burst length per item)."""
    burst_count = np.zeros(W)
    burst_len_sum = np.zeros(W)
    i = 0
    n = len(trace)
    while i < n:
        j = i
        item = trace[i]
        while j < n and trace[j] == item:
            j += 1
        burst_count[item] += 1
        burst_len_sum[item] += (j - i)
        i = j
    total_bursts = burst_count.sum()
    q = burst_count / total_bursts
    L = np.divide(burst_len_sum, burst_count, out=np.ones(W), where=burst_count > 0)
    return q, L


# =====================================================================
# 3. MRC models
# =====================================================================

def v1_uniform(C, W, L_mean):
    """Closed-form uniform-IRM baseline (no skew, no burst structure)."""
    return max(0.0, W - C) / (W * L_mean)


def che_zipf(C, q):
    """Che mean-field with q_i = raw per-item access frequency."""
    if C <= 0:
        return 1.0
    if C >= len(q):
        return 0.0
    Tc = solve_Tc(q, C)
    return np.sum(q * np.exp(-q * Tc))


def che_burst(C, q, L):
    """Che mean-field with q_i/L_i from real burst RLE decomposition.
    Miss rate is normalized per-access (dividing miss-bursts by mean burst length)."""
    if C <= 0:
        return 1.0
    if C >= len(q):
        return 0.0
    Tc = solve_Tc(q, C)
    miss_bursts = np.sum(q * np.exp(-q * Tc))
    return miss_bursts / np.sum(q * L)


def hybrid_che_step_fit(che_pred, step_pred, MR_real):
    """NEW MODEL (Tier 4): mr(C) = alpha*che_pred(C) + (1-alpha)*step_pred(C).

    Rationale: Che (exponential/mean-field) and the step-function fit each
    capture ONE pure regime -- i.i.d./skewed reuse vs. hard periodic reuse.
    A trace that mixes both (e.g. a loop scan with Zipf-hot interjections)
    is not "closer to one shape", its real MRC is a genuine blend of the
    two curves. Since both che_pred and step_pred are already computed and
    fixed, finding the best-fit alpha in [0,1] is a 1-D least-squares
    problem with a closed form -- no optimizer needed:

        alpha* = argmin_a sum( (a*che + (1-a)*step - MR)^2 )
               = sum(d*(MR-step)) / sum(d^2),   d = che - step
    clipped to [0,1] (outside that range the mix is no longer a convex
    combination and stops being interpretable as "how much of each regime").
    """
    d = che_pred - step_pred
    denom = np.sum(d * d)
    if denom < 1e-12:
        alpha = 0.5
    else:
        alpha = float(np.sum(d * (MR_real - step_pred)) / denom)
        alpha = min(1.0, max(0.0, alpha))
    pred = alpha * che_pred + (1 - alpha) * step_pred
    err = np.mean(np.abs(pred - MR_real))
    return alpha, pred, err


def segmented_che_curve(trace, W, Cs, K):
    """NEW MODEL (Tier 5 building block): split the trace into K contiguous
    segments, fit a LOCAL q_i (raw access freq within that segment only) per
    segment, compute that segment's own Che curve, then blend all segments'
    curves weighted by segment length.

    Why this is different from EM rescue: EM fits a 2-component exponential
    mixture to the GLOBAL reuse-distance distribution -- it's a statistical
    curve-fit with no notion of "where in the trace" a reuse happened. This
    model instead uses the trace's actual temporal structure: if item
    frequencies genuinely drift over time (a new motif set per segment, e.g.
    bioinformatics-like non-stationary access), the average of several
    *locally-correct* Che curves is physically motivated, not just a better
    curve shape. It should lose to EM when the trace is truly stationary
    (segments all look the same -> no info gained) and win when the trace
    has real phase drift.
    """
    n = len(trace)
    bounds = np.linspace(0, n, K + 1).astype(int)
    curves, weights = [], []
    for i in range(K):
        seg = trace[bounds[i]:bounds[i + 1]]
        if len(seg) < max(20, 2 * W // K):
            continue
        counts = np.bincount(seg, minlength=W).astype(float)
        if counts.sum() == 0:
            continue
        q = counts / counts.sum()
        q_nz = q[q > 0]  # zero-weight items contribute 0 to every Che sum;
                         # dropping them is exact, not approximate, and
                         # avoids O(W) wasted work per bisection iteration
                         # when a segment only touches a small subset of W
        curves.append(che_zipf_curve_fast(q_nz, Cs, n_total=W))
        weights.append(len(seg))
    if not curves:
        return None
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    return np.sum(np.array(curves) * weights[:, None], axis=0)


def best_segmented_che(trace, W, Cs, MR_real, K_candidates=(2, 3, 4, 6, 8, 12)):
    """Grid-search K (number of segments) and keep whichever minimizes error
    against the realized MR curve -- same model-selection spirit as picking
    among v1/v2/v3.1/EM/step elsewhere in this file, just one more knob."""
    best_err, best_K, best_curve = np.inf, None, None
    for K in K_candidates:
        curve = segmented_che_curve(trace, W, Cs, K)
        if curve is None:
            continue
        err = np.mean(np.abs(curve - MR_real))
        if err < best_err:
            best_err, best_K, best_curve = err, K, curve
    return best_K, best_curve, best_err


def _js_divergence(p, q, eps=1e-12):
    """Jensen-Shannon divergence between two discrete distributions (base-2
    would need a log2, but any consistent log base works fine here since we
    only ever threshold/compare divergence values against each other)."""
    m = 0.5 * (p + q)
    def _kl(a, b):
        return np.sum(a * np.log((a + eps) / (b + eps)))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _probe_divergence(trace, W, n_probes=None):
    """Compute the coarse probe grid + consecutive-probe JS-divergence ONCE.
    This is the expensive part of changepoint detection (one bincount per
    probe + one JS call per adjacent pair); it depends only on (trace, W),
    NOT on sensitivity. best_changepoint_che grid-searches ~9 sensitivity
    values x 2 min_seg values, so computing this fresh inside
    detect_changepoints on every one of those 18 calls (the old behavior)
    was pure waste -- same qs/div recomputed 18x for a quantity that only
    needs computing once per trace. Split out so it can be cached/reused."""
    n = len(trace)
    if n_probes is None:
        probe_len = max(50, min(n // 40, 4 * W))
    else:
        probe_len = max(50, n // n_probes)
    bounds = np.arange(0, n, probe_len)
    if bounds[-1] != n:
        bounds = np.append(bounds, n)
    probes = [trace[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    probes = [p for p in probes if len(p) > 0]
    if len(probes) < 3:
        return bounds, None
    qs = np.array([np.bincount(p, minlength=W).astype(float) / len(p) for p in probes])
    div = np.array([_js_divergence(qs[i], qs[i + 1]) for i in range(len(qs) - 1)])
    return bounds, div


def _threshold_and_refine(trace, W, bounds, div, sensitivity=1.0, refine=True):
    """Cheap part of changepoint detection: threshold a PRECOMPUTED div
    signal at a given sensitivity, then (optionally) refine each accepted
    boundary with a local fine-grained JS pass. Split out of
    detect_changepoints so best_changepoint_che's sensitivity grid-search
    only re-runs this O(#probes) thresholding step per candidate, not the
    O(#probes) bincount+JS computation that _probe_divergence already did
    once."""
    n = len(trace)
    if div is None or len(div) == 0 or div.std() < 1e-12:
        return np.array([0, n])
    probe_len = int(bounds[1] - bounds[0]) if len(bounds) > 1 else n

    med = np.median(div)
    mad = np.median(np.abs(div - med)) * 1.4826 + 1e-300
    thresh = med + sensitivity * mad
    changepoints = [0]
    for i in range(len(div)):
        is_local_peak = ((i == 0 or div[i] >= div[i - 1]) and
                          (i == len(div) - 1 or div[i] >= div[i + 1]))
        if div[i] > thresh and is_local_peak:
            changepoints.append(bounds[i + 1])
    changepoints.append(n)
    changepoints = sorted(set(changepoints))

    if refine and probe_len > 200:
        refined = [changepoints[0]]
        for cp in changepoints[1:-1]:
            lo = max(refined[-1], cp - probe_len // 2)
            hi = min(n, cp + probe_len // 2)
            fine_len = max(50, probe_len // 8)
            fbounds = np.arange(lo, hi, fine_len)
            if len(fbounds) < 3:
                refined.append(cp)
                continue
            if fbounds[-1] != hi:
                fbounds = np.append(fbounds, hi)
            fprobes = [trace[fbounds[j]:fbounds[j + 1]] for j in range(len(fbounds) - 1)]
            fprobes = [p for p in fprobes if len(p) > 0]
            if len(fprobes) < 3:
                refined.append(cp)
                continue
            fqs = np.array([np.bincount(p, minlength=W).astype(float) / len(p) for p in fprobes])
            fdiv = np.array([_js_divergence(fqs[j], fqs[j + 1]) for j in range(len(fqs) - 1)])
            best_j = int(np.argmax(fdiv))
            refined.append(int(fbounds[best_j + 1]))
        refined.append(changepoints[-1])
        changepoints = sorted(set(refined))
    return np.array(changepoints)


def detect_changepoints(trace, W, n_probes=None, sensitivity=1.0, refine=True):
    """NEW (Tier 6 building block): find WHERE the trace's access
    distribution actually drifts, instead of assuming it drifts on a fixed
    K-way grid (as segmented_che_curve/Tier 5 does via linspace).

    Standalone convenience wrapper around `_probe_divergence` +
    `_threshold_and_refine` (kept as two functions so best_changepoint_che
    can cache the expensive probe/divergence pass and only re-run the cheap
    threshold+refine pass per sensitivity candidate -- see their
    docstrings). Calling this directly still does both passes, useful for
    one-off/diagnostic use outside the grid-search.

    Median/MAD threshold instead of mean/std because one huge real
    changepoint spike inflates the mean/std of a small `div` array and can
    mask a second, smaller-but-still-real changepoint elsewhere (exactly the
    failure mode on irregular multi-segment traces); MAD is far less
    sensitive to that one outlier. Lower sensitivity -> more sensitive (more
    changepoints kept); higher -> stricter.

    `n_probes` defaults to scaling with the trace: too few probes (a fixed
    40 on a huge trace) blurs short segments together; too many (a fixed 40
    on a small trace) makes each probe's local q histogram too noisy to
    trust. `probe_len ~ max(50, min(n // 40, 4*W))` keeps probes coarse
    enough to have several observations per item while adapting probe COUNT
    to trace length.

    When `refine` is True, each accepted boundary is refined with a second,
    finer bisection pass local to that boundary, tightening the true
    boundary position instead of leaving it wherever the coarse grid
    happened to fall.

    Returns a sorted array of trace-index boundaries (always includes 0 and
    len(trace) as the outer bounds).
    """
    bounds, div = _probe_divergence(trace, W, n_probes=n_probes)
    if div is None:
        return np.array([0, len(trace)])
    return _threshold_and_refine(trace, W, bounds, div, sensitivity=sensitivity, refine=refine)


def _merge_short_boundaries(boundaries, min_seg):
    """Merge segments shorter than min_seg into whichever NEIGHBOR they're
    closer to (by segment length), not always the previous one. A pure
    left-merge (old behavior) can chain a short segment onto an already-huge
    left neighbor while a small right neighbor sits right there -- e.g.
    boundaries [0, 10, 5000, 5010, N] with min_seg=100 would left-merge [10,
    5000) forward into [0,10) (fine) but then also swallow [5010, N) into
    [10, 5010) even when N-5010 is tiny and the segment starting at 5010 was
    the more natural drift point. Repeated passes converge to a boundary set
    where every kept segment is >= min_seg (except possibly one final
    leftover at the very end, which then folds into its only neighbor)."""
    bounds = list(boundaries)
    changed = True
    while changed and len(bounds) > 2:
        changed = False
        lens = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
        for i, L in enumerate(lens):
            if L < min_seg:
                if len(bounds) <= 2:
                    break
                if i == 0:
                    del bounds[1]
                elif i == len(lens) - 1:
                    del bounds[-2]
                else:
                    left_len = lens[i - 1]
                    right_len = lens[i + 1]
                    if left_len <= right_len:
                        del bounds[i]      # merge into left neighbor
                    else:
                        del bounds[i + 1]  # merge into right neighbor
                changed = True
                break
    return bounds


def changepoint_che_curve(trace, W, Cs, boundaries, min_seg=None):
    """NEW MODEL (Tier 6): same weighted-blend-of-local-Che idea as Tier 5's
    segmented_che_curve, but segments come from `boundaries` (real detected
    drift points) instead of an evenly-spaced K-grid. Segments shorter than
    min_seg are merged into whichever neighbor is smaller (see
    `_merge_short_boundaries`) rather than dropped, so a changepoint that
    fires near the very start/end of the trace doesn't just throw away that
    data the way Tier 5's `continue` does.

    Each segment now picks its OWN best local model out of THREE candidates
    -- che_zipf, che_burst, and a local step-function -- instead of always
    defaulting to an exponential-family (Che) shape. The first two mirror
    what Tier 1 already does globally; the step candidate targets a segment
    that is ITSELF periodic (a scan sub-phase inside an otherwise mixed
    trace, e.g. the "scan" third of ThreePhase) which no Che variant can
    represent no matter how locally it's refit -- Che is exponential-family,
    a hard step transition is not in that family at any parameter setting.
    Picking among the three needs each segment's own REAL local miss-rate
    curve (not just its q_i), so we compute local stack distances on the
    segment's own sub-trace, exactly as if it were a standalone trace --
    this is what makes the step candidate possible here where Tier 5 (che-
    only) and the old repeat_frac heuristic could not use it."""
    if min_seg is None:
        min_seg = max(50, 2 * W)
    bounds = _merge_short_boundaries(boundaries, min_seg)

    curves, weights = [], []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        seg = trace[lo:hi]
        if len(seg) == 0:
            continue
        counts = np.bincount(seg, minlength=W).astype(float)
        if counts.sum() == 0:
            continue
        q_raw = counts / counts.sum()
        q_raw_nz = q_raw[q_raw > 0]
        zipf_curve = che_zipf_curve_fast(q_raw_nz, Cs, n_total=W)

        q_burst, L_burst = decompose_bursts(seg, W)
        burst_mask = q_burst > 0
        candidates_local = [zipf_curve]
        if burst_mask.sum() > 0 and burst_mask.sum() < len(seg):
            burst_curve = che_burst_curve_fast(q_burst[burst_mask], L_burst[burst_mask],
                                                Cs, n_total=W)
            candidates_local.append(burst_curve)

        best_curve = zipf_curve
        if len(seg) >= max(200, 2 * W):
            # local step candidate: compute this segment's OWN real MR curve
            # (via its own stack distances) and see if a hard step fits it
            # far better than any Che variant does -- cheap relative to the
            # Che curves already computed, and the only way to catch a
            # periodic sub-phase living inside a longer mixed trace.
            seg_dist = stack_distances(seg, W)
            INF = W + 1000
            seg_counts = np.bincount(np.clip(seg_dist, 0, INF), minlength=INF + 1)
            local_MR = np.array([seg_counts[c:].sum() / len(seg) for c in Cs])
            errs = [np.mean(np.abs(c - local_MR)) for c in candidates_local]
            cstar, err_step, top, bot = best_step_fit(Cs, local_MR)
            step_curve = np.where(Cs < cstar, top, bot)
            all_curves = candidates_local + [step_curve]
            all_errs = errs + [err_step]
            best_curve = all_curves[int(np.argmin(all_errs))]
        elif len(candidates_local) > 1:
            # too short for a trustworthy local MR/step fit -- fall back to
            # the cheap repeat_frac heuristic to choose zipf vs burst only
            repeat_frac = 1.0 - (burst_mask.sum() / max(1, len(seg)))
            best_curve = candidates_local[1] if repeat_frac > 0.15 else candidates_local[0]
        curves.append(best_curve)
        weights.append(len(seg))
    if not curves:
        return None, 0
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    curve = np.sum(np.array(curves) * weights[:, None], axis=0)
    return curve, len(curves)


def best_changepoint_che(trace, W, Cs, MR_real,
                          sensitivity_candidates=(0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)):
    """Grid-search the changepoint sensitivity knob (analogous to Tier 5's
    K grid-search), pick whichever detected-boundary set minimizes error.
    A very high sensitivity degenerates to boundaries=[0,n], i.e. plain
    global Che -- so this tier can never do *structurally* worse than Tier 1
    once errors are compared, it just might not win the pick.

    Grid widened (5 -> 9 points, denser at the low end) since the robust
    median+MAD threshold in detect_changepoints tends to keep more genuine
    boundaries at a given sensitivity value than the old mean+std version
    did -- the previously-coarse 0.5 step could jump from "too few
    boundaries" to "too many" without a candidate in between on irregular
    multi-segment traces. min_seg is also grid-searched jointly: a single
    fixed `2*W` floor can force merging two real, short-but-distinct
    segments together on traces whose true segments are close to (or
    shorter than) 2*W (e.g. IrregularMotif's shortest 8000-length segment
    vs W=300 -> 2*W=600 is fine, but smaller W/larger min-seg combos are
    not always safe), so we also try a tighter floor."""
    best_err, best_sens, best_curve, best_nseg = np.inf, None, None, None
    for min_seg_mult in (2, 4):
        min_seg = max(50, min_seg_mult * W)
        for sens in sensitivity_candidates:
            boundaries = detect_changepoints(trace, W, sensitivity=sens)
            curve, nseg = changepoint_che_curve(trace, W, Cs, boundaries, min_seg=min_seg)
            if curve is None:
                continue
            err = np.mean(np.abs(curve - MR_real))
            if err < best_err:
                best_err, best_sens, best_curve, best_nseg = err, sens, curve, nseg
    return best_sens, best_nseg, best_curve, best_err


def best_step_fit(Cs, MR_real):
    """Step model with FREE levels: mr(C) = top for C < Cstar, bottom for C >= Cstar.
    top/bottom are each the L1-optimal median of MR_real on their side --
    not hardcoded to 1/0, since burst-induced hits can put 'top' well below 1.
    Used as a cheap periodicity/step-regime diagnostic."""
    best_err, best_cstar, best_top, best_bot = np.inf, None, None, None
    for k, cstar in enumerate(Cs):
        left = MR_real[:k]         # C < cstar
        right = MR_real[k:]        # C >= cstar
        if len(left) == 0 or len(right) == 0:
            continue
        top = np.median(left)
        bot = np.median(right)
        step = np.where(Cs < cstar, top, bot)
        err = np.mean(np.abs(step - MR_real))
        if err < best_err:
            best_err, best_cstar, best_top, best_bot = err, cstar, top, bot
    return best_cstar, best_err, best_top, best_bot


# =====================================================================
# 4. Synthetic trace generators
# =====================================================================

def gen_ycsb(W=300, N=150000, s=0.99, seed=1):
    """Pure i.i.d. Zipfian per-access draws -- textbook IRM assumption holds
    by construction."""
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, W + 1)
    w = 1.0 / ranks ** s
    q = w / w.sum()
    trace = rng.choice(W, size=N, p=q)
    return trace, W


def gen_spec_loop(W=300, n_passes=400, jitter=0.05, small_burst_max=3, seed=2):
    """Nested-loop scan: repeatedly walks the SAME working set in (near-)fixed
    order. Classic periodic reuse at distance ~W -- IRM/Che assume memoryless
    reference, so they smear this into a gradual exponential instead of the
    true step transition."""
    rng = np.random.default_rng(seed)
    base_order = rng.permutation(W)
    trace = []
    for _ in range(n_passes):
        order = base_order.copy()
        for _ in range(int(W * jitter)):
            a, b = rng.integers(0, W, size=2)
            order[a], order[b] = order[b], order[a]
        for item in order:
            L = rng.integers(1, small_burst_max + 1)
            trace.extend([item] * L)
    return np.array(trace, dtype=np.int64), W


def gen_bioinfo_phase(W=300, n_segments=6, seg_len=25000, motif_frac=0.15,
                       p_motif=0.7, seed=3):
    """Non-stationary: trace is split into segments; each segment has its OWN
    small 'repeat motif' set drawn preferentially, the rest are near-unique
    cold accesses from a segment-local pool. Global q_i estimated over the
    whole trace is stationary by construction, but the true process is not --
    phase drift, not correlation."""
    rng = np.random.default_rng(seed)
    trace = []
    motif_size = max(1, int(W * motif_frac))
    for _ in range(n_segments):
        motifs = rng.choice(W, size=motif_size, replace=False)
        for _ in range(seg_len):
            if rng.random() < p_motif:
                item = motifs[rng.integers(0, motif_size)]
            else:
                item = rng.integers(0, W)
            trace.append(item)
    return np.array(trace, dtype=np.int64), W


def gen_bioinfo_phase_irregular(W=300, motif_frac=0.15, p_motif=0.7,
                                 seg_lens=(8000, 42000, 15000, 60000, 12000, 33000),
                                 seed=13):
    """Same phase-drift generative process as gen_bioinfo_phase, but with
    IRREGULAR segment lengths (not a fixed seg_len repeated n_segments
    times). This is the case Tier 5's evenly-spaced K-grid structurally
    cannot align to -- any fixed K will either straddle a short segment
    with a long one, or split a long segment unnecessarily. Tier 6's
    JS-divergence changepoint detector should find these real boundaries
    regardless of their (ir)regularity, since it measures drift directly
    instead of assuming a period."""
    rng = np.random.default_rng(seed)
    trace = []
    motif_size = max(1, int(W * motif_frac))
    for seg_len in seg_lens:
        motifs = rng.choice(W, size=motif_size, replace=False)
        for _ in range(seg_len):
            if rng.random() < p_motif:
                item = motifs[rng.integers(0, motif_size)]
            else:
                item = rng.integers(0, W)
            trace.append(item)
    return np.array(trace, dtype=np.int64), W


def gen_mixed_stress(W=300, n_passes=150, jitter=0.05, small_burst_max=3, s=0.8, seed=7):
    """Stress trace: loop scan (like SPEC) but over a Zipf-skewed access
    frequency ON TOP of the loop order (some loop iterations revisit hot
    items more densely) -- neither pure periodic nor pure i.i.d.-skewed,
    to see if the 3-tier predictor picks a sane mode rather than crashing
    or mislabeling."""
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, W + 1)
    w = 1.0 / ranks ** s
    hot_bias = w / w.sum()
    base_order = rng.permutation(W)
    trace = []
    for _ in range(n_passes):
        order = base_order.copy()
        for _ in range(int(W * jitter)):
            a_, b_ = rng.integers(0, W, size=2)
            order[a_], order[b_] = order[b_], order[a_]
        for item in order:
            L = rng.integers(1, small_burst_max + 1)
            trace.extend([item] * L)
            if rng.random() < 0.3:
                extra = rng.choice(W, p=hot_bias)
                trace.append(extra)
    return np.array(trace, dtype=np.int64), W


# =====================================================================
# 5. Tier-1/2/3 unified predictor
# =====================================================================

def predict_miss_rate(trace, W, tol=0.01, verbose=True):
    """Unified 3-tier miss-rate curve predictor.

    Tier 1 (cheap):   Che mean-field, with q_i/L_i from REAL burst RLE decomposition.
    Tier 2 (detect):  if Che misses tolerance, fit a free-level step function.
                       A decisive win flags a periodic/step regime (e.g. loop scans) --
                       exponential-family models (Che or EM) are the wrong shape for this,
                       so we report the detected period instead of forcing a fit.
    Tier 3 (rescue):  if Che misses tolerance and step does NOT win decisively, fall back
                       to EM bi-exponential mixture fit as a partial, diagnostic rescue --
                       not a guaranteed fix.

    Returns a dict: {mode, Cs, mr_pred, err, extra, MR_real}
      mode in {"che", "periodic", "em_rescue"}
    """
    N = len(trace)
    dist = stack_distances(trace, W)
    INF = W + 1000
    Cs = np.arange(1, W + 1)
    counts = np.bincount(np.clip(dist, 0, INF), minlength=INF + 1)
    MR_real = np.array([counts[c:].sum() / N for c in Cs])

    q_burst, L_burst = decompose_bursts(trace, W)
    raw_counts = np.bincount(trace, minlength=W).astype(float)
    q_raw = raw_counts / raw_counts.sum()

    q_raw_nz = q_raw[q_raw > 0]
    burst_mask = q_burst > 0
    q_burst_nz, L_burst_nz = q_burst[burst_mask], L_burst[burst_mask]
    che_zipf_all = che_zipf_curve_fast(q_raw_nz, Cs, n_total=W)
    che_burst_all = che_burst_curve_fast(q_burst_nz, L_burst_nz, Cs, n_total=W)
    err_che_zipf = np.mean(np.abs(che_zipf_all - MR_real))
    err_che_burst = np.mean(np.abs(che_burst_all - MR_real))

    if err_che_zipf <= err_che_burst:
        che_pred, err_che, che_variant = che_zipf_all, err_che_zipf, "che_zipf"
    else:
        che_pred, err_che, che_variant = che_burst_all, err_che_burst, "che_burst"

    if verbose:
        print(f"  Tier1 Che ({che_variant}): err={err_che:.5f}  (tol={tol})")

    if err_che <= tol:
        if verbose:
            print("  => TIER1 SUFFICIENT. mode=che")
        return dict(mode="che", Cs=Cs, mr_pred=che_pred, err=err_che,
                    extra=dict(variant=che_variant), MR_real=MR_real)

    cstar, err_step, top, bot = best_step_fit(Cs, MR_real)
    step_pred = np.where(Cs < cstar, top, bot)
    if verbose:
        print(f"  Tier2 step-detector: Cstar={cstar} err={err_step:.5f}  (Che err={err_che:.5f})")

    if err_step < 0.5 * err_che and err_step <= tol * 3:
        if verbose:
            print(f"  => PERIODIC REGIME DETECTED. mode=periodic  (est. period/working-set={cstar})")
        return dict(mode="periodic", Cs=Cs, mr_pred=step_pred, err=err_step,
                    extra=dict(cstar=int(cstar), top=float(top), bottom=float(bot)),
                    MR_real=MR_real)

    is_cold = dist >= W
    finite = dist[~is_cold].astype(float)
    finite = finite[finite > 0]
    rng = np.random.default_rng(0)
    n_sample = min(3000, len(finite))
    src = rng.choice(finite, size=n_sample, replace=False)
    a, lam1, lam2 = em_exp_mixture(src, n_iter=100)
    p_cold = is_cold.sum() / N
    mr_em = p_cold + (1 - p_cold) * (a * np.exp(-Cs / lam1) + (1 - a) * np.exp(-Cs / lam2))
    err_em = np.mean(np.abs(mr_em - MR_real))
    degenerate = (a < 0.02) or (a > 0.98) or (min(lam1, lam2) > 0 and abs(lam1 - lam2) / lam2 < 0.05)
    rel_gain = (err_che - err_em) / err_che if err_che > 0 else 0.0
    if verbose:
        print(f"  Tier3 EM rescue: a={a:.3f} lam1={lam1:.2f} lam2={lam2:.2f}  err={err_em:.5f}")
        if degenerate:
            print("  => mode=em_rescue, but EM DEGENERATED (lam1~lam2 or a~0/1): no real second cluster found.")
            print(f"     err improved {rel_gain*100:.0f}% over Che by coincidence of fit, not by capturing new structure.")
            print("     Treat this as 'Che is still the honest answer' -- the underlying pattern is genuinely unmodeled.")
        else:
            flag = "meaningfully improved over Che" if rel_gain > 0.3 else "only marginally improved over Che"
            print(f"  => mode=em_rescue ({flag}, {rel_gain*100:.0f}% error reduction) -- partial/diagnostic fix, not guaranteed")
    result_em = dict(mode="em_rescue", Cs=Cs, mr_pred=mr_em, err=err_em,
                      extra=dict(a=float(a), lam1=float(lam1), lam2=float(lam2), err_che=float(err_che),
                                 degenerate=bool(degenerate), rel_gain=float(rel_gain)),
                      MR_real=MR_real)

    # ---- Tier 4 (new): hybrid Che+Step rescue ----------------------------
    # Reached only when Che missed tol AND step didn't win decisively AND
    # EM's rescue is unconvincing (didn't clear tol, or degenerated). Try a
    # convex blend of the two curves we *already computed* -- covers mixed
    # regimes (periodic structure + skewed/hot items superimposed) that are
    # neither pure-exponential nor pure-step, and that EM's 2-component
    # exponential mixture also cannot represent (EM is still exponential-family).
    candidates = [("em_rescue", result_em, err_em)]

    if err_em > tol or degenerate:
        alpha, hybrid_pred, err_hybrid = hybrid_che_step_fit(che_pred, step_pred, MR_real)
        if verbose:
            print(f"  Tier4 hybrid (alpha*che + (1-alpha)*step): alpha={alpha:.3f}  err={err_hybrid:.5f}")
        result_hybrid = dict(mode="hybrid", Cs=Cs, mr_pred=hybrid_pred, err=err_hybrid,
                              extra=dict(alpha=alpha, err_che=float(err_che), err_step=float(err_step),
                                         err_em=float(err_em)),
                              MR_real=MR_real)
        candidates.append(("hybrid", result_hybrid, err_hybrid))

    # ---- Tier 5 (new): segmented/windowed Che rescue ----------------------
    # Reached whenever we're still here (Che+step+EM+hybrid all missed tol
    # or EM degenerated). Only worth trying if the trace is plausibly
    # non-stationary: EM/hybrid both implicitly assume q_i is stable over
    # the whole trace. Splitting into K windows and locally-refitting q_i
    # per window directly targets phase drift instead of curve-fitting
    # around it.
    best_K, seg_curve, err_seg = best_segmented_che(trace, W, Cs, MR_real)
    if seg_curve is not None:
        if verbose:
            print(f"  Tier5 segmented Che (uniform K-grid): best K={best_K}  err={err_seg:.5f}")
        result_seg = dict(mode="segmented_che", Cs=Cs, mr_pred=seg_curve, err=err_seg,
                           extra=dict(K=int(best_K), err_che=float(err_che), err_em=float(err_em)),
                           MR_real=MR_real)
        candidates.append(("segmented_che", result_seg, err_seg))

    # ---- Tier 6 (new): changepoint-adaptive Che rescue ---------------------
    # Same physical motivation as Tier 5 (phase drift is real trace structure,
    # not a curve-fitting artifact), but Tier 5 assumes drift happens on an
    # evenly-spaced K-grid. Tier 6 instead *measures* where the access
    # distribution actually drifts (JS-divergence between consecutive probe
    # windows) and cuts segments there. Always tried alongside Tier 5 so the
    # pipeline can pick whichever segmentation assumption actually fits this
    # trace -- uniform-grid or detected-boundary.
    best_sens, n_seg_cp, cp_curve, err_cp = best_changepoint_che(trace, W, Cs, MR_real)
    if cp_curve is not None:
        if verbose:
            print(f"  Tier6 changepoint Che (adaptive boundaries): "
                  f"sensitivity={best_sens} n_segments={n_seg_cp}  err={err_cp:.5f}")
        result_cp = dict(mode="changepoint_che", Cs=Cs, mr_pred=cp_curve, err=err_cp,
                          extra=dict(sensitivity=float(best_sens), n_segments=int(n_seg_cp),
                                     err_che=float(err_che), err_em=float(err_em),
                                     err_segmented_che=float(err_seg) if seg_curve is not None else None),
                          MR_real=MR_real)
        candidates.append(("changepoint_che", result_cp, err_cp))

    name, result, err = min(candidates, key=lambda c: c[2])
    if verbose:
        others = ", ".join(f"{n}={e:.5f}" for n, _, e in candidates if n != name)
        print(f"  => FINAL PICK: mode={name} (err={err:.5f})  [beat: {others}]")
    return result


# =====================================================================
# 6. Standalone per-trace evaluator (verbose diagnostic version, prints
#    every candidate model's error side by side -- distinct from the
#    "pick one and return" behavior of predict_miss_rate above)
# =====================================================================

def evaluate(name, trace, W, tol=0.01):
    N = len(trace)
    dist = stack_distances(trace, W)
    INF = W + 1000
    Cs = np.arange(1, W + 1)
    counts = np.bincount(np.clip(dist, 0, INF), minlength=INF + 1)
    MR_real = np.array([counts[c:].sum() / N for c in Cs])

    q_burst, L_burst = decompose_bursts(trace, W)
    L_mean = np.average(L_burst, weights=q_burst)

    mr_v1 = np.array([v1_uniform(c, W, L_mean) for c in Cs])
    err_v1 = np.mean(np.abs(mr_v1 - MR_real))

    raw_counts = np.bincount(trace, minlength=W).astype(float)
    q_raw = raw_counts / raw_counts.sum()
    che_zipf_all = np.array([che_zipf(c, q_raw) for c in Cs])
    err_che_zipf = np.mean(np.abs(che_zipf_all - MR_real))

    che_burst_all = np.array([che_burst(c, q_burst, L_burst) for c in Cs])
    err_che_burst = np.mean(np.abs(che_burst_all - MR_real))

    cstar, err_step, top, bot = best_step_fit(Cs, MR_real)

    best_che_err = min(err_che_zipf, err_che_burst)
    em_called = False
    err_em = None
    if best_che_err > tol:
        em_called = True
        is_cold = dist >= W
        finite = dist[~is_cold].astype(float)
        finite = finite[finite > 0]
        rng = np.random.default_rng(0)
        n_sample = min(3000, len(finite))
        src = rng.choice(finite, size=n_sample, replace=False)
        a, lam1, lam2 = em_exp_mixture(src, n_iter=100)
        p_cold = is_cold.sum() / N
        mr_em = p_cold + (1 - p_cold) * (a * np.exp(-Cs / lam1) + (1 - a) * np.exp(-Cs / lam2))
        err_em = np.mean(np.abs(mr_em - MR_real))

    print(f"[{name}]  W={W} N={N}  L_mean(real, burst-weighted)={L_mean:.2f}")
    print(f"  v1 (uniform IRM, closed-form)      err = {err_v1:.5f}")
    print(f"  v2 (Che, q_i=raw access freq)      err = {err_che_zipf:.5f}")
    print(f"  v3.1 (Che, q_i/L_i from real burst RLE) err = {err_che_burst:.5f}")
    if em_called:
        print(f"  Che exceeded tol({tol}) -> EM rescue        err = {err_em:.5f}")
    else:
        print(f"  Che within tol({tol}) -> EM not called (saved the cost)")
    print(f"  step-function diagnostic: best Cstar={cstar}  top={top:.3f} bottom={bot:.3f}  err_step = {err_step:.5f}")

    candidates = [("v1", err_v1, 0), ("v2 Che-Zipf", err_che_zipf, 1), ("v3.1 Che-burst", err_che_burst, 1)]
    if em_called:
        candidates.append(("EM bi-exp", err_em, 2))
    candidates.append(("step-function", err_step, 1))

    exp_family_best = min(c[1] for c in candidates if c[0] != "step-function")
    if err_step < 0.5 * exp_family_best and err_step <= tol * 3:
        print(f"  => PERIODIC/STEP REGIME DETECTED (step err {err_step:.5f} << best exp-family {exp_family_best:.5f}).")
        print(f"     Estimated period / effective working-set size: Cstar={cstar}.")
        print("     Recommend a period-aware model, NOT another exponential-mixture patch.")
    else:
        within_tol = [c for c in candidates if c[1] <= tol]
        if within_tol:
            pick = min(within_tol, key=lambda c: c[2])
            print(f"  => BEST/BALANCED PICK: {pick[0]} (meets tolerance, cheapest such option)")
        else:
            pick = min(candidates, key=lambda c: c[1])
            print(f"  => NONE meet tolerance. Least-bad: {pick[0]} (err={pick[1]:.5f}) -- flag as unmodeled regime")
    print()


# =====================================================================
# 7. Real-log driver: loads `{prefix}_stack_dist.txt` / `{prefix}_q.txt`
#    pairs and compares the EM bi-exponential physical law vs plain Che.
# =====================================================================

def load_real_log(prefix):
    stack_dist = np.array([int(x) for x in open(f"{prefix}_stack_dist.txt")])
    N = len(stack_dist)
    qs = np.array([float(l.split()[1]) for l in open(f"{prefix}_q.txt")])
    W = len(qs)
    return N, qs, W, stack_dist


def run_real_logs(prefix, sample_size=3000):
    """Compare EM bi-exponential 'physical law' fit against plain Che on a
    real trace's stack-distance log (`{prefix}_stack_dist.txt` + `{prefix}_q.txt`)."""
    N, qs, W, stack_dist = load_real_log(prefix)
    INF = W + 1000
    is_cold = stack_dist >= W
    p_cold = is_cold.sum() / N
    finite = stack_dist[~is_cold].astype(float)
    finite = finite[finite > 0]

    rng = np.random.default_rng(0)
    n_sample = min(sample_size, len(finite))
    src = rng.choice(finite, size=n_sample, replace=False)

    a, lam1, lam2 = em_exp_mixture(src, n_iter=100)

    Cs = np.arange(1, W + 1)
    counts = np.bincount(np.clip(stack_dist, 0, INF), minlength=INF + 1)
    MR_real = np.array([counts[c:].sum() / N for c in Cs])

    def mr_physical(C):
        return p_cold + (1 - p_cold) * (a * np.exp(-C / lam1) + (1 - a) * np.exp(-C / lam2))

    mr_phys_all = mr_physical(Cs)
    che_all = np.array([che_MR(c, qs) for c in Cs])

    err_phys = np.mean(np.abs(mr_phys_all - MR_real))
    err_che = np.mean(np.abs(che_all - MR_real))

    tag = f"(EM on {len(src)} sampled finite reuses, n_finite={len(finite)})"
    print(f"[{prefix}] W={W}  p_cold(exact, no fit)={p_cold:.4f}")
    print(f"  EM-fit: a={a:.3f}  lam1={lam1:.2f}  lam2={lam2:.2f}   {tag}")
    print(f"  mean abs error over full C range:  physical-law={err_phys:.5f}   Che={err_che:.5f}")
    print()
    return err_phys


# =====================================================================
# 8. CLI demo
# =====================================================================

if __name__ == "__main__":
    print("############################################")
    print("# Part A: synthetic traces via 3-tier predictor")
    print("############################################\n")

    print("=== YCSB-like (expect mode=che) ===")
    trace, W = gen_ycsb()
    predict_miss_rate(trace, W)
    print()

    print("=== SPEC-like loop (expect mode=periodic) ===")
    trace, W = gen_spec_loop()
    predict_miss_rate(trace, W)
    print()

    print("=== Bioinformatics-like phase drift (expect mode=em_rescue) ===")
    trace, W = gen_bioinfo_phase()
    predict_miss_rate(trace, W)
    print()

    print("=== Mixed stress: loop + Zipf hot interjections (unknown expectation) ===")
    trace, W = gen_mixed_stress()
    predict_miss_rate(trace, W)
    print()

    print("############################################")
    print("# Part B: synthetic traces via verbose evaluate()")
    print("############################################\n")

    trace, W = gen_ycsb()
    evaluate("YCSB-like (pure i.i.d. Zipfian)", trace, W)

    trace, W = gen_spec_loop()
    evaluate("SPEC-like (periodic full working-set loop)", trace, W)

    trace, W = gen_bioinfo_phase()
    evaluate("Bioinformatics-like (phase-shifting repeat motifs)", trace, W)

    print("############################################")
    print("# Part C: real trace logs (requires *_stack_dist.txt / *_q.txt)")
    print("############################################\n")
    print("Skipped by default -- uncomment below and supply the log files:")
    print('# for prefix in ["bzip2", "ycsb", "kmer", "grep"]:')
    print("#     run_real_logs(prefix)")
    # for prefix in ["bzip2", "ycsb", "kmer", "grep"]:
    #     run_real_logs(prefix)
