"""
GPU engine for sequential indicator (co-)simulation.

Each realization is one CUDA thread (``cupy.RawKernel``) walking its own
random path, as in `geolattice.geostatistics.sgs_gpu`; every launch advances all
threads by a chunk of steps. At each node the thread

1. keeps the first ``num_points`` informed locations of the node's
   precomputed nearest-candidate list,
2. for every class / threshold ``k`` builds that class's indicator kriging
   system (its own variogram), adds the collocated soft indicator under the
   Markov–Bayes model when given, solves it by Gaussian elimination and gets
   ``p_k = m_k + sum w (i - m_k)``,
3. categorical: clips and renormalises the probabilities and draws a class;
   continuous: applies the GSLIB order-relation correction (``ordrel``) and
   draws a value with `sisim`'s ``beyond`` interpolation and tail models
   (linear / power / hyperbolic).

The host generates each realization's path and uniforms in exactly the
order `IndicatorCoSimulator`'s CPU engine consumes them, and the kernel uses
the same covariances, diagonal jitter and draw rules, so GPU realizations
equal the CPU ones up to floating-point rounding.

Limits: ``num_points <= 48``; at most 16 classes / thresholds; spherical,
exponential and Gaussian variograms.
"""

import numpy as np

from ..simulation.sgsim import _get_random_generator

__all__ = ["sis_simulate_gpu", "MAX_POINTS", "MAX_CLASSES"]

MAX_POINTS = 48
MAX_CLASSES = 16
_VTYPES = {"spherical": 0, "exponential": 1, "gaussian": 2}
_TAILS = {"linear": 0, "power": 1, "hyperbolic": 2}

_SIS_KERNEL_SRC = r"""
#define MAXP 48
#define MAXN (MAXP + 1)
#define MAXK 16

__device__ double cov(double dx, double dy, const double* R, double sill, double nug, int vt) {
    const double ex = dx * R[0] + dy * R[2];
    const double ey = dx * R[1] + dy * R[3];
    const double h = sqrt(ex * ex + ey * ey);
    const double s = sill - nug;
    if (vt == 0) return h <= 1.0 ? s * (1.0 - 1.5 * h + 0.5 * h * h * h) : 0.0;
    if (vt == 1) return s * exp(-3.0 * h);
    return s * exp(-3.0 * h * h);
}

__device__ double solve_dot(double* A, double* b, const double* resid, int N) {
    // Gaussian elimination with partial pivoting on A (N x N, row-major) and b;
    // returns resid . x where A x = b.
    double x[MAXN];
    for (int k = 0; k < N; ++k) {
        int p = k; double big = fabs(A[k * N + k]);
        for (int i = k + 1; i < N; ++i) if (fabs(A[i * N + k]) > big) { big = fabs(A[i * N + k]); p = i; }
        if (p != k) {
            for (int c = 0; c < N; ++c) { double t = A[k * N + c]; A[k * N + c] = A[p * N + c]; A[p * N + c] = t; }
            double t = b[k]; b[k] = b[p]; b[p] = t;
        }
        const double piv = A[k * N + k];
        if (fabs(piv) < 1e-300) continue;
        for (int i = k + 1; i < N; ++i) {
            const double f = A[i * N + k] / piv;
            if (f == 0.0) continue;
            for (int c = k; c < N; ++c) A[i * N + c] -= f * A[k * N + c];
            b[i] -= f * b[k];
        }
    }
    double dot = 0.0;
    for (int k = N - 1; k >= 0; --k) {
        double s = b[k];
        for (int c = k + 1; c < N; ++c) s -= A[k * N + c] * x[c];
        x[k] = fabs(A[k * N + k]) < 1e-300 ? 0.0 : s / A[k * N + k];
    }
    for (int k = 0; k < N; ++k) dot += resid[k] * x[k];
    return dot;
}

__device__ double powint(double xl, double xh, double yl, double yh, double xv, double pw) {
    if (xh - xl < 1e-20) return 0.5 * (yh + yl);
    return yl + (yh - yl) * pow((xv - xl) / (xh - xl), pw);
}

__device__ double beyond(double u, const double* cut, const double* cdf, int n, double zmin, double zmax,
                         int lt, double ltp, int mid, double mp, int ut, double utp) {
    double z;
    if (u <= cdf[0]) {
        z = powint(0.0, cdf[0], zmin, cut[0], u, lt == 1 ? 1.0 / ltp : 1.0);
    } else if (u >= cdf[n - 1]) {
        if (ut == 2) {
            const double lam = pow(cut[n - 1], utp) * (1.0 - cdf[n - 1]);
            double den = 1.0 - u; if (den < 1e-15) den = 1e-15;
            z = pow(lam / den, 1.0 / utp);
        } else {
            z = powint(cdf[n - 1], 1.0, cut[n - 1], zmax, u, ut == 1 ? 1.0 / utp : 1.0);
        }
    } else {
        int lo = -1;
        for (int i = 0; i < n; ++i) if (cdf[i] <= u) lo = i;     // searchsorted(side="right") - 1
        if (lo < 0) lo = 0; if (lo > n - 2) lo = n - 2;
        z = powint(cdf[lo], cdf[lo + 1], cut[lo], cut[lo + 1], u, mid == 1 ? 1.0 / mp : 1.0);
    }
    if (z < zmin) z = zmin; if (z > zmax) z = zmax;
    return z;
}

extern "C" __global__ void sis_run(
    double* V, const long long n_loc, const int* perms, const double* unif,
    const long long n_nodes, const int step0, const int nsteps,
    const long long* nbrs, const int m, const double* loc, const int n1,
    const long long* coincident, const int num_points,
    const int K, const double* classes, const int categorical,
    const double* sill, const double* nug, const int* vt, const double* R, const double* mean,
    const int use_soft, const double* soft, const double* B,
    const double zmin, const double zmax, const int lt, const double ltp, const int mid,
    const double mp, const int ut, const double utp, const int nsim, double* work)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nsim) return;
    double* Vr = V + (long long)r * n_loc;
    double* A = work + (long long)r * (MAXN * MAXN);       // global per-realization workspace
    const int* pr = perms + (long long)r * n_nodes;
    const double* ur = unif + (long long)r * n_nodes;
    const int stop = step0 + nsteps < n_nodes ? step0 + nsteps : (int)n_nodes;
    int idx[MAXP]; double b[MAXN], resid[MAXN], c0[MAXP], p[MAXK];

    for (int s = step0; s < stop; ++s) {
        const long long j = pr[s];
        const long long cj = coincident[j];
        if (cj >= 0) { Vr[n1 + j] = Vr[cj]; continue; }
        int n = 0;
        const long long* nb = nbrs + j * m;
        for (int t = 0; t < m && n < num_points; ++t) {
            const long long c = nb[t];
            if (c < 0) continue;
            if (!isnan(Vr[c])) idx[n++] = (int)c;
        }
        const double tx = loc[2 * (n1 + j)], ty = loc[2 * (n1 + j) + 1];
        for (int k = 0; k < K; ++k) {
            const double* Rk = R + 4 * k;
            const double sk = sill[k], nk = nug[k], mk = mean[k];
            const double sv = use_soft ? soft[j * K + k] : 0.0;
            const int us = use_soft && !isnan(sv) && B[k] > 1e-6;
            if (n == 0) { p[k] = us ? sv : mk; continue; }
            const int N = n + us;
            for (int a = 0; a < n; ++a) {
                const double xa = loc[2 * idx[a]], ya = loc[2 * idx[a] + 1];
                for (int c = 0; c < n; ++c)
                    A[a * N + c] = (a == c) ? sk
                        : cov(xa - loc[2 * idx[c]], ya - loc[2 * idx[c] + 1], Rk, sk, nk, vt[k]);
                c0[a] = cov(xa - tx, ya - ty, Rk, sk, nk, vt[k]);
                b[a] = c0[a];
                const double v = Vr[idx[a]];
                const double ind = categorical ? (v == classes[k] ? 1.0 : 0.0) : (v <= classes[k] ? 1.0 : 0.0);
                resid[a] = ind - mk;
            }
            if (us) {
                for (int a = 0; a < n; ++a) { A[a * N + n] = A[n * N + a] = B[k] * c0[a]; }
                A[n * N + n] = B[k] * sk;
                b[n] = B[k] * sk;
                resid[n] = sv - mk;
            }
            for (int a = 0; a < N; ++a) A[a * N + a] += 1e-10 * sk;
            p[k] = mk + solve_dot(A, b, resid, N);
        }
        const double u = ur[s];
        double value;
        if (categorical) {
            double tot = 0.0;
            for (int k = 0; k < K; ++k) { if (p[k] < 0) p[k] = 0; if (p[k] > 1) p[k] = 1; tot += p[k]; }
            if (tot <= 0) {
                tot = 0.0;
                for (int k = 0; k < K; ++k) { p[k] = use_soft ? soft[j * K + k] : mean[k]; tot += p[k]; }
            }
            const double target = u * tot; double cdf = 0.0; int c = K - 1;
            for (int k = 0; k < K; ++k) { cdf += p[k]; if (cdf >= target) { c = k; break; } }
            value = classes[c];
        } else {
            double up[MAXK], dn[MAXK], cc[MAXK];
            for (int k = 0; k < K; ++k) { double q = p[k] < 0 ? 0 : (p[k] > 1 ? 1 : p[k]); up[k] = q; dn[k] = q; }
            for (int k = 1; k < K; ++k) if (up[k] < up[k - 1]) up[k] = up[k - 1];
            for (int k = K - 2; k >= 0; --k) if (dn[k] > dn[k + 1]) dn[k] = dn[k + 1];
            for (int k = 0; k < K; ++k) cc[k] = 0.5 * (up[k] + dn[k]);
            value = beyond(u, classes, cc, K, zmin, zmax, lt, ltp, mid, mp, ut, utp);
        }
        Vr[n1 + j] = value;
    }
}
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        import cupy

        _kernel = cupy.RawKernel(_SIS_KERNEL_SRC, "sis_run")
    return _kernel


def sis_simulate_gpu(sim, seeds, threads: int = 64, steps_per_launch: int = 1024) -> np.ndarray:
    """
    Run a fitted `IndicatorCoSimulator` on the GPU for every seed; returns
    ``(nsim, n_nodes)`` realizations (class values or continuous values).
    Realizations are batched to fit the free GPU memory.
    """
    import cupy as cp

    K = len(sim.classes)
    if K > MAX_CLASSES:
        raise ValueError(f"GPU SIS supports at most {MAX_CLASSES} classes / thresholds")
    if sim.num_points > MAX_POINTS:
        raise ValueError(f"GPU SIS supports num_points <= {MAX_POINTS}")
    for v in sim.varios:
        if str(v[5]).lower() not in _VTYPES:
            raise ValueError(f"GPU SIS supports {sorted(_VTYPES)} variograms, not {v[5]!r}")
    if sim.variable == "continuous":
        for key, allowed in (("ltail", ("linear", "power")), ("middle", ("linear", "power")),
                             ("utail", ("linear", "power", "hyperbolic"))):
            if sim.tails[key] not in allowed:
                raise ValueError(f"{key} must be one of {allowed}")

    nsim, n_nodes = len(seeds), sim.n_nodes_
    n_loc = len(sim._loc)
    per_real = 8 * n_loc + 8 * n_nodes + 4 * n_nodes + 8 * (MAX_POINTS + 1) ** 2
    free, _ = cp.cuda.runtime.memGetInfo()
    batch = int(max(1, min(nsim, (0.6 * free) // per_real)))
    out = np.empty((nsim, n_nodes))
    for b0 in range(0, nsim, batch):
        out[b0:b0 + batch] = _run_batch(sim, seeds[b0:b0 + batch], threads, steps_per_launch)
    return out


def _run_batch(sim, seeds, threads, steps_per_launch):
    import cupy as cp

    nsim, n_nodes = len(seeds), sim.n_nodes_
    n1 = len(sim._data_z)
    K = len(sim.classes)
    coincident = sim._coincident

    perms = np.empty((nsim, n_nodes), dtype=np.int32)
    unif = np.zeros((nsim, n_nodes))
    for r, s in enumerate(seeds):
        g = _get_random_generator(int(s))
        p = g.permutation(n_nodes)
        perms[r] = p
        draw = coincident[p] < 0
        unif[r, draw] = g.uniform(size=int(draw.sum()))

    n_loc = len(sim._loc)
    V0 = np.full((nsim, n_loc), np.nan)
    V0[:, :n1] = sim._data_z
    V = cp.asarray(V0)
    del V0

    def dev(a, dtype=np.float64):
        return cp.asarray(np.ascontiguousarray(a, dtype=dtype))

    use_soft = sim.soft_ is not None
    sill = dev([float(v[4]) for v in sim.varios])
    nug = dev([float(v[1]) for v in sim.varios])
    vt = dev([_VTYPES[str(v[5]).lower()] for v in sim.varios], np.int32)
    R = dev(np.concatenate([np.asarray(Rk, dtype=np.float64).ravel() for Rk in sim._rot]))
    soft = dev(sim.soft_.ravel()) if use_soft else cp.zeros(1)
    t = sim.tails
    zmin = getattr(sim, "_zmin", 0.0)
    zmax = getattr(sim, "_zmax", 1.0)

    args = (dev(sim._nbrs, np.int64), np.int32(sim._nbrs.shape[1]), dev(sim._loc), np.int32(n1),
            dev(coincident, np.int64), np.int32(sim.num_points), np.int32(K), dev(sim.classes),
            np.int32(sim.variable == "categorical"), sill, nug, vt, R, dev(sim.proportions_),
            np.int32(use_soft), soft, dev(sim.B_),
            np.float64(zmin), np.float64(zmax), np.int32(_TAILS[t["ltail"]]), np.float64(t["ltpar"]),
            np.int32(_TAILS[t["middle"]]), np.float64(t["mpar"]), np.int32(_TAILS[t["utail"]]),
            np.float64(t["utpar"]), np.int32(nsim))
    perms_d, unif_d = cp.asarray(perms), cp.asarray(unif)
    work = cp.empty(nsim * (MAX_POINTS + 1) ** 2)
    kern = _get_kernel()
    blocks = (nsim + threads - 1) // threads
    for step0 in range(0, n_nodes, steps_per_launch):
        kern((blocks,), (threads,), (V, np.int64(n_loc), perms_d, unif_d, np.int64(n_nodes),
                                     np.int32(step0), np.int32(steps_per_launch)) + args + (work,))
    return cp.asnumpy(V[:, n1:])
