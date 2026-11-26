"""
GPU engine for sequential Gaussian (co-)simulation.

Nodes of one realization are simulated in sequence, but realizations are
independent, so each realization is one CUDA thread (``cupy.RawKernel``)
that walks its own random path; every launch advances all threads by a
chunk of steps. At each node the thread

1. walks its node's precomputed nearest-candidate list and keeps the first
   ``num_points`` informed locations (hard data or already simulated nodes),
2. builds the simple- or ordinary-kriging system with the GSLIB
   practical-range covariance of `sgsim` (plus the collocated secondary
   under Markov Model 1 when given),
3. solves it by Gaussian elimination with partial pivoting,
4. draws ``estimate + sqrt(variance) * noise`` and stores it.

The host prepares every realization's random path and its normal deviates
in exactly the order `GaussianCoSimulator`'s CPU engine consumes them (no
draw at nodes that coincide with data), and the kernel uses the same
covariance and diagonal jitter, so GPU realizations equal CPU ones up to
floating-point rounding.

Limits: ``num_points <= 48``; spherical, exponential and Gaussian models;
collocated secondary only (``secondary_neighbors=1``).
"""

import numpy as np

from ..simulation.sgsim import _get_random_generator

__all__ = ["sgs_simulate_gpu", "MAX_POINTS"]

MAX_POINTS = 48
_VTYPES = {"spherical": 0, "exponential": 1, "gaussian": 2}

_SGS_KERNEL_SRC = r"""
#define MAXP 48
#define MAXN (MAXP + 2)

__device__ double cov(double dx, double dy, const double* R, double sill, double nug, int vt) {
    // effective lag = [dx dy] @ R   (R row-major 2x2)
    const double ex = dx * R[0] + dy * R[2];
    const double ey = dx * R[1] + dy * R[3];
    const double h = sqrt(ex * ex + ey * ey);
    const double s = sill - nug;
    if (vt == 0) return h <= 1.0 ? s * (1.0 - 1.5 * h + 0.5 * h * h * h) : 0.0;
    if (vt == 1) return s * exp(-3.0 * h);
    return s * exp(-3.0 * h * h);
}

__device__ void sgs_node(
    double* Vr, const long long j, const double z_noise,
    const long long* nbrs, const int m, const double* loc, const int n1,
    const long long* coincident, const int num_points,
    const double sill, const double nug, const int vt, const double* R,
    const int ordinary, const int use_y, const double* yr, const double rho, double* A)
{
    const long long c0 = coincident[j];
    if (c0 >= 0) { Vr[n1 + j] = Vr[c0]; return; }

    int idx[MAXP]; int n = 0;
    const long long* nb = nbrs + j * m;
    for (int t = 0; t < m && n < num_points; ++t) {
        const long long c = nb[t];
        if (c < 0) continue;
        if (!isnan(Vr[c])) idx[n++] = (int)c;
    }
    const double tx = loc[2 * (n1 + j)], ty = loc[2 * (n1 + j) + 1];
    const int uy = use_y && rho != 0.0;
    const int ok = ordinary && !uy && n > 0;
    const int N = n + uy + ok;
    double est = 0.0, var = sill;
    if (N > 0) {
        double b[MAXN], v[MAXN], w[MAXN], bb[MAXN];
        const double s0 = sqrt(sill);
        for (int a = 0; a < n; ++a) {
            const double xa = loc[2 * idx[a]], ya = loc[2 * idx[a] + 1];
            for (int c = 0; c < n; ++c) {
                A[a * N + c] = (a == c) ? sill
                    : cov(xa - loc[2 * idx[c]], ya - loc[2 * idx[c] + 1], R, sill, nug, vt);
            }
            A[a * N + a] += 1e-10 * sill;
            b[a] = cov(xa - tx, ya - ty, R, sill, nug, vt);
            v[a] = Vr[idx[a]];
        }
        if (uy) {
            for (int a = 0; a < n; ++a) { A[a * N + n] = A[n * N + a] = rho * b[a] / s0; }
            A[n * N + n] = 1.0 + 1e-10 * sill;
            b[n] = rho * s0;
            v[n] = yr[j];
        }
        if (ok) {
            for (int a = 0; a < n; ++a) { A[a * N + n] = A[n * N + a] = 1.0; }
            A[n * N + n] = 0.0;
            b[n] = 1.0;
            v[n] = 0.0;
        }
        for (int a = 0; a < N; ++a) bb[a] = b[a];
        // Gaussian elimination with partial pivoting (A and bb overwritten)
        for (int k = 0; k < N; ++k) {
            int p = k; double big = fabs(A[k * N + k]);
            for (int i = k + 1; i < N; ++i) if (fabs(A[i * N + k]) > big) { big = fabs(A[i * N + k]); p = i; }
            if (p != k) {
                for (int c = 0; c < N; ++c) { double tmp = A[k * N + c]; A[k * N + c] = A[p * N + c]; A[p * N + c] = tmp; }
                double tmp = bb[k]; bb[k] = bb[p]; bb[p] = tmp;
            }
            const double piv = A[k * N + k];
            if (fabs(piv) < 1e-300) continue;
            for (int i = k + 1; i < N; ++i) {
                const double f = A[i * N + k] / piv;
                if (f == 0.0) continue;
                for (int c = k; c < N; ++c) A[i * N + c] -= f * A[k * N + c];
                bb[i] -= f * bb[k];
            }
        }
        for (int k = N - 1; k >= 0; --k) {
            double s = bb[k];
            for (int c = k + 1; c < N; ++c) s -= A[k * N + c] * w[c];
            w[k] = fabs(A[k * N + k]) < 1e-300 ? 0.0 : s / A[k * N + k];
        }
        est = 0.0; var = sill;
        for (int a = 0; a < n + uy; ++a) est += w[a] * v[a];
        for (int a = 0; a < N; ++a) var -= w[a] * b[a];
        if (var < 0.0) var = 0.0;
    }
    Vr[n1 + j] = est + sqrt(var) * z_noise;
}

// One thread per realization; each runs `nsteps` consecutive steps of its own path.
extern "C" __global__ void sgs_run(
    double* V, const long long n_loc, const int* perms, const double* noise,
    const long long n_nodes, const int step0, const int nsteps,
    const long long* nbrs, const int m, const double* loc, const int n1,
    const long long* coincident, const int num_points,
    const double sill, const double nug, const int vt, const double* R,
    const int ordinary, const int use_y, const double* y, const long long y_stride,
    const double rho, const int nsim, double* work)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nsim) return;
    double* Vr = V + (long long)r * n_loc;
    // per-realization matrix in a global workspace: a large per-thread local
    // array would make CUDA reserve local memory for every resident thread
    double* A = work + (long long)r * (MAXN * MAXN);
    const int* pr = perms + (long long)r * n_nodes;
    const double* nr = noise + (long long)r * n_nodes;
    const double* yr = y + (long long)r * y_stride;
    const int stop = step0 + nsteps < n_nodes ? step0 + nsteps : (int)n_nodes;
    for (int s = step0; s < stop; ++s)
        sgs_node(Vr, (long long)pr[s], nr[s], nbrs, m, loc, n1, coincident, num_points,
                 sill, nug, vt, R, ordinary, use_y, yr, rho, A);
}
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        import cupy

        _kernel = cupy.RawKernel(_SGS_KERNEL_SRC, "sgs_run")
    return _kernel


def sgs_simulate_gpu(sim, seeds, y=None, threads: int = 64, steps_per_launch: int = 1024) -> np.ndarray:
    """
    Run a fitted `GaussianCoSimulator` on the GPU for every seed.

    `y` is None, the secondary normal scores ``(n_nodes,)``, or one row per
    realization ``(nsim, n_nodes)``. Returns ``(nsim, n_nodes)`` in the
    simulator's working (normal-score) units, before back-transformation.

    Realizations are processed in batches sized to the free GPU memory, and
    each kernel launch advances every realization by `steps_per_launch`
    steps (one thread per realization loops over them), so launch overhead
    is negligible and no single launch runs long.
    """
    import cupy as cp

    vtype = str(sim.vario[5]).lower()
    if vtype not in _VTYPES:
        raise ValueError(f"GPU SGS supports {sorted(_VTYPES)} variograms, not {vtype!r}")
    if sim.num_points > MAX_POINTS:
        raise ValueError(f"GPU SGS supports num_points <= {MAX_POINTS}")
    if y is not None and sim.secondary_neighbors > 1 and sim.rho_ != 0.0:
        raise ValueError("GPU SGS supports the collocated secondary only (secondary_neighbors=1)")

    nsim, n_nodes = len(seeds), sim.n_nodes_
    n_loc = len(sim._loc)
    # realizations per batch from the free GPU memory (V, noise, path, workspace)
    per_real = 8 * n_loc + 8 * n_nodes + 4 * n_nodes + 8 * (MAX_POINTS + 2) ** 2
    if y is not None and np.ndim(y) == 2:
        per_real += 8 * n_nodes
    free, _ = cp.cuda.runtime.memGetInfo()
    batch = int(max(1, min(nsim, (0.6 * free) // per_real)))
    out = np.empty((nsim, n_nodes))
    for b0 in range(0, nsim, batch):
        b1 = min(b0 + batch, nsim)
        yb = None if y is None else (y if np.ndim(y) == 1 else np.asarray(y)[b0:b1])
        out[b0:b1] = _run_batch(sim, seeds[b0:b1], yb, threads, steps_per_launch)
    return out


def _run_batch(sim, seeds, y, threads, steps_per_launch):
    import cupy as cp

    vtype = str(sim.vario[5]).lower()
    nsim, n_nodes = len(seeds), sim.n_nodes_
    n1 = len(sim._data_z)
    m_mean = sim.mean_
    coincident = sim._coincident

    # paths and normal deviates in the CPU engine's consumption order
    perms = np.empty((nsim, n_nodes), dtype=np.int32)
    noise = np.zeros((nsim, n_nodes))
    for r, s in enumerate(seeds):
        g = _get_random_generator(int(s))
        p = g.permutation(n_nodes)
        perms[r] = p
        draw = coincident[p] < 0
        noise[r, draw] = g.standard_normal(int(draw.sum()))

    n_loc = len(sim._loc)
    V0 = np.full((nsim, n_loc), np.nan)
    V0[:, :n1] = sim._data_z - m_mean
    V = cp.asarray(V0)
    del V0
    perms_d = cp.asarray(perms)
    noise_d = cp.asarray(noise)
    nbrs = cp.asarray(np.ascontiguousarray(sim._nbrs, dtype=np.int64))
    loc = cp.asarray(np.ascontiguousarray(sim._loc, dtype=np.float64))
    coinc = cp.asarray(np.ascontiguousarray(coincident, dtype=np.int64))
    R = cp.asarray(np.ascontiguousarray(sim._R, dtype=np.float64).ravel())

    use_y = y is not None and sim.rho_ != 0.0
    if use_y:
        y = np.asarray(y, dtype=np.float64)
        y_arr = cp.asarray(np.ascontiguousarray(y))
        y_stride = 0 if y.ndim == 1 else n_nodes
    else:
        y_arr, y_stride = cp.zeros(1), 0

    kern = _get_kernel()
    blocks = (nsim + threads - 1) // threads
    work = cp.empty(nsim * (MAX_POINTS + 2) ** 2)
    for step0 in range(0, n_nodes, steps_per_launch):
        kern((blocks,), (threads,), (
            V, np.int64(n_loc), perms_d, noise_d, np.int64(n_nodes), np.int32(step0),
            np.int32(steps_per_launch), nbrs, np.int32(sim._nbrs.shape[1]), loc, np.int32(n1), coinc,
            np.int32(sim.num_points), np.float64(sim.vario[4]), np.float64(sim.vario[1]),
            np.int32(_VTYPES[vtype]), R, np.int32(sim.kriging_type == "ordinary"),
            np.int32(use_y), y_arr, np.int64(y_stride), np.float64(sim.rho_), np.int32(nsim), work))
    return cp.asnumpy(V[:, n1:]) + m_mean
