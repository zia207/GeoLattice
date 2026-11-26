"""
GPU engine for direct sequential (co-)simulation.

Each realization walks its own random path on the GPU (as in
`geolattice.geostatistics.sgs_gpu`); each launch advances all realizations by a
chunk of steps. At every node

1. keeps the first ``num_points`` informed locations of the node's
   precomputed nearest-candidate list,
2. solves the simple (co)kriging system in original units -- the same
   collocated Markov Model 1 system as the CPU engine, with the same
   diagonal jitter -- for the estimate ``z*`` and variance,
3. draws the value from the global distribution:

   * ``method='soares'``: ``y* = G^-1(F(z*))``, ``z = F^-1(G(y* + s * e))``
     with ``s^2 = sigma2_sk / sigma2_Z``;
   * ``method='lookup'``: the Gaussian window ``(y_m, y_s)`` whose
     back-transformed mean / sd are nearest to ``(z*, sigma_sk)`` -- the
     same table point the CPU engine's KD-tree returns -- then
     ``z = F^-1(G(y_m + y_s * e))``. Here each realization is a thread
     *block*: thread 0 krigs, then all 128 threads scan strided slices of the
     table -- first in float32 for the minimum distance, then exactly in
     double precision for the entries within float32 rounding of it -- and
     a block reduction picks the nearest window (ties to the smallest index).

   ``F`` and ``F^-1`` replicate ``numpy.interp`` on the global table
   (including repeated values); ``G`` and ``G^-1`` are CUDA's ``normcdf`` /
   ``normcdfinv``.

The host generates each realization's path and normal deviates in the CPU
engine's order (one per simulated node), so GPU realizations equal the CPU
ones up to floating-point rounding.

Limits: ``num_points <= 48``; spherical, exponential and Gaussian variograms.
"""

import numpy as np

from ..simulation.sgsim import _get_random_generator

__all__ = ["dss_simulate_gpu", "MAX_POINTS"]

MAX_POINTS = 48
_VTYPES = {"spherical": 0, "exponential": 1, "gaussian": 2}

_DSS_KERNEL_SRC = r"""
#define MAXP 48
#define MAXN (MAXP + 2)
#define BLOCK 128

__device__ double cov(double dx, double dy, const double* R, double sill, double nug, int vt) {
    const double ex = dx * R[0] + dy * R[2];
    const double ey = dx * R[1] + dy * R[3];
    const double h = sqrt(ex * ex + ey * ey);
    const double s = sill - nug;
    if (vt == 0) return h <= 1.0 ? s * (1.0 - 1.5 * h + 0.5 * h * h * h) : 0.0;
    if (vt == 1) return s * exp(-3.0 * h);
    return s * exp(-3.0 * h * h);
}

// numpy.interp(x, xp, fp) for increasing (possibly repeated) xp
__device__ double interp(double x, const double* xp, const double* fp, int n) {
    if (x > xp[n - 1]) return fp[n - 1];
    if (x < xp[0]) return fp[0];
    int lo = 0, hi = n;                       // j = (number of xp <= x) - 1
    while (lo < hi) { int mid = (lo + hi) >> 1; if (xp[mid] <= x) lo = mid + 1; else hi = mid; }
    const int j = lo - 1;
    if (j >= n - 1) return fp[n - 1];
    if (xp[j] == x) return fp[j];
    const double slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    double res = slope * (x - xp[j]) + fp[j];
    if (isnan(res)) {
        res = slope * (x - xp[j + 1]) + fp[j + 1];
        if (isnan(res) && fp[j] == fp[j + 1]) res = fp[j];
    }
    return res;
}

__device__ double to_gaussian(double z, const double* zt, const double* pt, int nt) {
    double p = interp(z, zt, pt, nt);
    if (p < 1e-12) p = 1e-12; if (p > 1 - 1e-12) p = 1 - 1e-12;
    return normcdfinv(p);
}

__device__ double from_gaussian(double y, const double* zt, const double* pt, int nt) {
    return interp(normcdf(y), pt, zt, nt);
}

// Simple (co)kriging of node j for one realization; returns estimate (original
// units) and variance. Same system and jitter as the CPU engine.
__device__ void krige_node(
    const double* Vr, const long long j, const long long* nbrs, const int m, const double* loc,
    const int n1, const int num_points, const double sill, const double nug, const int vt,
    const double* R, const double mean, const int use_y, const double* y, const double rho,
    double* A, double* est_out, double* var_out)
{
    int idx[MAXP]; double b[MAXN], v[MAXN], w[MAXN], bb[MAXN];
    int n = 0;
    const long long* nb = nbrs + j * m;
    for (int t = 0; t < m && n < num_points; ++t) {
        const long long c = nb[t];
        if (c < 0) continue;
        if (!isnan(Vr[c])) idx[n++] = (int)c;
    }
    const double tx = loc[2 * (n1 + j)], ty = loc[2 * (n1 + j) + 1];
    const int uy = use_y && rho != 0.0;
    const int N = n + uy;
    const double s0 = sqrt(sill);
    double est = 0.0, var = sill;
    if (N > 0) {
        for (int a = 0; a < n; ++a) {
            const double xa = loc[2 * idx[a]], ya = loc[2 * idx[a] + 1];
            for (int c = 0; c < n; ++c)
                A[a * N + c] = (a == c) ? sill
                    : cov(xa - loc[2 * idx[c]], ya - loc[2 * idx[c] + 1], R, sill, nug, vt);
            b[a] = cov(xa - tx, ya - ty, R, sill, nug, vt);
            v[a] = Vr[idx[a]] - mean;
        }
        if (uy) {
            for (int a = 0; a < n; ++a) { A[a * N + n] = A[n * N + a] = rho * b[a] / s0; }
            A[n * N + n] = 1.0;
            b[n] = rho * s0;
            v[n] = y[j];
        }
        for (int a = 0; a < N; ++a) { A[a * N + a] += 1e-10 * sill; bb[a] = b[a]; }
        for (int k = 0; k < N; ++k) {
            int p = k; double big = fabs(A[k * N + k]);
            for (int i = k + 1; i < N; ++i) if (fabs(A[i * N + k]) > big) { big = fabs(A[i * N + k]); p = i; }
            if (p != k) {
                for (int c = 0; c < N; ++c) { double t = A[k * N + c]; A[k * N + c] = A[p * N + c]; A[p * N + c] = t; }
                double t = bb[k]; bb[k] = bb[p]; bb[p] = t;
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
            double t = bb[k];
            for (int c = k + 1; c < N; ++c) t -= A[k * N + c] * w[c];
            w[k] = fabs(A[k * N + k]) < 1e-300 ? 0.0 : t / A[k * N + k];
        }
        est = 0.0; var = sill;
        for (int a = 0; a < N; ++a) { est += w[a] * v[a]; var -= w[a] * b[a]; }
        if (var < 0.0) var = 0.0;
    }
    *est_out = mean + est;
    *var_out = var;
}

// Soares DSS: one thread per realization.
extern "C" __global__ void dss_soares(
    double* V, const long long n_loc, const int* perms, const double* noise,
    const long long n_nodes, const int step0, const int nsteps,
    const long long* nbrs, const int m, const double* loc, const int n1,
    const long long* coincident, const int num_points,
    const double sill, const double nug, const int vt, const double* R, const double mean,
    const int use_y, const double* y, const double rho,
    const double* zt, const double* pt, const int nt, const double var_z,
    const int nsim, double* work)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nsim) return;
    double* Vr = V + (long long)r * n_loc;
    double* A = work + (long long)r * (MAXN * MAXN);
    const int* pr = perms + (long long)r * n_nodes;
    const double* er = noise + (long long)r * n_nodes;
    const int stop = step0 + nsteps < n_nodes ? step0 + nsteps : (int)n_nodes;
    for (int s = step0; s < stop; ++s) {
        const long long j = pr[s];
        const long long cj = coincident[j];
        if (cj >= 0) { Vr[n1 + j] = Vr[cj]; continue; }
        double z0, var;
        krige_node(Vr, j, nbrs, m, loc, n1, num_points, sill, nug, vt, R, mean, use_y, y, rho, A, &z0, &var);
        const double y0 = to_gaussian(z0, zt, pt, nt);
        const double sd = var_z > 0 ? sqrt(var / var_z) : 0.0;
        Vr[n1 + j] = from_gaussian(y0 + sd * er[s], zt, pt, nt);
    }
}

// Lookup DSS: one block per realization; thread 0 krigs, all threads search
// the window table (strided) and a block reduction picks the nearest window.
extern "C" __global__ void dss_lookup(
    double* V, const long long n_loc, const int* perms, const double* noise,
    const long long n_nodes, const int step0, const int nsteps,
    const long long* nbrs, const int m, const double* loc, const int n1,
    const long long* coincident, const int num_points,
    const double sill, const double nug, const int vt, const double* R, const double mean,
    const int use_y, const double* y, const double rho,
    const double* zt, const double* pt, const int nt,
    const double* tab_mu, const double* tab_sd, const double* tab_ym, const double* tab_ys,
    const int ntab, const double scale, const float* tab_mu32, const float* tab_sd32,
    const int nsim, double* work)
{
    const int r = blockIdx.x, tid = threadIdx.x;
    if (r >= nsim) return;
    __shared__ double sh_z0, sh_var, sh_qm, sh_qs;
    __shared__ int sh_mode;                            // 0 coincident, 1 degenerate, 2 search
    __shared__ double red_d[BLOCK]; __shared__ int red_k[BLOCK]; __shared__ float red_f[BLOCK];
    double* Vr = V + (long long)r * n_loc;
    double* A = work + (long long)r * (MAXN * MAXN);
    const int* pr = perms + (long long)r * n_nodes;
    const double* er = noise + (long long)r * n_nodes;
    const int stop = step0 + nsteps < n_nodes ? step0 + nsteps : (int)n_nodes;
    for (int s = step0; s < stop; ++s) {
        const long long j = pr[s];
        if (tid == 0) {
            const long long cj = coincident[j];
            if (cj >= 0) { Vr[n1 + j] = Vr[cj]; sh_mode = 0; }
            else {
                double z0, var;
                krige_node(Vr, j, nbrs, m, loc, n1, num_points, sill, nug, vt, R, mean, use_y, y, rho, A, &z0, &var);
                sh_z0 = z0; sh_var = var;
                if (var <= 0.0) sh_mode = 1;
                else { sh_mode = 2; sh_qm = z0 / scale; sh_qs = sqrt(var) / scale; }
            }
        }
        __syncthreads();
        const int mode = sh_mode;
        if (mode == 2) {
            const double qm = sh_qm, qs = sh_qs;
            // pass 1: float32 minimum over the table (strided across the block)
            const float fm = (float)qm, fs = (float)qs;
            float b32 = 3.0e38f;
            for (int k = tid; k < ntab; k += blockDim.x) {
                const float dm = tab_mu32[k] - fm, ds = tab_sd32[k] - fs, d = dm * dm + ds * ds;
                b32 = d < b32 ? d : b32;
            }
            red_f[tid] = b32;
            __syncthreads();
            for (int h = blockDim.x / 2; h > 0; h >>= 1) {
                if (tid < h && red_f[tid + h] < red_f[tid]) red_f[tid] = red_f[tid + h];
                __syncthreads();
            }
            const float cut = red_f[0] * (1.0f + 1e-4f) + 1e-10f;
            // pass 2: exact double distance for entries within float32 rounding of it
            double best = 1e300; int bk = 0x7fffffff;
            for (int k = tid; k < ntab; k += blockDim.x) {
                const float dm = tab_mu32[k] - fm, ds = tab_sd32[k] - fs;
                if (dm * dm + ds * ds <= cut) {
                    const double em = tab_mu[k] - qm, es = tab_sd[k] - qs, d = em * em + es * es;
                    if (d < best) { best = d; bk = k; }      // k increases: first minimum kept
                }
            }
            red_d[tid] = best; red_k[tid] = bk;
            __syncthreads();
            for (int h = blockDim.x / 2; h > 0; h >>= 1) {
                if (tid < h) {
                    const double od = red_d[tid + h]; const int ok = red_k[tid + h];
                    if (od < red_d[tid] || (od == red_d[tid] && ok < red_k[tid])) { red_d[tid] = od; red_k[tid] = ok; }
                }
                __syncthreads();
            }
            if (tid == 0) {
                const int bi = red_k[0];
                Vr[n1 + j] = from_gaussian(tab_ym[bi] + tab_ys[bi] * er[s], zt, pt, nt);
            }
        } else if (mode == 1 && tid == 0) {
            const double z0 = sh_z0;
            Vr[n1 + j] = z0 < zt[0] ? zt[0] : (z0 > zt[nt - 1] ? zt[nt - 1] : z0);
        }
        __syncthreads();
    }
}
"""

_kernels = {}


def _get_kernel(name):
    if name not in _kernels:
        import cupy

        _kernels[name] = cupy.RawKernel(_DSS_KERNEL_SRC, name)
    return _kernels[name]


def dss_simulate_gpu(sim, seeds, threads: int = 64, steps_per_launch: int = 1024) -> np.ndarray:
    """
    Run a fitted `DirectSequentialSimulator` on the GPU for every seed;
    returns ``(nsim, n_nodes)`` realizations in original units. Realizations
    are batched to fit the free GPU memory.
    """
    import cupy as cp

    vtype = str(sim.vario[5]).lower()
    if vtype not in _VTYPES:
        raise ValueError(f"GPU DSS supports {sorted(_VTYPES)} variograms, not {vtype!r}")
    if sim.num_points > MAX_POINTS:
        raise ValueError(f"GPU DSS supports num_points <= {MAX_POINTS}")
    nsim, n_nodes = len(seeds), sim.n_nodes_
    n_loc = len(sim._loc)
    per_real = 8 * n_loc + 8 * n_nodes + 4 * n_nodes + 8 * (MAX_POINTS + 2) ** 2
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
    coincident = sim._coincident
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
    V0[:, :n1] = sim._data_z
    V = cp.asarray(V0)
    del V0

    def dev(a, dtype=np.float64):
        return cp.asarray(np.ascontiguousarray(a, dtype=dtype))

    dist = sim.distribution_
    use_y = sim._y is not None and sim.rho_ != 0.0
    common = (dev(sim._nbrs, np.int64), np.int32(sim._nbrs.shape[1]), dev(sim._loc), np.int32(n1),
              dev(coincident, np.int64), np.int32(sim.num_points),
              np.float64(sim.vario[4]), np.float64(sim.vario[1]),
              np.int32(_VTYPES[str(sim.vario[5]).lower()]), dev(np.asarray(sim._R).ravel()),
              np.float64(sim.mean_), np.int32(use_y), dev(sim._y) if use_y else cp.zeros(1),
              np.float64(sim.rho_), dev(dist.z_), dev(dist.p_), np.int32(len(dist.z_)))
    perms_d, noise_d = cp.asarray(perms), cp.asarray(noise)
    work = cp.empty(nsim * (MAX_POINTS + 2) ** 2)
    head = lambda step0: (V, np.int64(n_loc), perms_d, noise_d, np.int64(n_nodes),  # noqa: E731
                          np.int32(step0), np.int32(steps_per_launch))
    if sim.method == "soares":
        kern = _get_kernel("dss_soares")
        blocks = (nsim + threads - 1) // threads
        for step0 in range(0, n_nodes, steps_per_launch):
            kern((blocks,), (threads,), head(step0) + common + (np.float64(dist.var_), np.int32(nsim), work))
    else:
        tab = sim._table
        scale = tab._scale
        tabs = (dev(tab.mu / scale), dev(tab.sd / scale), dev(tab.ym), dev(tab.ys),
                np.int32(len(tab.mu)), np.float64(scale),
                dev(tab.mu / scale, np.float32), dev(tab.sd / scale, np.float32))
        kern = _get_kernel("dss_lookup")
        for step0 in range(0, n_nodes, steps_per_launch):
            kern((nsim,), (128,), head(step0) + common + tabs + (np.int32(nsim), work))
    return cp.asnumpy(V[:, n1:])
