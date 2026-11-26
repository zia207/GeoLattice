"""
Batched (GPU) engines for multiple-point simulation.

Nodes within one realization must be simulated in sequence, but different
realizations are independent. These engines advance all ``nsim``
realizations in lockstep -- each along its own random path -- and do the
work of each step for every realization at once with array operations, on
the GPU through CuPy (``xp = cupy``) or on the CPU with NumPy
(``xp = numpy``, used to validate the batched algorithm).

* `ds_simulate_batched` (Direct Sampling): gathers every realization's data
  event (first ``n_neighbors`` informed template nodes, compacted with a
  stable argsort), then evaluates the TI scan window in doubling chunks
  (128, 256, ... like the sequential scan) until every realization has
  found a match: the value
  pasted is that of the first candidate in scan order within ``threshold``,
  else the best candidate seen -- the same rule as the sequential scan in
  `DirectSampling`. Secondary co-simulation is supported.
On CuPy each step is a single fused CUDA kernel (``cupy.RawKernel``, one
thread block per realization): for Direct Sampling the block's threads scan
TI candidates in parallel, ``blockDim`` at a time in scan order, taking the
first hit by ``atomicMin`` on the scan index or else the best candidate by a
block reduction; for SNESIM the threads stride over the pattern table, each
finding how many leading template nodes its pattern matches (uninformed
nodes are wildcards) and adding its centre counts to a shared histogram by
match length, whose suffix sums are the prefix counts. On NumPy the same
steps are vectorized array operations (the reference implementation).

* `snesim_simulate_batched` (SNESIM): matches every realization's data
  event against the whole pattern table at once (equality, cumulative AND
  along template order, then an einsum with the centre counts), giving the
  counts of every data-event prefix; the longest prefix with at least
  ``min_replicates`` replicates is used, followed by the tau model,
  servosystem (applied only to categories with non-zero probability, as in
  `SNESIM`) and draw.

Differences from the sequential engines: the template's tie order is shared
by the realizations of a batch, and random numbers come from different
streams, so realizations are statistically equivalent but not
bit-identical to `use_gpu=False`. The speed-up grows with ``nsim``; for a
single realization the per-step launch overhead usually makes the
sequential CPU engine faster.
"""

import numpy as np

from ..simulation.sgsim import _get_random_generator

_MISSING = 255


def _to_numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _is_cupy(xp) -> bool:
    return getattr(xp, "__name__", "") == "cupy"


# ------------------------------------------------------------ CUDA kernels
# One thread block per realization; one launch per simulation step.

_DS_MAXN, _DS_MAXS, _SN_MAXT, _SN_MAXK = 128, 64, 64, 16

_DS_KERNEL_SRC = r"""
#define MAXN 128
#define MAXS 64
#define BIG 1e300
extern "C" __global__ void ds_step(
    double* S, const long long s_stride, const long long* nodes,
    const long long* sim_lag, const long long* ti_lag, const double* base_w, const int T,
    const int n_nb, const double* ti,
    const long long* scan, const long long scan_stride, const int n_valid, const int max_scan,
    const long long* starts, const double threshold, const int categorical, const double vrange,
    const int cosim, const double* sec, const double* sti,
    const long long* s_sim_lag, const long long* s_ti_lag, const int n_sec,
    const int sec_cat, const double srange, const double w_sec)
{
    const int r = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    __shared__ long long lagk[MAXN]; __shared__ double vk[MAXN], wk[MAXN];
    __shared__ long long slag[MAXS]; __shared__ double svk[MAXS];
    __shared__ int nk, ns, hit;
    __shared__ double red_d[256]; __shared__ int red_k[256];
    double* Sr = S + (long long)r * s_stride;
    const long long node = nodes[r];
    if (tid == 0) {
        int c = 0; double ws = 0.0;
        for (int t = 0; t < T && c < n_nb; ++t) {
            double v = Sr[node + sim_lag[t]];
            if (!isnan(v)) { lagk[c] = ti_lag[t]; vk[c] = v; wk[c] = base_w[t]; ws += base_w[t]; ++c; }
        }
        for (int i = 0; i < c; ++i) wk[i] /= ws;
        nk = c;
        int m = 0;
        if (cosim) for (int t = 0; t < n_sec; ++t) {
            double v = sec[node + s_sim_lag[t]];
            if (!isnan(v)) { svk[m] = v; slag[m] = s_ti_lag[t]; ++m; }
        }
        ns = m; hit = 0x7fffffff;
    }
    __syncthreads();
    const long long* scan_r = scan + (long long)r * scan_stride;
    const long long st = starts[r];
    double best_d = BIG; int best_k = 0x7fffffff;
    for (int base = 0; base < max_scan; base += nt) {
        const int k = base + tid;
        if (k < max_scan) {
            const long long c = scan_r[(st + k) % n_valid];
            double d1 = 0.0, d2 = 0.0;
            for (int i = 0; i < nk; ++i) {
                double tv = ti[c + lagk[i]], m;
                if (categorical) m = (isnan(tv) || tv != vk[i]) ? 1.0 : 0.0;
                else { m = isnan(tv) ? 1.0 : fabs(tv - vk[i]) / vrange; if (m > 1.0) m = 1.0; }
                d1 += m * wk[i];
            }
            if (cosim && ns > 0) {
                for (int i = 0; i < ns; ++i) {
                    double tv = sti[c + slag[i]], m;
                    if (sec_cat) m = (isnan(tv) || tv != svk[i]) ? 1.0 : 0.0;
                    else { m = isnan(tv) ? 1.0 : fabs(tv - svk[i]) / srange; if (m > 1.0) m = 1.0; }
                    d2 += m;
                }
                d2 /= ns;
            }
            double d;
            if (nk == 0) d = cosim ? d2 : 0.0;
            else d = cosim ? (1.0 - w_sec) * d1 + w_sec * d2 : d1;
            if (d <= threshold) atomicMin(&hit, k);
            if (d < best_d || (d == best_d && k < best_k)) { best_d = d; best_k = k; }
        }
        __syncthreads();
        if (hit != 0x7fffffff) break;
        __syncthreads();
    }
    int kk;
    if (hit != 0x7fffffff) kk = hit;
    else {
        red_d[tid] = best_d; red_k[tid] = best_k;
        __syncthreads();
        for (int s = nt / 2; s > 0; s >>= 1) {
            if (tid < s) {
                double od = red_d[tid + s]; int ok = red_k[tid + s];
                if (od < red_d[tid] || (od == red_d[tid] && ok < red_k[tid])) { red_d[tid] = od; red_k[tid] = ok; }
            }
            __syncthreads();
        }
        kk = red_k[0];
    }
    if (tid == 0) Sr[node] = ti[scan_r[(st + kk) % n_valid]];
}
"""

_SN_KERNEL_SRC = r"""
#define MAXT 64
#define MAXK 16
extern "C" __global__ void sn_step(
    unsigned char* S, const long long s_stride, const long long* nodes,
    const long long* lag, const int nt, const unsigned char* pats, const float* counts,
    const int U, const int K, const int min_rep, const double* target,
    const int use_soft, const double* soft, const double tau1, const double tau2,
    const double gain, double* n_sim, const double* unif)
{
    const int r = blockIdx.x, tid = threadIdx.x;
    __shared__ unsigned char ev[MAXT];
    __shared__ float hist[(MAXT + 1) * MAXK];
    unsigned char* Sr = S + (long long)r * s_stride;
    const long long node = nodes[r];
    for (int j = tid; j < nt; j += blockDim.x) ev[j] = Sr[node + lag[j]];
    for (int i = tid; i < (nt + 1) * K; i += blockDim.x) hist[i] = 0.f;
    __syncthreads();
    for (int u = tid; u < U; u += blockDim.x) {
        const unsigned char* pu = pats + (long long)u * nt;
        int m = 0;
        while (m < nt && (ev[m] == 255 || pu[m] == ev[m])) ++m;
        for (int k = 0; k < K; ++k) {
            float c = counts[(long long)u * K + k];
            if (c != 0.f) atomicAdd(&hist[m * K + k], c);
        }
    }
    __syncthreads();
    if (tid != 0) return;
    double acc[MAXK], p[MAXK];
    for (int k = 0; k < K; ++k) acc[k] = hist[nt * K + k];
    int found = 0; double tot = 0.0;
    // prefix ending at template position j is matched by patterns with m > j
    for (int j = nt - 1; j >= 0; --j) {
        if (ev[j] != 255) {
            tot = 0.0; for (int k = 0; k < K; ++k) tot += acc[k];
            if (tot >= min_rep) { found = 1; break; }
        }
        for (int k = 0; k < K; ++k) acc[k] += hist[j * K + k];
    }
    for (int k = 0; k < K; ++k) p[k] = found ? acc[k] / tot : target[k];
    if (use_soft) {
        const double eps = 1e-9; double s = 0.0;
        for (int k = 0; k < K; ++k) {
            double p0 = fmin(fmax(target[k], eps), 1 - eps), p1 = fmin(fmax(p[k], eps), 1 - eps);
            double p2 = fmin(fmax(soft[node * K + k], eps), 1 - eps);
            double x0 = (1 - p0) / p0, x1 = (1 - p1) / p1, x2 = (1 - p2) / p2;
            double x = x0 * pow(x1 / x0, tau1) * pow(x2 / x0, tau2);
            p[k] = 1.0 / (1.0 + x); s += p[k];
        }
        for (int k = 0; k < K; ++k) p[k] /= s;
    }
    double* nr = n_sim + (long long)r * K;
    if (gain != 0.0) {
        double tn = 0.0; for (int k = 0; k < K; ++k) tn += nr[k];
        if (tn > 0) {
            double s = 0.0, q[MAXK];
            for (int k = 0; k < K; ++k) {
                q[k] = p[k] > 0 ? p[k] + gain * (target[k] - nr[k] / tn) : 0.0;
                if (q[k] < 0) q[k] = 0; s += q[k];
            }
            if (s > 0) for (int k = 0; k < K; ++k) p[k] = q[k] / s;
        }
    }
    double s = 0.0; for (int k = 0; k < K; ++k) s += p[k];
    const double u = unif[r] * s; double cdf = 0.0; int c = K - 1;
    for (int k = 0; k < K; ++k) { cdf += p[k]; if (cdf >= u) { c = k; break; } }
    Sr[node] = (unsigned char)c;
    nr[c] += 1.0;
}
"""

_kernel_cache = {}


def _kernel(name, src):
    import cupy

    if name not in _kernel_cache:
        _kernel_cache[name] = cupy.RawKernel(src, name)
    return _kernel_cache[name]


# ----------------------------------------------------------- Direct Sampling


def ds_simulate_batched(ds, seeds, xp, first_chunk: int = 128, max_chunk: int = 4096) -> np.ndarray:
    """Lockstep Direct Sampling for all realizations; returns (nsim, nx, ny) on the host."""
    from ..mps.direct_sampling import _template

    nsim = len(seeds)
    nx, ny = ds.lattice_["nx"], ds.lattice_["ny"]
    rngs = [_get_random_generator(int(s)) for s in seeds]
    template = _template(ds.search_radius, rngs[0])
    tdist = np.hypot(template[:, 0], template[:, 1])
    r = int(np.ceil(ds.search_radius))
    n_nb = ds.n_neighbors

    P0 = np.full((nx + 2 * r, ny + 2 * r), np.nan)
    P0[r:r + nx, r:r + ny] = ds._hard
    ncol_s = P0.shape[1]
    S = xp.asarray(np.tile(P0.ravel(), (nsim, 1)))
    sim_lag = xp.asarray(template[:, 0] * ncol_s + template[:, 1])

    tpad = np.full((ds.ti.shape[0] + 2 * r, ds.ti.shape[1] + 2 * r), np.nan)
    tpad[r:-r, r:-r] = ds.ti
    ncol_t = tpad.shape[1]
    ti_flat = xp.asarray(tpad.ravel())
    ti_lag = xp.asarray(template[:, 0] * ncol_t + template[:, 1])
    valid = (ds._ti_valid[:, 0] + r) * ncol_t + (ds._ti_valid[:, 1] + r)
    n_valid = len(valid)
    max_scan = max(1, int(np.ceil(ds.max_scan_fraction * n_valid)))
    scan = xp.asarray(np.stack([valid[g.permutation(n_valid)] for g in rngs]))

    free = np.flatnonzero(np.isnan(ds._hard).ravel())
    free = (free // ny + r) * ncol_s + (free % ny + r)
    passes = 1 + ds.n_postprocess
    paths = xp.asarray(np.stack([np.concatenate([free[g.permutation(len(free))] for _ in range(passes)])
                                 for g in rngs]))
    L = paths.shape[1]
    starts = xp.asarray(np.stack([g.integers(0, n_valid, size=L) for g in rngs]))

    base_w = xp.asarray(tdist ** -ds.distance_power if ds.distance_power else np.ones(len(template)))
    categorical = ds.kind == "categorical"

    cosim = getattr(ds, "_sec_lattice", None) is not None
    if cosim:
        spad = np.full(P0.shape, np.nan)
        spad[r:r + nx, r:r + ny] = ds._sec_lattice
        s_flat = xp.asarray(spad.ravel())
        stpad = np.full(tpad.shape, np.nan)
        stpad[r:-r, r:-r] = ds.secondary_ti
        st_flat = xp.asarray(stpad.ravel())
        soff = np.vstack([[0, 0], template[: ds.n_secondary - 1]])
        s_sim_lag = xp.asarray(soff[:, 0] * ncol_s + soff[:, 1])
        s_ti_lag = xp.asarray(soff[:, 0] * ncol_t + soff[:, 1])
        sec_cat = ds.secondary_kind == "categorical"
        w_sec = ds.secondary_weight

    if _is_cupy(xp):
        if n_nb > _DS_MAXN or (cosim and ds.n_secondary > _DS_MAXS):
            raise ValueError(f"GPU Direct Sampling supports n_neighbors <= {_DS_MAXN} "
                             f"and n_secondary <= {_DS_MAXS}")
        kern = _kernel("ds_step", _DS_KERNEL_SRC)
        paths_t = xp.ascontiguousarray(paths.T)
        starts_t = xp.ascontiguousarray(starts.T)
        dummy = xp.zeros(1)
        dummy_l = xp.zeros(1, dtype=xp.int64)
        args_tail = (
            (s_flat, st_flat, s_sim_lag, s_ti_lag, np.int32(len(soff)), np.int32(sec_cat),
             np.float64(ds._sec_range), np.float64(w_sec)) if cosim else
            (dummy, dummy, dummy_l, dummy_l, np.int32(0), np.int32(0), np.float64(1.0), np.float64(0.0)))
        s_stride = np.int64(S.shape[1])
        for step in range(L):
            kern((nsim,), (256,), (
                S, s_stride, paths_t[step], sim_lag, ti_lag, base_w, np.int32(len(template)),
                np.int32(n_nb), ti_flat, scan, np.int64(n_valid), np.int32(n_valid),
                np.int32(max_scan), starts_t[step], np.float64(ds.threshold),
                np.int32(categorical), np.float64(ds._range), np.int32(cosim)) + args_tail)
        S = _to_numpy(S).reshape(nsim, nx + 2 * r, ny + 2 * r)
        return S[:, r:r + nx, r:r + ny].copy()

    rows = xp.arange(nsim)
    ar = xp.arange(max_chunk)

    def mismatch(tv, v, cat, rng_):
        if cat:
            return (tv != v).astype(xp.float64)
        m = xp.abs(tv - v) / rng_
        return xp.where(xp.isnan(m), 1.0, xp.minimum(m, 1.0))

    for step in range(L):
        nodes = paths[:, step]
        vals = S[rows[:, None], nodes[:, None] + sim_lag[None, :]]        # (nsim, T)
        inf = ~xp.isnan(vals)
        sel = inf & (xp.cumsum(inf, axis=1) <= n_nb)
        order = xp.argsort(~sel, axis=1, kind="stable")[:, :n_nb]          # first selected first
        ok_k = xp.take_along_axis(sel, order, axis=1)
        v_k = xp.take_along_axis(vals, order, axis=1)
        lag_k = ti_lag[order]
        w_k = base_w[order] * ok_k
        wsum = w_k.sum(axis=1)
        has_inf = wsum > 0
        w_k = w_k / xp.where(has_inf, wsum, 1.0)[:, None]
        if cosim:
            sv = s_flat[nodes[:, None] + s_sim_lag[None, :]]
            s_ok = ~xp.isnan(sv)
            s_w = s_ok / xp.maximum(s_ok.sum(axis=1), 1)[:, None]

        found = xp.zeros(nsim, dtype=bool)
        chosen = xp.zeros(nsim, dtype=xp.int64)
        best_d = xp.full(nsim, xp.inf)
        best_c = xp.zeros(nsim, dtype=xp.int64)
        st = starts[:, step]
        c0, chunk = 0, first_chunk
        while c0 < max_scan:                    # same doubling schedule as the sequential scan
            m = min(chunk, max_scan - c0)
            pos = (st[:, None] + c0 + ar[None, :m]) % n_valid
            centres = scan[rows[:, None], pos]                              # (nsim, m)
            tv = ti_flat[centres[:, :, None] + lag_k[:, None, :]]           # (nsim, m, k)
            d = (mismatch(tv, v_k[:, None, :], categorical, ds._range) * w_k[:, None, :]).sum(-1)
            if cosim:
                stv = st_flat[centres[:, :, None] + s_ti_lag[None, None, :]]
                d2 = (mismatch(stv, sv[:, None, :], sec_cat, ds._sec_range) * s_w[:, None, :]).sum(-1)
                d = xp.where(has_inf[:, None], (1.0 - w_sec) * d + w_sec * d2, d2)
            else:
                d = xp.where(has_inf[:, None], d, 0.0)
            hit = (d <= ds.threshold) & ~found[:, None]
            row_hit = hit.any(axis=1)
            first = xp.argmax(hit, axis=1)
            chosen = xp.where(row_hit, centres[rows, first], chosen)
            found = found | row_hit
            k = xp.argmin(d, axis=1)
            dk = d[rows, k]
            better = dk < best_d
            best_d = xp.where(better, dk, best_d)
            best_c = xp.where(better, centres[rows, k], best_c)
            if bool(found.all()):
                break
            c0 += m
            chunk = min(chunk * 2, max_chunk)
        final = xp.where(found, chosen, best_c)
        S[rows, nodes] = ti_flat[final]

    S = _to_numpy(S).reshape(nsim, nx + 2 * r, ny + 2 * r)
    return S[:, r:r + nx, r:r + ny].copy()


# -------------------------------------------------------------------- SNESIM


def snesim_simulate_batched(sn, seeds, xp, max_elements: float = 2e8) -> np.ndarray:
    """Lockstep SNESIM for all realizations; returns (nsim, nx, ny) category values."""
    nsim = len(seeds)
    nx, ny = sn.lattice_["nx"], sn.lattice_["ny"]
    K = len(sn.categories_)
    rngs = [_get_random_generator(int(s)) for s in seeds]
    hard = sn._hard
    hard_mask = hard != _MISSING
    sim = np.tile(hard, (nsim, 1, 1)).astype(np.uint8)
    n_sim = xp.asarray(np.tile(np.bincount(hard[hard_mask], minlength=K).astype(float), (nsim, 1)))
    target = xp.asarray(sn.target_proportions_)
    gain = sn.servo / (1.0 - sn.servo) if sn.servo else 0.0
    soft = getattr(sn, "_soft", None)
    tau1, tau2 = sn.tau
    rows = xp.arange(nsim)
    eps = 1e-9

    for g in reversed(range(sn.n_multigrid)):
        step = 2 ** g
        off = sn._template * step
        relocated = []
        if g > 0 and hard_mask.any():
            for i, j in np.argwhere(hard_mask):
                ci = min(int(round(i / step)) * step, (nx - 1) // step * step)
                cj = min(int(round(j / step)) * step, (ny - 1) // step * step)
                if sim[0, ci, cj] == _MISSING:
                    sim[:, ci, cj] = sim[0, i, j]
                    relocated.append((ci, cj))
        ii, jj = np.meshgrid(np.arange(0, nx, step), np.arange(0, ny, step), indexing="ij")
        nodes = np.column_stack([ii.ravel(), jj.ravel()])
        nodes = nodes[sim[0, nodes[:, 0], nodes[:, 1]] == _MISSING]
        n_l = len(nodes)
        if n_l == 0:
            for ci, cj in relocated:
                sim[:, ci, cj] = _MISSING
            continue

        rpad = int(np.abs(off).max())
        Sp = np.full((nsim, nx + 2 * rpad, ny + 2 * rpad), _MISSING, dtype=np.uint8)
        Sp[:, rpad:rpad + nx, rpad:rpad + ny] = sim
        ncol = Sp.shape[2]
        Sflat = xp.asarray(Sp.reshape(nsim, -1))
        lag = xp.asarray(off[:, 0] * ncol + off[:, 1])
        node_flat = (nodes[:, 0] + rpad) * ncol + (nodes[:, 1] + rpad)
        perm_d = xp.asarray(np.stack([node_flat[g_.permutation(n_l)] for g_ in rngs]))   # (nsim, n_l)
        unif = xp.asarray(np.stack([g_.uniform(size=n_l) for g_ in rngs]))

        pats_np, counts_np = sn._levels[g]
        pats = xp.asarray(pats_np)
        counts = xp.asarray(counts_np.astype(np.float32))
        U, nt = pats.shape
        r_chunk = max(1, int(max_elements // max(U * nt, 1)))

        if soft is not None:                    # soft probabilities on the same padded flat index
            sp = np.tile(sn.target_proportions_, (nx + 2 * rpad, ny + 2 * rpad, 1))
            sp[rpad:rpad + nx, rpad:rpad + ny] = sn._soft
            soft_flat = xp.asarray(sp.reshape(-1, K))

        if _is_cupy(xp):
            if nt > _SN_MAXT or K > _SN_MAXK:
                raise ValueError(f"GPU SNESIM supports n_template <= {_SN_MAXT} and <= {_SN_MAXK} categories")
            kern = _kernel("sn_step", _SN_KERNEL_SRC)
            perm_t = xp.ascontiguousarray(perm_d.T)
            unif_t = xp.ascontiguousarray(unif.T)
            pats_c = xp.ascontiguousarray(pats)
            counts_c = xp.ascontiguousarray(counts)
            soft_arg = xp.ascontiguousarray(soft_flat).ravel() if soft is not None else xp.zeros(1)
            tgt = xp.ascontiguousarray(target, dtype=xp.float64)
            n_sim = xp.ascontiguousarray(n_sim, dtype=xp.float64)
            for s in range(n_l):
                kern((nsim,), (256,), (
                    Sflat, np.int64(Sflat.shape[1]), perm_t[s], lag, np.int32(nt), pats_c, counts_c,
                    np.int32(U), np.int32(K), np.int32(sn.min_replicates), tgt,
                    np.int32(soft is not None), soft_arg, np.float64(tau1), np.float64(tau2),
                    np.float64(gain), n_sim, unif_t[s]))
        for s in range(n_l if not _is_cupy(xp) else 0):
            nd = perm_d[:, s]
            vals = Sflat[rows[:, None], nd[:, None] + lag[None, :]]                  # (nsim, nt)
            inf = vals != _MISSING
            c_sel = xp.empty((nsim, K), dtype=xp.float32)
            ok_any = xp.empty(nsim, dtype=bool)
            for a in range(0, nsim, r_chunk):
                b = min(a + r_chunk, nsim)
                E = (pats[None, :, :] == vals[a:b, None, :]) | ~inf[a:b, None, :]
                cum = xp.cumprod(E.astype(xp.uint8), axis=2).astype(xp.float32)      # (r, U, nt)
                cnt = xp.einsum("ruj,uk->rjk", cum, counts)                            # (r, nt, K)
                tot = cnt.sum(axis=2)
                ok = (tot >= sn.min_replicates) & inf[a:b]
                rev = ok[:, ::-1]
                jsel = nt - 1 - xp.argmax(rev, axis=1)
                rr = xp.arange(b - a)
                c_sel[a:b] = cnt[rr, jsel]
                ok_any[a:b] = ok.any(axis=1)
            tot_sel = c_sel.sum(axis=1, keepdims=True)
            p = xp.where(ok_any[:, None], c_sel / xp.where(tot_sel > 0, tot_sel, 1.0), target[None, :])
            if soft is not None:
                p0 = xp.clip(target[None, :], eps, 1 - eps)
                p1 = xp.clip(p, eps, 1 - eps)
                p2 = xp.clip(soft_flat[nd], eps, 1 - eps)
                x0, x1, x2 = (1 - p0) / p0, (1 - p1) / p1, (1 - p2) / p2
                x = x0 * (x1 / x0) ** tau1 * (x2 / x0) ** tau2
                p = 1.0 / (1.0 + x)
                p = p / p.sum(axis=1, keepdims=True)
            if gain:
                tot_n = n_sim.sum(axis=1, keepdims=True)
                adj = xp.clip(p + gain * (target[None, :] - n_sim / xp.where(tot_n > 0, tot_n, 1.0)), 0.0, None)
                adj = xp.where(p > 0, adj, 0.0)
                ssum = adj.sum(axis=1, keepdims=True)
                p = xp.where((tot_n > 0) & (ssum > 0), adj / xp.where(ssum > 0, ssum, 1.0), p)
            cdf = xp.cumsum(p, axis=1)
            c = xp.minimum((cdf < (unif[:, s] * cdf[:, -1])[:, None]).sum(axis=1), K - 1)
            Sflat[rows, nd] = c.astype(xp.uint8)
            n_sim[rows, c] += 1.0

        sim = _to_numpy(Sflat).reshape(nsim, nx + 2 * rpad, ny + 2 * rpad)[:, rpad:rpad + nx,
                                                                         rpad:rpad + ny].copy()
        for ci, cj in relocated:
            sim[:, ci, cj] = _MISSING
    return sn.categories_[sim]
