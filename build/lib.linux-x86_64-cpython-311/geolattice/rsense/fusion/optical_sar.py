"""Optical–SAR fusion: feature stacking and SAR-driven gap filling of cloudy optical pixels."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from ..backend import to_cpu
from ..optical.bands import as_dataset


def stack_features(*sources: xr.Dataset | xr.DataArray, reference: int = 0,
                   prefixes: Sequence[str] | None = None) -> xr.Dataset:
    """Align datasets to the grid of ``sources[reference]`` and merge their variables.

    Each source is reprojected/resampled (``reproject_match``, bilinear). Use ``prefixes``
    (e.g. ``["s2_", "s1_"]``) to keep variable names unique.
    """
    from rasterio.enums import Resampling

    dss = [as_dataset(to_cpu(s)) for s in sources]
    ref = dss[reference]
    tmpl = ref[list(ref.data_vars)[0]]
    out = {}
    for i, ds in enumerate(dss):
        pre = prefixes[i] if prefixes else ""
        for v in ds.data_vars:
            da = ds[v]
            if i != reference:
                da = da.rio.reproject_match(tmpl, resampling=Resampling.bilinear)
                da = da.assign_coords({d: tmpl[d] for d in tmpl.dims[-2:]})
            out[f"{pre}{v}"] = da
    return xr.Dataset(out).rio.write_crs(ref.rio.crs)


def sar_gap_fill(optical: xr.Dataset, sar: xr.Dataset, *, bands: Sequence[str] | None = None,
                 model: str = "rf", max_train: int = 50_000, seed: int = 0,
                 **model_kw: Any) -> tuple[xr.Dataset, dict]:
    """Fill cloudy (NaN) optical pixels from co-registered SAR features of the same date.

    A regressor per optical band (default random forest) is trained on clear pixels with
    SAR variables (and pixel coordinates) as predictors, then predicts masked pixels.
    Returns the filled Dataset and per-band out-of-sample R² on a 20 % holdout.
    """
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    from ..classify import Classifier

    bands = list(bands or optical.data_vars)
    sar = stack_features(optical[bands[0]], sar).drop_vars(bands[0])
    ydim, xdim = optical[bands[0]].dims[-2:]
    yy, xx = np.meshgrid(optical[ydim].values, optical[xdim].values, indexing="ij")
    feats = np.stack([np.asarray(to_cpu(sar[v]).values, "float32").ravel() for v in sar.data_vars]
                     + [yy.ravel().astype("float32"), xx.ravel().astype("float32")], 1)
    fok = np.all(np.isfinite(feats), 1)
    rng = np.random.default_rng(seed)
    out, scores = optical.copy(), {}
    for b in bands:
        y = np.asarray(to_cpu(optical[b]).values, "float32").ravel()
        train = np.flatnonzero(fok & np.isfinite(y))
        fill = np.flatnonzero(fok & ~np.isfinite(y))
        if train.size < 100 or fill.size == 0:
            continue
        sel = rng.choice(train, min(max_train, train.size), replace=False)
        Xtr, Xte, ytr, yte = train_test_split(feats[sel], y[sel], test_size=0.2, random_state=seed)
        reg = Classifier(model, task="regression", **model_kw).fit(Xtr, ytr)
        scores[b] = float(r2_score(yte, reg.predict(Xte)))
        reg.fit(feats[sel], y[sel])
        yf = y.copy()
        yf[fill] = reg.predict(feats[fill])
        out[b] = optical[b].copy(data=yf.reshape(optical[b].shape))
    out.attrs["pyrsense:gap_fill_r2"] = str(scores)
    return out, scores
