import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geolattice.rsense import classify, fusion, xai


def test_stack_features(reflectance):
    opt = reflectance.isel(time=0)[["red", "nir"]]
    sar = opt[["red"]].coarsen(x=2, y=2).mean().rename(red="vv").rio.write_crs(32618)
    st = fusion.stack_features(opt, sar, prefixes=["s2_", "s1_"])
    assert set(st.data_vars) == {"s2_red", "s2_nir", "s1_vv"}
    assert st.s1_vv.shape == opt.red.shape


def test_starfm_constant_change(single_band):
    two_level = np.where(np.arange(80) < 40, 0.1, 0.3)[None, :].repeat(64, 0)
    f1 = single_band.copy(data=two_level.astype("float32"))
    c1 = f1.copy(data=np.full(f1.shape, float(f1.mean()), "float32"))
    c2 = c1 + 0.05
    pred = fusion.starfm(f1, c1, c2, window=7)
    np.testing.assert_allclose(pred.values, f1.values + 0.05, atol=1e-5)


def test_upscale_and_compare(single_band):
    fine = single_band.copy(data=(single_band.values > 0).astype("float32"))
    coarse = single_band.coarsen(x=4, y=4).mean().rio.write_crs(32618)
    frac = fusion.upscale(fine, coarse, method="fraction")
    assert frac.shape == coarse.shape and 0 <= float(frac.min()) and float(frac.max()) <= 1
    np.testing.assert_allclose(frac.values, fine.coarsen(x=4, y=4).mean().values, atol=1e-6)
    m = fusion.compare(frac, frac)
    assert m["r2"] == pytest.approx(1.0) and m["rmse"] == 0


def test_sar_gap_fill(reflectance):
    opt = reflectance.isel(time=0)[["red"]].copy(deep=True)
    rng = np.random.default_rng(0)
    vv = opt.red.copy(data=(opt.red.values * 2 + rng.normal(0, 0.001, opt.red.shape)).astype("float32"))
    sar = xr.Dataset({"vv": vv}).rio.write_crs(32618)
    truth = opt.red.values.copy()
    opt["red"][:10, :10] = np.nan
    filled, r2 = fusion.sar_gap_fill(opt, sar, n_estimators=50)
    assert r2["red"] > 0.9
    assert np.abs(filled.red.values[:10, :10] - truth[:10, :10]).mean() < 0.01


def _xy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.uniform(0, 10_000, (n, 2))
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    X.attrs["coords"] = c
    y = 3 * X.a + rng.normal(0, 0.1, n)
    return X, y.to_numpy(), c


def test_spatial_cv_splitters():
    X, y, c = _xy()
    for cv in (xai.SpatialBlockCV(5, 2000), xai.SpatialClusterCV(5), xai.BufferedKFold(5, 300)):
        folds = list(cv.split(X))
        assert len(folds) >= 4
        for tr, te in folds:
            assert not set(tr) & set(te)
    from scipy.spatial import cKDTree

    for tr, te in xai.BufferedKFold(5, 300).split(X):
        assert cKDTree(c[te]).query(c[tr])[0].min() > 300
    res = xai.cross_validate_spatial(classify.Classifier("rf", task="regression", n_estimators=30),
                                     X, y, xai.SpatialBlockCV(4, 2500))
    assert res["mean"] > 0.8


def test_permutation_importance():
    X, y, _ = _xy()
    m = classify.Classifier("rf", task="regression", n_estimators=50).fit(X, y)
    imp = xai.permutation_importance(m, X, y, n_repeats=3)
    assert imp.feature.iloc[0] == "a"


def test_aoa(single_band):
    X, y, _ = _xy()
    tpl = single_band.isel(y=slice(0, 10), x=slice(0, 10))
    a = np.zeros((10, 10))
    a[:, 5:] = 50.0                         # far outside the training range
    feats = xr.Dataset({"a": tpl.copy(data=a), "b": tpl.copy(data=np.zeros((10, 10)))})
    out = xai.aoa(feats, X)
    assert bool(out.AOA[0, 0]) and not bool(out.AOA[0, 9])


def test_conformal_coverage():
    X, y, _ = _xy(2000)
    cr = xai.ConformalRegressor(classify.Classifier("rf", task="regression", n_estimators=50),
                                alpha=0.1).fit(X.iloc[:1500], y[:1500])
    iv = cr.predict_interval(X.iloc[1500:])
    cov = xai.coverage(y[1500:], iv.lower, iv.upper)
    assert 0.85 <= cov <= 0.97
    lab = np.where(X.a > 0, "hi", "lo")
    cc = xai.ConformalClassifier(classify.Classifier("rf", n_estimators=50), alpha=0.1)
    cc.fit(X.iloc[:1500], lab[:1500])
    sets = cc.predict_sets(X.iloc[1500:])
    idx = np.searchsorted(cc.classes_, lab[1500:])
    assert sets[np.arange(500), idx].mean() >= 0.85
