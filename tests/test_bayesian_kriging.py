"""Tests for geolattice.geostatistics.kriging.bayesian_kriging."""

import numpy as np
import pandas as pd
import pytest

from geolattice.geostatistics import (
    BayesianGLMKriging,
    BayesianKriging,
    BayesianTransformedKriging,
    ConjugateBayesianKriging,
    Variogram,
    bayesian_kriging,
    correlation_function,
)


def _field(n=120, range_=0.15, seed=0, nugget_sd=0.3):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 1, (n, 2))
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    L = np.linalg.cholesky(correlation_function(d, "exponential", range_) + 1e-9 * np.eye(n))
    latent = L @ rng.standard_normal(n)
    z = 5.0 + 2.0 * latent + nugget_sd * rng.standard_normal(n)
    return xy, z, latent, L, rng


def _ok_reference(xy, z, xp, range_):
    """Ordinary kriging with correlation R (independent implementation)."""
    n = len(xy)
    R = correlation_function(np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1)), "exponential", range_)
    r = correlation_function(np.sqrt(((xp[:, None] - xy[None]) ** 2).sum(-1)), "exponential", range_)
    K = np.block([[R, np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
    w = np.linalg.solve(K, np.hstack([r, np.ones((len(xp), 1))]).T).T
    return w[:, :n] @ z, 1.0 - np.einsum("ij,ij->i", w[:, :n], r) - w[:, n]


def test_correlation_function_basics():
    h = np.array([0.0, 0.1, 1.0, 10.0])
    for model in ["exponential", "spherical", "gaussian", "matern", "cubic"]:
        rho = correlation_function(h, model, 1.0)
        assert rho[0] == 1.0
        assert np.all(np.diff(rho) <= 1e-12)
        assert rho[-1] < 0.05
    with pytest.raises(ValueError):
        correlation_function(h, "nope", 1.0)


def test_conjugate_mean_equals_ordinary_kriging_and_t_scale():
    xy, z, *_ = _field()
    xp = np.random.default_rng(5).uniform(0, 1, (25, 2))
    ck = ConjugateBayesianKriging("exponential", range_=0.15, nugget_ratio=0.0).fit(xy, z)
    ok_mean, ok_var = _ok_reference(xy, z, xp, 0.15)
    loc, scale, dof = ck._component_predictions(xp)
    np.testing.assert_allclose(loc[0], ok_mean, atol=1e-5)
    # Student-t scale^2 = S^2 x (ordinary kriging variance in correlation units)
    np.testing.assert_allclose(scale[0] ** 2, ck._comps[0]["s2"] * ok_var, rtol=1e-4)
    assert dof[0] == len(z) - 1


def test_exact_interpolation_without_nugget():
    xy, z, *_ = _field()
    ck = ConjugateBayesianKriging("exponential", range_=0.15).fit(xy, z)
    mean, sd = ck.predict(xy[:10], return_std=True)
    np.testing.assert_allclose(mean, z[:10], atol=1e-6)
    assert sd.max() < 1e-3


def test_grid_posterior_concentrates_near_true_range():
    xy, z, *_ = _field(n=200, seed=1)
    bk = BayesianKriging("exponential", nugget_grid=[0.0, 0.05, 0.1, 0.25]).fit(xy, z)
    post = bk.posterior_
    assert np.isclose(post.weight.sum(), 1.0)
    by_range = post.groupby("range").weight.sum()
    mode = float(by_range.idxmax())
    assert 0.05 < mode < 0.45
    # posterior mean of the intercept near 5
    assert abs((post.weight * post.beta_intercept).sum() - 5.0) < 1.5


def test_mixture_predictions_consistent_with_samples():
    xy, z, *_ = _field(seed=2)
    xp = np.random.default_rng(3).uniform(0, 1, (15, 2))
    bk = BayesianKriging("exponential", range_grid=np.geomspace(0.03, 1.0, 12)).fit(xy, z)
    mean, sd = bk.predict(xp, return_std=True)
    q = bk.predict_quantiles(xp, [0.05, 0.5, 0.95])
    assert np.all(q[:, 0] < q[:, 1]) and np.all(q[:, 1] < q[:, 2])
    s = bk.predictive_samples(xp, 6000, seed=4)
    np.testing.assert_allclose(s.mean(0), mean, atol=0.08)
    np.testing.assert_allclose(s.std(0), sd, rtol=0.08)
    np.testing.assert_allclose(np.quantile(s, 0.95, axis=0), q[:, 2], atol=0.2)
    p = bk.predict_proba(xp, 6.0)
    np.testing.assert_allclose(p, (s > 6.0).mean(0), atol=0.03)
    np.testing.assert_allclose(bk.predict_proba(xp, 6.0, greater=False), 1 - p)


def test_joint_samples_have_conditional_covariance():
    xy, z, *_ = _field(seed=6)
    ck = ConjugateBayesianKriging("exponential", range_=0.15).fit(xy, z)
    xp = np.array([[0.5, 0.5], [0.51, 0.5], [0.9, 0.1]])
    s = ck.predictive_samples(xp, 20000, joint=True, seed=0)
    c = np.corrcoef(s.T)
    assert c[0, 1] > 0.8            # close points strongly correlated
    assert abs(c[0, 2]) < c[0, 1]


def test_uncertainty_wider_than_plugin():
    xy, z, *_ = _field(n=60, seed=7)
    xp = np.random.default_rng(8).uniform(0, 1, (30, 2))
    full = BayesianKriging("exponential").fit(xy, z)
    best = full.posterior_.sort_values("weight").iloc[-1]
    plug = ConjugateBayesianKriging("exponential", range_=best.range, nugget_ratio=best.nugget_ratio).fit(xy, z)
    _, sd_full = full.predict(xp, return_std=True)
    _, sd_plug = plug.predict(xp, return_std=True)
    assert sd_full.mean() > sd_plug.mean() * 0.98


def test_coverage_of_held_out_values():
    xy, z, latent, _, _ = _field(n=260, seed=9, nugget_sd=0.0)
    train, test = np.arange(200), np.arange(200, 260)
    bk = BayesianKriging("exponential", nugget_grid=[0.0, 0.05, 0.2]).fit(xy[train], z[train])
    q = bk.predict_quantiles(xy[test], [0.05, 0.95], signal=False)
    cover = np.mean((z[test] >= q[:, 0]) & (z[test] <= q[:, 1]))
    assert 0.75 <= cover <= 1.0


def test_informative_priors_and_trend():
    xy, z, *_ = _field(seed=10)
    flat = ConjugateBayesianKriging("exponential", range_=0.15, trend=1).fit(xy, z)
    vague = ConjugateBayesianKriging("exponential", range_=0.15, trend=1,
                                     beta_prior=("normal", np.zeros(3), 1e8 * np.eye(3))).fit(xy, z)
    xp = np.random.default_rng(1).uniform(0, 1, (10, 2))
    np.testing.assert_allclose(flat.predict(xp), vague.predict(xp), atol=1e-3)
    assert flat.beta_names_ == ["intercept", "x", "y"]
    strong = ConjugateBayesianKriging("exponential", range_=0.15,
                                      beta_prior=("normal", [100.0], [[1e-6]]),
                                      sigma2_prior=("scaled_inv_chi2", 5.0, 1.0)).fit(xy, z)
    assert abs(strong._comps[0]["beta"][0] - 100.0) < 1.0      # prior dominates the mean


def test_covariates_and_one_call():
    xy, z, *_ = _field(seed=11)
    cov = xy[:, 0] * 3.0
    bk = BayesianKriging("exponential", range_grid=[0.1, 0.2], nugget_grid=[0.0, 0.1]).fit(xy, z + cov, covariates=cov)
    assert bk.beta_names_ == ["intercept", "cov1"]
    xp = xy[:5]
    bk.predict(xp, covariates=cov[:5])
    with pytest.raises(ValueError):
        bk.predict(xp)
    out = bayesian_kriging(xy, z, xp, range_grid=[0.1, 0.2], nugget_grid=[0.0])
    assert list(out.columns) == ["mean", "sd", "q05", "q50", "q95"]


def test_from_variogram():
    xy, z, *_ = _field(seed=12)
    vg = Variogram(xy, z, model="exponential", n_lags=12).fit()
    ck = BayesianKriging.from_variogram(vg)
    assert isinstance(ck, ConjugateBayesianKriging)
    assert np.isclose(ck.range_grid[0], vg.fitted_params[2])
    ck.fit(xy, z).predict(xy[:3])


def test_sample_parameters():
    xy, z, *_ = _field(seed=13)
    bk = BayesianKriging("exponential", range_grid=[0.1, 0.2], nugget_grid=[0.0, 0.2]).fit(xy, z)
    s = bk.sample_parameters(200, seed=0)
    assert {"range", "nugget_ratio", "sigma2", "nugget", "beta_intercept"} <= set(s.columns)
    assert (s.sigma2 > 0).all()


def test_transformed_kriging_lognormal():
    xy, _, latent, _, _ = _field(n=150, seed=14)
    z = np.exp(0.5 * latent)
    bt = BayesianTransformedKriging("exponential", range_grid=np.geomspace(0.03, 1, 10),
                                    nugget_grid=[0.0, 0.1]).fit(xy, z)
    lam = bt.lambda_posterior_
    assert np.isclose(lam.sum(), 1.0)
    assert lam.loc[[-0.25, 0.0, 0.25]].sum() > 0.6        # near log
    pred = bt.predict(xy[:5], seed=0)
    assert (pred["q05"] > 0).all()
    assert np.all(pred["q05"] <= pred["median"]) and np.all(pred["median"] <= pred["q95"])
    with pytest.raises(ValueError):
        BayesianTransformedKriging().fit(xy, np.r_[-1.0, z[1:]])


def test_glm_poisson_recovers_latent_field():
    xy, _, latent, _, rng = _field(n=150, seed=15)
    eta = 0.5 + 0.8 * latent
    y = rng.poisson(np.exp(eta))
    g = BayesianGLMKriging("poisson", range_grid=np.geomspace(0.03, 1, 12)).fit(
        xy, y, n_iter=3000, burn=1000, thin=4, seed=0)
    assert 0.3 < g.acceptance_["gamma"] < 0.9
    assert np.corrcoef(g.latent_samples().mean(0), eta)[0, 1] > 0.5
    pred = g.predict(xy[:5], seed=0)
    assert (pred["mean"] > 0).all()
    pp = g.predict_proba(xy[:5], 1.0)
    assert np.all((pp >= 0) & (pp <= 1))
    assert len(g.trace_) == (3000 - 1000) // 4


def test_glm_binomial_and_validation():
    xy, _, latent, _, rng = _field(n=100, seed=16)
    trials = np.full(100, 8)
    y = rng.binomial(trials, 1 / (1 + np.exp(-latent)))
    g = BayesianGLMKriging("binomial", range_grid=np.geomspace(0.03, 1, 10)).fit(
        xy, y, trials_or_exposure=trials, n_iter=1500, burn=500, seed=1)
    p = g.predict(xy[:5], seed=0)
    assert ((p["mean"] > 0) & (p["mean"] < 1)).all()
    with pytest.raises(ValueError):
        BayesianGLMKriging("binomial").fit(xy, trials + 1, trials_or_exposure=trials)
    with pytest.raises(ValueError):
        BayesianGLMKriging("gamma")


def test_input_validation():
    with pytest.raises(ValueError):
        BayesianKriging("nope")
    with pytest.raises(ValueError):
        BayesianKriging(range_grid=[0.0, 1.0])
    with pytest.raises(ValueError):
        BayesianKriging(nugget_grid=[-0.1])
    xy, z, *_ = _field(n=30)
    with pytest.raises(ValueError):
        BayesianKriging(trend=3).fit(xy, z)
    with pytest.raises(ValueError):
        BayesianKriging(range_prior="weird").fit(xy, z)
    with pytest.raises(ValueError):
        BayesianKriging(sigma2_prior=("scaled_inv_chi2", -1, 1)).fit(xy, z)
    with pytest.raises(RuntimeError):
        BayesianKriging().predict(xy)
    with pytest.raises(ValueError):
        BayesianKriging().fit(xy, z[:-1])


def test_glm_interweaving_improves_intercept_mixing_and_options():
    xy, _, latent, _, rng = _field(n=120, seed=21)
    y = rng.poisson(np.exp(1.0 + 0.6 * latent))
    kw = dict(range_grid=np.geomspace(0.03, 1, 10))
    plain = BayesianGLMKriging("poisson", interweave=False, **kw).fit(xy, y, n_iter=3000, burn=1000, thin=1, seed=0)
    woven = BayesianGLMKriging("poisson", interweave=True, **kw).fit(xy, y, n_iter=3000, burn=1000, thin=1, seed=0)

    def lag1(x):
        x = np.asarray(x) - np.mean(x)
        return float((x[1:] * x[:-1]).sum() / (x * x).sum())

    assert lag1(woven.trace_["beta_intercept"]) < lag1(plain.trace_["beta_intercept"])
    assert abs(woven.trace_["beta_intercept"].mean() - 1.0) < 0.6
    with pytest.raises(ValueError):
        BayesianGLMKriging(range_gibbs_every=0)
    with pytest.raises(ValueError):
        BayesianGLMKriging(sigma2_prior=(0.0, 1.0))


def test_transformed_negative_lambda_warns_on_infinite_draws():
    xy, _, latent, _, _ = _field(n=100, seed=22)
    z = np.exp(1.5 * latent) + 0.01
    bt = BayesianTransformedKriging("exponential", lambda_grid=[-1.0], range_grid=[0.15], nugget_grid=[0.5]).fit(xy, z)
    far = np.array([[5.0, 5.0]])                      # far from data: wide predictive
    with pytest.warns(UserWarning, match="Box-Cox domain"):
        s = bt.predictive_samples(far, 4000, seed=0)
    assert np.isinf(s).any()
    assert np.isfinite(np.median(s))


def test_one_call_signal_and_return_model():
    xy, z, *_ = _field(seed=23)
    kw = dict(range_grid=[0.1, 0.2], nugget_grid=[0.0, 0.5])
    sig = bayesian_kriging(xy, z, xy[:5], **kw)
    obs, model = bayesian_kriging(xy, z, xy[:5], signal=False, return_model=True, **kw)
    assert isinstance(model, BayesianKriging)
    np.testing.assert_allclose(sig["mean"], obs["mean"])
    assert np.all(obs["sd"] >= sig["sd"] - 1e-12)          # observations include the nugget
    np.testing.assert_allclose(obs["mean"], model.predict(xy[:5], signal=False))


def test_stable_alpha_is_used():
    h = np.linspace(0, 3, 50)
    np.testing.assert_allclose(correlation_function(h, "stable", 1.0, alpha=1.0), correlation_function(h, "exponential", 1.0))
    np.testing.assert_allclose(correlation_function(h, "stable", 1.0, alpha=2.0), correlation_function(h, "gaussian", 1.0))
    assert not np.allclose(correlation_function(h, "stable", 1.0, alpha=1.5), correlation_function(h, "exponential", 1.0))
    with pytest.raises(ValueError):
        correlation_function(h, "stable", 1.0, alpha=2.5)
    xy, z, *_ = _field(seed=24)
    a = ConjugateBayesianKriging("stable", range_=0.15, nu=1.0).fit(xy, z).predict(xy[:3] + 0.01)
    b = ConjugateBayesianKriging("exponential", range_=0.15).fit(xy, z).predict(xy[:3] + 0.01)
    np.testing.assert_allclose(a, b, atol=1e-8)                      # nu doubles as alpha for 'stable'
