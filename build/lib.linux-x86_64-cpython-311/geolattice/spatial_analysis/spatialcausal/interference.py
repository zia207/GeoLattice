"""Direct, spillover and total effects under spatial interference.

Two estimators are provided, both built on an ``ExposureMapping``:

``method="aipw"`` (observational data)
    Doubly robust estimator for the mean potential outcome under each
    exposure condition. The generalised propensity ``P(C_i = c | X)`` is
    modelled from a unit's own covariates *and* the spatial lag of its
    neighbours' covariates, because neighbours' treatment depends on their
    covariates. Nuisance models are cross-fitted on spatial blocks.

``method="hajek"`` (randomised designs)
    Design-based Hájek estimator (Aronow & Samii, 2017) with exposure
    probabilities computed by simulating the known randomisation design.

Contrasts reported:

* ``direct``           mu(c10) - mu(c00): own treatment, low neighbour exposure
* ``spillover``        mu(c01) - mu(c00): neighbour exposure among controls
* ``spillover_treated``mu(c11) - mu(c10): neighbour exposure among treated
* ``total``            mu(c11) - mu(c00)
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from geolattice.spatial_analysis.spatialcausal._utils import (
    as_1d,
    as_2d,
    as_binary,
    as_coords,
    as_sparse_weights,
    check_lengths,
    make_learner,
)
from geolattice.spatial_analysis.spatialcausal.core.data import SpatialCausalData
from geolattice.spatial_analysis.spatialcausal.core.result import EffectEstimate, SpatialCausalResult
from geolattice.spatial_analysis.spatialcausal.crossfit import make_folds
from geolattice.spatial_analysis.spatialcausal.exposure import CONDITIONS, ExposureMapping, exposure_probabilities
from geolattice.spatial_analysis.spatialcausal.inference import standard_error

CONTRASTS = {
    "direct": (2, 0),
    "spillover": (1, 0),
    "spillover_treated": (3, 2),
    "total": (3, 0),
}


def _unpack(data_or_X, treatment, y, coords):
    if isinstance(data_or_X, SpatialCausalData):
        d = data_or_X
        return d.X, d.treatment, d.y, d.coords
    if treatment is None or y is None or coords is None:
        raise ValueError("pass a SpatialCausalData, or X, treatment, y and coords")
    return data_or_X, treatment, y, coords


class InterferenceEstimator:
    """Estimate direct and spillover effects under spatial interference.

    Parameters
    ----------
    exposure : ExposureMapping
        Defines neighbourhoods and the high/low exposure threshold.
    method : {"aipw", "hajek"}
    learner : str or sklearn regressor
        Outcome model (AIPW only).
    propensity_learner : str or sklearn classifier
        Multinomial model for the generalised propensity (AIPW only).
    design : BernoulliDesign / CompleteDesign / ClusterDesign
        Known randomisation design (Hájek only).
    cv, n_splits : cross-fitting scheme (default spatial blocks).
    trim : float
        Generalised propensities are clipped to ``[trim, 1]``.
    se : {"iid", "conley", "block"}
    cutoff : float
        Conley distance cutoff when ``se="conley"``.
    lag_features : bool
        Add spatially lagged covariates ``W X`` to the nuisance models.
    """

    def __init__(self, exposure: ExposureMapping, method: str = "aipw",
                 learner="gbr", propensity_learner="logistic", design=None,
                 cv="spatial_block", n_splits: int = 5, n_estimators: int = 100,
                 trim: float = 0.01, se: str = "iid", cutoff: Optional[float] = None,
                 lag_features: bool = True, n_sims: int = 2000, random_state: int = 0):
        if method not in ("aipw", "hajek"):
            raise ValueError("method must be 'aipw' or 'hajek'")
        if method == "hajek" and design is None:
            raise ValueError("method='hajek' requires the randomisation design")
        self.exposure = exposure
        self.method = method
        self.learner = learner
        self.propensity_learner = propensity_learner
        self.design = design
        self.cv = cv
        self.n_splits = n_splits
        self.n_estimators = n_estimators
        self.trim = trim
        self.se = se
        self.cutoff = cutoff
        self.lag_features = lag_features
        self.n_sims = n_sims
        self.random_state = random_state

    # ------------------------------------------------------------------
    def fit(self, data_or_X, treatment=None, y=None, coords=None):
        X, t, y, coords = _unpack(data_or_X, treatment, y, coords)
        X, t, y, coords = as_2d(X), as_binary(t), as_1d(y), as_coords(coords)
        n = check_lengths(X=X, t=t, y=y, coords=coords)
        C = self.exposure.conditions(t)
        self.conditions_ = C
        self.counts_ = {CONDITIONS[c]: int(np.sum(C == c)) for c in range(4)}
        for name, k in self.counts_.items():
            if k < 10:
                warnings.warn(f"only {k} units in exposure condition {name}; "
                              "estimates involving it will be unstable", RuntimeWarning)
        if self.method == "aipw":
            gamma, pi = self._aipw_scores(X, t, y, coords, C)
            self.scores_ = gamma
            self.propensity_ = pi
            mu = gamma.mean(axis=0)
            psis = {k: gamma[:, a] - gamma[:, b] for k, (a, b) in CONTRASTS.items()}
            used = {k: np.ones(n, bool) for k in CONTRASTS}
        else:
            pi = exposure_probabilities(self.exposure, self.design, n, self.n_sims, self.random_state)
            self.propensity_ = pi
            mu, psis, used = self._hajek(y, C, pi)
        self.condition_means_ = dict(zip(CONDITIONS, mu))

        effects = {}
        for k, (a, b) in CONTRASTS.items():
            if np.isnan(mu[a]) or np.isnan(mu[b]):
                continue
            m = used[k]
            se = standard_error(psis[k][m], coords[m], self.se, self.cutoff)
            effects[k] = EffectEstimate(k, mu[a] - mu[b], se)
        self.result_ = SpatialCausalResult(
            f"Spatial interference ({self.method.upper()})", effects, se_method=self.se, n=n,
            exposure=repr(self.exposure), condition_counts=self.counts_)
        return self

    # ------------------------------------------------------------------
    def _features(self, X, coords, n):
        F = X
        if self.lag_features:
            W = as_sparse_weights(self.exposure.w, n, row_standardize=True)
            F = np.hstack([X, np.asarray(W @ X)])
        # standardise: the multinomial propensity model converges far faster on scaled
        # features (8x in the county tutorial); tree learners are unaffected
        sd = F.std(0)
        return (F - F.mean(0)) / np.where(sd > 0, sd, 1.0)

    def _aipw_scores(self, X, t, y, coords, C):
        n = len(y)
        F = self._features(X, coords, n)
        folds = make_folds(coords, self.cv, self.n_splits, self.random_state)
        pi = np.full((n, 4), np.nan)
        m = np.full((n, 4), np.nan)
        for train, test in folds:
            clf = make_learner(self.propensity_learner, "classification", self.n_estimators, self.random_state)
            clf.fit(F[train], C[train])
            proba = np.zeros((len(test), 4))
            proba[:, clf.classes_.astype(int)] = clf.predict_proba(F[test])
            pi[test] = proba
            for c in range(4):
                tr_c = train[C[train] == c]
                if len(tr_c) >= 10:
                    reg = make_learner(self.learner, "regression", self.n_estimators, self.random_state)
                    reg.fit(F[tr_c], y[tr_c])
                    m[test, c] = reg.predict(F[test])
                else:  # pooled fallback with condition indicators
                    D = np.eye(4)[C[train]]
                    reg = make_learner(self.learner, "regression", self.n_estimators, self.random_state)
                    reg.fit(np.hstack([F[train], D]), y[train])
                    Dc = np.zeros((len(test), 4))
                    Dc[:, c] = 1
                    m[test, c] = reg.predict(np.hstack([F[test], Dc]))
        if np.isnan(pi).any():
            raise RuntimeError("some units were never in a test fold; check the cv argument")
        raw_min = pi.min()
        pi = np.clip(pi, self.trim, 1.0)
        if raw_min < self.trim:
            warnings.warn(f"generalised propensities below {self.trim} were trimmed "
                          f"(min {raw_min:.4f}); check overlap", RuntimeWarning)
        ind = np.eye(4)[C]
        gamma = m + ind * (y[:, None] - m) / pi
        return gamma, pi

    @staticmethod
    def _hajek(y, C, pi):
        n = len(y)
        ind = np.eye(4)[C]
        mu = np.full(4, np.nan)
        phi = np.zeros((n, 4))
        for c in range(4):
            ok = pi[:, c] > 0
            if not np.any(ind[ok, c]):
                continue
            wts = np.where(ok, ind[:, c] / np.where(ok, pi[:, c], 1.0), 0.0)
            mu[c] = np.sum(wts * y) / np.sum(wts)
            phi[:, c] = wts * (y - mu[c]) / np.mean(wts)
        psis, used = {}, {}
        for k, (a, b) in CONTRASTS.items():
            used[k] = (pi[:, a] > 0) & (pi[:, b] > 0)
            psis[k] = phi[:, a] - phi[:, b]
        return mu, psis, used

    # ------------------------------------------------------------------
    def effect(self, name: str = "direct") -> float:
        return self.result_[name].estimate

    def summary(self) -> str:
        return self.result_.summary()
