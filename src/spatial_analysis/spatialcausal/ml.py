"""Causal machine learning with spatial structure.

Three things change relative to ordinary DML / DR-learning:

1. **Spatial features.** Nuisance models get coordinates, spatially lagged
   covariates and/or Moran eigenvectors, so a smooth unmeasured confounder
   can be absorbed by ``E[Y | X, s]`` and ``E[T | X, s]``.
2. **Spatial cross-fitting.** Folds are spatial blocks (optionally with a
   buffer), so out-of-fold predictions are really out of sample.
3. **Spatially robust SEs.** Conley or block-cluster variance of the
   orthogonal score.

``PyCausalMLAdapter`` passes the same spatial features to any PyCausalML
estimator.
"""
from __future__ import annotations

import importlib
from typing import Optional, Sequence, Union

import numpy as np

from geolattice.spatial_analysis.spatialcausal._utils import (
    as_1d,
    as_2d,
    as_coords,
    as_sparse_weights,
    check_lengths,
    clip_propensity,
    default_weights,
    make_learner,
)
from geolattice.spatial_analysis.spatialcausal.core.data import SpatialCausalData
from geolattice.spatial_analysis.spatialcausal.core.result import EffectEstimate, SpatialCausalResult
from geolattice.spatial_analysis.spatialcausal.crossfit import cross_fit_predict, make_folds
from geolattice.spatial_analysis.spatialcausal.inference import standard_error

FeatureSpec = Union[str, Sequence[str], None]


# ----------------------------------------------------------------------------
# Spatial feature engineering
# ----------------------------------------------------------------------------
def spatial_features(X, coords, w=None, spatial_features: FeatureSpec = ("coords", "lag"),
                     k_eigen="auto", k_rbf="auto", random_state: int = 0,
                     coord_ref=None) -> np.ndarray:
    """Append spatial features to ``X``.

    ``spatial_features`` is any combination of:

    * ``"coords"`` — standardised coordinates (lets trees learn a surface)
    * ``"lag"``    — spatial lag ``W X`` (neighbourhood covariates)
    * ``"eigen"``  — Moran eigenvectors of ``W``
    * ``"rbf"``    — radial basis functions on coordinates

    ``coord_ref=(mean, std)`` standardises coordinates with training-set
    values, so features for new units are on the same scale.

    ``k_eigen`` / ``k_rbf`` default to ``"auto"``: ``min(200, max(20, n // 30))``
    eigenvectors and ``min(200, max(30, n // 20))`` radial basis functions.
    A basis that is too coarse cannot represent a confounder that varies
    over short distances; with 3,109 counties and a ~350 km confounder,
    30 basis functions left most of the confounding bias in place.
    """
    from geolattice.spatial_analysis.spatialcausal.confounding import moran_eigenvectors, rbf_basis

    X = as_2d(X)
    if spatial_features is None:
        return X
    if isinstance(spatial_features, str):
        spatial_features = (spatial_features,)
    coords = as_coords(coords)
    n = len(X)
    k_eigen = min(200, max(20, n // 30)) if k_eigen == "auto" else int(k_eigen)
    k_rbf = min(200, max(30, n // 20)) if k_rbf == "auto" else int(k_rbf)
    blocks = [X]
    w = w if w is not None else (default_weights(coords) if {"lag", "eigen"} & set(spatial_features) else None)
    for f in spatial_features:
        if f == "coords":
            mu, sd = coord_ref if coord_ref is not None else (coords.mean(0), coords.std(0))
            blocks.append((coords - mu) / sd)
        elif f == "lag":
            W = as_sparse_weights(w, len(X), row_standardize=True)
            blocks.append(np.asarray(W @ X))
        elif f == "eigen":
            blocks.append(moran_eigenvectors(w, k_eigen, len(X)))
        elif f == "rbf":
            blocks.append(rbf_basis(coords, k_rbf, random_state=random_state))
        else:
            raise ValueError(f"unknown spatial feature '{f}'")
    return np.hstack(blocks)


def _unpack(data_or_X, treatment, y, coords, w):
    if isinstance(data_or_X, SpatialCausalData):
        d = data_or_X
        return d.X, d.treatment, d.y, d.coords, (w if w is not None else d.w)
    if treatment is None or y is None or coords is None:
        raise ValueError("pass a SpatialCausalData, or X, treatment, y and coords")
    return as_2d(data_or_X), as_1d(treatment), as_1d(y), as_coords(coords), w


def _is_binary(t) -> bool:
    return bool(np.all(np.isin(t, (0.0, 1.0))))


# ----------------------------------------------------------------------------
# Shared nuisance stages (also used by geolattice.spatial_analysis.spatialcausal.local)
# ----------------------------------------------------------------------------
def dml_residuals(F, t, y, folds, model_y="rf", model_t="rf", n_estimators=200,
                  trim=0.01, random_state=0):
    """Cross-fitted partialling-out residuals ``u = y - E[y|F]``, ``v = t - E[t|F]``.

    Returns ``(u, v, l_hat, m_hat)``. A binary ``t`` uses a classifier for ``E[t|F]``.
    """
    rs, ne = random_state, n_estimators
    l_hat = cross_fit_predict(lambda: make_learner(model_y, "regression", ne, rs), F, y, folds)
    if _is_binary(t):
        m_hat = cross_fit_predict(lambda: make_learner(model_t, "classification", ne, rs),
                                  F, t.astype(int), folds, method="predict_proba")
        m_hat = clip_propensity(m_hat, trim)
    else:
        m_hat = cross_fit_predict(lambda: make_learner(model_t, "regression", ne, rs), F, t, folds)
    return y - l_hat, t - m_hat, l_hat, m_hat


def aipw_pseudo_outcomes(F, t, y, folds, model_y="gbr", model_t="gbr", n_estimators=200,
                         trim=0.01, random_state=0):
    """Cross-fitted AIPW pseudo-outcomes for a binary treatment.

    ``gamma_i = mu1 - mu0 + t (y - mu1) / e - (1 - t)(y - mu0) / (1 - e)``;
    ``E[gamma | X, s] = tau(X, s)``. Returns ``(gamma, e_hat, mu0, mu1)``.
    """
    t = np.asarray(t).astype(int)
    n = len(y)
    rs, ne = random_state, n_estimators
    e = clip_propensity(cross_fit_predict(
        lambda: make_learner(model_t, "classification", ne, rs), F, t, folds, "predict_proba"), trim)
    mu0, mu1 = np.full(n, np.nan), np.full(n, np.nan)
    for train, test in folds:
        for arm, store in ((0, mu0), (1, mu1)):
            tr = train[t[train] == arm]
            reg = make_learner(model_y, "regression", ne, rs).fit(F[tr], y[tr])
            store[test] = reg.predict(F[test])
    gamma = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    return gamma, e, mu0, mu1


# ----------------------------------------------------------------------------
# Spatial double machine learning (partially linear model)
# ----------------------------------------------------------------------------
class SpatialDML:
    """Partially linear DML with spatial features, folds and SEs.

    Model: ``Y = theta * T + g(X, s) + e``,  ``T = m(X, s) + v``.

    Parameters
    ----------
    model_y, model_t : str or sklearn estimator
        Learners for ``E[Y|X,s]`` and ``E[T|X,s]``. For a binary
        treatment ``model_t`` is used as a classifier.
    spatial_features : see :func:`spatial_features`
    cv : {"spatial_block", "random"}, ``SpatialBlockKFold``, or folds list
    n_splits : int
    buffer : float
        Buffer distance for spatial blocks (ignored for other cv).
    se : {"iid", "conley", "block"}
    cutoff : float
        Conley cutoff (coordinate units).
    """

    def __init__(self, model_y="rf", model_t="rf", spatial_features: FeatureSpec = ("coords", "lag"),
                 cv="spatial_block", n_splits: int = 5, buffer: float = 0.0,
                 n_estimators: int = 200, se: str = "iid", cutoff: Optional[float] = None,
                 trim: float = 0.01, random_state: int = 0):
        self.model_y, self.model_t = model_y, model_t
        self.spatial_features = spatial_features
        self.cv, self.n_splits, self.buffer = cv, n_splits, buffer
        self.n_estimators = n_estimators
        self.se, self.cutoff, self.trim = se, cutoff, trim
        self.random_state = random_state

    def fit(self, data_or_X, treatment=None, y=None, coords=None, w=None):
        X, t, y, coords, w = _unpack(data_or_X, treatment, y, coords, w)
        n = check_lengths(X=X, t=t, y=y, coords=coords)
        F = spatial_features(X, coords, w, self.spatial_features, random_state=self.random_state)
        kw = {"buffer": self.buffer} if self.cv == "spatial_block" else {}
        folds = make_folds(coords, self.cv, self.n_splits, self.random_state, **kw)
        u, v, l_hat, m_hat = dml_residuals(F, t, y, folds, self.model_y, self.model_t,
                                           self.n_estimators, self.trim, self.random_state)
        theta = float(np.sum(v * u) / np.sum(v * v))
        psi = v * (u - theta * v) / np.mean(v * v)
        se = standard_error(psi, coords, self.se, self.cutoff)
        self.residuals_ = {"y": u, "t": v}
        self.nuisance_ = {"l_hat": l_hat, "m_hat": m_hat}
        self.score_ = psi
        r2_t = 1 - np.var(v) / np.var(t)
        self.result_ = SpatialCausalResult(
            "Spatial DML (partially linear)", {"ate": EffectEstimate("ate", theta, se)},
            se_method=self.se, n=n, features=str(self.spatial_features),
            cv=str(self.cv), treatment_r2_from_nuisance=round(float(r2_t), 3))
        return self

    def effect(self) -> float:
        return self.result_["ate"].estimate

    def confidence_interval(self, level: float = 0.95):
        e = self.result_["ate"]
        return EffectEstimate(e.name, e.estimate, e.se, level).ci

    def summary(self) -> str:
        return self.result_.summary()


# ----------------------------------------------------------------------------
# Spatial DR-learner (binary treatment; ATE + CATE surface)
# ----------------------------------------------------------------------------
class SpatialDRLearner:
    """Doubly robust learner with spatial cross-fitting.

    Produces an ATE (AIPW) with spatially robust SE and a CATE function
    from regressing the AIPW pseudo-outcome on ``cate_features``:
    ``"X"``, ``"coords"`` (a pure map of effect heterogeneity) or
    ``"X+coords"``. Include only features you believe modify the effect;
    pseudo-outcomes are noisy and irrelevant features dilute the fit.

    ``cate_`` holds *out-of-fold* CATE predictions. Nuisance models are
    cross-fitted on spatial blocks; the final stage is cross-fitted with
    ``final_cv`` (default random folds, because the CATE surface should be
    interpolated, not extrapolated into held-out regions).
    ``final_model="auto"`` is a random forest with ``min_samples_leaf =
    max(15, n // 100)``.
    """

    def __init__(self, model_y="gbr", model_t="gbr", final_model="auto",
                 spatial_features: FeatureSpec = ("coords", "lag"), cate_features: str = "X+coords",
                 final_cv: str = "random",
                 cv="spatial_block", n_splits: int = 5, buffer: float = 0.0, n_estimators: int = 200,
                 se: str = "iid", cutoff: Optional[float] = None, trim: float = 0.01,
                 random_state: int = 0):
        self.model_y, self.model_t, self.final_model = model_y, model_t, final_model
        self.spatial_features = spatial_features
        if cate_features not in ("X", "coords", "X+coords"):
            raise ValueError("cate_features must be 'X', 'coords' or 'X+coords'")
        self.cate_features = cate_features
        self.final_cv = final_cv
        self.cv, self.n_splits, self.buffer = cv, n_splits, buffer
        self.n_estimators = n_estimators
        self.se, self.cutoff, self.trim = se, cutoff, trim
        self.random_state = random_state

    def _cate_features(self, X, coords):
        X = as_2d(X)
        if self.cate_features == "X":
            return X
        c = (as_coords(coords) - self._cmean) / self._cstd
        return c if self.cate_features == "coords" else np.hstack([X, c])

    def fit(self, data_or_X, treatment=None, y=None, coords=None, w=None):
        X, t, y, coords, w = _unpack(data_or_X, treatment, y, coords, w)
        if not _is_binary(t):
            raise ValueError("SpatialDRLearner needs a binary treatment; use SpatialDML for continuous")
        t = t.astype(int)
        n = check_lengths(X=X, t=t, y=y, coords=coords)
        F = spatial_features(X, coords, w, self.spatial_features, random_state=self.random_state)
        kw = {"buffer": self.buffer} if self.cv == "spatial_block" else {}
        folds = make_folds(coords, self.cv, self.n_splits, self.random_state, **kw)
        rs, ne = self.random_state, self.n_estimators
        gamma, e, _, _ = aipw_pseudo_outcomes(F, t, y, folds, self.model_y, self.model_t,
                                              ne, self.trim, rs)
        ate = float(gamma.mean())
        se = standard_error(gamma, coords, self.se, self.cutoff)
        self._cmean, self._cstd = coords.mean(0), coords.std(0)
        Fc = self._cate_features(X, coords)

        def final():
            if self.final_model == "auto":
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(n_estimators=max(ne, 200), min_samples_leaf=max(15, n // 100),
                                             random_state=rs, n_jobs=-1)
            return make_learner(self.final_model, "regression", ne, rs)

        final_folds = folds if self.final_cv == "same" else make_folds(coords, self.final_cv, self.n_splits, rs)
        self.cate_ = cross_fit_predict(final, Fc, gamma, final_folds)
        self.final_ = final().fit(Fc, gamma)
        self.pseudo_outcome_ = gamma
        self.propensity_ = e
        self.result_ = SpatialCausalResult(
            "Spatial DR-learner", {"ate": EffectEstimate("ate", ate, se)}, se_method=self.se, n=n,
            propensity_range=(round(float(e.min()), 3), round(float(e.max()), 3)))
        return self

    def predict(self, X, coords=None) -> np.ndarray:
        """CATE for new units (coords required unless ``cate_features='X'``)."""
        if self.cate_features != "X" and coords is None:
            raise ValueError("coords are required for this cate_features setting")
        return self.final_.predict(self._cate_features(X, coords))

    def effect(self) -> float:
        return self.result_["ate"].estimate

    def summary(self) -> str:
        return self.result_.summary()


# ----------------------------------------------------------------------------
# Adapter for PyCausalML estimators
# ----------------------------------------------------------------------------
_PYCAUSALML_MODULES = (
    "pycausalml",
    "pycausalml.meta_learners",
    "pycausalml.double_machine_learning",
    "pycausalml.tree_based_models",
    "pycausalml.uplift_learners",
)


def _resolve_pycausalml(name: str):
    errors = []
    for mod in _PYCAUSALML_MODULES:
        try:
            m = importlib.import_module(mod)
        except Exception as exc:  # torch missing, etc.
            errors.append(f"{mod}: {exc}")
            continue
        if hasattr(m, name):
            return getattr(m, name)
    raise ImportError(
        f"Could not load PyCausalML estimator '{name}'. PyCausalML is not on PyPI; install it with "
        "`pip install https://github.com/zia207/PyCausalML/releases/download/v0.2.0/pycausalml-0.2.0-py3-none-any.whl`. "
        "Details: " + "; ".join(errors[:2]))


class PyCausalMLAdapter:
    """Run any PyCausalML estimator on spatially augmented features.

    Parameters
    ----------
    estimator : str or estimator instance
        A PyCausalML class name (``"CausalForest"``, ``"DRLearner"``,
        ``"PartiallyLinearDML"``, ``"XLearner"``…) or an instance with a
        ``fit(X, treatment, y)`` method.
    spatial_features : see :func:`spatial_features`
    **kwargs : passed to the estimator constructor when ``estimator`` is a name.

    Notes
    -----
    PyCausalML estimators cross-fit with random folds internally, and
    their confidence intervals assume independent units. Use this adapter
    for heterogeneity (CATE) and model comparison; use :class:`SpatialDML`
    or :class:`SpatialDRLearner` when you need spatially valid inference.
    """

    def __init__(self, estimator, spatial_features: FeatureSpec = ("coords", "lag"), **kwargs):
        self.estimator = estimator
        self.spatial_features = spatial_features
        self.kwargs = kwargs

    def fit(self, data_or_X, treatment=None, y=None, coords=None, w=None):
        X, t, y, coords, w = _unpack(data_or_X, treatment, y, coords, w)
        self._w, self._coords_train = w, coords
        self._coord_ref = (coords.mean(0), coords.std(0))
        F = spatial_features(X, coords, w, self.spatial_features, coord_ref=self._coord_ref)
        est = self.estimator
        if isinstance(est, str):
            est = _resolve_pycausalml(est)(**self.kwargs)
        self.estimator_ = est.fit(F, t, y) or est
        if not hasattr(self.estimator_, "predict") and hasattr(est, "predict"):
            self.estimator_ = est
        return self

    def transform(self, X, coords, w=None) -> np.ndarray:
        """Spatial features for new data.

        ``"eigen"`` and ``"rbf"`` features are specific to the training sample
        and cannot be computed for new units; ``"coords"`` and ``"lag"`` can.
        For ``"lag"``, pass ``w`` for the new units (default: 8-NN among them).
        """
        feats = self.spatial_features or ()
        feats = (feats,) if isinstance(feats, str) else tuple(feats)
        same = len(as_2d(X)) == len(self._coords_train) and np.allclose(as_coords(coords), self._coords_train)
        if not same and {"eigen", "rbf"} & set(feats):
            raise ValueError("'eigen'/'rbf' features cannot be built for new units; "
                             "refit with spatial_features=('coords', 'lag') to predict out of sample")
        if same:
            w = self._w
        elif w is None and "lag" in feats:
            w = default_weights(as_coords(coords))
        return spatial_features(X, coords, w, self.spatial_features, coord_ref=self._coord_ref)

    def predict(self, X, coords, w=None):
        return self.estimator_.predict(self.transform(X, coords, w))

    def __getattr__(self, item):
        if item.startswith("_") or item in ("estimator_",):
            raise AttributeError(item)
        return getattr(self.estimator_, item)
