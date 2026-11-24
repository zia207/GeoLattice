"""
Model validation and spatial cross-validation.

Modules: ``cross_validation`` (`loo_cv`, `kfold_cv`), ``spatial_cv_rk``,
``spatial_cv_pytorch_rk``, ``spatial_cv_cokriging``, ``spatial_cv_tf``,
``spatial_cv_IK``, ``spatial_cv_h2o_rk`` and ``spatial_cv_automl``.

``geolattice.geostatistics.validation`` used to be the cross-validation module;
`loo_cv` and `kfold_cv` are re-exported here so old imports keep working.
"""

from .cross_validation import kfold_cv, loo_cv

__all__ = ["loo_cv", "kfold_cv"]
