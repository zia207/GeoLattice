"""
Geostatistics module for GeoLattice.

GPU-capable variograms, the kriging family, sequential simulation, GSLIB
helpers, and validation utilities (ported from ``pygstat`` / gstat)::

    from geolattice.geostatistics import Variogram, OrdinaryKriging
    from geolattice.geostatistics import RegressionKriging, IndicatorKriging
    import geolattice.geostatistics as gstat

Optional deep-learning / AutoML backends require extras such as
``geolattice[geostatistics-pytorch]`` or ``geolattice[geostatistics-h2o]``; when they
are missing the corresponding classes are stubs that raise ``ImportError``
on use.
"""

import importlib
import importlib.abc

from geolattice.install_options import install_hint
import importlib.util
import sys


def _optional_import(module_path, names, extra_name):
    """
    Import `names` from `module_path`. If the module (or one of its own heavy
    dependencies, e.g. h2o/torch/tensorflow) fails to import, bind each name
    to a stub that raises a clear, actionable ImportError only when actually
    *used* -- instead of breaking `import geolattice.geostatistics` entirely for everyone,
    including users who only need core Variogram/Kriging functionality.
    """
    try:
        mod = importlib.import_module(module_path, __name__)
        return {n: getattr(mod, n) for n in names}
    except Exception as e:
        # Broad on purpose: a broken optional backend can fail with all sorts
        # of exceptions, not just ImportError (e.g. protobuf VersionError from
        # a TensorFlow/protobuf version mismatch, OSError for a missing CUDA
        # shared library, etc.). None of those should break `import geolattice.geostatistics`.
        reason = f"{type(e).__name__}: {e}"

        def _make_stub(n):
            class _MissingOptionalDependency:
                def __init__(self, *a, **kw):
                    raise ImportError(
                        f"geolattice.geostatistics.{n} could not be loaded because an "
                        f"optional dependency is missing or broken in this "
                        f"environment ({reason}). Install it with: "
                        f"{install_hint('geostatistics-' + extra_name)}"
                    )
            _MissingOptionalDependency.__name__ = n
            return _MissingOptionalDependency

        return {n: _make_stub(n) for n in names}


# Core variogram and kriging (required deps only -- always available)
from .ml.regression_kriging import (
    RegressionKriging,
    GAMRegressor,
    ENSEMBLE_REGRESSORS,
    STATISTICAL_REGRESSORS,
    AVAILABLE_REGRESSORS,
    enforce_quantile_monotonicity,
)
from .core.variogram import Variogram
from .core.kriging import OrdinaryKriging, SimpleKriging
from .core.variogram_cloud import VariogramCloud, compute_variogram_cloud, plot_variogram_cloud
from .kriging.cokriging import (
    Cokriging,
    CrossVariogram,
    MultivariateCokriging,
    ColocatedCokriging,
    fit_lmc,
    check_lmc_validity,
)
from .kriging.universal_kriging import UniversalKriging
from .core.indicator_variogram import fit_indicator_variogram
from .kriging.indicator_kriging import IndicatorKriging
from .kriging.disjunctive_kriging import (
    DisjunctiveKriging,
    hermite_polynomials,
    hermite_indicator_coefficients,
)
from .kriging.ebk import EmpiricalBayesianKriging
from .kriging.bayesian_kriging import (
    BayesianKriging,
    ConjugateBayesianKriging,
    BayesianTransformedKriging,
    BayesianGLMKriging,
    bayesian_kriging,
    correlation_function,
)
from .kriging.etype_estimates import compute_etype_from_probabilities
from .kriging.poisson_kriging import PoissonKriging, PointSupport, AreaPoissonKriging
from .kriging import multi_poisson_kriging
from .spacetime.krige_st import (
    krige_st,
    krige_st_df,
    krige_st_local,
    krige_st_tg,
    krige_st_tg_local,
    cov_fn_st,
    vgm_area_st,
)
from .spacetime.st_variogram_models import (
    vgm,
    vgm_st,
    variogram_line,
    variogram_surface,
    fit_st_variogram,
    extract_par,
    insert_par,
    extract_par_names,
    empirical_st_variogram,
    fit_metric_st_variogram,
    joint_nugget_sill_range,
)
from .spacetime.STPoisson_kriging import STPoissonKriging, load_atlantic_panel
from .spacetime.krigeST import STKriging, load_ca_pm25_panel, make_prediction_grid

# Validation (updated to expose loo_cv and kfold_cv separately)
from .validation.cross_validation import loo_cv, kfold_cv
from .validation.spatial_cv_rk import KFoldCV_RK, LOOCV_RK
from .validation.spatial_cv_pytorch_rk import KFoldCV_PyTorch_RK
from .validation.spatial_cv_cokriging import KFoldCV_Cokriging
from .validation.spatial_cv_tf import KFoldCV_TF_RK
from .validation.spatial_cv_IK import IndicatorKrigingCV

# Optional heavy backends: H2O AutoML, PyTorch, TensorFlow.
# Each is only imported by the module(s) that need it, so a missing/broken
# backend never breaks the rest of geolattice.geostatistics.
globals().update(_optional_import(
    ".ml.regression_kriging_h2o", ["RegressionKrigingH2O", "H2O_MODEL_TYPES"], "h2o"
))
globals().update(_optional_import(
    ".ml.regression_kriging_pytorch_gpu", ["DeepRegressionKriging"], "pytorch"
))
globals().update(_optional_import(
    ".ml.regression_kriging_gnn", ["GNNRegressionKriging", "KCN", "IGNNK"], "pytorch"
))
globals().update(_optional_import(
    ".ml.kriging_STGNN", ["STGNNRegressionKriging", "TGCN", "DCRNN"], "pytorch"
))
globals().update(_optional_import(
    ".ml.kriging_STGTN", ["STGTNRegressionKriging", "STTN", "GMAN"], "pytorch"
))
globals().update(_optional_import(
    ".ml.kriging_CNNLSTM", ["CNNLSTMRegressionKriging", "ConvLSTM"], "pytorch"
))
# TensorFlow's GPU init loads a second CUDA stack and then CuPy's cuSOLVER
# handle creation fails (CUSOLVER_STATUS_INTERNAL_ERROR). Import it only when
# DeepRegressionKrigingTF is actually used, so ordinary CuPy kriging keeps
# the GPU.
_LAZY_EXPORTS = {
    "DeepRegressionKrigingTF": (
        ".ml.regression_kriging_tf",
        "DeepRegressionKrigingTF",
        "tensorflow",
    ),
    # JAX ports of the core variogram and kriging classes. Distinct names so
    # they do not replace the NumPy/CuPy Variogram and kriging classes.
    "JaxVariogram": (".jax_variogram_kriging", "Variogram", "jax"),
    "JaxOrdinaryKriging": (".jax_variogram_kriging", "OrdinaryKriging", "jax"),
    "JaxSimpleKriging": (".jax_variogram_kriging", "SimpleKriging", "jax"),
    "JaxUniversalKriging": (".jax_variogram_kriging", "UniversalKriging", "jax"),
    "JaxCokriging": (".jax_variogram_kriging", "Cokriging", "jax"),
}
globals().update(_optional_import(
    ".validation.spatial_cv_h2o_rk",
    ["KFoldCV_H2O_RK", "KFoldCV_H2O_ResidualRK", "H2OResidualKriging"],
    "h2o",
))
globals().update(_optional_import(
    ".validation.spatial_cv_automl",
    ["KFoldCV_AutoML", "LOOCV_AutoML"],
    "h2o",
))

# I/O helpers
from .io.geopandas_io import from_geodataframe

# GSLIB-equivalent helper utilities (Geo-EAS I/O, normal-score transform,
# log/Box-Cox transform, cell declustering, grid coordinate generation).
from .gslib.common import read_gslib, write_gslib, read_gslib_grid
from .gslib.nscore import nscore_forward, nscore_back, nscore_file
# `backtr.py`'s public function is also named `backtr`; import it under a
# distinct name so it doesn't collide with the `geolattice.geostatistics.backtr` submodule.
from .gslib.backtr import backtr as backtr_transform
from .gslib.trans import transform_log, back_transform_log, transform_boxcox, back_transform_boxcox
from .gslib.declus import declus_cell
from .gslib.addcoord import add_coordinates_to_grid, addcoord
from .simulation.sgsim import sgsim
from .simulation.sisim import sisim
from .simulation.sis_local_means import (
    order_relations,
    residual_covariances,
    sis_local_means,
    sk_correction,
    sklm_cdf,
)
from .simulation.lusim import LUSimulator, lusim, lusim_unconditional
from .simulation.tbsim import TBSimulator, tbsim, tbsim_unconditional
from .simulation.fftsim import FFTSimulator, fftsim, fftsim_unconditional
from .mps.direct_sampling import (
    DirectSampling,
    channel_training_image,
    connectivity,
    dssim,
    pattern_histogram,
    pattern_histogram_distance,
)
from .mps.snesim import SNESIM, snesim, tau_model
from .simulation.sicosim import IndicatorCoSimulator, SoftCalibration, sicosim
from .simulation.sgcosim import GaussianCoSimulator, sgcosim, sgcosim_hierarchical, sgsim_gpu
from .simulation.direct_sequential import DirectSequentialSimulator, GlobalDistribution, dssim_direct
from .simulation.annealing import SimulatedAnnealing, sasim
from .simulation.pgs import (
    PluriGaussianSimulator,
    TruncationRule,
    category_probabilities,
    category_proportions,
    pgsim,
)
from .kriging.kt3d import kt3d, kt3d_grid, kt3d_cross_validate, build_grid_3d
from .kriging.soft_kriging import (
    SoftKriging,
    indicator_transform,
    markov_bayes_calibrate,
    fit_soft_probability_model,
)
from .kriging.factorial_kriging import FactorialKriging, fit_nested_variogram

# Core variogram model helpers, multivariate Poisson cokriging helpers, and
# backend / anisotropy utilities.
from .core import (  # noqa: E402
    MODEL_FUNCS,
    covariance_from_variogram,
    cubic,
    exponential,
    gaussian,
    matern,
    spherical,
    stable,
)
from .kriging.multi_poisson_kriging import (  # noqa: E402
    LMC,
    c_exp,
    c_sph,
    covariance_model,
    eb_risk,
    fit_lmc as fit_poisson_lmc,
    pck_crossvariogram,
    pck_variogram,
    poisson_cokrige,
    poisson_krige,
    shared_risk,
)
from .utils import (  # noqa: E402
    CUPY_AVAILABLE,
    asarray,
    resolve_torch_device,
    seed_torch,
    to_numpy,
    transform_aniso,
)

__version__ = "0.3.49"


# ---------------------------------------------------------------------------
# Backward-compatible module paths
# ---------------------------------------------------------------------------
# Modules are grouped in subpackages (core, kriging, spacetime, ml,
# validation, simulation, mps, gpu, gslib, io, utils). Every former flat
# module path -- e.g. ``geolattice.geostatistics.sgsim`` or
# ``from geolattice.geostatistics.kriging_STGNN import DCRNNCell`` -- still
# resolves to the module's new home (the same module object). Modules
# already imported above are registered in ``sys.modules`` directly, which
# also stops ``import geolattice.geostatistics.sgsim`` from rebinding the
# package's ``sgsim`` function to the module; the optional / GPU modules
# resolve lazily through a small import hook, so nothing heavy is imported.
_LEGACY_MODULES = {
    "indicator_variogram": "core.indicator_variogram",
    "dss_gpu": "gpu.dss_gpu",
    "mps_gpu": "gpu.mps_gpu",
    "sgs_gpu": "gpu.sgs_gpu",
    "sis_gpu": "gpu.sis_gpu",
    "tbsim_gpu": "gpu.tbsim_gpu",
    "addcoord": "gslib.addcoord",
    "backtr": "gslib.backtr",
    "common": "gslib.common",
    "declus": "gslib.declus",
    "nscore": "gslib.nscore",
    "trans": "gslib.trans",
    "cokriging": "kriging.cokriging",
    "disjunctive_kriging": "kriging.disjunctive_kriging",
    "ebk": "kriging.ebk",
    "etype_estimates": "kriging.etype_estimates",
    "factorial_kriging": "kriging.factorial_kriging",
    "indicator_kriging": "kriging.indicator_kriging",
    "kt3d": "kriging.kt3d",
    "multi_poisson_kriging": "kriging.multi_poisson_kriging",
    "poisson_kriging": "kriging.poisson_kriging",
    "soft_kriging": "kriging.soft_kriging",
    "universal_kriging": "kriging.universal_kriging",
    "kriging_CNNLSTM": "ml.kriging_CNNLSTM",
    "kriging_STGNN": "ml.kriging_STGNN",
    "kriging_STGTN": "ml.kriging_STGTN",
    "regression_kriging": "ml.regression_kriging",
    "regression_kriging_gnn": "ml.regression_kriging_gnn",
    "regression_kriging_h2o": "ml.regression_kriging_h2o",
    "regression_kriging_pytorch_gpu": "ml.regression_kriging_pytorch_gpu",
    "regression_kriging_tf": "ml.regression_kriging_tf",
    "snesim": "mps.snesim",
    "_conditioning": "simulation._conditioning",
    "annealing": "simulation.annealing",
    "direct_sequential": "simulation.direct_sequential",
    "fftsim": "simulation.fftsim",
    "lusim": "simulation.lusim",
    "pgs": "simulation.pgs",
    "sgcosim": "simulation.sgcosim",
    "sgsim": "simulation.sgsim",
    "sicosim": "simulation.sicosim",
    "sis_local_means": "simulation.sis_local_means",
    "sisim": "simulation.sisim",
    "tbsim": "simulation.tbsim",
    "STPoisson_kriging": "spacetime.STPoisson_kriging",
    "krigeST": "spacetime.krigeST",
    "krige_st": "spacetime.krige_st",
    "st_variogram_models": "spacetime.st_variogram_models",
    "spatial_cv_IK": "validation.spatial_cv_IK",
    "spatial_cv_automl": "validation.spatial_cv_automl",
    "spatial_cv_cokriging": "validation.spatial_cv_cokriging",
    "spatial_cv_h2o_rk": "validation.spatial_cv_h2o_rk",
    "spatial_cv_pytorch_rk": "validation.spatial_cv_pytorch_rk",
    "spatial_cv_rk": "validation.spatial_cv_rk",
    "spatial_cv_tf": "validation.spatial_cv_tf",
}


class _LegacyModuleFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve former flat module paths to their new subpackage modules."""

    _prefix = __name__ + "."

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(self._prefix) and fullname[len(self._prefix):] in _LEGACY_MODULES:
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return importlib.import_module(self._prefix + _LEGACY_MODULES[spec.name[len(self._prefix):]])

    def exec_module(self, module):
        pass


for _old, _new in _LEGACY_MODULES.items():
    _mod = sys.modules.get(f"{__name__}.{_new}")
    if _mod is not None:
        sys.modules.setdefault(f"{__name__}.{_old}", _mod)
if not any(isinstance(_f, _LegacyModuleFinder) for _f in sys.meta_path):
    sys.meta_path.append(_LegacyModuleFinder())
del _old, _new, _mod

# Backward-compatible alias: modules used to live in
# ``geolattice.geostatistics.pygstat``; ``gstat.pygstat.Variogram`` still works.
pygstat = sys.modules[__name__]

__all__ = [
    # Variogram & Kriging
    "Variogram",
    "OrdinaryKriging",
    "SimpleKriging",
    "JaxVariogram",
    "JaxOrdinaryKriging",
    "JaxSimpleKriging",
    "JaxUniversalKriging",
    "JaxCokriging",
    "VariogramCloud",
    "compute_variogram_cloud",
    "plot_variogram_cloud",
    "RegressionKriging",
    "GAMRegressor",
    "ENSEMBLE_REGRESSORS",
    "STATISTICAL_REGRESSORS",
    "AVAILABLE_REGRESSORS",
    "enforce_quantile_monotonicity",
    "RegressionKrigingH2O",
    "H2O_MODEL_TYPES",
    "DeepRegressionKriging",
    "GNNRegressionKriging",
    "KCN",
    "IGNNK",
    "STGNNRegressionKriging",
    "TGCN",
    "DCRNN",
    "STGTNRegressionKriging",
    "STTN",
    "GMAN",
    "CNNLSTMRegressionKriging",
    "ConvLSTM",
    "DeepRegressionKrigingTF",
    "UniversalKriging",
    "Cokriging",
    "CrossVariogram",
    "MultivariateCokriging",
    "ColocatedCokriging",
    "fit_lmc",
    "check_lmc_validity",
    "fit_indicator_variogram",
    "IndicatorKriging",
    "DisjunctiveKriging",
    "hermite_polynomials",
    "hermite_indicator_coefficients",
    "EmpiricalBayesianKriging",
    "BayesianKriging",
    "ConjugateBayesianKriging",
    "BayesianTransformedKriging",
    "BayesianGLMKriging",
    "bayesian_kriging",
    "correlation_function",
    "compute_etype_from_probabilities",
    "PoissonKriging",
    "PointSupport",
    "AreaPoissonKriging",
    "multi_poisson_kriging",
    # Spatio-temporal kriging
    "krige_st",
    "krige_st_df",
    "krige_st_local",
    "krige_st_tg",
    "krige_st_tg_local",
    "cov_fn_st",
    "vgm_area_st",
    # Spatio-temporal variogram models
    "vgm",
    "vgm_st",
    "variogram_line",
    "variogram_surface",
    "fit_st_variogram",
    "extract_par",
    "insert_par",
    "extract_par_names",
    "empirical_st_variogram",
    "fit_metric_st_variogram",
    "joint_nugget_sill_range",
    # Spatio-temporal Poisson kriging
    "STPoissonKriging",
    "load_atlantic_panel",
    # Spatio-temporal (Ordinary) kriging
    "STKriging",
    "load_ca_pm25_panel",
    "make_prediction_grid",
    # Validation
    "loo_cv",
    "kfold_cv",
    "KFoldCV_RK",
    "LOOCV_RK",
    "KFoldCV_H2O_RK",
    "KFoldCV_H2O_ResidualRK",
    "H2OResidualKriging",
    "KFoldCV_AutoML",
    "LOOCV_AutoML",
    "KFoldCV_PyTorch_RK",
    "KFoldCV_Cokriging",
    "KFoldCV_TF_RK",
    "IndicatorKrigingCV",
    # I/O
    "from_geodataframe",
    # GSLIB-equivalent helpers
    "read_gslib",
    "write_gslib",
    "read_gslib_grid",
    "nscore_forward",
    "nscore_back",
    "nscore_file",
    "backtr_transform",
    "transform_log",
    "back_transform_log",
    "transform_boxcox",
    "back_transform_boxcox",
    "declus_cell",
    "add_coordinates_to_grid",
    "addcoord",
    "sgsim",
    "sisim",
    "sis_local_means",
    "sklm_cdf",
    "sk_correction",
    "residual_covariances",
    "order_relations",
    "lusim",
    "lusim_unconditional",
    "LUSimulator",
    "tbsim",
    "tbsim_unconditional",
    "TBSimulator",
    "fftsim",
    "fftsim_unconditional",
    "FFTSimulator",
    "pgsim",
    "PluriGaussianSimulator",
    "TruncationRule",
    "category_probabilities",
    "category_proportions",
    "dssim",
    "DirectSampling",
    "pattern_histogram",
    "pattern_histogram_distance",
    "connectivity",
    "channel_training_image",
    "snesim",
    "SNESIM",
    "tau_model",
    "sicosim",
    "IndicatorCoSimulator",
    "SoftCalibration",
    "sgcosim",
    "sgcosim_hierarchical",
    "sgsim_gpu",
    "GaussianCoSimulator",
    "dssim_direct",
    "DirectSequentialSimulator",
    "GlobalDistribution",
    "sasim",
    "SimulatedAnnealing",
    "kt3d",
    "kt3d_grid",
    "kt3d_cross_validate",
    "build_grid_3d",
    "SoftKriging",
    "indicator_transform",
    "markov_bayes_calibrate",
    "fit_soft_probability_model",
    "FactorialKriging",
    "fit_nested_variogram",
    "pygstat",
    # Core variogram model helpers
    "spherical",
    "exponential",
    "gaussian",
    "matern",
    "stable",
    "cubic",
    "MODEL_FUNCS",
    "covariance_from_variogram",
    # Multivariate Poisson cokriging helpers
    "c_exp",
    "c_sph",
    "covariance_model",
    "shared_risk",
    "eb_risk",
    "pck_variogram",
    "pck_crossvariogram",
    "LMC",
    "fit_poisson_lmc",
    "poisson_cokrige",
    "poisson_krige",
    # Backend / anisotropy utilities
    "asarray",
    "to_numpy",
    "CUPY_AVAILABLE",
    "resolve_torch_device",
    "seed_torch",
    "transform_aniso",
]


def __getattr__(name):
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr, extra_name = spec
    loaded = _optional_import(module_path, [attr], extra_name)
    # Bind under the public name. The JAX classes are exported as Jax*
    # while the module attribute keeps the core class name.
    globals()[name] = loaded[attr]
    return globals()[name]
