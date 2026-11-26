"""
CUDA (CuPy) engines behind the ``use_gpu`` option of the simulators.

Modules: ``mps_gpu`` (Direct Sampling, SNESIM), ``sgs_gpu`` (sequential
Gaussian), ``sis_gpu`` (sequential indicator), ``dss_gpu`` (direct
sequential), ``tbsim_gpu`` (turning bands). They are imported only when a
simulator runs with ``use_gpu`` and a usable GPU.
"""
