"""
Regression kriging with machine-learning trend models.

Modules: ``regression_kriging`` (scikit-learn, GAM and ensembles),
``regression_kriging_h2o`` (H2O AutoML), ``regression_kriging_pytorch_gpu``,
``regression_kriging_tf``, ``regression_kriging_gnn`` (graph networks),
``kriging_STGNN``, ``kriging_STGTN`` and ``kriging_CNNLSTM`` (space-time deep
models). The H2O / PyTorch / TensorFlow modules need their optional extras;
nothing is imported eagerly here.

Import the classes from `geolattice.geostatistics`.
"""
