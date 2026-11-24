"""Distributed data parallel (DDP) training of a small soil carbon network, for notebook 08_09.

``train(rank, world, X, y, epochs)`` runs in each process started by
``geolattice.distributed.ml.train_ddp``: it wraps the model with ``ddp_model``, gives each rank
its own shard of the data with a ``DistributedSampler`` and returns the loss history, so the
notebook can check that every rank ended with the same model.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import geolattice.distributed as pdist


def train(rank, world, X, y, epochs=200):
    torch.manual_seed(0)  # same starting weights on every rank
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.ReLU(),
                              torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    model = pdist.ml.ddp_model(net, rank)  # replica on this rank's device, gradients all-reduced
    dev = next(model.parameters()).device
    data = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=True, seed=0)
    loader = DataLoader(data, batch_size=64, sampler=sampler)  # this rank's shard only
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    history = []
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # a new shuffle each epoch, the same on all ranks
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(xb).squeeze(1), yb)
            loss.backward()  # DDP averages the gradients of all ranks here
            opt.step()
            total += float(loss.detach()) * len(xb)
            n += len(xb)
        history.append(total / n)
    checksum = float(sum(p.detach().double().sum().cpu() for p in model.parameters()))
    return {"rank": rank, "device": str(dev), "rows": len(sampler), "history": np.array(history),
            "weights_checksum": checksum}
