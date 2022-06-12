"""
Compute the eigenvalue decomposition of thousands of matrices

"""

# -- imports --
import time
import torch as th
import numpy as np
import tiny_eigh
import cupy
from easydict import EasyDict as edict

# -- Params --
nreps = 3
device = "cuda:1"
size = (15000,49,49)
th.cuda.set_device(device)
names = ["torch","tiny_eigh"]

# -- Create Data --
cov = th.randn(size).to(device)
cov = (cov.transpose(2,1) + cov)/2.
th.cuda.synchronize()

# -- Run for Reps --
runtimes = edict()
runtimes.time_th = []
runtimes.time_te = []

for rep in range(nreps):

    # -- check outputs --
    eigVecs,eigVals = [],[]

    # -- New Order Each Time --
    names.reverse()
    for name in names:
         if name == "torch":
             # -- PyTorch --
             t_start = time.perf_counter()
             eigVals_th,eigVecs_th = th.linalg.eigh(cov)
             th.cuda.synchronize()
             time_th = time.perf_counter() - t_start
             eigVals.append(eigVals_th)
             eigVecs.append(eigVecs_th)
             runtimes.time_th.append(time_th)
         else:
             # -- TinyEigh --
             t_start = time.perf_counter()
             eigVals_te,eigVecs_te = cupy.linalg.eigh(cov)
             # eigVals_te,eigVecs_te = tiny_eigh.run(cov)
             th.cuda.synchronize()
             time_te = time.perf_counter() - t_start
             eigVals.append(eigVals_te)
             eigVecs.append(eigVecs_te)
             runtimes.time_te.append(time_te)

    # -- Compare --
    error = th.dist(eigVecs[0].abs(),eigVecs[1].abs())
    assert error < 1e-1
    error = th.dist(eigVals[0].abs(),eigVals[1].abs())
    assert error < 1e-1

# -- Compute Means --
print(runtimes)
mean_th = np.mean(runtimes.time_th)
mean_te = np.mean(runtimes.time_te)
print("Runtime Pytorch: ",mean_th)
print("Runtime TinyEigh: ",mean_te)
print("Sometimes this is x10 faster. Sometimes this is x2/3 faster.")
