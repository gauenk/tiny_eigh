

import torch as th
from einops import rearrange
import tiny_eigh_cuda

def run(covMat):

    # -- compute covMat --
    num,pdim,pdim = covMat.shape
    tf64 = th.float64
    tf32 = th.float32
    device = covMat.device
    tfdt = tf32

    # -- type --
    covMat = covMat.contiguous().type(tfdt)

    # -- create shells --
    eigVals = th.zeros((num,pdim),dtype=tfdt,device=device)
    eigVecs = covMat.clone()

    # -- faiss stream --
    tiny_eigh_cuda.run(eigVecs,eigVals)

    # -- formatting --
    eigVecs = eigVecs.transpose(2,1)
    # eigVecs = th.flip(eigVecs,dims=(2,))

    return eigVals,eigVecs
