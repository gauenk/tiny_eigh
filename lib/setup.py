from setuptools import setup
# from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='tiny_eigh',
      packages=["tiny_eigh"],
      ext_modules=[
          CUDAExtension('tiny_eigh_cuda', [
              'csrc/tiny_eigh_cuda.cpp',
              'csrc/tiny_eigh_kernel.cu',
              'csrc/pybind.cpp',
          ])
      ],
      cmdclass={'build_ext': BuildExtension}
)
