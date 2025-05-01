from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ppfd_compute',
    ext_modules=[
        CUDAExtension(
            name='ppfd_compute',
            sources=['ppfd_compute.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
