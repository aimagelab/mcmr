from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = []

ext_modules=[
    CUDAExtension('soft_renderer.cuda.load_textures', [
        'soft_renderer/cuda/load_textures_cuda.cpp',
        'soft_renderer/cuda/load_textures_cuda_kernel.cu',
        ],
        extra_compile_args={'cxx': [], 'nvcc': [
            '--generate-code=arch=compute_35,code=sm_35',
            '--generate-code=arch=compute_37,code=sm_37',
            '--generate-code=arch=compute_50,code=sm_50',
            '--generate-code=arch=compute_52,code=sm_52',
            '--generate-code=arch=compute_53,code=sm_53',
            '--generate-code=arch=compute_60,code=sm_60',
            '--generate-code=arch=compute_61,code=sm_61',
            '--generate-code=arch=compute_62,code=sm_62',
            '--generate-code=arch=compute_70,code=sm_70',
            '--generate-code=arch=compute_72,code=sm_72',
            '--generate-code=arch=compute_75,code=sm_75',
        ]}
                  ),
    CUDAExtension('soft_renderer.cuda.create_texture_image', [
        'soft_renderer/cuda/create_texture_image_cuda.cpp',
        'soft_renderer/cuda/create_texture_image_cuda_kernel.cu',
        ],
        extra_compile_args={'cxx': [], 'nvcc': [
            '--generate-code=arch=compute_35,code=sm_35',
            '--generate-code=arch=compute_37,code=sm_37',
            '--generate-code=arch=compute_50,code=sm_50',
            '--generate-code=arch=compute_52,code=sm_52',
            '--generate-code=arch=compute_53,code=sm_53',
            '--generate-code=arch=compute_60,code=sm_60',
            '--generate-code=arch=compute_61,code=sm_61',
            '--generate-code=arch=compute_62,code=sm_62',
            '--generate-code=arch=compute_70,code=sm_70',
            '--generate-code=arch=compute_72,code=sm_72',
            '--generate-code=arch=compute_75,code=sm_75',
        ]}
                  ),
    CUDAExtension('soft_renderer.cuda.soft_rasterize', [
        'soft_renderer/cuda/soft_rasterize_cuda.cpp',
        'soft_renderer/cuda/soft_rasterize_cuda_kernel.cu',
        ],
        extra_compile_args={'cxx': [], 'nvcc': [
            '--generate-code=arch=compute_35,code=sm_35',
            '--generate-code=arch=compute_37,code=sm_37',
            '--generate-code=arch=compute_50,code=sm_50',
            '--generate-code=arch=compute_52,code=sm_52',
            '--generate-code=arch=compute_53,code=sm_53',
            '--generate-code=arch=compute_60,code=sm_60',
            '--generate-code=arch=compute_61,code=sm_61',
            '--generate-code=arch=compute_62,code=sm_62',
            '--generate-code=arch=compute_70,code=sm_70',
            '--generate-code=arch=compute_72,code=sm_72',
            '--generate-code=arch=compute_75,code=sm_75',
        ]}
                  ),
    CUDAExtension('soft_renderer.cuda.voxelization', [
        'soft_renderer/cuda/voxelization_cuda.cpp',
        'soft_renderer/cuda/voxelization_cuda_kernel.cu',
        ],
        extra_compile_args={'cxx': [], 'nvcc': [
            '--generate-code=arch=compute_35,code=sm_35',
            '--generate-code=arch=compute_37,code=sm_37',
            '--generate-code=arch=compute_50,code=sm_50',
            '--generate-code=arch=compute_52,code=sm_52',
            '--generate-code=arch=compute_53,code=sm_53',
            '--generate-code=arch=compute_60,code=sm_60',
            '--generate-code=arch=compute_61,code=sm_61',
            '--generate-code=arch=compute_62,code=sm_62',
            '--generate-code=arch=compute_70,code=sm_70',
            '--generate-code=arch=compute_72,code=sm_72',
            '--generate-code=arch=compute_75,code=sm_75',
        ]}
                  ),
    ]

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision', 'scikit-image', 'tqdm', 'imageio']

setup(
    description='PyTorch implementation of "Soft Rasterizer"',
    author='Shichen Liu',
    author_email='liushichen95@gmail.com',
    license='MIT License',
    version='1.0.0',
    name='soft_renderer',
    packages=['soft_renderer', 'soft_renderer.cuda', 'soft_renderer.functional'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExtension}
)
