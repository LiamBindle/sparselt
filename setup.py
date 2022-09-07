from setuptools import setup

setup(
    name='sparselt',
    version='0.1.3',
    author="Liam Bindle",
    author_email="liam.bindle@gmail.com",
    description="A small library for regridding Earth system model data.",
    url="https://github.com/LiamBindle/sparselt",
    project_urls={
        "Bug Tracker": "https://github.com/LiamBindle/sparselt/issues",
    },
    packages=['sparselt'],
    install_requires=[
        'numpy',
        'netcdf4',
        'xarray',
        'scipy',
    ],
)
