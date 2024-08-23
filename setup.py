import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(
    name='mdvcontainment',
    version='0.1.1',
    packages=['mdvcontainment'],
    ext_modules=cythonize(["mdvcontainment/find_label_contacts.pyx", 'mdvcontainment/find_bridges.pyx']),
    include_dirs=[numpy.get_include()],
    install_requires=[
        "numpy",
        "networkx",
        "scipy",
        "MDAnalysis",
        "matplotlib",
    ],
)
