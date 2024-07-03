from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    name='mdvcontainment',
    version='0.1',
    packages=['mdvcontainment'],
    ext_modules=cythonize(["mdvcontainment/find_label_contacts.pyx", 'mdvcontainment/find_bridges.pyx']),
    install_requires=[
        "numpy",
        "networkx",
        "scipy",
        "MDAnalysis",
        "matplotlib",
    ],
)
