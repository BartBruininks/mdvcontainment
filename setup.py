import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(
    name='mdvcontainment',
    version='0.1.1',
    packages=['mdvcontainment'],
    author='BMH Bruininks',
    description = "Containment analysis for (periodic) point clouds.",
    long_description = 'file: README.md',
    long_description_content_type = 'text/markdown',
    url = "https://github.com/BartBruininks/mdvcontainment",
    python_requires = ">=3.12",
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
