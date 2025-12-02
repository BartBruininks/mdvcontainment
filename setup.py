import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(
    name='mdvcontainment',
    version='v1.1.1',
    packages=['mdvcontainment'],
    author='BMH Bruininks',
    description = "Containment analysis for (periodic) point clouds.",
    long_description = 'README.md',
    long_description_content_type = 'text/markdown',
    url = "https://github.com/BartBruininks/mdvcontainment",
    python_requires = ">=3.12",
    ext_modules=cythonize(["mdvcontainment/find_label_contacts.pyx", 'mdvcontainment/find_bridges.pyx', "mdvcontainment/atoms_voxels_mapping.pyx"], language_level=3),
    include_dirs=[numpy.get_include()],
    install_requires=[
        "numpy>=2.2",
        "networkx>=3.4",
        "scipy>=1.15",
        "MDAnalysis>=2.8",
        "matplotlib>=3.10",
    ],
)
