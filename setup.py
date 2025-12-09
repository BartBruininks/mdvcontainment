import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(
    name='mdvcontainment',
    version='v2.0.0',
    packages=['src'],
    author='BMH Bruininks',
    description = "Containment analysis for (periodic) point clouds.",
    long_description = 'README.md',
    long_description_content_type = 'text/markdown',
    url = "https://github.com/BartBruininks/mdvcontainment",
    python_requires = ">=3.12",
    ext_modules=cythonize(["src/find_label_contacts.pyx", 'src/find_bridges.pyx', "src/atoms_voxels_mapping.pyx"], language_level=3),
    include_dirs=[numpy.get_include()],
    install_requires=[
        "numpy>=2.2",
        "networkx>=3.4",
        "scipy>=1.15",
        "MDAnalysis>=2.8",
        "matplotlib>=3.10",
    ],
)
