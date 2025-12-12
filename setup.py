import numpy
from Cython.Build import cythonize
from setuptools import setup
from pathlib import Path

# Read version from _version.py
version_file = Path(__file__).parent / "mdvcontainment" / "_version.py"
version_dict = {}
exec(version_file.read_text(), version_dict)

setup(
    name='mdvcontainment',
    version=version_dict["__version__"],
    packages=['mdvcontainment'],
    author='BMH Bruininks',
    description = "Containment analysis for (periodic) point clouds.",
    readme = 'README.md',
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
