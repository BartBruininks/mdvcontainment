Welcome to MDVContainment's Documentation
=========================================

This library provides classes for hierarchical spatial containment analysis of molecular systems.

The containment hierarchy represents nested spatial compartments detected in molecular structures,
enabling analysis of volumes, boundaries, and relationships between different regions. The module
supports efficient voxel-based representations with memory-efficient views. There are two main 
classes exposed in the API (`Containment` and `VoxelContainment`). The `Containment` class is a 
wrapper around `VoxelContainment` which takes care of all the mapping of atoms from an `MDAnalysis.AtomGroup`
to the voxel representation (and back).

Key Features
------------
- Voxel-based spatial decomposition of molecular systems
- Hierarchical containment graph construction
- Memory-efficient views for filtered/merged structures
- Bidirectional mapping between voxels and atoms
- Volume calculations and spatial queries
- Integration with MDAnalysis for molecular dynamics analysis

Examples
--------
Basic usage with an MDAnalysis AtomGroup:

>>> import MDAnalysis as mda
>>> u = mda.Universe('topology.pdb', 'trajectory.xtc')
>>> membrane = u.select_atoms('resname POPC')
>>> containment = Containment(membrane, resolution=0.5)
>>> print(containment)

Create a filtered view:

>>> # Keep only large compartments (>200 nm³)
>>> view = containment.node_view(min_size=200)
>>> atoms = view.get_atomgroup_from_nodes([1, 2, 3])

Access the containment graph from VoxelContainment

>>> containment.voxel_containment.containment_graph

Notes
-----
The module uses nanometer (nm) units for lengths and volumes. MDAnalysis 
atomgroups using Angstrom units are automatically converted internally.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
   developer
