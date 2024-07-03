# MDVContainment — OPEN BETA —
Robust and fast containment charaterization for periodic point clouds and voxel masks.

Molecular Dynamics Voxel Containment yields a robust characterization of the inside and outside hierarchy for point clouds in periodic spaces of dimensionality three or lower (e.g. R^3/Z^3).

Using an MDAnalysis atomgroup, bead selection and resolution, a density grid is created. This density grid is segmented using connected components and graph logic is utilized to solve the topological identification of inside and outside. This algorithm is applicable to both periodic and non-periodic segments. The final output is a set of Directed Acyclic Graphs (DAGs) running from the largest container to the smallest (from outside inwards in graph space). This containment logic can then be used to analyse or manipulate the systems.

![alt text](https://private-user-images.githubusercontent.com/1488903/258806872-be5fff63-a967-47c2-a933-a3ecb7dcd5de.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjAwNDE1OTcsIm5iZiI6MTcyMDA0MTI5NywicGF0aCI6Ii8xNDg4OTAzLzI1ODgwNjg3Mi1iZTVmZmY2My1hOTY3LTQ3YzItYTkzMy1hM2VjYjdkY2Q1ZGUucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI0MDcwMyUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNDA3MDNUMjExNDU3WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9OGQyZjU1MWJkMDUzNDE4MzgyYzlmMDQ4MGRjODEzNTc2MmIzY2JjNTNlZWRjMDM0ZDA0ZTJiZjFhNTZkYjc3YyZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmYWN0b3JfaWQ9MCZrZXlfaWQ9MCZyZXBvX2lkPTAifQ.6HlI9IuboZP8_OsNRdXr_Jqaxxe_9HXFinF9bw_3HS4)

**Figure 1 | Containment hierarchy in self-assembled acyl chain bicelles.** The main solvent (component -2) is the most outer component in this system. It containes three non-periodic components (seg. 1, 2, 3), where component 1 is split over the periodic boundary. Component 3 contains a piece of inner solvent (component -1).

$~$

![cylinder_containment](https://github.com/BartBruininks/mdvcontainment/assets/1488903/da3d8cdb-682f-4fe3-b7dc-bced188b390d)
 
**Figure 2 | A periodic hollow cylinder in solution**. The cylinder (component 1) splits the solution into two components (component -2, -1), the solid cylinder inside the hollow cylinder (component -1), and all of the space outside of the cylinder (component -2). Any complex configuration of periodic/nonperiodic objects is supported by this algorithm in a robust manner.

$~$

# Citation and License
This work is currently in the process of publication, the DOI of the mansucript will be placed here once it is available. 

Please cite this work if you use it for scientific publications. It helps me to continue to work on this kind of software, thanks!

MDVContainment is available under the Apache-2.0 license.

# Requirements
MDVContainment has been tested to work with python >= 3.8 and Ubuntu 20.04.6 LTS. 

# Installation
## Using git
Direct install into the current python environment libraries:
```
pip install git+https://github.com/BartBruininks/mdvcontainment
```

Create a folder in a custom location to easily find the examples, or modify the code:

```
git clone git@github.com:BartBruininks/mdvcontainment.git
cd mdvcontainment
pip install .
```
# Minimal example
### Input
```python
# `minimal_example.py` for a CG Martini structure file
# Import the required libraries
import MDAnalysis as mda
from mdvcontainment import Containment

# Import the structure file
path = 'your_structure.gro'
universe = mda.Universe(path)
selection_string = 'not resname W WF ION' # Useful for CG Martini
selection = universe.select_atoms(selection_string)

# Run the containment analysis
containment = Containment(selection, resolution=0.5, closure=True)

# Show the containment graph with voxel counts
print(containment)
```

### Output
```
Containment Graph with 3 components (component: nvoxels):
└── [-2: 54461]
    └── [1: 15403]
        └── [-1: 5136]
```

# Extensive examples
For worked examples in jupyter notebooks, take a look at the `examples/notebooks` folder. Some example structure files are added under examples/structures.

