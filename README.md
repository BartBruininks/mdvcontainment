# MDVContainment—OPEN BETA
Robust and fast containment charaterization for (periodic) point clouds and voxel masks.

**M**olecular **D**ynamics **V**oxel **Containment** yields a robust characterization of the inside and outside hierarchy for point clouds in periodic spaces of dimensionality three or lower (e.g. R^3/Z^3).

Using an MDAnalysis atomgroup, bead selection and resolution, a density grid is created. This density grid is segmented using connected components and graph logic is utilized to solve the topological identification of containment (insides and outsides). The final output is a set of Directed Acyclic Graphs (DAGs) running from the largest container to the smallest (from outside inwards in graph space). This containment logic can then be used to analyse or manipulate the systems.

 *Any complex configuration of (non)nonperiodic segments is supported by this algorithm in a fast, robust, unambiguous, deterministic and rot+trans invarient (up to voxel discretization) manner.*

<p align="center">
  <img src="https://github.com/BartBruininks/mdvcontainment/assets/1488903/be5fff63-a967-47c2-a933-a3ecb7dcd5de">
</p>

**Figure 1 | Containment hierarchy in self-assembled acyl chain bicelles.** The main solvent (segment -2) is the most outer segment in this system. It containes three non-periodic segments (1, 2, 3), where segment 1 is split over the periodic boundary. Segment 3 *contains* a bubble of inner solvent (segment -1).

$~$

<p align="center">
  <img src="https://github.com/BartBruininks/mdvcontainment/assets/1488903/da3d8cdb-682f-4fe3-b7dc-bced188b390d">
</p>
 
**Figure 2 | A periodic hollow cylinder in solution**. The cylinder (segment 1) splits the solution into two segments (segment -2, -1), the solid cylinder inside the hollow cylinder (segment -1), and all of the space outside of the cylinder (segment -2). Both cylindrical segments (1 and -1) are said to be *contained* by the solvent segment (-2), although only the hollow cylinder (1) is a child of the solvent segment (-2).

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

```console
pip install git+https://github.com/BartBruininks/mdvcontainment
```

Create a folder in a custom location to have access to the examples folder:

```console
git clone git@github.com:BartBruininks/mdvcontainment.git
cd mdvcontainment
pip install .
```

> [!IMPORTANT]
> If you need any help with MDVContainment or have ideas for future functionalities, please raise an issue!

# Minimal example CG Martini
<details open>
<summary>Input</summary>
<br>

```python
# `minimal_example.py` for a CG Martini structure file
# Import the required libraries
import MDAnalysis as mda
from mdvcontainment import Containment

# Import the structure file
path = 'your_structure.pdb' # Or any MDA supported structures file
universe = mda.Universe(path)
selection_string = 'not resname W WF ION' # Useful for CG Martini
selection = universe.select_atoms(selection_string)

# Run the containment analysis
containment = Containment(selection, resolution=0.5, closure=True)

# Show the containment graph with voxel counts
print(containment)
```
</details>

> [!NOTE]
> For atomistic structures use `closure=False`. Take a look at
> [closing](https://en.wikipedia.org/wiki/Closing_(morphology)) (link to wikipedia) to lean more about what it does.

<details>
<summary>Output</summary>
<br>

```
Containment Graph with 3 components (component: nvoxels):
└── [-2: 54461]
    └── [1: 15403]
        └── [-1: 5136]
```
</details>


# Extensive examples
For worked examples in jupyter notebooks, take a look at the [examples/notebooks](https://github.com/BartBruininks/mdvcontainment/tree/main/examples/notebooks) folder. Some example structure files are added under [examples/structures](https://github.com/BartBruininks/mdvcontainment/tree/main/examples/structures).





