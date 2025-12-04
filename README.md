<p align="center">
  <img src="https://github.com/user-attachments/assets/354b70dc-9069-493f-8c5b-fc756b7d14d0" width="60%">
</p>

# MDVContainment v2.0.0
Robust and fast containment charaterization for (periodic) point clouds and voxel masks for binary density labeling.

**M**olecular **D**ynamics **V**oxel **Containment** yields a robust characterization of the inside and outside hierarchy for point clouds in periodic spaces of dimensionality three or lower (e.g. R^3/Z^3).

Using an MDAnalysis atomgroup and resolution, a density grid is created by pbc aware binning of the particle positions. This density grid is segmented using connected components, and graph logic is utilized to solve the topological identification of containment (insides and outsides). The final output is a set of Directed Acyclic Graphs (DAGs) running from the largest container to the smallest (from outside inwards in graph space). This containment logic can then be used to analyse or manipulate the systems.

 *Any complex configuration of (non)periodic segments is supported by this algorithm in a fast, robust, unambiguous, deterministic and rot+trans invarient (up to voxel discretization) manner.*

> [!NOTE]
> MDVContainment is undergoing a functional overhaul, supporting integer labeling (e.g. MDVLeafletSegmentation labeling)!
> When the code is ready to be tested a branch will be created called `integer_containment` which will probably become the new main branch over time.

<p align="center">
  <img src="https://github.com/BartBruininks/mdvcontainment/assets/1488903/be5fff63-a967-47c2-a933-a3ecb7dcd5de">
</p>

**Figure 1 | Containment hierarchy in self-assembled acyl chain bicelles.** The main solvent (segment -2) is the most outer segment in this system. It containes three non-periodic segments (1, 2, 3), where segment 1 is split over the periodic boundary. Segment 3 *contains* a bubble of inner solvent (segment -1).

$~$

<p align="center">
  <img src="https://github.com/BartBruininks/mdvcontainment/assets/1488903/da3d8cdb-682f-4fe3-b7dc-bced188b390d">
</p>
 
**Figure 2 | A periodic hollow cylinder in solution**. The cylinder (segment 1) splits the solution into two segments (segment -2, -1), the solid cylinder inside the hollow cylinder (segment -1), and all of the space outside of the cylinder (segment -2). Both cylindrical segments (1 and -1) are said to be *contained* by the solvent segment (-2), although only the hollow cylinder (1) is a *child* of the solvent segment (-2).

$~$

# Citation and License
```
@article {Bruininks2025.08.06.668936,
	author = {Bruininks, Bart M. H. and Vattulainen, Ilpo},
	title = {Classification of containment hierarchy for point clouds in periodic space},
	elocation-id = {2025.08.06.668936},
	year = {2025},
	doi = {10.1101/2025.08.06.668936},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/08/09/2025.08.06.668936},
	eprint = {https://www.biorxiv.org/content/early/2025/08/09/2025.08.06.668936.full.pdf},
	journal = {bioRxiv}
}
```

Please cite this work if you use it for scientific publications. It helps me to continue to work on this kind of software, thanks! On that note, if you would offer me a job to work on this, I would take your offer very seriously.

MDVContainment is available under the Apache-2.0 license.

# Requirements
MDVContainment has been tested to work with python >= 3.12. 

# Installation
## Using pypi
Install v2.0.0 in the current python environment:
```console
pip install mdvcontainment==v2.0.0
```

## Using git
Direct install from the github main branch into the current python environment:

```console
pip install git+https://github.com/BartBruininks/mdvcontainment
```

Create a folder in a custom location using git clone:

```console
git clone git@github.com:BartBruininks/mdvcontainment.git
cd mdvcontainment
pip install .
```

> [!IMPORTANT]
> If you need any help with MDVContainment or have ideas for future functionalities, please raise an issue!

# Minimal example CG Martini
## Calculate the containment
### Input
```python
# Import the required libraries
import MDAnalysis as mda
from mdvcontainment import Containment

# Import the structure file
path = 'your_structure.pdb' # Or any MDA supported structures file
universe = mda.Universe(path)
selection_string = 'not resname W WF ION' # Useful for CG Martini
selection = universe.select_atoms(selection_string)

# Run the containment analysis
containment = Containment(selection, resolution=0.5, closing=True)

# Show the containment graph with voxel counts
print(containment)
```

> [!NOTE]
> For atomistic structures use `closing=False` at 0.5 nm resolution. Take a look at
> [closing](https://en.wikipedia.org/wiki/Closing_(morphology)) (link to wikipedia) to learn more about what it does.

### Output
```console
Containment Graph with 3 components (component: nm^3: rank):
└── [-2: 7350: 3]
    └── [1: 477: 0]
        └── [-1: 65: 0]
```

## Plot the compositions
### Input
```python
# Plot the compositions
composition, fig, axs = cl.analyze_composition(containment, mode='names') # or 'resnames' / 'molar'
```

### Output
<p align="center">
  <img width="713" height="809" alt="image" src="https://github.com/user-attachments/assets/d221d70b-7626-4be8-b872-a44e9e3bcc04" width="60%">
</p>
</details>




# Extensive examples
For worked examples in jupyter notebooks, take a look at the [examples/notebooks](https://github.com/BartBruininks/mdvcontainment/tree/main/examples/notebooks) folder. Some example structure files are added under [examples/structures](https://github.com/BartBruininks/mdvcontainment/tree/main/examples/structures).

I still need to add this tutorial to this repo, but for now a very detailed and up to date tutorial can be found on the cgmartini website [here](https://cgmartini.nl/docs/tutorials/Martini3/MDVContainment/).





