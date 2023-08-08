# MDVContainment
**M**olecular **D**ynamics **V**oxel **Containment** yields a robust characterization of the inside and outside hierarchy for point clouds in periodic spaces (*e.g.* R^3/Z^3).

Using a MDAnalysis supported structure file, bead selection and resolution, a density grid is created. This density grid is segmented using connected components and graph logic is utilized to solve the topological identification of inside and outside. This algorithm is applicable to both periodic and non-periodic segments.  

![blobs_containment_small](https://github.com/BartBruininks/mdvcontainment/assets/1488903/be5fff63-a967-47c2-a933-a3ecb7dcd5de)

**Figure 1 | Containment hierarchy in self-assembled acyl chain bicelles.** The main void (seg. -2) is the most outside segment in this system. It containes three non-periodic segments (seg. 1, 2, 3), where seg. 1 is split over the periodic boundary. Seg. 3 contains a piece of inner void (seg. -1). In list notation we represent the graph as {[-2 1 2 [3 -1]]}, where the first element in each list is the container of the following elements.

$~$

![blobs_containment_nodes](https://github.com/BartBruininks/mdvcontainment/assets/1488903/3769a16d-1beb-45a1-8e98-6e9eac088a4a)

**Figure 2 | Containment hierarchy in self-assembled acyl chain bicelles - nodes only** Using the same system as Fig. 1 we can represent de data as nodes only, using their particle counts for the node sizes. The advantage of this representation is that it does not require VMD to be installed. Making is a stricly python dependent method.

$~$

![cylinder_containment](https://github.com/BartBruininks/mdvcontainment/assets/1488903/da3d8cdb-682f-4fe3-b7dc-bced188b390d)
 
**Figure 3 | A periodic hollow cylinder in solution**. The cylinder (seg. 1) splits the solution into two segments, the solid cylinder inside the hollow cylinder (seg. -1), and all of the space outside of the cylinder (seg. -2). This results in the following containment graph {[-2, [1, -1]]}. Any complex configuration of periodic objects is supported by this algorithm in a robust manner.

$~$

# License
MDVContainment is available under the Apache-2.0 license.

# Requirements
MDVContainment has been tested to work with python >= 3.8 and Ubuntu 20.04.6 LTS. 

# Installation
## Using git
```
git clone git@github.com:BartBruininks/mdvcontainment.git
cd mdvcontainment
pip install .
```
# Usage
```
## Minimum required imports
import mdvcontainment as mdvc
import MDAnalysis as mda
## Displaying the containment graph
import webbrowser
from IPython.display import display

## A TPR can be used in combination with the GRO to add the boneded information.
#  One can also use a PDB annotated file for the bonds, although the maximum bonds
#  is then coupled to the PDB fixed file format (max 9999?).
TPR = 'blobs.tpr'
GRO = 'whole_blobs.gro'
selection_string = '(name C2A C2B C3A C3B C4A C5A C5B C4B D2A D2B D3A D3B D4A D4B D5A D5B)'
resolution = 0.5 # This is in nm if you are working with GRO/PDB files

## Loading the files using MDA.
universe = mda.Universe(TPR, GRO) # The TPR adds bonded information
selection = universe.select_atoms(selection_string)

## Calculating the containment.
containers = mdvc.Containers(selection, resolution)

## Plottin the containment without renders.
containers.plot()

## Plotting the containment with renders (this makes use of VMD and expects the presence of the render scripts).
containers.render()

## Opening the interactive graph in a browser.
webbrowser.open('containment_img.html')

## Extract the container and its content by using the container ID (renders are useful here).
contained_nodes = containers.get_downstream_nodes([-2]) # -2 is the ID in this example
print(f'The container has a volume of {containers.get_volume(contained_nodes)} nm^3')

## Writing the atomgroup. 
#  Using the most common label per residue is pretty slow and is usually not needed.
output_atomgroup = containers.get_atomgroup_from_nodes(contained_nodes, b_factor=True, residue=False)

## In VMD the selection "beta 'id'" can be used to select the desired container. The quotes around the ID are
#  required if the ID is negative! 
output_atomgroup.write('combined_container.pdb')
```
