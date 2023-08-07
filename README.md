# mdvcontainment
Robust characterization of the inside and outside hierarchy for point clouds in periodic spaces (e.g. R3/Z3).

![blobs_containment_small](https://github.com/BartBruininks/mdvcontainment/assets/1488903/be5fff63-a967-47c2-a933-a3ecb7dcd5de)

**Figure 1 | Containment hierarchy in self-assembled acyl chain bicelles.** The main void (-2) is the most outside segment in this system. It containes three non-periodic segments (1,2,3), segment 1 is split over the periodic boundary. Segment 3 contains a piece of void itself (-2). In list notation we represent the graph as [-2 1 2 [3 -1]], where the first element in each list is the container of the following elements.

# License
MDVContainment is available under the Apache-2.0 license.

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
import containment as mdvc
import MDAnalysis as mda
## Displaying the containment graph
import webbrowser
from IPython.display import display

## A TPR can be used in combination with the GRO to add the boneded information.
#  One can also use a PDB annotated file for the bonds, although the maximum bonds
#  is then coupled to the PDB fixed file format (max 9999?)
TPR = 'blobs.tpr'
GRO = 'whole_blobs.gro'
selection_string = '(name C2A C2B C3A C3B C4A C5A C5B C4B D2A D2B D3A D3B D4A D4B D5A D5B)'
resolution = 0.5 # This is in nm if you are working with GRO/PDB files

## Loading the files using MDA
universe = mda.Universe(TPR, GRO) # The TPR adds bonded information
selection = universe.select_atoms(selection_string)

## Calculating the containment
containers = mdvc.Containers(selection, resolution)

## Plottin the containment without renders
containers.plot()

## Plotting the containment with renders (this makes use of VMD and expects the presence of the render scripts).
containers.render()

## Opening the interactive graph in a browser
webbrowser.open('containment_img.html')

## Extract the container and its content by using the container ID (renders are useful here).
contained_nodes = containers.get_downstream_nodes([-2]) # -2 is the ID in this example
print(f'The container has a volume of {containers.get_volume(contained_nodes)} nm^3')

## Writing the atomgroup 
# Using the most common label per residue is pretty slow and is usually not needed.
output_atomgroup = containers.get_atomgroup_from_nodes(contained_nodes, b_factor=True, residue=False)

## In VMD the selection "beta 'id'" can be used to select the desired container. The quotes around the ID are
# required if the ID is negative! 
output_atomgroup.write('combined_container.pdb')


```
