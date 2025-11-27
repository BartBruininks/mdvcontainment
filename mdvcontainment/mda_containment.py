import numpy as np

import MDAnalysis as mda

from .atomgroup_to_voxels import close_voxels, create_voxels, voxels2atomgroup
from .containment_main import VoxelContainment

class Containment():
    def __init__(self, atomgroup, resolution, closing=False, slab=False, max_offset=0.05, 
                 verbose=False, write_structures=False, no_mapping=False, return_counts=True, 
                 betafactors=True):
        """
        Atomgroup specifies the selected atoms. All atoms in the universe
        will be mapped, therefore removing waters from a file can make this
        faster if you do not need them.
        
        Resolution is taken in nn for MDA atomgroups. You do 
        not need to preprocess the positions, the units of the 
        atomgroup are converted from A to nm internally.

        Closing applies binary dilation and erosion, often called
        binary closure. This closes small gaps in the voxel mask.
        This is usually required for CG Martini simulations when,
        using a resolution of 0.5 nm. For a resolution of 1 nm it
        is not required and disadvised.

        Slab can be set to 'True' to process slices of a larger whole,
        this will disable any treatment of PBC.

        Max_offset can be set to 0 to allow any resolution scaling 
        artefact for voxelization.

        Verbose can be used to set developer prints.

        Write_structures can be set to write the input, label and component voxel masks as a gro.

        No_mapping can be used to skip the generation of the voxel2atoms mapping. Making 
        the mapping is somewhat costly and if you already know you are interested in the voxel masks,
        why bother making it.

        Betafactors can be set to 'True' to store the component id per atom in the beta factors column
        of the universe. This makes it easy to extract atoms per component later on. However, this overwrites
        any existing beta factors in the universe and it is a rather costly operation for large systems.
        It cannot be set to 'True' when no_mapping is 'True'.

        Return counts is set to 'True' by default, this counts the amount of voxels per component
        this can be useful to get an estimate for the volume of a component.

        All voxel logic (selecting by component etc) can be found under self.voxel_containment.
        """
        self.atomgroup = atomgroup
        self.resolution = resolution
        self.closing = closing
        self._slab = slab
        self._max_offset = max_offset
        self._verbose = verbose
        self._write_structures = write_structures
        self._no_mapping = no_mapping
        self._return_counts = return_counts
        self._betafactors = betafactors

        # Check for compatibility in settings
        if self._no_mapping and self._betafactors:
            raise ValueError("betafactors='True' requires no_mapping='False'.")

        self.universe = atomgroup.universe
        self.negative_atomgroup = self.universe.atoms - self.atomgroup
        self.boolean_grid, self.voxel2atom = self._voxelize_atomgroup()
        # Set type to match what cython expects it to be.
        self.voxel_containment = VoxelContainment(
            self.boolean_grid, verbose=self._verbose, write_structures=self._write_structures, slab=self._slab, counts=self._return_counts)

        # Set the beta factors in the universe.atoms
        if self._betafactors:
            self.set_betafactors()
        
    def __str__(self):
        return str(self.voxel_containment)

    def _voxelize_atomgroup(self):
        """
        Creates a boolean grid from the atomgroup and returns
        the boolean grid, voxel2atom.
        """
        if not self._no_mapping:
            # Need universe-wide mapping, but grid only for atomgroup
            grid, voxel2atom = create_voxels(
                self.universe.atoms, self.resolution, max_offset= self._max_offset, return_mapping=True)
            grid, _ = create_voxels(
                self.atomgroup, self.resolution, max_offset= self._max_offset, return_mapping=False)    
        else:
            # No mapping needed, just voxelize the atomgroup
            grid, voxel2atom  = create_voxels(
                self.atomgroup, self.resolution, max_offset= self._max_offset, return_mapping=False)
        
        if self.closing:
            grid = close_voxels(grid)       
        return grid, voxel2atom

    def get_atomgroup_from_voxel_positions(self, voxels):
        """
        Converts the voxels in a voxel array back to an atomgroup.
        Takes a voxel position array and uses the stored mapping to generate 
        a corresponding atomgroup. This is the inverse of create_voxels.
        
        Parameters
        ----------
        voxels: array-like of shape (M, 3)
            Voxel positions to convert back to atoms.

        Returns
        -------
        atomgroup: MDAnalysis AtomGroup
            The atomgroup corresponding to the provided voxel positions.   
        """
        if self._no_mapping:
            raise ValueError(
                "Voxel to atomgroup transformations are not possible when using the 'no_mapping' flag.\n"
                "no_mapping is only useful to speed up generating the voxel level containment,\n"
                "for it does not create breadcrumbs to work its way back to the atomgroup."
            )
        return voxels2atomgroup(voxels, self.voxel2atom, self.atomgroup)

    def get_atomgroup_from_nodes(self, nodes, containment=False):
        """
        Returns an atomgroup for the specified containment nodes.

        Parameters
        ----------
        nodes: list of int
            The containment node ids to extract atoms for.
        containment: bool
            If True, retrieves all inner compartments as well. 

        Returns
        -------
        atomgroup: MDAnalysis AtomGroup
            The atomgroup corresponding to the provided containment nodes.
        """
        if self._no_mapping:
            raise ValueError("Voxel to atomgroup transformations are not possible when using the 'no_mapping' flag.\nno_mapping is only useful to speed up generating the voxel level containment,\nfor it does not create breadcrumps to work its way back to the atomgroup.")
        # Retrieves all inner compartments as well.
        if containment:
            nodes = self.voxel_containment.get_downstream_nodes(nodes)
        
        # Uses the precalculated components per atom index in the tempfactors.
        if self._betafactors:
            mask = np.isin(self.universe.atoms.tempfactors, nodes)
            atomgroup = self.universe.atoms[mask]
        # Fallback to using the voxel mapping if the betafactors
        #  have not been set for performance reasons. This is useful
        #  when one only wants to extract/interact with a small part
        #  of the universe.
        else:
            voxel_positions = self.voxel_containment.get_voxel_positions(nodes)
            atomgroup = self.get_atomgroup_from_voxel_positions(voxel_positions)
        return atomgroup

    def set_betafactors(self):
        """
        Sets the component id per atom in the beta factors column of the universe.
        """
        betafactors = np.zeros(len(self.universe.atoms))
        all_nodes = self.voxel_containment.nodes
        atom_voxels, atom_indices = self.voxel2atom
        
        for node in all_nodes:
            voxels = self.voxel_containment.get_voxel_positions(node)
            voxels_array = np.asarray(voxels)
            
            # Vectorized comparison to find atoms in these voxels
            mask = (atom_voxels[:, None] == voxels_array[None, :]).all(axis=2).any(axis=1)
            selected_indices = atom_indices[mask]
            
            betafactors[selected_indices] = node
        
        try:
            self.universe.atoms.tempfactors = betafactors
            print('NOTE: tempfactors already set in the universe, and will be overwritten with the component ids.')
        except AttributeError:
            self.universe.add_TopologyAttr(
                mda.core.topologyattrs.Tempfactors(betafactors))
            print('Writing component ids in the tempfactors of universe.')

