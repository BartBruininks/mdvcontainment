import numpy as np

import MDAnalysis as mda

from .atomgroup_to_voxels import close_voxels, create_voxels
from .containment_main import VoxelContainment

class Containment():
    def __init__(self, atomgroup, resolution, closure=False, slab=False, max_offset=0.05, 
                 verbose=False, write_structures=False, no_mapping=False, return_counts=True, 
                 betafactors=True):
        """
        Atomgroup specifies the selected atoms. All atoms in the universe
        will be mapped, therefore removing waters from a file can make this
        faster if you do not need them.
        
        Resolution is taken in nn for MDA atomgroups. You do 
        not need to preprocess the positions, the units of the 
        atomgroup are converted from A to nm internally.

        Closure applies binary dilation and erosion, often called
        binary closure. This closes small gaps in the voxel mask.
        This is useually required for CG Martini simulations when,
        using a resolution of 0.5 nm. For a resolution of 1 nm it
        is not required and disadviced.

        Slab can be set to 'True' to process slices of a larger whole,
        this will disable any treatment of PBC.

        Max_offset can be set to 0 to allow any resolution scaling 
        artefact for voxelization.

        Verbose can be used to set developer prints.

        Write_structures can be set to write the input, label and component voxel masks as a gro.

        No_mapping can be used to skip the generation of the voxel2atoms dictionary. Making 
        dict is rather slow and if you already know you are interested in the voxel masks,
        why bother making it. This will prevent any mapping from voxels2atoms to work!

        Return counts is set to 'True' by default, this counts the amount of voxels per component
        this can be useful to get an estimate for the volume of a component.

        All voxel logic (selecting by comopnent etc) can be found under self.voxel_containment.
        """
        self.atomgroup = atomgroup
        self.resolution = resolution
        self.closure = closure
        self._slab = slab
        self._max_offset = max_offset
        self._verbose = verbose
        self._write_structures = write_structures
        self._no_mapping = no_mapping
        self._return_counts = return_counts
        self._betafactors = betafactors

        self.universe = atomgroup.universe
        self.negative_atomgroup = self.universe.atoms - self.atomgroup
        self.boolean_grid, self.voxel2atom, self.nbox = self._voxelize_atomgroup()
        # Set type to match what cython expects it to be.
        self.nbox = self.nbox.astype(np.float64)
        self.voxel_containment = VoxelContainment(
            self.boolean_grid, self.nbox, verbose=self._verbose, write_structures=self._write_structures, slab=self._slab, counts=self._return_counts)
        # Check for compatibility in settings
        if self._no_mapping and self._betafactors:
            raise ValueError("betafactors='True' requires no_mapping='False'.")
        # Set the beta factors in the universe.atoms
        self.set_betafactors()
        

    def __str__(self):
        return str(self.voxel_containment)

    def _voxelize_atomgroup(self):
        """
        Creates a boolean grid from the atomgroup and returns
        the boolean grid, voxel2atom and nbox.
        """
        if not self._no_mapping:
            grid, voxel2atom, nbox = create_voxels(
                self.universe.atoms, self.resolution, hyperres=False, max_offset= self._max_offset, return_mapping=True)
            grid, _, nbox = create_voxels(
                self.atomgroup, self.resolution, hyperres=False, max_offset= self._max_offset, return_mapping=False)
        else:
            grid, voxel2atom, nbox = create_voxels(
                self.universe.atoms, self.resolution, hyperres=False, max_offset= self._max_offset, return_mapping=False)
            grid, _, nbox = create_voxels(
                self.atomgroup, self.resolution, hyperres=False, max_offset= self._max_offset, return_mapping=False)
        
        if self.closure:
            grid = close_voxels(grid, nbox)
        return grid, voxel2atom, nbox

    def get_atomgroup_from_voxel_positions(self, voxels):
        """
        Converts the voxels in a voxel arrag back to an atomgroup.
        
        Takes a voxel position array and uses the voxel2atom mapping with respect to the
        atomgroup.universe to generate a corresponding atomgroup with the voxel 
        list. This is the inverse of gen_explicit_matrix.
        
        Returns an atomgroup.
        """
        if self._no_mapping:
            raise ValueError("Voxel to atomgroup transformations are not possible when using the 'no_mapping' flag.\nno_mapping is only useful to speed up generating the voxel level containment,\nfor it does not create breadcrumps to work its way back to the atomgroup.")
        # It is not important that every index only occurs onec,
        # as long as each atom is only selected once.
        indices = { idx for v in voxels for idx in self.voxel2atom[tuple(v)] }
        return self.universe.atoms[list(indices)]

    def get_atomgroup_from_nodes(self, nodes, containment=False):
        """
        Returns the atomgroup with all atoms inside the components grid which have their
        value in nodes. Containment can be set to True to also include all downstream 
        nodes of the selected nodes in the containmnt graph.
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
        Sets the component id per atom in the beta factors column.
        """
        betafactors = np.zeros(len(self.universe.atoms))

        all_nodes = self.voxel_containment.nodes
        for node in all_nodes:
            voxels = self.voxel_containment.get_voxel_positions(node)
            atom_indices = list({ idx for v in voxels for idx in self.voxel2atom[tuple(v)] })
            betafactors[atom_indices] = node

        try:
            print('NOTE: beta/tempfactors already set in the universe, and will be overwritten with the component ids.')
            self.unverse.tempfactors = betafactors
        except AttributeError:
            self.universe.add_TopologyAttr(
                mda.core.topologyattrs.Tempfactors(betafactors))

