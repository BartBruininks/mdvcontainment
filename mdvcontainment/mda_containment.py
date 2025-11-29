from abc import ABC
import numpy as np

import MDAnalysis as mda

from .atomgroup_to_voxels import create_voxels, voxels2atomgroup, morph_voxels
from .containment_main import VoxelContainment


class ContainmentBase(ABC):
    """
    Abstract base class for Containment and ContainmentView.
    
    Contains all shared functionality that works on any containment (original or view).
    Subclasses must set:
    - self._base: Reference to the data owner (Containment instance)
    - self.voxel_containment: VoxelContainment or VoxelContainmentView instance
    """
    
    def __str__(self):
        return str(self.voxel_containment)
    
    # Properties - all delegate to _base for data access
    
    @property
    def atomgroup(self):
        """Reference to the original atomgroup."""
        return self._base._atomgroup
    
    @property
    def universe(self):
        """Reference to the original universe."""
        return self._base._universe
    
    @property
    def negative_atomgroup(self):
        """Reference to the negative atomgroup from base."""
        return self._base._negative_atomgroup
    
    @property
    def resolution(self):
        """Resolution from base containment."""
        return self._base._resolution
    
    @property
    def closing(self):
        """Closing parameter from base containment."""
        return self._base._closing
    
    @property
    def morph(self):
        """Morph parameter from base containment."""
        return self._base._morph
    
    @property
    def boolean_grid(self):
        """Reference to the boolean grid from base."""
        return self._base._boolean_grid
    
    @property
    def voxel2atom(self):
        """Reference to the voxel2atom mapping from base."""
        return self._base._voxel2atom
    
    # Shared methods - work identically for both Containment and ContainmentView
    
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
        if self._base._no_mapping:
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
        if self._base._no_mapping:
            raise ValueError(
                "Voxel to atomgroup transformations are not possible when using the 'no_mapping' flag.\n"
                "no_mapping is only useful to speed up generating the voxel level containment,\n"
                "for it does not create breadcrumbs to work its way back to the atomgroup."
            )
        
        # Retrieves all inner compartments as well.
        if containment:
            nodes = self.voxel_containment.get_downstream_nodes(nodes)
        
        # TODO: Proper benchmarking of both methods, for now it seems like needless complexity.
        ## Uses the precalculated components per atom index in the tempfactors.
        #if self._base._betafactors:
        #    mask = np.isin(self.universe.atoms.tempfactors, nodes)
        #    atomgroup = self.universe.atoms[mask]
        ## Fallback to using the voxel mapping if the betafactors
        ## have not been set for performance reasons. This is useful
        ## when one only wants to extract/interact with a small part
        ## of the universe.
        #else:

        voxel_positions = self.voxel_containment.get_voxel_positions(nodes)
        atomgroup = self.get_atomgroup_from_voxel_positions(voxel_positions)
        return atomgroup
    
    def node_view(self, keep_nodes):
        """
        Create a view where only keep_nodes are visible.
        
        Nodes not in keep_nodes are merged upstream to their nearest kept ancestor.
        This creates a ContainmentView that shares the underlying data (voxel grids,
        atom mappings) with the base Containment for memory efficiency.
        
        Parameters
        ----------
        keep_nodes : list
            Nodes to keep in the view. Other nodes are merged upstream.
        
        Returns
        -------
        ContainmentView
            A view with the same API but merged nodes.
        
        Examples
        --------
        >>> containment = Containment(atomgroup, resolution=0.5)
        >>> # Remove small compartments
        >>> view = containment.node_view([1, 3, 5, 7])
        >>> 
        >>> # Work with the simplified structure
        >>> atoms = view.get_atomgroup_from_nodes([1])
        >>> print(view)  # Shows merged structure
        >>> 
        >>> # Original containment is unchanged
        >>> print(containment)  # Shows full structure
        """
        # Always create view from the original base, not from intermediate views
        return ContainmentView(self._base, keep_nodes)
    
    def set_betafactors(self):
        """
        Sets the component id per atom in the beta factors column of the universe.
        
        For Containment: Sets original node IDs.
        For ContainmentView: Sets view node IDs (merged nodes get same ID).
        
        Note: This modifies the universe and overwrites any existing tempfactors.
        """
        if self._base._no_mapping:
            raise ValueError(
                "set_betafactors requires no_mapping='False'."
            )
        
        betafactors = np.zeros(len(self.universe.atoms))
        all_nodes = self.voxel_containment.nodes
        
        for node in all_nodes:
            voxels = self.voxel_containment.get_voxel_positions([node])
            
            # Get corresponding atoms
            selected_atoms = voxels2atomgroup(voxels, self.voxel2atom, self.universe.atoms)
            
            if len(selected_atoms) > 0:
                betafactors[selected_atoms.ix] = node
        
        # Determine if this is a view or original containment
        is_view = isinstance(self, ContainmentView)
        
        try:
            self.universe.atoms.tempfactors = betafactors
            if is_view:
                print('NOTE: tempfactors already set in the universe, and will be overwritten with the VIEW component ids.')
            else:
                print('NOTE: tempfactors already set in the universe, and will be overwritten with the component ids.')
        except AttributeError:
            self.universe.add_TopologyAttr(
                mda.core.topologyattrs.Tempfactors(betafactors))
            if is_view:
                print('Writing VIEW component ids in the tempfactors of universe.')
            else:
                print('Writing component ids in the tempfactors of universe.')


class Containment(ContainmentBase):
    """
    Main containment class that creates and owns voxelized data.
    
    This class performs voxelization of an atomgroup and creates the containment
    hierarchy. It owns all the data (voxel grids, atom mappings, etc.).
    """
    
    def __init__(self, atomgroup, resolution, closing=False, slab=False, morph="", max_offset=0.05, 
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

        Morph can be used to apply additional morphological operations
        on the voxel mask. It is a string containing 'd' and 'e' characters,
        where 'd' is dilation and 'e' is erosion. The operations are
        applied in the order they appear in the string. For example,
        morph='de' will first dilate the voxel mask, and then erode it.

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
        assert isinstance(atomgroup, mda.core.groups.AtomGroup), "Atomgroup must be an MDAnalysis AtomGroup."
        assert type(resolution) in [float, int], "Resolution must be a float or int."
        assert type(closing) == bool, "Closing must be a boolean."
        assert type(morph) == str, "Morph must be a string."
        assert set(morph).issubset({'d', 'e'}), "Morph strings can only contain 'd' and 'e' characters."
        assert type(slab) == bool, "Slab must be a boolean."
        assert type(max_offset) in [float, int], "Max_offset must be a float or int."
        assert type(verbose) == bool, "Verbose must be a boolean."
        assert type(write_structures) == bool, "Write_structures must be a boolean (write to mask.gro)."
        assert type(no_mapping) == bool, "No_mapping must be a boolean."
        assert type(return_counts) == bool, "Return_counts must be a boolean."
        assert type(betafactors) == bool, "Betafactors must be a boolean."

        if closing:
            print("WARNING: The 'closing' parameter is deprecated and will be removed in future versions. Please use the 'morph' parameter with value 'de' instead.")
            if morph != "":
                print("WARNING: Both 'closing' and 'morph' parameters are set. The 'closing' operation will be applied before the 'morph' operations.")

        # Check for compatibility in settings
        if no_mapping and betafactors:
            raise ValueError("betafactors='True' requires no_mapping='False'.")

        # Store all parameters as instance attributes
        self._atomgroup = atomgroup
        self._resolution = resolution
        self._closing = closing
        self._morph = morph
        self._slab = slab
        self._max_offset = max_offset
        self._verbose = verbose
        self._write_structures = write_structures
        self._no_mapping = no_mapping
        self._return_counts = return_counts
        self._betafactors = betafactors
        
        # Store universe and derived atomgroups
        self._universe = atomgroup.universe
        self._negative_atomgroup = self._universe.atoms - self._atomgroup
        
        # Voxelize the atomgroup
        self._boolean_grid, self._voxel2atom = self._voxelize_atomgroup()
        
        # Create voxel containment
        self.voxel_containment = VoxelContainment(
            self._boolean_grid, verbose=self._verbose, write_structures=self._write_structures, 
            slab=self._slab, counts=self._return_counts)
        
        # CRITICAL: Set self-reference for base class properties to work
        self._base = self

        # Set the beta factors in the universe.atoms
        if self._betafactors:
            self.set_betafactors()
    
    def _voxelize_atomgroup(self):
        """
        Creates a boolean grid from the atomgroup and returns
        the boolean grid, voxel2atom.
        """
        if not self._no_mapping:
            # Need universe-wide mapping, but grid only for atomgroup
            grid, voxel2atom = create_voxels(
                self._universe.atoms, self._resolution, max_offset=self._max_offset, return_mapping=True)
            grid, _ = create_voxels(
                self._atomgroup, self._resolution, max_offset=self._max_offset, return_mapping=False)    
        else:
            # No mapping needed, just voxelize the atomgroup
            grid, voxel2atom = create_voxels(
                self._atomgroup, self._resolution, max_offset=self._max_offset, return_mapping=False)
        
        # This is a legacy method and will be removed in future versions.
        if self._closing:
            grid = morph_voxels(grid, 'de')

        if self._morph:
            grid = morph_voxels(grid, self._morph)
        return grid, voxel2atom


class ContainmentView(ContainmentBase):
    """
    A view on a Containment that merges unspecified nodes with their parents.
    
    This provides the same API as Containment but operates on a view of the
    underlying VoxelContainment. The voxel2atom mapping and atomgroup references
    are shared with the base Containment for memory efficiency.
    
    Parameters
    ----------
    base_containment : Containment
        The base Containment object to create a view on.
    keep_nodes : list or set
        Nodes to keep visible in the view. Other nodes are merged upstream
        to their nearest kept ancestor. If no ancestor is kept, the node is dropped.
    
    Examples
    --------
    >>> containment = Containment(atomgroup, resolution=0.5)
    >>> # Remove small compartments (nodes B, D, E)
    >>> view = containment.node_view([A, C, F])
    >>> # Now can work with simplified structure
    >>> atoms = view.get_atomgroup_from_nodes([A])  # Includes merged nodes
    """
    
    def __init__(self, base_containment, keep_nodes):
        # Store reference to the base Containment (the data owner)
        self._base = base_containment._base
        
        # Create the VoxelContainment view - this does the heavy lifting
        self.voxel_containment = self._base.voxel_containment.node_view(keep_nodes)
    
    def get_original_nodes(self, view_node):
        """
        Get all original nodes that are merged into a view node.
        
        Parameters
        ----------
        view_node : int
            A node ID in the view.
        
        Returns
        -------
        list
            List of original node IDs merged into this view node.
        """
        return self.voxel_containment.get_original_nodes(view_node)
    
    def get_view_node(self, original_node):
        """
        Get the view node that an original node is mapped to.
        
        Parameters
        ----------
        original_node : int
            A node ID from the original containment.
        
        Returns
        -------
        int or None
            The view node ID, or None if the original node was dropped.
        """
        return self.voxel_containment.get_view_node(original_node)
