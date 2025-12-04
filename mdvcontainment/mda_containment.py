# Python
from abc import ABC

# Python External
import numpy as np
import MDAnalysis as mda

# Python Module
from .atomgroup_to_voxels import create_voxels, voxels2atomgroup, morph_voxels
from .containment_main import VoxelContainment
from .graph_logic import format_dag_structure


class ContainmentBase(ABC):
    """
    Abstract base class for Containment and ContainmentView.
    
    Contains all shared functionality that works on any containment (original or view).
    Subclasses must set:
    - self._base: Reference to the data owner (Containment instance)
    - self.voxel_containment: VoxelContainment or VoxelContainmentView instance
    """
    
    def __str__(self):
        return format_dag_structure(
            self.voxel_containment.containment_graph, 
            self.voxel_containment.component_ranks, 
            self.voxel_volumes, 
            unit='nm^3')
    
    # Properties - all delegate to _base for data access

    @property
    def nodes(self):
        """Containment nodes."""
        return self.voxel_containment.nodes
    
    @property
    def voxel_volume(self):
        """Volume of a single voxel in nm^3."""
        total_volume = np.prod(self.universe.dimensions[:3])
        total_voxels = np.prod(self.boolean_grid.shape)
        return (total_volume / total_voxels) / 1000  # Convert A^3 to nm^3
    
    @property
    def voxel_volumes(self):
        """
        Voxel volumes dict (nm^3) for each node in the current containment graph.
        """
        voxel_counts = self.voxel_containment.voxel_counts
        volumes = {key: value*self._base.voxel_volume for key, value in voxel_counts.items()}
        return volumes
    
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
    
    def _filter_nodes_on_volume(self, nodes, min_size):
        """
        Filter nodes based on their volumes (in nm^3) -- includes their downstream nodes.
        
        Parameters
        ----------
        nodes : list
            Nodes to include in the filtering
        min_size : int
            Minimum size (in voxels) for a node plus its downstream nodes to be kept.
        
        Returns
        -------
        list
            List of nodes that meet the size requirement.
        """
        filtered_nodes = []
        for node in nodes:
            downstream = list(self.voxel_containment.get_downstream_nodes([node]))
            voxel_count = sum([self.voxel_volumes[n] for n in downstream])
            if voxel_count >= min_size:
                filtered_nodes.append(int(node))
        return filtered_nodes
    
    def node_view(self, keep_nodes=None, min_size=0):
        """
        Create a view where only keep_nodes > min_size are exposed.
        
        Nodes not in keep_nodes or smaller than min_size are merged upstream to 
        their nearest kept ancestor. This creates a ContainmentView that shares 
        the underlying data (voxel grids, atom mappings) with the base Containment 
        for memory efficiency.
        
        Parameters
        ----------
        keep_nodes : list
            Nodes to keep in the view. Other nodes are merged upstream.
        min_size : int
            Minimum size (in voxels) for a node plus its downstream nodes to be kept.
        
        Returns
        -------
        ContainmentView
            A view with the same API but merged nodes.
        
        Examples
        --------
        >>> containment = Containment(atomgroup, resolution=0.5)
        >>> # Remove small compartments and at most keep provided node list
        >>> view = containment.node_view([1, 3, 5, 7], min_size=200)
        >>> 
        >>> # Work with the simplified structure
        >>> atoms = view.get_atomgroup_from_nodes([1])
        >>> print(view)  # Shows merged structure
        >>> 
        >>> # Original containment is unchanged
        >>> print(containment)  # Shows full structure
        """
        if keep_nodes is None:
            keep_nodes = self.nodes
        else:
            unknown_nodes = set(keep_nodes) - set(self.nodes)
            assert len(unknown_nodes) == 0, f"Specified nodes not present in current nodes {unknown_nodes}."

        # Filter nodes on volume as well if a min_size is provided
        if min_size > 0:
            keep_nodes = self._filter_nodes_on_volume(keep_nodes, min_size)
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
    
    def __init__(self, atomgroup, resolution, closing=False, morph="", max_offset=0.05, 
                 verbose=False, no_mapping=False, 
                 ):
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
        morph='de' will first dilate the voxel mask, and then erode it, 
        which is equivalent to closing. 

        Max_offset can be set to 0 to allow any resolution scaling 
        artefact for voxelization.

        Verbose can be used to set progress prints.

        No_mapping can be used to skip the generation of the voxel2atoms mapping. Making 
        the mapping is somewhat costly and if you already know you are not interested in 
        backmapping to atoms, why bother making it.

        All voxel logic (selecting by component etc) can be found under self.voxel_containment.
        """
        assert isinstance(atomgroup, mda.core.groups.AtomGroup), "Atomgroup must be an MDAnalysis AtomGroup."
        assert type(resolution) in [float, int], "Resolution must be a float or int."
        assert type(closing) == bool, "Closing must be a boolean."
        assert type(morph) == str, "Morph must be a string."
        assert set(morph).issubset({'d', 'e'}), "Morph strings can only contain 'd' and 'e' characters."
        assert type(max_offset) in [float, int], "Max_offset must be a float or int."
        assert type(verbose) == bool, "Verbose must be a boolean."
        assert type(no_mapping) == bool, "No_mapping must be a boolean."

        if closing:
            print("WARNING: The 'closing' parameter is deprecated and will be removed in future versions. Please use the 'morph' parameter with value 'de' instead.")
            if morph != "":
                print("WARNING: Both 'closing' and 'morph' parameters are set. The 'closing' operation will be applied before the 'morph' operations.")

        # Store all parameters as instance attributes
        self._atomgroup = atomgroup
        self._resolution = resolution
        self._closing = closing
        self._morph = morph
        self._max_offset = max_offset
        self._verbose = verbose
        self._no_mapping = no_mapping
        
        # Store universe and derived atomgroups
        self._universe = atomgroup.universe
        self._negative_atomgroup = self._universe.atoms - self._atomgroup
        
        # Voxelize the atomgroup
        self._boolean_grid, self._voxel2atom = self._voxelize_atomgroup()
        
        # Create voxel containment
        self.voxel_containment = VoxelContainment(self._boolean_grid, verbose=self._verbose)
        
        # CRITICAL: Set self-reference for base class properties to work
        self._base = self
    
    def __repr__(self):
        return f'<Containment with {len(self.universe.atoms)} atoms in a {self.voxel_containment.grid.shape} grid with a resolution of {self.resolution} nm>'
    
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

    def __repr__(self):
        return f'<ContainmentView with {len(self.universe.atoms)} atoms in a {self.voxel_containment.grid.shape} grid with a resolution of {self.resolution} nm>'
    
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
