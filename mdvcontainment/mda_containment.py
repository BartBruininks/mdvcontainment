"""
MDAnalysis level wrapper for containment of Atomgroups, still uses VoxelContainment for the containment computation.
"""

# Python
from abc import ABC
from typing import Union, List, Set, Dict, Tuple, Optional

# Python External
import numpy as np
import numpy.typing as npt
import MDAnalysis as mda

# Python Module
from .voxel_logic import create_voxels, voxels2atomgroup, morph_voxels
from .voxel_containment import VoxelContainment
from .graph_logic import format_dag_structure


class ContainmentBase(ABC):
    """
    Abstract base class for Containment and ContainmentView.
    
    This class provides all shared functionality that operates on containment data,
    whether it's the original full containment or a filtered view. Concrete subclasses
    must initialize two attributes:
    
    - ``self._base``: Reference to the Containment instance that owns the data
    - ``self.voxel_containment``: VoxelContainment or VoxelContainmentView instance
    
    The base class provides properties that delegate to the base containment for
    data access, ensuring views share memory with the original containment.
    """
    
    def __str__(self) -> str:
        """
        Return a formatted string representation of the containment hierarchy.
        
        Returns
        -------
        str
            Multi-line string showing the DAG structure with node volumes.
        """
        return format_dag_structure(
            self.voxel_containment.containment_graph, 
            self.voxel_containment.component_ranks, 
            self.voxel_volumes, 
            unit='nm³')
    
    # Properties - all delegate to _base for data access

    @property
    def nodes(self) -> List[int]:
        """
        List of containment node IDs.
        
        Returns
        -------
        list of int
            Node identifiers in the current containment structure.
        """
        return self.voxel_containment.nodes
    
    @property
    def voxel_volume(self) -> float:
        """
        Volume of a single voxel.
        
        Returns
        -------
        float
            Volume in nm³ calculated from universe dimensions and grid shape.
        """
        total_volume = np.prod(self.universe.dimensions[:3])
        total_voxels = np.prod(self.boolean_grid.shape)
        return (total_volume / total_voxels) / 1000  # Convert A^3 to nm^3
    
    @property
    def voxel_volumes(self) -> Dict[int, float]:
        """
        Volume of each node in the containment graph.
        
        Returns
        -------
        dict of {int: float}
            Mapping from node ID to volume in nm³.
        """
        voxel_counts = self.voxel_containment.voxel_counts
        volumes = {key: value*self._base.voxel_volume for key, value in voxel_counts.items()}
        return volumes
    
    @property
    def atomgroup(self) -> mda.AtomGroup:
        """
        Original atomgroup used to create the containment.
        
        Returns
        -------
        MDAnalysis.AtomGroup
            The atomgroup that was voxelized.
        """
        return self._base._atomgroup
    
    @property
    def universe(self) -> mda.Universe:
        """
        Universe containing all atoms.
        
        Returns
        -------
        MDAnalysis.Universe
            The complete molecular system.
        """
        return self._base._universe
    
    @property
    def negative_atomgroup(self) -> mda.AtomGroup:
        """
        Atoms not included in the main atomgroup.
        
        Returns
        -------
        MDAnalysis.AtomGroup
            Atoms in the universe minus the original atomgroup.
        """
        return self._base._negative_atomgroup
    
    @property
    def resolution(self) -> float:
        """
        Voxel resolution used for discretization.
        
        Returns
        -------
        float
            Resolution in nanometers.
        """
        return self._base._resolution
    
    @property
    def closing(self) -> bool:
        """
        Whether binary closing was applied.
        
        Returns
        -------
        bool
            True if binary closing (dilation + erosion) was used.
        
        .. deprecated:: 
           Use the ``morph`` parameter instead.
        """
        return self._base._closing
    
    @property
    def morph(self) -> str:
        """
        Morphological operations applied to the voxel grid.
        
        Returns
        -------
        str
            String of operations: 'd' for dilation, 'e' for erosion.
        """
        return self._base._morph
    
    @property
    def boolean_grid(self) -> npt.NDArray[np.bool_]:
        """
        3D boolean array representing the voxelized structure.
        
        Returns
        -------
        numpy.ndarray of bool
            Boolean grid where True indicates occupied voxels.
        """
        return self._base._boolean_grid
    
    @property
    def voxel2atom(self) -> Dict[Tuple[int, int, int], Set[int]]:
        """
        Mapping from voxel positions to atom indices.
        
        Returns
        -------
        dict of {tuple: set}
            Keys are voxel position tuples (i, j, k), values are sets of atom indices.
        """
        return self._base._voxel2atom
    
    # Shared methods - work identically for both Containment and ContainmentView
    
    def get_atomgroup_from_voxel_positions(
        self, 
        voxels: npt.ArrayLike
    ) -> mda.AtomGroup:
        """
        Convert voxel positions back to an atomgroup.
        
        This is the inverse operation of voxelization, using the stored mapping
        to retrieve the atoms that occupy the specified voxel positions.
        
        Parameters
        ----------
        voxels : array_like, shape (M, 3)
            Voxel positions as (i, j, k) indices.
        
        Returns
        -------
        MDAnalysis.AtomGroup
            Atoms corresponding to the provided voxel positions.
        
        Raises
        ------
        ValueError
            If the containment was created with ``no_mapping=True``.
        
        Examples
        --------
        >>> voxels = np.array([[10, 20, 30], [11, 20, 30]])
        >>> atoms = containment.get_atomgroup_from_voxel_positions(voxels)
        >>> print(len(atoms))
        """
        if self._base._no_mapping:
            raise ValueError(
                "Voxel to atomgroup transformations are not possible when using the 'no_mapping' flag.\n"
                "no_mapping is only useful to speed up generating the voxel level containment,\n"
                "for it does not create breadcrumbs to work its way back to the atomgroup."
            )
        return voxels2atomgroup(voxels, self.voxel2atom, self.atomgroup)
    
    def get_atomgroup_from_nodes(
        self, 
        nodes: List[int], 
        containment: bool = False
    ) -> mda.AtomGroup:
        """
        Extract atoms belonging to specified containment nodes.
        
        Parameters
        ----------
        nodes : list of int
            Containment node IDs to extract atoms for.
        containment : bool, optional
            If True, also includes all atoms from nodes contained within
            the specified nodes (downstream in the hierarchy). Default is False.
        
        Returns
        -------
        MDAnalysis.AtomGroup
            Atoms corresponding to the specified nodes.
        
        Raises
        ------
        ValueError
            If the containment was created with ``no_mapping=True``.
        
        Examples
        --------
        Get atoms for specific nodes:
        
        >>> atoms = containment.get_atomgroup_from_nodes([1, 2, 3])
        
        Get atoms including all contained compartments:
        
        >>> atoms = containment.get_atomgroup_from_nodes([1], containment=True)
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
        
        voxel_positions = self.voxel_containment.get_voxel_positions(nodes)
        atomgroup = self.get_atomgroup_from_voxel_positions(voxel_positions)
        return atomgroup
    
    def _filter_nodes_on_volume(
        self, 
        nodes: List[int], 
        min_size: float
    ) -> List[int]:
        """
        Filter nodes based on their total volume including contained nodes.
        
        Parameters
        ----------
        nodes : list of int
            Node IDs to filter.
        min_size : float
            Minimum volume in nm³ for a node plus its downstream nodes.
        
        Returns
        -------
        list of int
            Node IDs that meet the size requirement.
        """
        filtered_nodes = []
        for node in nodes:
            downstream = list(self.voxel_containment.get_downstream_nodes([node]))
            voxel_count = sum([self.voxel_volumes[n] for n in downstream])
            if voxel_count >= min_size:
                filtered_nodes.append(int(node))
        return filtered_nodes
    
    def node_view(
        self, 
        keep_nodes: Optional[List[int]] = None, 
        min_size: float = 0
    ) -> 'ContainmentView':
        """
        Create a view with filtered or merged nodes.
        
        This creates a ContainmentView that shares underlying data with the base
        Containment for memory efficiency. Nodes not in ``keep_nodes`` or smaller
        than ``min_size`` are merged upstream to their nearest kept ancestor.
        
        Parameters
        ----------
        keep_nodes : list of int, optional
            Nodes to keep in the view. If None, keeps all nodes. Nodes not
            in this list are merged to their nearest kept ancestor.
        min_size : float, optional
            Minimum volume in nm³ for a node (including downstream nodes) to
            be kept. Nodes smaller than this are merged upstream. Default is 0.
        
        Returns
        -------
        ContainmentView
            A view with the same API as Containment but with merged nodes.
        
        Raises
        ------
        AssertionError
            If ``keep_nodes`` contains node IDs not present in the current structure.
        
        Examples
        --------
        Remove small compartments:
        
        >>> # Keep only compartments larger than 200 nm³
        >>> view = containment.node_view(min_size=200)
        
        Keep specific nodes:
        
        >>> # Keep only nodes 1, 3, 5, 7
        >>> view = containment.node_view([1, 3, 5, 7])
        
        Combine both filters:
        
        >>> # Keep specific nodes that are also large enough
        >>> view = containment.node_view([1, 3, 5, 7], min_size=200)
        >>> 
        >>> # Work with the simplified structure
        >>> atoms = view.get_atomgroup_from_nodes([1])
        >>> print(view)  # Shows merged structure
        >>> 
        >>> # Original containment is unchanged
        >>> print(containment)  # Shows full structure
        
        Notes
        -----
        Views always reference the original base Containment, not intermediate
        views. This ensures consistent data sharing and prevents chains of views.
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
    
    def set_betafactors(self) -> None:
        """
        Store node IDs in the universe's tempfactors (B-factors) column.
        
        This assigns each atom's containment node ID to its tempfactor field,
        enabling visualization in molecular viewers and quick node-based selections.
        For views, this stores the merged view node IDs.
        
        Raises
        ------
        ValueError
            If the containment was created with ``no_mapping=True``.
        
        Warnings
        --------
        This modifies the universe and overwrites any existing tempfactor values.
        
        Examples
        --------
        >>> containment.set_betafactors()
        >>> # Now tempfactors contain node IDs
        >>> u.atoms.write('structure_with_nodes.pdb')
        
        >>> # Use in selections
        >>> node_1_atoms = u.select_atoms('tempfactor 1')
        
        Notes
        -----
        If tempfactors don't exist in the universe, they will be created.
        A message is printed indicating whether original or view node IDs
        are being written.
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
    Hierarchical spatial containment analysis of molecular systems.
    
    This class creates a voxelized representation of an atomgroup and constructs
    a containment hierarchy showing which spatial regions contain others. It owns
    all the underlying data including voxel grids and atom mappings.
    
    The containment hierarchy is represented as a directed acyclic graph (DAG) where
    edges point from containing regions to contained regions. This enables analysis
    of volumes, boundaries, and spatial relationships.
    
    Parameters
    ----------
    atomgroup : MDAnalysis.AtomGroup
        The atoms to voxelize and analyze. All atoms in the universe will be
        mapped, so removing unnecessary atoms (e.g., water) before creating
        the universe can improve performance.
    resolution : float
        Voxel size in nanometers. Smaller values give finer detail but require
        more memory and computation time.
    closing : bool, optional
        If True, applies binary closing (dilation then erosion) to fill small
        gaps. Usually required for coarse-grained simulations at 0.5 nm resolution.
        Default is False. **Deprecated**: Use ``morph='de'`` instead.
    morph : str, optional
        Morphological operations to apply: 'd' for dilation, 'e' for erosion.
        Operations are applied in order. For example, ``morph='de'`` is equivalent
        to ``closing=True``. Default is empty string (no operations).
    max_offset : float, optional
        Maximum allowed voxel alignment offset as a fraction of resolution.
        Set to 0 to allow any offset. Default is 0.05.
    verbose : bool, optional
        If True, prints progress information during construction. Default is False.
    no_mapping : bool, optional
        If True, skips creating the voxel-to-atom mapping, improving performance
        when atom-level operations are not needed. Methods that require this mapping
        will raise errors. Default is False.
    
    Raises
    ------
    AssertionError
        If parameters have incorrect types.
    
    Examples
    --------
    Basic usage:
    
    >>> import MDAnalysis as mda
    >>> u = mda.Universe('membrane.gro')
    >>> lipids = u.select_atoms('resname POPC POPE')
    >>> containment = Containment(lipids, resolution=0.5, morph='de')
    >>> print(containment)
    
    Access volumes:
    
    >>> volumes = containment.voxel_volumes
    >>> print(f"Node 1 volume: {volumes[1]} nm³")
    
    Extract atoms for specific compartments:
    
    >>> outer_shell = containment.get_atomgroup_from_nodes([1])
    >>> inner_core = containment.get_atomgroup_from_nodes([2], containment=True)
    
    Create simplified views:
    
    >>> # Keep only large compartments
    >>> view = containment.node_view(min_size=100)
    >>> print(view)
    
    Performance optimization:
    
    >>> # Skip mapping if only analyzing volumes
    >>> fast = Containment(lipids, resolution=1.0, no_mapping=True)
    >>> print(fast.voxel_volumes)  # Works fine
    >>> # fast.get_atomgroup_from_nodes([1])  # Would raise error
    
    Notes
    -----
    - Resolution units are nanometers, while MDAnalysis uses Angstroms internally
    - The universe's box dimensions must be defined for voxelization
    - Memory usage scales with ``(box_size / resolution)³``
    - For CG Martini simulations, typical resolutions are 0.5-1.0 nm
    """
    
    def __init__(
        self, 
        atomgroup: mda.AtomGroup, 
        resolution: Union[float, int], 
        closing: bool = False, 
        morph: str|None = None, 
        max_offset: Union[float, int] = 0.05, 
        verbose: bool = False, 
        no_mapping: bool = False
    ) -> None:
        assert isinstance(atomgroup, mda.core.groups.AtomGroup), "AtomGroup must be an MDAnalysis.AtomGroup."
        assert type(resolution) in [float, int], "Resolution must be a float or int."
        assert type(closing) == bool, "Closing must be a boolean."
        assert type(morph) == type(None) or type(morph) == str, "Morph must be a string or None."
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
    
    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f'<Containment with {len(self.universe.atoms)} atoms in a {self.voxel_containment.grid.shape} grid with a resolution of {self.resolution} nm>'
    
    def _voxelize_atomgroup(
        self
    ) -> Tuple[npt.NDArray[np.bool_], Dict[Tuple[int, int, int], Set[int]]]:
        """
        Create boolean grid from atomgroup.
        
        Returns
        -------
        tuple of (numpy.ndarray, dict)
            (boolean_grid, voxel2atom) where boolean_grid is a 3D numpy array
            and voxel2atom is a dict mapping voxel positions to atom indices.
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
    Memory-efficient view on a Containment with merged nodes.
    
    A ContainmentView provides the same API as Containment but operates on a
    filtered or simplified version of the hierarchy. Nodes not in the view are
    merged upstream to their nearest kept ancestor, creating a coarser hierarchy
    while preserving the containment relationships.
    
    Views share all underlying data (voxel grids, atom mappings) with the base
    Containment for memory efficiency. The original Containment remains unchanged.
    
    Parameters
    ----------
    base_containment : Containment
        The Containment object to create a view on.
    keep_nodes : list or set of int
        Node IDs to keep visible in the view. Other nodes are merged upstream
        to their nearest kept ancestor. Nodes without a kept ancestor are dropped.
    
    Examples
    --------
    Create a simplified view:
    
    >>> containment = Containment(atomgroup, resolution=0.5)
    >>> print(containment.nodes)  # [1, 2, 3, 4, 5, 6]
    >>> 
    >>> # Keep only nodes 1, 3, 6
    >>> view = containment.node_view([1, 3, 6])
    >>> print(view.nodes)  # [1, 3, 6]
    
    Work with the view:
    
    >>> # Nodes 2, 4, 5 are merged into their kept ancestors
    >>> atoms = view.get_atomgroup_from_nodes([1])  # Includes merged nodes
    >>> print(view)  # Shows simplified hierarchy
    
    Track merged nodes:
    
    >>> # See which original nodes are in view node 1
    >>> originals = view.get_original_nodes(1)
    >>> print(f"View node 1 contains original nodes: {originals}")
    >>> 
    >>> # Check where an original node ended up
    >>> view_node = view.get_view_node(2)  # Returns the ancestor it merged into
    
    Original is unchanged:
    
    >>> print(containment.nodes)  # Still [1, 2, 3, 4, 5, 6]
    
    Notes
    -----
    - Views always reference the original base Containment, not intermediate views
    - Multiple views can exist simultaneously on the same base Containment
    - All atom operations work identically on views and original containments
    - Volume calculations include merged nodes automatically
    """
    
    def __init__(
        self, 
        base_containment: Containment, 
        keep_nodes: Union[List[int], Set[int]]
    ) -> None:
        # Store reference to the base Containment (the data owner)
        self._base = base_containment._base
        
        # Create the VoxelContainment view - this does the heavy lifting
        self.voxel_containment = self._base.voxel_containment.node_view(keep_nodes)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f'<ContainmentView with {len(self.universe.atoms)} atoms in a {self.voxel_containment.grid.shape} grid with a resolution of {self.resolution} nm>'
    
    def get_original_nodes(self, view_node: int) -> List[int]:
        """
        Get all original nodes merged into a view node.
        
        Parameters
        ----------
        view_node : int
            A node ID in the view.
        
        Returns
        -------
        list of int
            Original node IDs that are merged into this view node.
        
        Examples
        --------
        >>> view = containment.node_view([1, 5])
        >>> originals = view.get_original_nodes(1)
        >>> print(f"View node 1 contains: {originals}")
        # Might output: [1, 2, 3] if nodes 2 and 3 were merged into 1
        """
        return self.voxel_containment.get_original_nodes(view_node)
    
    def get_view_node(self, original_node: int) -> Optional[int]:
        """
        Get the view node that an original node is mapped to.
        
        Parameters
        ----------
        original_node : int
            A node ID from the original containment.
        
        Returns
        -------
        int or None
            The view node ID that contains this original node, or None if
            the original node was dropped (no kept ancestor).
        
        Examples
        --------
        >>> view = containment.node_view([1, 5])
        >>> view_node = view.get_view_node(3)
        >>> # Might output: 1, if the original node 3 was merged into node 1 upon 
        >>> # making the view, due to it being downstream of 1.
        """
        return self.voxel_containment.get_view_node(original_node)