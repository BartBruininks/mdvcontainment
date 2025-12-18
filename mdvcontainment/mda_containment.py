"""
MDAnalysis level wrapper for containment of AtomGroups, still uses VoxelContainment for the containment computation.
"""

# Python
from typing import Union, List, Set, Dict, Tuple, Optional, TypeAlias, TYPE_CHECKING
import numpy as np
import numpy.typing as npt
import MDAnalysis as mda  # type: ignore
from MDAnalysis.core.topologyattrs import Tempfactors

# Python Module
from .voxel_logic import create_voxels, voxels2atomgroup, morph_voxels
from .voxel_containment import VoxelContainment
from .graph_logic import format_dag_structure

if TYPE_CHECKING:
    from .voxel_containment import VoxelContainmentView

# Type aliases for clarity and reusability
NDArrayBool: TypeAlias = npt.NDArray[np.bool_]


class ContainmentBase:
    """
    Base class for Containment and ContainmentView.
    
    Contains all shared functionality that works on any containment
    (original or view).
    """
    # Type hints for the type checker - these attributes will exist at runtime
    # in concrete subclasses but are defined here for static type checking
    if TYPE_CHECKING:
        _base: 'Containment'
        voxel_containment: Union[VoxelContainment, 'VoxelContainmentView']

    def __str__(self) -> str:
        return format_dag_structure(
            self.voxel_containment.containment_graph,
            self.voxel_containment.component_ranks,
            self.component_volumes, 
            unit='nm³')
    
    # Properties - all delegate to _base for data access

    @property
    def nodes(self) -> List[int]:
        """List of all component/node IDs."""
        return self.voxel_containment.nodes

    @property
    def voxel_volume(self) -> float:
        """Effective voxel volume in nm³ (this might deviate from the resolution³)."""
        total_volume: float = float(np.prod(self.universe.dimensions[:3]))
        total_voxels: int = int(np.prod(self.boolean_grid.shape))
        return (total_volume / total_voxels) / 1000  # Convert A^3 to nm^3

    @property
    def component_volumes(self) -> Dict[int, float]:
        """Component volumes in nm³ as a dict."""
        voxel_counts: Dict[int, int] = self.voxel_containment.voxel_counts
        volumes: Dict[int, float] = {
            key: value * self._base.voxel_volume
            for key, value in voxel_counts.items()
        }
        return volumes

    @property
    def atomgroup(self) -> mda.AtomGroup:
        """
        Reference to the input `AtomGroup` used for the creation of the boolean grid.

        A voxel is set to `True` if at least one atom lies inside it.
        """
        return self._base._atomgroup

    @property
    def universe(self) -> mda.Universe:
        """Reference to the input `Universe`."""
        return self._base._universe

    @property
    def negative_atomgroup(self) -> mda.AtomGroup:
        """Reference to all atoms in the `Universe` NOT in the input `AtomGroup`."""
        return self._base._negative_atomgroup

    @property
    def resolution(self) -> float:
        """
        Reference to the input target `resolution` in nm for the voxel size.
        
        The resolution might deviate slightly (`max_offset`) from the real voxel shape. 
        As the voxel size has to be an integer divider of the box size.
        
        The resolution should be about twice the average distance (LJ sigma),
        of the condensed phase. E.g. for CG Martini use 1.0 nm, or use 0.5 nm
        with `closing`. For atomistic systems a resolution of 0.5 nm can
        often be used without `closing`.
        """
        return self._base._resolution

    @property
    def closing(self) -> bool:
        """
        Reference to the input argument `closing` (bool).
        
        Closing performs binary dilation and erosion on the boolean grid
        to close small holes. Closing is advised when using CG Martini
        with a `resolution` of 0.5 nm.
        """
        return self._base._closing

    @property
    def morph(self) -> Optional[str]:
        """
        Reference to the input morph string.
        
        Morphing is the general case for performing any amount of binary 
        dilation and or erosion operations. The operations are performed 
        in order of occurrence in the string.

        Example
        -------
        >>> # Perform dilation first, then erosion followed by one last dilation. 
        >>> morph = 'ded' 
        """
        return self._base._morph

    @property
    def max_offset(self) -> float:
        """Reference to the input `max_offset`, specifies the maximum deviation 
        ratio between the target and resulting voxel resolution."""
        return self._base._max_offset

    @property
    def boolean_grid(self) -> NDArrayBool:
        """
        Created boolean grid by atom occupancy and or morphing.
        
        Initially a voxel in the grid is `True` if any atom in the input AtomGroup lies in it.
        This can be affected by `closing` and `morph` as well, which would alter the boolean values
        as indicated.
        """
        return self._base._boolean_grid

    @property
    def mapping_dicts(self) -> Optional[Dict[str, Union[npt.NDArray, Dict]]]:
        """
        AtomGroup-Voxels mapping dicts.
        
        If self._return_mapping=True, contains:
            - 'atom_voxels': (N, 3) array mapping each atom to voxel coordinates
            - 'voxel_to_atoms': dict mapping voxel tuple to atom indices
            - 'atom_indices': original atom indices from atomgroup
        """
        return self._base._mapping

    # Shared methods

    def get_atomgroup_from_voxel_positions(
        self,
        voxels: npt.NDArray[np.int32],
        ) -> mda.AtomGroup:
        """Returns the `AtomGroup` of all atoms which lie in the specified voxel positions."""
        if self._base._no_mapping:
            raise ValueError(
                "Voxel to atomgroup transformations are not possible when using the 'no_mapping' flag.\n"
                "no_mapping is only useful to speed up generating the voxel level containment,\n"
                "for it does not create breadcrumbs to work its way back to the atomgroup."
            )

        assert self.mapping_dicts is not None, "No mapping available this is probably because `no_mapping` was used."
        return voxels2atomgroup(voxels, self.mapping_dicts, self.atomgroup)

    def get_atomgroup_from_nodes(
        self,
        nodes: List[int],
        containment: bool = False,
        ) -> mda.AtomGroup:
        """Returns the `AtomGroup` of all atoms which lie in voxels whose label ID is in the provided list of nodes."""
        if self._base._no_mapping:
            raise ValueError(
                "Voxel to atomgroup transformations are not possible when using the 'no_mapping' flag.\n"
                "no_mapping is only useful to speed up generating the voxel level containment,\n"
                "for it does not create breadcrumbs to work its way back to the atomgroup."
            )

        if containment:
            nodes = self.voxel_containment.get_downstream_nodes(nodes)

        voxel_positions: npt.NDArray[np.int32] = self.voxel_containment.get_voxel_positions(nodes)
        atomgroup: mda.AtomGroup = self.get_atomgroup_from_voxel_positions(voxel_positions)
        return atomgroup

    def _filter_nodes_on_volume(
        self,
        nodes: List[int],
        min_size: float,
        ) -> List[int]:
        """Returns all nodes who have a downstream volume larger than the cutoff (includes self)."""
        filtered_nodes: List[int] = []
        for node in nodes:
            downstream: List[int] = list(self.voxel_containment.get_downstream_nodes([node]))
            voxel_count: float = sum([self.component_volumes[n] for n in downstream])
            if voxel_count >= min_size:
                filtered_nodes.append(int(node))
        return filtered_nodes

    def node_view(
        self,
        keep_nodes: Optional[Union[List[int], Set[int]]] = None,
        min_size: float = 0,
        ) -> 'ContainmentView':
        """Returns a `ContainmentView` which collapses all nodes upstream which are not selected."""
        if keep_nodes is None:
            keep_nodes_list = self.nodes
        else:
            # Convert to list if it's a set
            keep_nodes_list = list(keep_nodes) if isinstance(keep_nodes, set) else keep_nodes
            unknown_nodes: Set[int] = set(keep_nodes_list) - set(self.nodes)
            assert len(unknown_nodes) == 0, f"Specified nodes not present in current nodes {unknown_nodes}."

        if min_size > 0:
            keep_nodes_list = self._filter_nodes_on_volume(keep_nodes_list, min_size)
        
        return ContainmentView(self._base, keep_nodes_list)

    def set_betafactors(self) -> None:
        """Sets the component IDs of the atoms in the betafactor of the `Universe`."""
        if self._base._no_mapping:
            raise ValueError("set_betafactors requires no_mapping='False'.")

        assert self.universe is not None, 'Universe is None.'
        assert self.universe.atoms is not None, 'Universe.atoms is None.'
        assert self.mapping_dicts is not None, "No mapping available this is probably because `no_mapping` was used."

        betafactors: npt.NDArray[np.float64] = np.zeros(len(self.universe.atoms), dtype=np.float64)
        all_nodes: List[int] = self.voxel_containment.nodes

        for node in all_nodes:
            voxels: npt.NDArray[np.int32] = self.voxel_containment.get_voxel_positions([node])
            selected_atoms: mda.AtomGroup = voxels2atomgroup(voxels, self.mapping_dicts, self.universe.atoms)
            if len(selected_atoms) > 0:
                betafactors[selected_atoms.ix] = node

        is_view: bool = isinstance(self, ContainmentView)

        try:
            self.universe.atoms.tempfactors = betafactors  # type: ignore[attr-defined]
            if is_view:
                print('NOTE: tempfactors already set in the `Universe`, and will be overwritten with the VIEW component ids.')
            else:
                print('NOTE: tempfactors already set in the `Universe`, and will be overwritten with the component ids.')
        except AttributeError:
            self.universe.add_TopologyAttr(
                Tempfactors(betafactors))
            if is_view:
                print('Writing VIEW component ids in the tempfactors of Universe.')
            else:
                print('Writing component ids in the tempfactors of Universe.')


class Containment(ContainmentBase):
    """
    Hierarchical spatial containment analysis of molecular systems.

    Examples
    --------
    Basic usage with an MDAnalysis AtomGroup:

    >>> import MDAnalysis as mda
    >>> u = mda.Universe('topology.pdb', 'trajectory.xtc')
    >>> membrane = u.select_atoms('resname POPC')
    >>> containment = Containment(membrane, resolution=0.5)
    >>> print(containment)

    Notes
    -----
    The module uses nanometer (nm) units for lengths and volumes. MDAnalysis 
    atomgroups using Angstrom units are automatically converted internally.
    """

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        resolution: Union[float, int],
        closing: bool = False,
        morph: Optional[str] = None,
        max_offset: Union[float, int] = 0.05,
        verbose: bool = False,
        no_mapping: bool = False
    ) -> None:
        assert isinstance(atomgroup, mda.core.groups.AtomGroup), "AtomGroup must be an MDAnalysis.AtomGroup."
        assert isinstance(resolution, (float, int)), "Resolution must be a float or int."
        assert isinstance(closing, bool), "Closing must be a boolean."
        assert morph is None or isinstance(morph, str), "Morph must be a string or None."
        assert set(morph or "").issubset({'d', 'e'}), "Morph strings can only contain 'd' and 'e' characters."
        assert isinstance(max_offset, (float, int)), "Max_offset must be a float or int."
        assert isinstance(verbose, bool), "Verbose must be a boolean."
        assert isinstance(no_mapping, bool), "No_mapping must be a boolean."
        
        if closing:
            print("WARNING: The 'closing' parameter is deprecated and will be removed in future versions. Please use the 'morph' parameter with value 'de' instead.")
            if morph != "":
                print("WARNING: Both 'closing' and 'morph' parameters are set. The 'closing' operation will be applied before the 'morph' operations.")

        self._atomgroup: mda.AtomGroup = atomgroup
        self._resolution: float = float(resolution)
        self._closing: bool = closing
        self._morph: str = morph or ""
        self._max_offset: float = float(max_offset)
        self._verbose: bool = verbose
        self._no_mapping: bool = no_mapping

        self._universe: mda.Universe = atomgroup.universe
        self._negative_atomgroup: mda.AtomGroup = self._universe.atoms - self._atomgroup

        self._boolean_grid: NDArrayBool
        self._mapping: Optional[Dict[str, Union[npt.NDArray, Dict]]]
        
        self._boolean_grid, self._mapping = self._voxelize_atomgroup()

        self.voxel_containment = VoxelContainment(self._boolean_grid, verbose=self._verbose)
        self._base: 'Containment' = self

    def __repr__(self) -> str:
        # Tying around universes is a bit messy...
        assert self.universe is not None, 'Universe is None.'
        assert self.universe.atoms is not None, 'Universe.atoms is None.'
        return f'<Containment with {len(self.universe.atoms)} atoms in a {self.voxel_containment.grid.shape} grid with a resolution of {self.resolution} nm>'

    def _voxelize_atomgroup(self) -> Tuple[NDArrayBool, Optional[Dict[str, Union[npt.NDArray, Dict]]]]:
        """Returns the boolean occupancy grid by the atoms and the associated mapping if required."""
        # Tying around universes is a bit messy...
        assert self._universe is not None, 'Universe is None.'
        assert self._universe.atoms is not None, 'Universe.atoms is None.'
        
        if not self._no_mapping:
            grid, mapping = create_voxels(
                self._universe.atoms, self._resolution, max_offset=self._max_offset, return_mapping=True)
            grid, _ = create_voxels(
                self._atomgroup, self._resolution, max_offset=self._max_offset, return_mapping=False)
        else:
            grid, mapping = create_voxels(
                self._atomgroup, self._resolution, max_offset=self._max_offset, return_mapping=False)

        if self._closing:
            grid = morph_voxels(grid, 'de')
        if self._morph:
            grid = morph_voxels(grid, self._morph)
        return grid, mapping


class ContainmentView(ContainmentBase):
    """
    Memory-efficient view on a Containment with merged nodes.

    Example
    -------
    Create a filtered view:

    >>> # Keep only large compartments (>200 nm³)
    >>> view = containment.node_view(min_size=200)
    >>> atoms = view.get_atomgroup_from_nodes([1, 2, 3])

    Access the containment graph from VoxelContainment

    >>> containment.voxel_containment.containment_graph
    """

    def __init__(
        self,
        base_containment: Containment,
        keep_nodes: List[int]
    ) -> None:
        self._base: Containment = base_containment._base
        self.voxel_containment = self._base.voxel_containment.node_view(keep_nodes)
        

    def __repr__(self) -> str:
        # Tying around universes is a bit messy...
        assert self.universe is not None, 'Universe is None.'
        assert self.universe.atoms is not None, 'Universe.atoms is None.'
        return f'<ContainmentView with {len(self.universe.atoms)} atoms in a {self.voxel_containment.grid.shape} grid with a resolution of {self.resolution} nm>'

    def get_original_nodes(self, view_node: int) -> List[int]:
        """Returns the original nodes that are mapped to the view node."""
        if TYPE_CHECKING:
            assert type(self.voxel_containment) == VoxelContainmentView
        return self.voxel_containment.get_original_nodes(view_node)

    def get_view_node(self, original_node: int) -> Optional[int]:
        """Returns the view node the original node is mapped to."""
        if TYPE_CHECKING:
            assert type(self.voxel_containment) == VoxelContainmentView
        return self.voxel_containment.get_view_node(original_node)