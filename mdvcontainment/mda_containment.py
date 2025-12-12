"""
MDAnalysis level wrapper for containment of AtomGroups, still uses VoxelContainment for the containment computation.
"""

# Python
from abc import ABC, abstractmethod
from typing import Union, List, Set, Dict, Tuple, Optional, TypeAlias, Self, TYPE_CHECKING

# Python External
import numpy as np
import numpy.typing as npt
import MDAnalysis as mda  # type: ignore

# Python Module
from .voxel_logic import create_voxels, voxels2atomgroup, morph_voxels
from .voxel_containment import VoxelContainment
from .graph_logic import format_dag_structure

if TYPE_CHECKING:
    from .voxel_containment import VoxelContainmentView

# Type aliases for clarity and reusability
ArrayLike: TypeAlias = npt.ArrayLike
NDArrayBool: TypeAlias = npt.NDArray[np.bool_]
NDArrayInt: TypeAlias = npt.NDArray[np.int_]
VoxelPosition: TypeAlias = Tuple[int, int, int]

class ContainmentBase(ABC):
    """
    Abstract base class for Containment and ContainmentView.
    """

    def __str__(self) -> str:
        return format_dag_structure(
            self.voxel_containment.containment_graph,
            self.voxel_containment.component_ranks,
            self.voxel_volumes,
            unit='nm³')
    
    # Properties - all delegate to _base for data access

    @property
    def nodes(self) -> List[int]:
        return self.voxel_containment.nodes

    @property
    def voxel_volume(self) -> float:
        total_volume: float = float(np.prod(self.universe.dimensions[:3]))
        total_voxels: int = int(np.prod(self.boolean_grid.shape))
        return (total_volume / total_voxels) / 1000  # Convert A^3 to nm^3

    @property
    def voxel_volumes(self) -> Dict[int, float]:
        voxel_counts: Dict[int, int] = self.voxel_containment.voxel_counts
        volumes: Dict[int, float] = {
            key: value * self._base.voxel_volume
            for key, value in voxel_counts.items()
        }
        return volumes

    @property
    def atomgroup(self) -> mda.AtomGroup:
        return self._base._atomgroup

    @property
    def universe(self) -> mda.Universe:
        return self._base._universe

    @property
    def negative_atomgroup(self) -> mda.AtomGroup:
        return self._base._negative_atomgroup

    @property
    def resolution(self) -> float:
        return self._base._resolution

    @property
    def closing(self) -> bool:
        return self._base._closing

    @property
    def morph(self) -> str:
        return self._base._morph

    @property
    def boolean_grid(self) -> NDArrayBool:
        return self._base._boolean_grid

    @property
    def voxel2atom(self) -> Dict[VoxelPosition, Set[int]]:
        return self._base._voxel2atom

    # Shared methods

    def get_atomgroup_from_voxel_positions(
        self,
        voxels: ArrayLike
    ) -> mda.AtomGroup:
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
        if self._base._no_mapping:
            raise ValueError(
                "Voxel to atomgroup transformations are not possible when using the 'no_mapping' flag.\n"
                "no_mapping is only useful to speed up generating the voxel level containment,\n"
                "for it does not create breadcrumbs to work its way back to the atomgroup."
            )

        if containment:
            nodes = self.voxel_containment.get_downstream_nodes(nodes)

        voxel_positions: ArrayLike = self.voxel_containment.get_voxel_positions(nodes)
        atomgroup: mda.AtomGroup = self.get_atomgroup_from_voxel_positions(voxel_positions)
        return atomgroup

    def _filter_nodes_on_volume(
        self,
        nodes: List[int],
        min_size: float
    ) -> List[int]:
        filtered_nodes: List[int] = []
        for node in nodes:
            downstream: List[int] = list(self.voxel_containment.get_downstream_nodes([node]))
            voxel_count: float = sum([self.voxel_volumes[n] for n in downstream])
            if voxel_count >= min_size:
                filtered_nodes.append(int(node))
        return filtered_nodes

    def node_view(
        self,
        keep_nodes: Optional[List[int]] = None,
        min_size: float = 0
    ) -> 'ContainmentView':
        if keep_nodes is None:
            keep_nodes = self.nodes
        else:
            unknown_nodes: Set[int] = set(keep_nodes) - set(self.nodes)
            assert len(unknown_nodes) == 0, f"Specified nodes not present in current nodes {unknown_nodes}."

        if min_size > 0:
            keep_nodes = self._filter_nodes_on_volume(keep_nodes, min_size)
        return ContainmentView(self._base, keep_nodes)

    def set_betafactors(self) -> None:
        if self._base._no_mapping:
            raise ValueError("set_betafactors requires no_mapping='False'.")

        betafactors: npt.NDArray[np.float_] = np.zeros(len(self.universe.atoms))
        all_nodes: List[int] = self.voxel_containment.nodes

        for node in all_nodes:
            voxels: ArrayLike = self.voxel_containment.get_voxel_positions([node])
            selected_atoms: mda.AtomGroup = voxels2atomgroup(voxels, self.voxel2atom, self.universe.atoms)
            if len(selected_atoms) > 0:
                betafactors[selected_atoms.ix] = node

        is_view: bool = isinstance(self, ContainmentView)

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
        self._voxel2atom: Dict[VoxelPosition, Set[int]]
        self._boolean_grid, self._voxel2atom = self._voxelize_atomgroup()

        self.voxel_containment: VoxelContainment = VoxelContainment(self._boolean_grid, verbose=self._verbose)
        self._base: Self = self

    def __repr__(self) -> str:
        return f'<Containment with {len(self.universe.atoms)} atoms in a {self.voxel_containment.grid.shape} grid with a resolution of {self.resolution} nm>'

    def _voxelize_atomgroup(self) -> Tuple[NDArrayBool, Dict[VoxelPosition, Set[int]]]:
        if not self._no_mapping:
            grid, voxel2atom = create_voxels(
                self._universe.atoms, self._resolution, max_offset=self._max_offset, return_mapping=True)
            grid, _ = create_voxels(
                self._atomgroup, self._resolution, max_offset=self._max_offset, return_mapping=False)
        else:
            grid, voxel2atom = create_voxels(
                self._atomgroup, self._resolution, max_offset=self._max_offset, return_mapping=False)

        if self._closing:
            grid = morph_voxels(grid, 'de')
        if self._morph:
            grid = morph_voxels(grid, self._morph)
        return grid, voxel2atom

class ContainmentView(ContainmentBase):
    """
    Memory-efficient view on a Containment with merged nodes.
    """

    def __init__(
        self,
        base_containment: Containment,
        keep_nodes: Union[List[int], Set[int]]
    ) -> None:
        self._base: Containment = base_containment._base
        self.voxel_containment: VoxelContainmentView = self._base.voxel_containment.node_view(keep_nodes)

    def __repr__(self) -> str:
        return f'<ContainmentView with {len(self.universe.atoms)} atoms in a {self.voxel_containment.grid.shape} grid with a resolution of {self.resolution} nm>'

    def get_original_nodes(self, view_node: int) -> List[int]:
        return self.voxel_containment.get_original_nodes(view_node)

    def get_view_node(self, original_node: int) -> Optional[int]:
        return self.voxel_containment.get_view_node(original_node)
