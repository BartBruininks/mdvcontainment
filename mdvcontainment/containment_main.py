# Python External 
from abc import ABC
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Python Module
from .graph_logic import format_dag_structure


class VoxelContainmentBase(ABC):
    """
    Abstract base class for VoxelContainment and VoxelContainmentView.
    
    Contains all shared functionality that works on any voxel containment 
    (original or view). Subclasses must set:
    - self._base: Reference to the data owner (VoxelContainment instance)
    - self.containment_graph: The graph to operate on (property)
    """
    
    def __str__(self):
        return format_dag_structure(self.containment_graph, self.component_ranks, self.voxel_counts, unit='nvoxels')
    
    # Properties - delegate to _base or compute from current graph state
    
    @property
    def grid(self):
        """Reference to the original grid."""
        return self._base._grid
    
    @property
    def slab(self):
        """Slab setting from base containment."""
        return self._base._slab
    
    @property
    def components_grid(self):
        """Reference to the base components grid."""
        return self._base._components_grid
    
    @property
    def component_contact_graph(self):
        """Contact graph from base containment."""
        return self._component_contact_graph
    
    @property
    def nonp_label_contact_graph(self):
        """Non-periodic label contact graph from base containment."""
        return self._nonp_label_contact_graph
    
    @property
    def containment_graph(self):
        """Return the containment graph computed during construction."""
        return self._containment_graph
    
    @property
    def component_ranks(self):
        """Return the ranks computed during construction."""
        return self._base._component_ranks
    
    @property
    def voxel_counts(self):
        """
        Voxel counts for each node in the current containment graph.
        Computed based on the actual voxels represented by each node.
        """
        if not self._base.voxel_counts:
            return False
        
        counts = {}
        for node in self.nodes:
            # Get all original nodes this node represents
            original_nodes = self._resolve_nodes_to_original([node])
            counts[node] = sum(
                self._base.voxel_counts.get(orig, 0) for orig in original_nodes
            )
        return counts
    
    @property
    def nodes(self):
        """All nodes in the current containment graph."""
        return list(self.containment_graph.nodes)
    
    @property
    def root_nodes(self):
        """Root nodes (nodes with no parents) in the current containment graph."""
        return [n for n, d in self.containment_graph.in_degree() if d == 0]
    
    @property
    def leaf_nodes(self):
        """Leaf nodes (nodes with no children) in the current containment graph."""
        return [n for n, d in self.containment_graph.out_degree() if d == 0]
    
    # Helper method for node resolution (override in views)
    
    def _resolve_nodes_to_original(self, nodes):
        """
        Resolve nodes to their original node IDs in the base containment.
        
        For VoxelContainment: nodes are already original, return as-is.
        For VoxelContainmentView: expand view nodes to all merged original nodes.
        
        Parameters
        ----------
        nodes : list
            Node IDs in the current graph.
        
        Returns
        -------
        list
            Original node IDs in the base containment.
        """
        return nodes  # Default: no transformation
    
    # Shared methods - work for both VoxelContainment and VoxelContainmentView
    
    def _get_nodes(self):
        """Returns all nodes (components) in the containment graph."""
        return self.nodes
    
    def get_counts(self, grid=None):
        """
        Returns the number of elements in each component as a dict.

        Parameters
        ----------
        grid: int32 3D array, optional
            The components grid to calculate counts for. If None, uses
            cached voxel_counts.
        
        Returns
        -------
        counts_dict: dict
            Dict of component:int -> voxel_count:int 
        """
        if grid is not None:
            # If a custom grid is provided, count it directly
            return dict(zip(*np.unique(grid, return_counts=True)))
        else:
            # Return the cached counts
            return self.voxel_counts if self.voxel_counts else {}
    
    def _get_roots(self):
        """Returns a list of root nodes (absolute outsides)."""
        return self.root_nodes
    
    def _get_leaves(self):
        """Returns a list of leaf nodes (deepest level of containment)."""
        return self.leaf_nodes
    
    def _get_inverted_containment_graph(self):
        """
        Returns the inverted containment graph, creating and caching it if needed.
        """
        if not hasattr(self, '_inv_containment_graph') or not self._inv_containment_graph:
            self._inv_containment_graph = self.containment_graph.reverse()
        return self._inv_containment_graph
    
    def get_child_nodes(self, start_nodes):
        """
        Returns the children (neighbors) of the given nodes in the containment graph. 
        Taking directionality of the edges into account.
        """
        all_children = []
        for start_node in start_nodes:
            if start_node in self.containment_graph:
                all_children += list(self.containment_graph.neighbors(start_node))
        return sorted(set(all_children))
    
    def get_downstream_nodes(self, start_nodes):
        """
        Returns all nodes contained by the specified nodes (list), including the nodes itself.
        """
        downstream_nodes = []
        for node in start_nodes:
            if node in self.containment_graph:
                # Perform a depth-first search traversal from the start_node
                downstream_nodes += list(nx.dfs_postorder_nodes(self.containment_graph, node))[::-1]
        return sorted(set(downstream_nodes))
    
    def get_upstream_nodes(self, start_nodes):
        """
        Returns all nodes which contain the specified nodes (list), including the nodes itself.
        """
        # Create a reversed version of the graph
        reversed_G = self._get_inverted_containment_graph()
        upstream_nodes = []
        for node in start_nodes:
            if node in reversed_G:
                # Perform a depth-first search traversal from the start_node
                upstream_nodes += list(nx.dfs_postorder_nodes(reversed_G, node))[::-1]
        return sorted(set(upstream_nodes))
    
    def get_parent_nodes(self, start_nodes):
        """
        Returns the parents (neighbors) of the given nodes in the containment graph. 
        Taking inverted directionality of the edges into account.
        """
        reversed_G = self._get_inverted_containment_graph()
        all_parents = []
        for start_node in start_nodes:
            if start_node in reversed_G:
                all_parents += list(reversed_G.neighbors(start_node))
        return sorted(set(all_parents))
    
    def get_total_voxel_count(self, nodes):
        """
        Returns the total amount of voxels in the list of nodes.
        """
        if not self.voxel_counts:
            return 0
        total_count = 0
        for node in nodes:
            total_count += self.voxel_counts.get(node, 0)
        return total_count
    
    def get_voxel_mask(self, nodes):
        """
        Returns a boolean mask over the components grid where the voxel
        value is in the list of provided nodes.
        
        This method automatically handles view node expansion.
        """
        original_nodes = self._resolve_nodes_to_original(nodes)
        return np.isin(self.components_grid, original_nodes)
    
    def get_voxel_positions(self, nodes):
        """
        Returns all the voxel indices as a Nx3 array which are part of the specified nodes.
        
        This method automatically handles view node expansion.
        """
        mask = self.get_voxel_mask(nodes)
        indices = np.where(mask)
        return np.array(indices).T
    
    def format_containment(self, nodes=False):
        """
        Returns the selected nodes as a containment graph.

        By default prints the complete graph.
        """
        if nodes is not False:
            string = format_dag_structure(self.containment_graph.subgraph(nodes), 
                                        self.component_ranks, self.voxel_counts)
        else:
            string = format_dag_structure(self.containment_graph, 
                                        self.component_ranks, self.voxel_counts)
        return string
    
    def draw(self, nodes=False):
        """Draw the containment graph."""
        if nodes is False:
            nx.draw_networkx(self.containment_graph)
        else:
            nx.draw_networkx(self.containment_graph.subgraph(nodes))
        plt.show()

    def filter_nodes_on_size(self, min_size):
        """
        Filter nodes based on their size (in voxels) -- includes their downstream nodes.
        
        Parameters
        ----------
        min_size : int
            Minimum size (in voxels) for a node plus its downstream nodes to be kept.
        
        Returns
        -------
        list
            List of nodes that meet the size requirement.
        """
        filtered_nodes = []
        for node in self.nodes:
            downstream = list(self.get_downstream_nodes([node]))
            voxel_count = sum([self.voxel_counts[n] for n in downstream])
            if voxel_count >= min_size:
                filtered_nodes.append(int(node))
        return filtered_nodes
    
    def node_view(self, keep_nodes=None, min_size=0):
        """
        Create a view where only keep_nodes are visible.
        
        Parameters
        ----------
        keep_nodes : list
            Nodes to keep in the view. Other nodes are merged upstream.
        min_size : int
            Minimum size (in voxels) for a node plus its downstream nodes to be kept.
        
        Returns
        -------
        VoxelContainmentView
            A view with the same API but merged nodes.
        """
        if keep_nodes is None:
            keep_nodes = self.nodes
        else:
            unknown_nodes = set(keep_nodes) - set(self.nodes)
            assert len(unknown_nodes) == 0, f"Specified nodes not present in current nodes {unknown_nodes}."

        # Filter nodes on size if a min_size is provided
        if min_size > 0:
            keep_nodes = self.filter_nodes_on_size(min_size)
        
        # Always create view from the original base
        return VoxelContainmentView(self._base, keep_nodes)


class VoxelContainment(VoxelContainmentBase):
    """
    Main voxel containment class that creates and owns the containment graph.
    
    A Containment graph is a DAG which has a parent pointing to its children with an edge.
    """
    
    def __init__(self, grid, verbose=False, write_structures=False, draw_graphs=False, 
                 counts=True, slab=False):
        """
        Parameters
        ----------
        grid : ndarray
            Boolean 3D grid representing the voxelized structure.
        verbose : bool
            Enable verbose output.
        write_structures : bool
            Write structure files.
        draw_graphs : bool
            Draw graphs during computation.
        counts : bool
            Calculate voxel counts per component.
        slab : bool
            Process as a slab (disable PBC treatment).
        """
        # Store input parameters
        self._grid = grid
        self._verbose = verbose
        self._write_structures = write_structures
        self._draw_graphs = draw_graphs
        self._slab = slab
        
        # CRITICAL: Set self-reference FIRST so properties work during init
        self._base = self
        
        # Calculate containment
        (self._containment_graph, 
         self._component_contact_graph, 
         self._components_grid, 
         self._component_ranks,
         self._nonp_label_contact_graph) = self._calc_containment()
        
        # Initialize inverted graph flag
        self._inv_containment_graph = False
        
        # Compute and cache voxel counts if required
        if counts:
            self._voxel_counts = self._compute_counts(self._components_grid)
        else:
            self._voxel_counts = False
    
    def _calc_containment(self):
        """Calculate the containment graph."""
        return calc_containment_graph(
            self._grid, verbose=self._verbose, write_structures=self._write_structures, 
            draw_graphs=self._draw_graphs, slab=self._slab)
    
    def _compute_counts(self, grid):
        """Compute voxel counts for each component."""
        return dict(zip(*np.unique(grid, return_counts=True)))
    
    # Override properties to return owned data or cached values
    
    @property
    def voxel_counts(self):
        """Return the voxel counts computed during construction."""
        return self._voxel_counts


class VoxelContainmentView(VoxelContainmentBase):
    """
    A view on a VoxelContainment that merges specified nodes with their parents.
    Nodes not in keep_nodes are merged upstream, maintaining the DAG structure.
    
    This view provides the same API as VoxelContainment but with lazy remapping
    of nodes. Memory overhead is minimal as the underlying components_grid is
    shared with the base containment.
    
    Parameters
    ----------
    base_containment : VoxelContainment
        The base containment object to create a view on.
    keep_nodes : list or set
        Nodes to keep visible in the view. Other nodes are merged upstream
        to their nearest kept ancestor. If a removed node has no kept ancestor,
        it is dropped entirely.
    
    Examples
    --------
    >>> # Original: A -> B -> C, where B is small
    >>> containment = VoxelContainment(grid)
    >>> # Create view without B (merges B into A)
    >>> view = containment.node_view([A, C])
    >>> # Now A contains all of B's voxels
    >>> view.containment_graph.edges()  # [(A, C)]
    """
    
    def __init__(self, base_containment, keep_nodes):
        # Store reference to base (handles nested views automatically)
        self._base = base_containment._base
        self._keep_nodes = set(keep_nodes)
        
        # Build the remapping once at construction
        self._node_map = self._build_node_map()
        self._reverse_node_map = self._build_reverse_node_map()
        # For now we rebuild the graphs, I think this is often cheap enough to make this viable
        self._component_contact_graph = self._build_view_graph(self._base.component_contact_graph)
        self._nonp_label_contact_graph = self._build_view_graph(self._base.nonp_label_contact_graph)
        self._containment_graph = self._build_view_graph(
            self._base.containment_graph, 
            directed=True, 
            reduce_transitive=True,
            )
        
        # Initialize inverted graph flag
        self._inv_containment_graph = None
    
    def _resolve_nodes_to_original(self, nodes):
        """
        Expand view nodes to all original nodes they represent.
        This is the key method that makes views work transparently.
        """
        original_nodes = []
        for view_node in nodes:
            original_nodes.extend(self._reverse_node_map.get(view_node, []))
        return sorted(set(original_nodes))
    
    def _build_node_map(self):
        """
        Creates mapping: original_node -> view_node
        Nodes not in keep_nodes are mapped to their nearest kept ancestor.
        If no kept ancestor exists, the node is dropped (maps to None).
        """
        node_map = {}
        
        # First pass: map kept nodes to themselves
        for node in self._keep_nodes:
            node_map[node] = node
        
        # Second pass: map removed nodes to nearest kept ancestor
        try:
            topo_order = list(nx.topological_sort(self._base.containment_graph))
        except:
            topo_order = self._base.nodes
        
        for node in topo_order:
            if node in self._keep_nodes:
                continue
            
            parents = self._base.get_parent_nodes([node])
            
            if not parents:
                node_map[node] = None
            else:
                mapped_ancestor = None
                for parent in parents:
                    if parent in self._keep_nodes:
                        mapped_ancestor = parent
                        break
                    elif parent in node_map:
                        mapped_ancestor = node_map[parent]
                        if mapped_ancestor:
                            break
                
                node_map[node] = mapped_ancestor
        
        return node_map
    
    def _build_reverse_node_map(self):
        """
        Creates reverse mapping: view_node -> [original_nodes]
        This tells us which original nodes are merged into each view node.
        """
        reverse_map = {node: [] for node in self._keep_nodes}
        
        for orig_node, view_node in self._node_map.items():
            if view_node is not None:
                reverse_map[view_node].append(orig_node)
        
        return reverse_map
    
    def _build_view_graph(self, source_graph, directed=False, reduce_transitive=False):
        """
        Build a view graph with remapped node IDs from a source graph.
        
        Parameters
        ----------
        source_graph : networkx.Graph or networkx.DiGraph
            The original graph to build the view from.
        directed : bool, default=False
            Whether to create a directed graph (DiGraph) or undirected graph (Graph).
        reduce_transitive : bool, default=False
            Whether to apply transitive reduction (only for directed graphs).
        
        Returns
        -------
        networkx.Graph or networkx.DiGraph
            View graph with remapped node IDs.
        """
        view_graph = nx.DiGraph() if directed else nx.Graph()
        view_graph.add_nodes_from(self._keep_nodes)
        
        # Iterate over all edges in the source graph
        for u, v in source_graph.edges():
            mapped_u = self._node_map.get(u)
            mapped_v = self._node_map.get(v)
            # Only add edge if both nodes are kept and they're different
            if mapped_u and mapped_v and mapped_u != mapped_v:
                view_graph.add_edge(mapped_u, mapped_v)
        
        # Apply transitive reduction if requested (only for directed graphs)
        if reduce_transitive and directed:
            view_graph = nx.transitive_reduction(view_graph)
        
        return view_graph
    
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
        return self._reverse_node_map.get(view_node, [])
    
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
        return self._node_map.get(original_node)