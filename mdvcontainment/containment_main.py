# Python External 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Python
from .rank_logic import get_ranks, get_is_contained
from .graph_logic import draw_graph, create_contact_graph, get_mapping_dicts, get_subgraphs, create_component_contact_graph, create_containment_graph
from .voxels_to_gro import voxels_to_gro
from .label import label_3d_grid, create_components_grid

# Cython
from .find_label_contacts import find_label_contacts
from .find_bridges import find_bridges 

def nonp_is_contained(labeled_grid, nonp_unique_labels, write_structures=False):
    """
    Returns containment status for the slab case and the component ranks.
    
    Each label is set to be either contained or not, based on the following
    criteria:
        1) The relevant dimensions for the outside are picked dynamically
            based on the two largest axis in the shape of the
            input array. The thinnest dimension is assumed to be the slice
            thickness.
        2) Any label which has a voxel in the relevant hollow cylinder
            formed by the minimal and maximal positions of the two picked
            dimensions is regarded an absolute outside.    

    Parameters
    ----------
    labeled_grid: int32 3D array
        The labeled grid where each voxel has a label integer.
    nonp_unique_labels: array-like
        The unique labels in the labeled grid.
    write_structures: bool
        Whether to write a gro file of the mask.

    Returns
    -------
    is_contained_dict: dict
        Dict of label:int -> is_contained:bool
    component_ranks: dict
        Dict of label:int -> rank:int
    """
    # Find the relevant dimensions which define the bounding cylinder.
    shape = labeled_grid.shape
    relevant_dimensions = np.argsort(shape)[1:]
    
    # Create the edge mask to assign outside.
    mask = np.zeros(shape, dtype=bool)
    if 0 in relevant_dimensions:
        mask[ 0, :, :] = True
        mask[-1, :, :] = True
    if 1 in relevant_dimensions:
        mask[ :, 0, :] = True
        mask[ :,-1, :] = True
    if 2 in relevant_dimensions:
        mask[ :, :, 0] = True
        mask[ :, :,-1] = True
    
    # Write mask if required
    if write_structures:
        voxels_to_gro('mask.gro', mask)
    
    # Create the is contained dict
    component_ranks = {}    
    is_contained_dict = {}
    outsides = np.unique(labeled_grid[mask])
    for label in nonp_unique_labels:
        if label in outsides:
            component_ranks[label] = 1 
            is_contained_dict[label] = False
        else:
            component_ranks[label] = 0
            is_contained_dict[label] = True      
    return is_contained_dict, component_ranks

def calc_containment_graph(boolean_grid, verbose=False, write_structures=False, draw_graphs=False, slab=False):
    """
    Creates the containment graph nx.MultiDiGraph(), taking right angled PBC into account.
    
    Parameters
    ----------
    boolean_grid: bool 3D array
        Boolean 3D array of voxel occupancy.
    verbose: bool
        Whether to print developer information.
    write_structures: bool
        Whether to write gro files of the input, labeled and components grids.
    draw_graphs: bool
        Whether to draw the contact graphs.
    slab: bool
        Whether to treat the data as a slab (non periodic in one dimension).

    Returns
    -------
    containment_graph: nx.DiGraph
        The containment graph with directed edges from parent to child.
    component_contact_graph: nx.Graph
        The undirected component contact graph.
    components_grid: int32 3D array
        The components grid where each voxel has a component integer.
    component_ranks: dict
        Dict of component:int -> rank:int
    contact_graph: nx.Graph
        The undirected label contact graph. 
    """
    # Return the basic counts of Trues and Falses in the input grid if verbose.
    if verbose:
        counts_boolean_grid = dict(zip(*np.unique(boolean_grid, return_counts=True)))
        print(f'Value prevalence in boolean_grid: {counts_boolean_grid}')\

    # Finding all unique label ids in the labeled grid.
    nonp_labeled_grid = label_3d_grid(boolean_grid)
    nonp_unique_labels = np.unique(nonp_labeled_grid)
    if verbose:
        print(f'Unique labels in labeled_grid: {nonp_unique_labels}')

    # Write the input and labeled structures
    if write_structures:
        voxels_to_gro('input.gro', boolean_grid)
        voxels_to_gro('nonp_labels.gro', nonp_labeled_grid)

    # Find all non periodic label contacts
    nonp_contacts = find_label_contacts(nonp_labeled_grid)

    if not slab:
        # Find all bridges (contacts between labels over PBC).
        bridges = find_bridges(nonp_labeled_grid)

    # Generate the label contact graph with bridge annotation.
    if not slab:
        contact_graph = create_contact_graph(nonp_contacts, nonp_unique_labels, bridges)
    else:
        contact_graph = create_contact_graph(nonp_contacts, nonp_unique_labels)
    if draw_graphs:
        print('=== NON PERIODIC LABEL CONTACT GRAPH ===')
        draw_graph(contact_graph)

    if not slab:
        # Get all the mappings between labels and components plus their subgraphs
        positive_subgraphs, negative_subgraphs = get_subgraphs(nonp_unique_labels, contact_graph)
        component2labels, labels2component, label2component = get_mapping_dicts(positive_subgraphs, negative_subgraphs)
        if verbose:
            print(f'component2labels {component2labels}')
            print(f'labels2component {labels2component}')
            print(f'label2component {label2component}')

    # Relabel the labeled_grid to take pbc into account.
    if not slab:
        components_grid = create_components_grid(nonp_labeled_grid, component2labels)
    else:
        components_grid = nonp_labeled_grid
    # Write the components structure file
    if write_structures:
        voxels_to_gro('components.gro', components_grid)
    
    # Calculate the ranks for the components and complements.
    if not slab:
        component_ranks, complement_ranks = get_ranks(positive_subgraphs, negative_subgraphs, nonp_unique_labels, component2labels, labels2component, contact_graph)
        if verbose:
            print(f'All rank of components: {component_ranks}')
            print(f'Complement ranks: {complement_ranks}')
        
        # Determine weather a component is contained or not, based on the fact
        #  that a component is contained if its complement is of higher rank.
        is_contained_dict = get_is_contained(component_ranks, complement_ranks)
        if verbose:
            print(f'Containment status: {is_contained_dict}')
    else:
        is_contained_dict, component_ranks = nonp_is_contained(nonp_labeled_grid, nonp_unique_labels, write_structures)
        
    if not slab:
        # Create the component level contact graph. Using an iterative approach we 
        #  trickly down the containment arrow started by a non-contained component
        #  being in contact with a contained component, therefore containing it. 
        #  Since every contained component can only have on parent, this means it
        #  must be the parent of all its other contacts in the component contact
        #  graph. This iterates until all nodes have been processed.
        component_contact_graph = create_component_contact_graph(
            component2labels, label2component, contact_graph)
        # Draw the component contacts graph
        if draw_graphs:
            print('=== COMPONENT CONTACTS GRAPH ===')
            draw_graph(component_contact_graph)
    else:
        component_contact_graph = nx.Graph(contact_graph)
    
    # Finally create the containment graph by directing the edges in the 
    #  component contact graph (also breaking edges if they do not represent,
    #  a containment hierarchy).
    unique_components = np.unique(components_grid)
    containment_graph = create_containment_graph(is_contained_dict, unique_components, component_contact_graph)
    return containment_graph, component_contact_graph, components_grid, component_ranks, contact_graph

def format_dag(G, node, counts, prefix='', is_last=True):
    """
    Recursively format a DAG node and its children as a string.
    
    Args:
        G: NetworkX DAG
        node: Current node to format
        counts: Dictionary of counts per node, or False to omit counts
        prefix: String prefix for current line (for tree structure)
        is_last: Whether this node is the last child of its parent
    
    Returns:
        String representation of the node and its subtree
    """
    connector = '└── ' if is_last else '├── '
    
    if counts is not False:
        result = f"{prefix}{connector}[{node}: {counts[node]}]\n"
    else:
        result = f"{prefix}{connector}[{node}]\n"
    
    children = list(G.successors(node))
    for i, child in enumerate(children):
        new_prefix = prefix + ('    ' if is_last else '│   ')
        result += format_dag(G, child, counts, new_prefix, i == len(children) - 1)
    return result


def format_dag_structure(G, counts=False):
    """
    Format the entire DAG structure as a string.
    
    Args:
        G: NetworkX graph
        counts: Dictionary of counts per node, or False to omit counts
    
    Returns:
        String representation of the entire DAG
    """
    if counts is not False:
        result = f'Containment Graph with {len(G.nodes())} components (component: nvoxels):\n'
    else:
        result = f'Containment Graph with {len(G.nodes())} components:\n'
    
    roots = [node for node, in_degree in G.in_degree() if in_degree == 0]
    
    for i, root in enumerate(roots):
        result += format_dag(G, root, counts, '', i == len(roots) - 1)
    return result

class VoxelContainment():
    def __init__(self, grid, verbose=False, write_structures=False, draw_graphs=False, counts=True, slab=False):
        """
        A Containment graph is a DAG which has a parent pointing to its children with an edge.
        """
        # Read input
        self.grid = grid
        self._verbose = verbose
        self._write_structures = write_structures
        self._draw_graphs = draw_graphs
        self.slab = slab

        # Set all containment output
        self.containment_graph, self.component_contact_graph, self.components_grid, self.component_ranks, self.nonp_label_contact_graph = self._calc_containment()
        # Instantiate the inverted graph to False, if it is needed
        #  it is generated only once.
        self._inv_containment_graph = False

        # Set the voxel counts per label if required
        self.voxel_counts = counts
        if self.voxel_counts:
            self.voxel_counts = self.get_counts(self.components_grid)
        # Sets all nodes
        self.nodes = self._get_nodes()
        # Sets the root nodes
        self.root_nodes = self._get_roots()
        # Set leaf nodes
        self.leaf_nodes = self._get_leaves()

    def _calc_containment(self):
        return calc_containment_graph(
            self.grid, verbose=self._verbose, write_structures=self._write_structures, draw_graphs=self._draw_graphs, slab=self.slab)

    def __str__(self):
        return format_dag_structure(self.containment_graph, self.voxel_counts)

    def _get_nodes(self):
        """
        Returns all nodes (components) in the containment graph.
        """
        return list(self.containment_graph.nodes)

    def get_counts(self, grid=None):
        """
        Returns the number of elements in each component as a dict.

        Parameters
        ----------
        grid: int32 3D array, optional
            The components grid to calculate counts for. If None, uses
            self.components_grid. 
        
        Returns
        -------
        counts_dict: dict
            Dict of component:int -> voxel_count:int 
        """
        if grid is None:
            grid = self.components_grid
        return dict(zip(*np.unique(grid, return_counts=True)))

    def _get_roots(self):
        """
        Returns a list of root nodes (absolute outsides).
        """
        return list((node for node, in_degree in self.containment_graph.in_degree() if in_degree == 0))

    def _get_leaves(self):
        """
        Returns a list of leaf nodes (deepest level of containment).
        """
        return list((node for node, out_degree in self.containment_graph.out_degree() if out_degree == 0))

    def _get_inverted_containment_graph(self):
        """
        Returns the inverted containment graph, if it is not
        yet present it creates it and binds it to self._inv_containment_graph.
        """
        if self._inv_containment_graph:
            return self._inv_containment_graph
        else:
            self._inv_containment_graph = self.containment_graph.reverse()
            return self._inv_containment_graph

    def get_child_nodes(self, start_nodes):
        """
        Returns the children (neighbors) of the given nodes in the containment graph. 
        Taking directionality of the edges into account.
        """
        all_children = []
        for start_node in start_nodes:
            all_children += list(self.containment_graph.neighbors(start_node))
        return sorted(set(all_children))
    
    def get_downstream_nodes(self, start_nodes):
        """
        Returns all nodes contained by the specified nodes (list), including the nodes itself.
        """
        downstream_nodes = []
        for node in start_nodes:
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
            all_parents += list(reversed_G.neighbors(start_node))
        return sorted(set(all_parents))

    def get_total_voxel_count(self, nodes):
        """
        Returns the total amount of voxels in the list of nodes.
        """
        total_count = 0
        for node in nodes:
            total_count += self.voxel_counts[node]
        return total_count

    def get_voxel_mask(self, nodes):
        """
        Returns as boolean mask over the components grid where the voxel
        value is in the list of provided nodes.
        """
        return np.isin(self.components_grid, nodes)

    def get_voxel_positions(self, nodes):
        """
        Returns all the voxel indices as a Nx3 array which are part of the specified
        nodes.
        """
        # Create a boolean mask where True indicates the values match the specified values
        mask = self.get_voxel_mask(nodes)

        # Use np.where to find the indices where mask is True
        indices = np.where(mask)
        return np.array(indices).T

    def format_containment(self, nodes=False):
        """
        Returns the selected nodes as a containment graph.

        By default prints the complete graph.
        """
        if nodes is not False:
            string = format_dag_structure(self.containment_graph.subgraph(nodes), self.voxel_counts)
        else:
            string = format_dag_structure(self.containment_graph, self.voxel_counts)
        return string

    def draw(self, nodes=False):
        if nodes is False:
            nx.draw_networkx(self.containment_graph)
        else:
            nx.draw_networkx(self.containment_graph.subgraph(nodes))
        plt.show()
    
    
    
