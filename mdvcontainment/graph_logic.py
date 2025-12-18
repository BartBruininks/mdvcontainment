"""
Graph handling functions for creating and formatting graphs.
"""
# Python 
from typing import cast

# Python External
from typing import List, Tuple, Dict, Set, Optional
from networkx import Graph, DiGraph, MultiDiGraph
import numpy as np
import numpy.typing as npt

# Python Module
from .rank_logic import make_relabel_dicts, get_ranks, get_is_contained
from .label_logic import label_3d_grid, create_components_grid
from .wrappers import connected_components

# Cython Module
from .wrappers import find_label_contacts, find_bridges


def create_contact_graph(
    contacts: Optional[npt.NDArray[np.int32]], 
    nodes: npt.NDArray[np.int32],
    bridges: Optional[npt.NDArray[np.int32]] = None
) -> MultiDiGraph:
    """
    Returns the contact graph using the contacts and bridges.

    Parameters
    ----------
    contacts: list of tuples
        The contact edges between nodes.
    nodes: ndarray of int
        The nodes in the graph.
    bridges: list of tuples, optional
        The bridge edges between nodes with their values.
        Each tuple contains (node1, node2, x, y, z) where x, y, z are bridge values.
        Default is None.

    Returns
    -------
    graph: networkx.MultiDiGraph
        The created contact graph.
    """
    graph: MultiDiGraph = MultiDiGraph()
    for node in nodes:
        if node == 0:
            continue
        graph.add_node(int(node))

    if contacts is not None:
        value = np.array([0, 0, 0])
        for contact in contacts:
            if 0 in contact:
                continue
            graph.add_edge(int(contact[0]), int(contact[1]), label=str(value), cost=value)
            graph.add_edge(int(contact[1]), int(contact[0]), label=str(value), cost=value)
    
    if bridges is not None:
        for bridge in bridges:
            if 0 in bridge[:2]:
                continue
            value = np.array(bridge[2:])
            graph.add_edge(int(bridge[0]), int(bridge[1]), label=str(value), cost=value)
            graph.add_edge(int(bridge[1]), int(bridge[0]), label=str(-value), cost=-value)
   
    return graph


def collapse_nodes(
    graph: MultiDiGraph, 
    query_node: int, 
    target_nodes: List[int]
) -> MultiDiGraph:
    """
    Collapse target nodes into the query node recursively.

    Parameters
    ----------
    graph: networkx.MultiDiGraph
        The graph containing the nodes to be collapsed.
    query_node: int
        The node into which the target nodes will be collapsed.
    target_nodes: list of int
        The nodes to be collapsed into the query node.

    Returns
    -------
    graph: networkx.MultiDiGraph
        The modified graph with the target nodes collapsed into the query node.
    """
    # Initialize the set of nodes to be collapsed
    nodes_to_collapse: Set[int] = set(target_nodes)

    # Initialize a set to keep track of visited nodes
    visited: Set[int] = set()

    # Recursive function to find all nodes connected to the query node via target nodes
    def find_connected_nodes(node: int) -> None:
        if node in visited:
            return
        visited.add(node)
        for neighbor in graph.neighbors(node):
            if neighbor in target_nodes and neighbor not in visited:
                nodes_to_collapse.add(neighbor)
                find_connected_nodes(neighbor)

    # Start the recursion from the query node
    find_connected_nodes(query_node)

    # Collapse all identified nodes into the query node
    for node in nodes_to_collapse:
        if node != query_node:
            # Move all edges from the node to the query node
            for neighbor in list(graph.neighbors(node)):
                graph.add_edge(query_node, neighbor)
                graph.remove_edge(node, neighbor)
            # Remove the node from the graph
            graph.remove_node(node)
  
    return graph


def get_subgraphs(
    nonp_unique_labels: npt.NDArray[np.int32], 
    contact_graph: MultiDiGraph
) -> Tuple[List[MultiDiGraph], List[MultiDiGraph]]:
    """
    Returns two lists for the positive and negative subgraphs in the contact graph.

    Parameters
    ----------
    nonp_unique_labels: ndarray of int
        The unique non-periodic labels in the contact graph.
    contact_graph: networkx.MultiDiGraph
        The contact graph containing the labels.
    
    Returns
    -------
    positive_subgraphs: list of networkx.MultiDiGraph
        The list of positive contact subgraphs.
    negative_subgraphs: list of networkx.MultiDiGraph
        The list of negative contact subgraphs.
    """
    # Obtain the positive and negative label ids.
    positive_labels = nonp_unique_labels[nonp_unique_labels > 0]
    negative_labels = nonp_unique_labels[nonp_unique_labels < 0]
    # Create the positive and negative contacts subgraphs.
    positive_contact_graph = cast(MultiDiGraph, contact_graph.subgraph(positive_labels))
    negative_contact_graph = cast(MultiDiGraph, contact_graph.subgraph(negative_labels))
    # Create the connected_components subgraphs (components)
    positive_subgraphs_raw = [positive_contact_graph.subgraph(c) for c in connected_components(Graph(positive_contact_graph))] 
    negative_subgraphs_raw = [negative_contact_graph.subgraph(c) for c in connected_components(Graph(negative_contact_graph))] 
    
    # Cast to MultiDiGraph for type compatibility
    positive_subgraphs = [cast(MultiDiGraph, sg) for sg in positive_subgraphs_raw]
    negative_subgraphs = [cast(MultiDiGraph, sg) for sg in negative_subgraphs_raw]

    return positive_subgraphs, negative_subgraphs


def get_mapping_dicts(
    positive_subgraphs: List[MultiDiGraph], 
    negative_subgraphs: List[MultiDiGraph]
) -> Tuple[Dict[int, List[int]], Dict[Tuple[int, ...], int], Dict[int, int]]:
    """
    Returns the component -> label maps.

    A label is a non periodic connected component.
    A component is a (potentially) periodic connected component. Therefore 
    a component can consist of multiple labels.

    Parameters
    ----------
    positive_subgraphs: list of networkx.MultiDiGraph
        The list of positive contact subgraphs.
    negative_subgraphs: list of networkx.MultiDiGraph
        The list of negative contact subgraphs.

    Returns
    -------
    component2labels: dict
        The mapping from component id to list of labels.
    labels2component: dict
        The mapping from sorted labels tuple to component id.
    label2component: dict
        The mapping from single label to component id.
    """
    # Create the mapping from labels to connected components (a cc can consist of multiple labels)
    component2labels_raw, labels2component = make_relabel_dicts(positive_subgraphs, negative_subgraphs)
    
    # Convert tuples to lists for component2labels
    component2labels: Dict[int, List[int]] = {k: list(v) for k, v in component2labels_raw.items()}

    # Create the single label -> component map
    label2component: Dict[int, int] = {}
    for component, labels in component2labels.items():
        for label in labels:
            label2component[label] = component
 
    return component2labels, labels2component, label2component


def create_component_contact_graph(
    component2labels: Dict[int, List[int]], 
    label2component: Dict[int, int], 
    contact_graph: MultiDiGraph
) -> Graph:
    """
    Returns the undirected component level contact graph.

    Parameters
    ----------
    component2labels: dict
        The mapping from component id to list of labels.
    label2component: dict
        The mapping from single label to component id.
    contact_graph: networkx.MultiDiGraph
        The contact graph containing the labels.

    Returns
    -------
    component_contact_graph: networkx.Graph
        The created component level contact graph.
    """
    # Creating a component level contact graph.
    component_contact_graph: Graph = Graph()
    for component, labels in component2labels.items():
        # Add the component node
        component_contact_graph.add_node(component)
        # Find all the component neighbors
        contacts: Set[int] = set()
        for label in labels:
            neighbor_list = list(contact_graph.neighbors(label))
            for label_neighbor in neighbor_list:
                component_neighbor = label2component[label_neighbor]
                if component_neighbor != component:
                    contacts.add(component_neighbor)
        # Add the component-contact edges
        for contact in contacts:
            component_contact_graph.add_edge(component, contact)
 
    return component_contact_graph


def create_containment_graph(
    is_contained_dict: Dict[int, bool], 
    unique_components: npt.NDArray[np.int32], 
    component_contact_graph: Graph
) -> DiGraph:
    """
    Returns the directed containment graph. A parent points to its children.

    Parameters
    ----------
    is_contained_dict: dict
        The mapping from component id to its is_contained status (True/False).
    unique_components: ndarray of int
        The array of unique component ids.
    component_contact_graph: networkx.Graph
        The undirected component level contact graph.

    Returns
    -------
    containment_graph: networkx.DiGraph
        The created directed containment graph.
    """
    # Use the is_contained status to propagate containment. Start with finding all
    #  is_not_contained nodes.
    is_not_contained: Set[int] = set([component for component, value in is_contained_dict.items() if value == False])
    is_contained: Set[int] = set([component for component, value in is_contained_dict.items() if value == True])

    # Instantiate all nodes for the containment_graph
    containment_graph: DiGraph = DiGraph()
    for node in unique_components:
        containment_graph.add_node(int(node))

    counter = 0  # for checking if recursion is not getting out of hand.
    stack: Set[int] = is_not_contained.copy()  # Start with non-contained nodes.
    while len(stack):
        assert counter < 100000, 'During orientation of the containment graph we got in an iterative death (max iterations 100,000).'
        counter += 1
        query_node = stack.pop()
        neighbors = component_contact_graph.neighbors(query_node)
        for neighbor in neighbors:
            if neighbor not in is_not_contained:
                containment_graph.add_edge(query_node, neighbor)
                is_contained.remove(neighbor)
                is_not_contained.add(neighbor)
                stack.add(neighbor)
 
    return containment_graph


def format_dag(
    G: DiGraph, 
    node: int, 
    ranks: Dict[int, int], 
    counts: Dict[int, int|float],
    prefix: str = '', 
    is_last: bool = True
) -> str:
    """
    Recursively format a DAG node and its children as a string.
    
    Args:
        G: NetworkX DiGraph
        node: Current node to format
        ranks: Dictionary of ranks per node
        counts: Dictionary of counts per node
        prefix: String prefix for current line (for tree structure)
        is_last: Whether this node is the last child of its parent
    
    Returns:
        String representation of the node and its subtree
    """
    connector = '└── ' if is_last else '├── '
    
    result = f"{prefix}{connector}[{node}: {int(counts[node])}: {ranks[node]}]\n"
    
    children = sorted(list(G.successors(node)))
    for i, child in enumerate(children):
        new_prefix = prefix + ('    ' if is_last else '│   ')
        result += format_dag(G, child, ranks, counts, new_prefix, i == len(children) - 1)

    return result


def format_dag_structure(
    G: DiGraph, 
    ranks: Dict[int, int], 
    counts: Dict[int, int|float],
    unit: str = 'nvoxels'
) -> str:
    """
    Format the entire DAG structure as a string.
    
    Args:
        G: NetworkX DiGraph
        ranks: Dictionary of ranks per node
        counts: Dictionary of counts per node
        unit: Unit string for counts display
    
    Returns:
        String representation of the entire DAG
    """
    result = f'Containment Graph with {len(G.nodes())} components (component: {unit}: rank):\n'
    
    roots = sorted([node for node, in_degree in G.in_degree() if in_degree == 0])
    
    for i, root in enumerate(roots):
        result += format_dag(G, root, ranks, counts, '', i == len(roots) - 1)
 
    return result


def calc_containment_graph(
    boolean_grid: npt.NDArray[np.bool_], 
    verbose: bool = False
) -> Tuple[DiGraph, Graph, npt.NDArray[np.int32], Dict[int, int]]:
    """
    Creates the containment graph DiGraph(), taking right angled PBC into account.
    
    Parameters
    ----------
    boolean_grid: bool 3D array
        Boolean 3D array of voxel occupancy.
    verbose: bool
        Whether to print progress information.

    Returns
    -------
    containment_graph: DiGraph
        The containment graph with directed edges from parent to child.
    component_contact_graph: Graph
        The undirected component contact graph.
    components_grid: int32 3D array
        The components grid where each voxel has a component integer.
    component_ranks: dict
        Dict of component:int -> rank:int
    """
    # Finding all unique label ids in the labeled grid.
    if verbose:
        print('Calculating non-periodic labels...')
    nonp_labeled_grid: npt.NDArray[np.int32] = label_3d_grid(boolean_grid).astype(np.int32)
    nonp_unique_labels: npt.NDArray[np.int32] = np.unique(nonp_labeled_grid)

    # Find all non periodic label contacts
    if verbose:
        print('Calculating non-periodic label contacts...')
    nonp_contacts: Optional[npt.NDArray[np.int32]] = find_label_contacts(nonp_labeled_grid)

    # Find all bridges (contacts between labels over PBC).
    if verbose:
        print('Calculating periodic label contacts...')
    bridges: Optional[npt.NDArray[np.int32]] = find_bridges(nonp_labeled_grid)

    # Generate the label contact graph with bridge annotation.
    if verbose:
        print('Creating label contact graph...')
    contact_graph: MultiDiGraph = create_contact_graph(nonp_contacts, nonp_unique_labels, bridges)

    # Get all the mappings between labels and components plus their subgraphs.
    if verbose:
        print('Creating label subgraphs...')
    positive_subgraphs, negative_subgraphs = get_subgraphs(nonp_unique_labels, contact_graph)
    component2labels, labels2component, label2component = get_mapping_dicts(positive_subgraphs, negative_subgraphs)

    # Relabel the labeled_grid to take pbc into account.
    if verbose:
        print('Calculating component grid...')
    components_grid: npt.NDArray[np.int32] = create_components_grid(nonp_labeled_grid, component2labels)
    
    # Calculate the ranks for the components and complements.
    if verbose:
        print('Calculating component and complement ranks...')
    # Convert component2labels back to tuples for get_ranks
    component2labels_tuples: Dict[int, Tuple[int, ...]] = {k: tuple(v) for k, v in component2labels.items()}
    component_ranks, complement_ranks = get_ranks(
        positive_subgraphs, 
        negative_subgraphs, 
        list(nonp_unique_labels), 
        component2labels_tuples, 
        labels2component, 
        contact_graph
    )
    
    # Determine whether a component is contained or not, based on the fact
    #  that a component is contained if its complement is of higher rank.
    if verbose:
        print('Calculating is contained...')
    is_contained_dict: Dict[int, bool] = get_is_contained(component_ranks, complement_ranks)
        
    # Create the component level contact graph.
    if verbose:
        print('Creating component contact graph...')
    component_contact_graph: Graph = create_component_contact_graph(
        component2labels, label2component, contact_graph)
    
    # Finally create the containment graph by directing the edges in the 
    #  component contact graph (also breaking edges if they do not represent
    #  a containment hierarchy).
    if verbose:
        print('Creating containment_graph...')
    unique_components: npt.NDArray[np.int32] = np.unique(components_grid)
    containment_graph: DiGraph = create_containment_graph(is_contained_dict, unique_components, component_contact_graph)

    if verbose:
        print('Done!')
 
    return containment_graph, component_contact_graph, components_grid, component_ranks