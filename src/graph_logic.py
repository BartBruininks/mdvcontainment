"""
Graph handling functions for creating and formatting graphs.
"""

# Python External
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Python Module
from .rank_logic import make_relabel_dicts, get_ranks, get_is_contained
from .label_logic import label_3d_grid, create_components_grid

# Cython Module
from .find_label_contacts import find_label_contacts
from .find_bridges import find_bridges 


def draw_graph(graph):
    """Draws the provided graph using a Kamada-Kawai layout."""
    nx.draw(graph, pos=nx.kamada_kawai_layout(graph))
    nx.draw_networkx_labels(graph, nx.kamada_kawai_layout(graph))
    plt.show()

def create_contact_graph(contacts, nodes, bridges=False):
    """
    Returns the contact graph using the contacts and bridges.

    Parameters
    ----------
    contacts: list of tuples
        The contact edges between nodes.
    nodes: list of int
        The nodes in the graph.
    bridges: list of tuples, optional
        The bridge edges between nodes with their values.

    Returns
    -------
    graph: networkx.MultiDiGraph
        The created contact graph.
    """
    graph = nx.MultiDiGraph()
    for node in nodes:
        if node == 0:
            continue
        graph.add_node(int(node))

    value = np.array([0,0,0])
    for contact in contacts:
        if 0 in contact:
            continue
        graph.add_edge(int(contact[0]), int(contact[1]), label=str(value), cost=value)
        graph.add_edge(int(contact[1]), int(contact[0]), label=str(value), cost=value)
    
    if bridges is not False:
        for bridge in bridges:
            if 0 in bridge[:2]:
                continue
            value = bridge[2:]
            graph.add_edge(int(bridge[0]), int(bridge[1]), label=str(value), cost=value)
            graph.add_edge(int(bridge[1]), int(bridge[0]), label=str(-value), cost=-value)
    return graph

def collapse_nodes(graph, query_node, target_nodes):
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
    nodes_to_collapse = set(target_nodes)

    # Initialize a set to keep track of visited nodes
    visited = set()

    # Recursive function to find all nodes connected to the query node via target nodes
    def find_connected_nodes(node):
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

def get_subgraphs(nonp_unique_labels, contact_graph):
    """
    Returns to lists for the positive and negative subgraphs in the contact graph.

    Parameters
    ----------
    nonp_unique_labels: array-like
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
    positive_contact_graph = contact_graph.subgraph(positive_labels)
    negative_contact_graph = contact_graph.subgraph(negative_labels)
    # Create the connected_components subgraphs (components)
    positive_subgraphs = [positive_contact_graph.subgraph(c) for c in nx.connected_components(nx.Graph(positive_contact_graph))]
    negative_subgraphs = [negative_contact_graph.subgraph(c) for c in nx.connected_components(nx.Graph(negative_contact_graph))]
    return positive_subgraphs, negative_subgraphs

def get_mapping_dicts(positive_subgraphs, negative_subgraphs):
    """
    Returns the component -> label maps.

    A label is a non periodic connected component.
    A component is a (potentially) periodic connected component. Therefore 
    a component can consists out of multiple labels.

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
    component2labels, labels2component  = make_relabel_dicts(positive_subgraphs, negative_subgraphs)

    # Create the single label -> component map
    label2component = {}
    for component, labels in component2labels.items():
        for label in labels:
            label2component[label] = component
    return component2labels, labels2component, label2component

def create_component_contact_graph(component2labels, label2component, contact_graph):
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
    component_contact_graph = nx.Graph()
    for component, labels in component2labels.items():
        # Add the component node
        component_contact_graph.add_node(component)
        # Find all the component neighbors
        contacts = set()
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

def create_containment_graph(is_contained_dict, unique_components, component_contact_graph):
    """
    Returns the directed containment graph. A parent points to its children.

    Parameters
    ----------
    is_contained_dict: dict
        The mapping from component id to its is_contained status (True/False).
    unique_components: list of int
        The list of unique component ids.
    component_contact_graph: networkx.Graph
        The undirected component level contact graph.

    Returns
    -------
    containment_graph: networkx.MultiDiGraph
        The created directed containment graph.
    """
    # Use the is_contained status to propagate containment. Start with finding all
    #  is_not_contained nodes.
    is_not_contained = set([component for component, value in is_contained_dict.items() if value == False])
    is_contained = set([component for component, value in is_contained_dict.items() if value == True])

    # Instantiate all nodes for the containment_graph
    containment_graph = nx.MultiDiGraph()
    for node in unique_components:
        containment_graph.add_node(node)

    counter = 0 # for checking if recursion is not getting out of hand.
    stack = is_not_contained.copy() # Start with non-contained nodes.
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

def format_dag(G, node, ranks, counts, prefix='', is_last=True):
    """
    Recursively format a DAG node and its children as a string.
    
    Args:
        G: NetworkX DAG
        node: Current node to format
        ranks: Dictionary of ranks per node
        counts: Dictionary of counts per node, or False to omit counts
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

def format_dag_structure(G, ranks, counts, unit='nvoxels'):
    """
    Format the entire DAG structure as a string.
    
    Args:
        G: NetworkX graph
        ranks: Dictionary of ranks per node
        counts: Dictionary of counts per node, or False to omit counts
        unit: Unit string for counts display
    
    Returns:
        String representation of the entire DAG
    """
    result = f'Containment Graph with {len(G.nodes())} components (component: {unit}: rank):\n'
    
    roots = sorted([node for node, in_degree in G.in_degree() if in_degree == 0])
    
    for i, root in enumerate(roots):
        result += format_dag(G, root, ranks, counts, '', i == len(roots) - 1)
    return result

def calc_containment_graph(boolean_grid, verbose=False):
    """
    Creates the containment graph nx.MultiDiGraph(), taking right angled PBC into account.
    
    Parameters
    ----------
    boolean_grid: bool 3D array
        Boolean 3D array of voxel occupancy.
    verbose: bool
        Whether to progress information.

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
    # Finding all unique label ids in the labeled grid.
    if verbose:
        print('Calculating non-periodic labels...')
    nonp_labeled_grid = label_3d_grid(boolean_grid)
    nonp_unique_labels = np.unique(nonp_labeled_grid)

    # Find all non periodic label contacts
    if verbose:
        print('Calculating non-periodic label contacts...')
    nonp_contacts = find_label_contacts(nonp_labeled_grid)

    # Find all bridges (contacts between labels over PBC).
    if verbose:
        print('Calculating periodic label contacts...')
    bridges = find_bridges(nonp_labeled_grid)

    # Generate the label contact graph with bridge annotation.
    if verbose:
        print('Creating label contact graph...')
    contact_graph = create_contact_graph(nonp_contacts, nonp_unique_labels, bridges)

    # Get all the mappings between labels and components plus their subgraphs.
    if verbose:
        print('Creating label subgraphs...')
    positive_subgraphs, negative_subgraphs = get_subgraphs(nonp_unique_labels, contact_graph)
    component2labels, labels2component, label2component = get_mapping_dicts(positive_subgraphs, negative_subgraphs)

    # Relabel the labeled_grid to take pbc into account.
    if verbose:
        print('Calculating component grid...')
    components_grid = create_components_grid(nonp_labeled_grid, component2labels)
    
    # Calculate the ranks for the components and complements.
    if verbose:
        print('Calculating component and complement ranks...')
    component_ranks, complement_ranks = get_ranks(positive_subgraphs, negative_subgraphs, nonp_unique_labels, component2labels, labels2component, contact_graph)
    
    # Determine weather a component is contained or not, based on the fact
    #  that a component is contained if its complement is of higher rank.
    if verbose:
        print('Calculating is contained...')
    is_contained_dict = get_is_contained(component_ranks, complement_ranks)
        
    # Create the component level contact graph.
    if verbose:
        print('Creating component contact graph...')
    component_contact_graph = create_component_contact_graph(
        component2labels, label2component, contact_graph)
    
    # Finally create the containment graph by directing the edges in the 
    #  component contact graph (also breaking edges if they do not represent,
    #  a containment hierarchy).
    if verbose:
        print('Creating containment_graph...')
    unique_components = np.unique(components_grid)
    containment_graph = create_containment_graph(is_contained_dict, unique_components, component_contact_graph)

    if verbose:
        print('Done!')
    return containment_graph, component_contact_graph, components_grid, component_ranks
