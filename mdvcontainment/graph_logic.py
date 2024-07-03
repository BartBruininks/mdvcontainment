import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from .rank_logic import make_relabel_dicts

def draw_graph(graph):
    nx.draw(graph, pos=nx.kamada_kawai_layout(graph))
    nx.draw_networkx_labels(graph, nx.kamada_kawai_layout(graph))
    plt.show()

def create_contact_graph(contacts, nodes, bridges=False):
    """
    Returns the contact graph using the contacts and bridges.
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

    Parameters:
    graph (networkx.Graph): The graph containing the nodes.
    query_node (any hashable type): The node into which target nodes should be collapsed.
    target_nodes (list of any hashable type): The nodes to be collapsed into the query node.

    Returns:
    networkx.Graph: The graph after collapsing the nodes.
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
    Returns to lists for the positive and negrative subgraphs in the contact graph.
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
    Returns the component - label maps.

    A label is a non periodic connected compoponent.
    A component is a (potentially) periodic connected component. Therefore 
        a component can exist out of multiple labels.

    components2labels, labels2components, label2component
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
    """
    # Creating a component level contact graph.
    component_contact_graph = nx.Graph()
    for component, labels in component2labels.items():
        # Add the component node
        component_contact_graph.add_node(component)
        # Find all the component neighbours
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
    """
    # Use the is_contained status to propagate containment. Start with finding all
    #  is_not_contained nodes.
    is_not_contained = set([component for component, value in is_contained_dict.items() if value == False])
    is_contained = set([component for component, value in is_contained_dict.items() if value == True])

    # Instantiate all nodes for the containment_graph
    containment_graph = nx.MultiDiGraph()
    for node in unique_components:
        containment_graph.add_node(node)

    counter = 0 # for checing if recursion is not getting out of hand.
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
