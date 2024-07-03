import networkx as nx
import numpy as np


def get_path_matrix(graph, key='value'):
    """
    Returns an array with the edge and edge weight in order of the eulerian path.
    """
    eulerian = list(nx.eulerian_circuit(graph, keys='value'))
    path = []
    for step in eulerian:
        source = step[0] # start node
        sink = step[1] # stop node
        edge = step[2]
        weight = graph.get_edge_data(source, sink, edge)[key] # edge weight
        path.append((source, sink, *weight))
    return np.array(path)


def get_path_node_array(path_matrix):
    """
    Returns a list of the order of nodes in the path matrix.
    """
    node_list = []
    first = True
    for edge in path_matrix:
        if first:
            node_list.append(edge[0])
            first = False
        node_list.append(edge[1])
    return np.array(node_list)


def get_path_edge_array(path_matrix):
    """
    Returns a list of the order of nodes in the path matrix.
    """
    edge_list = []
    for edge in path_matrix:
        edge_list.append(edge[2:])
    return np.vstack(edge_list)


def get_cycles(path_matrix):
    """
    Returns the node_cycles and edge_cycles as two lists of tuples.
    """
    path_node_array = get_path_node_array(path_matrix)
    # Handle the case where there is no path.
    if len(path_node_array) == 0:
        return [], []
    path_edge_array = get_path_edge_array(path_matrix)
    visited = dict() # the node id and position in the path.
    node_cycles = []
    edge_cycles = []
    for idx, node in enumerate(path_node_array):
        # Check if the node was visited before.
        try:
            last_index = visited[node]
        except KeyError:
            # If the node was not visite before add it and continue.
            visited[node] = idx
            continue
        # Backtrack the path in the cycle if the node was visited before.
        node_cycle = path_node_array[last_index:idx+1]
        node_cycles.append(node_cycle)
        # Keep in mind that the edges are of len(nodes) - 1.
        edge_cycle = path_edge_array[last_index:idx]
        edge_cycles.append(edge_cycle)
        # Update the last visited position of the node.
        visited[node] = idx
    return node_cycles, edge_cycles


def get_edge_cycle_matrix(edge_cycles):
    """
    Returns the matrix of the total cycle vectors.
    """
    cycle_vectors = []
    for cycle in edge_cycles:
        cycle_vector = np.sum(cycle, 0)
        cycle_vectors.append(cycle_vector)
    return np.array(cycle_vectors)


def get_rank_from_edge_cycle_matrix(edge_cycle_matrix):
    """
    Returns the rank of the object described by the cycles.
    """
    # Handle the case where there are no cycles and the rank is 
    #  0 for sure.
    if len(edge_cycle_matrix) == 0:
        return 0
    singular_values = np.linalg.svd(edge_cycle_matrix)[1]
    rank = 0
    for singular_value in singular_values:
        # This is a tolerance for rounding errors which cause
        #  0 not to be exactly zero.
        if singular_value > 0.0001:
            rank += 1
    return rank


def get_rank(graph, key='cost'):
    """
    Returns the rank of the object graph.
    """
    # Determine the eulerian path.
    path_matrix = get_path_matrix(graph, key)
    # Find the cycles.
    node_cycles, edge_cycles = get_cycles(path_matrix)
    # Find the total displacement of the edge cycles.
    edge_cycle_matrix = get_edge_cycle_matrix(edge_cycles)
    # Obtain the rank of the object.
    rank = get_rank_from_edge_cycle_matrix(edge_cycle_matrix)
    return rank

def make_relabel_dicts(positive_subgraphs, negative_subgraphs):
    relabel_dict_c2l = {}
    relabel_dict_l2c = {}
    
    # Positive components
    positive_components = [tuple(sorted(x.nodes)) for x in positive_subgraphs]
    positive_ranks = [get_rank(x) for x in positive_subgraphs]
    positive_component_ids = np.arange(1,len(positive_components)+1)

    for component, labels in zip(positive_component_ids, positive_components):
        relabel_dict_c2l[component] = labels
        relabel_dict_l2c[labels] = component
    
    # Negative components
    negative_components = [tuple(sorted(x.nodes)) for x in negative_subgraphs]
    negative_ranks = [get_rank(x) for x in negative_subgraphs]
    negative_component_ids = np.arange(-1,-len(negative_components)-1, -1)

    for component, labels in zip(negative_component_ids, negative_components):
        relabel_dict_c2l[component] = labels
        relabel_dict_l2c[labels] = component
        
    return relabel_dict_c2l, relabel_dict_l2c


#@profile
def get_ranks(positive_subgraphs, negative_subgraphs, nonp_unique_labels, component2labels, labels2component, contact_graph):
    """
    Returns the ranks for the components and their complements as two dicts.
    """
    positive_ranks = {labels2component [tuple(sorted(x.nodes))]: get_rank(x) for x in positive_subgraphs}
    negative_ranks = {labels2component [tuple(sorted(x.nodes))]: get_rank(x) for x in negative_subgraphs}
    component_ranks = {}
    component_ranks.update(positive_ranks)
    component_ranks.update(negative_ranks)
    
    # Obtaining the ranks for the complements
    complements = {}
    unique_labels = set(nonp_unique_labels)
    for component, nodes in component2labels.items():
        # Skip the rank 0 components, for they can be auto assigned to have a complement of 3.
        if component_ranks[component] < 2:
            continue
        component_labels = set(nodes)
        complement_labels = unique_labels - component_labels
        complement_graph = contact_graph.subgraph(complement_labels)
        complements[component] = [complement_graph.subgraph(c) for c in nx.connected_components(nx.Graph(complement_graph))]
    
    # Effective complement ranks
    complement_ranks = {}
    for component in component_ranks.keys():
        # Auto assign the rank 0 components for efficiency.
        if component not in complements.keys():
            tmp_component_rank = component_ranks[component]
            if tmp_component_rank == 0:
                complement_ranks[component] = 3
            # This is a drity trick, it is not really 2, it could be
            #  3 as well, but we can be sure that the complement 
            #  rank will be bigger than 1 and thus this will lead
            #  to the contained status, this could be made more 
            #  truthful... We do return the complement ranks anyway.
            elif tmp_component_rank == 1:
                complement_ranks[component] = 2
            continue
        graphs = complements[component]
        complement_ranks[component] = 0
        for graph in graphs:
            rank = get_rank(graph)
            if rank > complement_ranks[component]:
                complement_ranks[component] = rank
    
    return component_ranks, complement_ranks


def get_is_contained(component_ranks, complement_ranks):
    """
    Returns a dict of the components with a True or False value for
    each component, indicating if it is contained or not.
    """
    # Use the containment logic to establish is a component is contained
    #  This is the case when the rank of the complement of the component is higher
    #  then the rank of the component.
    is_contained_dict = {}
    for component in component_ranks.keys():
        is_contained = False
        if component_ranks[component] < complement_ranks[component]:
            is_contained = True
        is_contained_dict[component] = is_contained
    return is_contained_dict
