"""
Graph rank analysis using Eulerian paths and cycle detection.
"""
import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Set

# Constants
SINGULAR_VALUE_TOLERANCE = 0.0001


def get_path_matrix(graph: nx.MultiGraph, key: str = 'value') -> np.ndarray:
    """
    Returns an array with the edge and edge weight in order of the Eulerian path.
    
    Parameters
    ----------
    graph
        A MultiGraph that must have an Eulerian circuit
    key, default='value'x
        The edge attribute key to extract weights from
        
    Returns
    -------
    Array where each row is (source, sink, weight_components...)
        
    Raises
    ------
    ValueError
        If the graph doesn't have an Eulerian circuit
    """
    if not nx.is_eulerian(graph):
        raise ValueError("Graph must be Eulerian to compute path matrix")
    
    eulerian = list(nx.eulerian_circuit(graph, keys=True))
    path = []
    
    for source, sink, edge in eulerian:
        weight = graph.get_edge_data(source, sink, edge)[key]
        # Handle both scalar and iterable weights
        if hasattr(weight, '__iter__') and not isinstance(weight, str):
            path.append((source, sink, *weight))
        else:
            path.append((source, sink, weight))
    return np.array(path)

def get_path_node_array(path_matrix: np.ndarray) -> np.ndarray:
    """
    Returns an array of nodes in the order they appear in the path.
    
    Parameters
    ----------
    path_matrix
        Output from get_path_matrix()
        
    Returns
    -------
    Array of node IDs in path order
    """
    if len(path_matrix) == 0:
        return np.array([])
    
    # First node from first edge, then all sink nodes
    first_node = [path_matrix[0, 0]]
    remaining_nodes = path_matrix[:, 1].tolist()
    return np.array(first_node + remaining_nodes)

def get_path_edge_array(path_matrix: np.ndarray) -> np.ndarray:
    """
    Returns an array of edge weights in the order they appear in the path.
    
    Parameters
    ----------
    path_matrix
        Output from get_path_matrix()
        
    Returns
    -------
    Array of edge weight vectors
    """
    if len(path_matrix) == 0:
        return np.array([])
    return path_matrix[:, 2:]

def get_cycles(path_matrix: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Detects cycles in the Eulerian path by finding repeated node visits.
    
    Parameters
    ----------
    path_matrix
        Output from get_path_matrix()
        
    Returns
    -------
    node_cycles
        List of arrays, each containing nodes in a detected cycle
    edge_cycles
        List of arrays, each containing edge weights in a detected cycle
    """
    path_node_array = get_path_node_array(path_matrix)
    
    if len(path_node_array) == 0:
        return [], []
    
    path_edge_array = get_path_edge_array(path_matrix)
    visited: Dict[int, int] = {}  # node_id -> last_index
    node_cycles = []
    edge_cycles = []
    
    for idx, node in enumerate(path_node_array):
        if node in visited:
            # Found a cycle - extract it
            last_index = visited[node]
            node_cycle = path_node_array[last_index:idx + 1]
            edge_cycle = path_edge_array[last_index:idx]
            
            node_cycles.append(node_cycle)
            edge_cycles.append(edge_cycle)
        
        visited[node] = idx 
    return node_cycles, edge_cycles

def get_edge_cycle_matrix(edge_cycles: List[np.ndarray]) -> np.ndarray:
    """
    Computes the total displacement vector for each cycle.
    
    Parameters
    ----------
    edge_cycles
        List of edge cycle arrays
        
    Returns
    -------
    Matrix where each row is the sum of edges in a cycle
    """
    if not edge_cycles:
        return np.array([])
    
    cycle_vectors = [np.sum(cycle, axis=0) for cycle in edge_cycles]
    return np.array(cycle_vectors)

def get_rank_from_edge_cycle_matrix(edge_cycle_matrix: np.ndarray) -> int:
    """
    Computes the rank using SVD on the cycle displacement matrix.
    
    Parameters
    ----------
    edge_cycle_matrix
        Matrix of cycle displacement vectors
        
    Returns
    -------
    The rank (number of non-zero singular values)
    """
    if len(edge_cycle_matrix) == 0:
        return 0
    
    singular_values = np.linalg.svd(edge_cycle_matrix, compute_uv=False)
    rank = np.sum(singular_values > SINGULAR_VALUE_TOLERANCE)
    return int(rank)

def get_rank(graph: nx.MultiGraph, key: str = 'cost') -> int:
    """
    Computes the topological rank of a graph object.
    
    Parameters
    ----------
    graph
        A MultiGraph with an Eulerian circuit
    key, default='cost'
        Edge attribute key for weights
        
    Returns
    -------
    The rank of the object represented by the graph
    """
    path_matrix = get_path_matrix(graph, key)
    node_cycles, edge_cycles = get_cycles(path_matrix)
    edge_cycle_matrix = get_edge_cycle_matrix(edge_cycles)
    rank = get_rank_from_edge_cycle_matrix(edge_cycle_matrix)
    return rank

def make_relabel_dicts(
    positive_subgraphs: List[nx.MultiGraph],
    negative_subgraphs: List[nx.MultiGraph]
) -> Tuple[Dict[int, Tuple], Dict[Tuple, int]]:
    """
    Creates bidirectional mapping between component IDs and node labels.
    
    Parameters
    ----------
    positive_subgraphs
        List of connected components with positive IDs
    negative_subgraphs
        List of connected components with negative IDs
        
    Returns
    -------
    component_to_labels
        Mapping from component ID to node labels tuple
    labels_to_component
        Mapping from node labels tuple to component ID
    """
    component_to_labels = {}
    labels_to_component = {}
    
    # Positive components (IDs: 1, 2, 3, ...)
    for idx, subgraph in enumerate(positive_subgraphs, start=1):
        labels = tuple(sorted(subgraph.nodes))
        component_to_labels[idx] = labels
        labels_to_component[labels] = idx
    
    # Negative components (IDs: -1, -2, -3, ...)
    for idx, subgraph in enumerate(negative_subgraphs, start=1):
        labels = tuple(sorted(subgraph.nodes))
        component_id = -idx
        component_to_labels[component_id] = labels
        labels_to_component[labels] = component_id
    return component_to_labels, labels_to_component

def _compute_complement_rank(
    component_labels: Set,
    all_labels: Set,
    contact_graph: nx.Graph
) -> int:
    """
    Computes the maximum rank among all connected components of the complement.
    
    Parameters
    ----------
    component_labels
        Set of node labels in the component
    all_labels
        Set of all node labels
    contact_graph
        The full contact graph
        
    Returns
    -------
    Maximum rank among complement's connected components
    """
    complement_labels = all_labels - component_labels
    complement_graph = contact_graph.subgraph(complement_labels)
    
    max_rank = 0
    for component in nx.connected_components(nx.Graph(complement_graph)):
        subgraph = complement_graph.subgraph(component)
        rank = get_rank(subgraph)
        max_rank = max(max_rank, rank)
    return max_rank

def get_ranks(
    positive_subgraphs: List[nx.MultiGraph],
    negative_subgraphs: List[nx.MultiGraph],
    nonp_unique_labels: List,
    component_to_labels: Dict[int, Tuple],
    labels_to_component: Dict[Tuple, int],
    contact_graph: nx.Graph
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Computes ranks for all components and their complements.
    
    Parameters
    ----------
    positive_subgraphs
        List of positive components
    negative_subgraphs
        List of negative components
    nonp_unique_labels
        All unique node labels
    component_to_labels
        Mapping from component ID to node labels
    labels_to_component
        Mapping from node labels to component ID
    contact_graph
        The full contact graph
        
    Returns
    -------
    component_ranks
        Mapping from component ID to rank
    complement_ranks
        Mapping from component ID to complement rank
    """
    # Compute component ranks
    component_ranks = {}
    for subgraph in positive_subgraphs + negative_subgraphs:
        labels = tuple(sorted(subgraph.nodes))
        component_id = labels_to_component[labels]
        component_ranks[component_id] = get_rank(subgraph)
    
    # Compute complement ranks
    all_labels = set(nonp_unique_labels)
    complement_ranks = {}
    
    for component_id, labels in component_to_labels.items():
        rank = component_ranks[component_id]
        
        # Fast path for rank 0 and 1 components
        if rank == 0:
            # A rank-0 component's complement is always at least rank 3
            complement_ranks[component_id] = 3
        elif rank == 1:
            # A rank-1 component's complement is always at least rank 2
            # Note: This is a conservative estimate - actual rank could be higher
            complement_ranks[component_id] = 2
        else:
            # Compute actual complement rank for rank >= 2
            component_labels = set(labels)
            complement_ranks[component_id] = _compute_complement_rank(
                component_labels, all_labels, contact_graph
            )
    
    return component_ranks, complement_ranks

def get_is_contained(
    component_ranks: Dict[int, int],
    complement_ranks: Dict[int, int]
) -> Dict[int, bool]:
    """
    Determines which components are topologically contained.
    
    A component is contained if its complement has a higher rank than the component itself.
    
    Parameters
    ----------
    component_ranks
        Mapping from component ID to rank
    complement_ranks
        Mapping from component ID to complement rank
        
    Returns
    -------
    Mapping from component ID to containment status (True if contained)
    """
    return {
        component: component_ranks[component] < complement_ranks[component]
        for component in component_ranks
    }