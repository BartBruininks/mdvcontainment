"""
Wrappers are function bindings around third party untyped functions. This file neatly contains all 
`unsafe` typing ignores and rebinds them to what they are expected to be.
"""
# Python
from typing import List, Optional, Tuple, Set

# Python External
import numpy as np
import numpy.typing as npt

## networkx types
from networkx import Graph
from networkx import DiGraph
from networkx import MultiDiGraph

# Import the functions at module level with type ignores if needed
from networkx import ancestors as nx_ancestors  # type: ignore[attr-defined]
from networkx import descendants as nx_descendants  # type: ignore[attr-defined]
from networkx import topological_sort as nx_topological_sort  # type: ignore[attr-defined]
from networkx import transitive_reduction as nx_transitive_reduction  # type: ignore[attr-defined]
from networkx import is_eulerian as nx_is_eulerian  # type: ignore[attr-defined]
from networkx import eulerian_circuit as nx_eulerian_circuit  # type: ignore[attr-defined]
from networkx import connected_components as nx_connected_components  # type: ignore[attr-defined]
from scipy.ndimage import label as scipy_label 
from .find_label_contacts import find_label_contacts as cy_find_label_contacts 
from .find_bridges import find_bridges as cy_find_bridges

def ancestors(digraph: DiGraph, node: int) -> Set[int]:
    """Returns the set of the upstream nodes of the given node in the directed graph."""
    return nx_ancestors(digraph, node) 


def descendants(digraph: DiGraph, node: int) -> Set[int]:
    """Returns the set of the downstream nodes of the given node in the directed graph."""
    return nx_descendants(digraph, node) 


def topological_sort(digraph: DiGraph) -> List[int]:
    """Returns a list of the topological order of the nodes in the directed acyclic graph."""
    return nx_topological_sort(digraph)


def transitive_reduction(digraph: DiGraph) -> DiGraph:
    """
    Returns transitive reduction of a directed acyclic graph.
    
    The transitive reduction of G = (V,E) is a graph G- = (V,E-) such that for all v,w in V 
    there is an edge (v,w) in E- if and only if (v,w) is in E and there is no path from 
    v to w in G with length greater than 1.
    """
    return nx_transitive_reduction(digraph) 


def is_eulerian(graph: Graph|DiGraph|MultiDiGraph) -> bool:
    """Returns whether the (Multi)(Di)Graph is Eulerian."""
    return nx_is_eulerian(graph)  


def eulerian_circuit(graph: Graph|DiGraph|MultiDiGraph, keys: bool) -> List[Tuple[int, int, Tuple[int, int]]]:
    """
    Returns an iterator over the edges of an Eulerian circuit in G.

    An Eulerian circuit is a closed walk that includes each edge of a graph exactly once.
    """
    return nx_eulerian_circuit(graph, keys=keys)


def connected_components(graph: Graph|DiGraph|MultiDiGraph) -> List[Set[int]]:
    """
    Generate connected components.

    The connected components of a (Multi)(Di)Graph partition the graph into disjoint sets of 
    nodes. Each of these sets induces a subgraph of graph G that is connected and not part of 
    any larger connected subgraph.
    
    A graph is connected if, for every pair of distinct nodes, there is a path 
    between them. If there is a pair of nodes for which such path does not exist, the graph is 
    not connected (also referred to as "disconnected").

    A graph consisting of a single node and no edges is connected. Connectivity is undefined 
    for the null graph (graph with no nodes).
    """
    assert len(graph.nodes), "A graph without nodes has no connected components"
    return nx_connected_components(Graph(graph))


## scipy.ndimage wrappers
def label(grid: npt.NDArray[np.bool_], structure: npt.NDArray[np.bool_]) -> Tuple[npt.NDArray[np.int32], int]:
    """
    Returns the relabeled boolean grid and the amount of unique labels.
    
    The structure is used to determine the neighbors for the determination of
    the connected components.
    """
    relabeled_grid = np.zeros_like(grid, dtype=np.int32)
    ncomponents = scipy_label(grid, structure=structure, output=relabeled_grid)
    return relabeled_grid, int(ncomponents) # type: ignore[return-value]


## cython wrappers
def find_label_contacts(labeled_grid: npt.NDArray[np.int32]) -> Optional[npt.NDArray[np.int32]]:
    """Find which labels are in contact with each other without periodic boundary conditions."""
    return cy_find_label_contacts(labeled_grid)


def find_bridges(labeled_grid: npt.NDArray[np.int32]) -> Optional[npt.NDArray[np.int32]]:
    """Find which labels are in contact with each other without periodic boundary conditions."""
    return cy_find_bridges(labeled_grid)