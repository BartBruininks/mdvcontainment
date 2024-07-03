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

#@profile
def nonp_is_contained(labeled_grid, nonp_unique_labels, write_structures=False):
    """
    Returns containment status for the slab case and the component ranks.
    
    Each label is set to be either contained or not, based on the following
    critera:
        1) The relevent dimensions for the outside are picked dynamically
            based on is set to the two largest axis in the shape of the
            input array. The thinnest dimension is assumed to be the slice
            thickness.
        2) Any label which has a voxel in the relevent hollow cylinder
            formed by the minimal and maximal positions of the two picked
            dimensions is regarded an absolute outside.    
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

#@profile
def calc_containment_graph(boolean_grid, nbox, verbose=False, write_structures=False, draw_graphs=False, slab=False):
    """
    Returns the containment graph nx.MultiDiGraph(), 
    taking right angled PBC into account.
    
    It also returns the component contact graph nx.Graph(), the component grid np.array(int32),
    the ranks of the components dict(component:rank) and the labels contact graph.
    """
    # Return the basic counts of Trues and Falses in the input grid if verbose.
    if verbose:
        counts_boolean_grid = dict(zip(*np.unique(boolean_grid, return_counts=True)))
        print(f'Value prevalence in booelan_grid: {counts_boolean_grid}')\

    # Finding all unique label ids in the labaled grid.
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
        bridges = find_bridges(nonp_labeled_grid, nbox)

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
        # Create the component level contact graph. Using an iterative aproach we 
        #  trickly down the containment arrow started by a non-contained component
        #  being incontact with a contained component, therefore containing it. 
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
    #  componnet contact graph (also breaking edges if they do not represent,
    #  a containment hierarchy).
    unique_components = np.unique(components_grid)
    containment_graph = create_containment_graph(is_contained_dict, unique_components, component_contact_graph)
    
    return containment_graph, component_contact_graph, components_grid, component_ranks, contact_graph



def print_dag(G, node, counts, prefix='', is_last=True):
    connector = '└── ' if is_last else '├── '
    if counts is not False:
        print(f"{prefix}{connector}[{node}: {counts[node]}]")
    else:
        print(f"{prefix}{connector}[{node}]")
    children = list(G.successors(node))
    for i, child in enumerate(children):
        new_prefix = prefix + ('    ' if is_last else '│   ')
        print_dag(G, child, counts, new_prefix, i == len(children) - 1)

def print_dag_structure(G, counts=False):
    if counts is not False:
        print(f'Containment Graph with {len(G.nodes())} components (component: nvoxels):')
    else:
        print(f'Containment Graph with {len(G.nodes())} components:')
    roots = [node for node, in_degree in G.in_degree() if in_degree == 0]
    for i, root in enumerate(roots):
        print_dag(G, root, counts, '', i == len(roots) - 1)


class VoxelContainment():
    def __init__(self, grid, nbox=False, verbose=False, write_structures=False, draw_graphs=False, counts=True, slab=False):
        """
        A Containment graph is a DAG which has a parent pointing to its children with an edge.
        """
        # Read input
        self.grid = grid
        self.nbox = nbox
        self._verbose = verbose
        self._write_structures = write_structures
        self._draw_graphs = draw_graphs
        self.slab = slab

        # Automatic nbox for rectangular PBC
        if self.nbox is False:
            if self._verbose:
                print('Setting nbox to rectangular PBC (wrap).')
            self.nbox = np.zeros((3,3))
            self.nbox[0,0] = self.grid.shape[0]
            self.nbox[1,1] = self.grid.shape[1]
            self.nbox[2,2] = self.grid.shape[2]

        # Set all containment ouput
        self.containment_graph, self.component_contact_graph, self.components_grid, self.component_ranks, self.nonp_label_contact_graph = self._calc_containment()
        # Instantiate the inverted graph to False, if it is needed
        #  it is generated only once.
        self._inv_containment_graph = False

        # Set the voxel counts per label if required
        self.voxel_counts = counts
        if self.voxel_counts:
            self.voxel_counts = self.get_counts(self.components_grid)

        self.nodes = self._set_nodes()

        # Sets the root nodes
        self.root_nodes = self._set_roots()

        # Set leave nodes
        self.leaf_nodes = self._set_leaves()

    def _calc_containment(self):
        return calc_containment_graph(
            self.grid, self.nbox, verbose=self._verbose, write_structures=self._write_structures, draw_graphs=self._draw_graphs, slab=self.slab)

    def __str__(self):
        print_dag_structure(self.containment_graph, self.voxel_counts)
        return ''

    def _set_nodes(self):
        """
        Returns all nodes (components) in the contaiment graph.
        """
        return list(self.containment_graph.nodes)

    def get_counts(self, grid=None):
        """
        Returns the number of elements in each component as a dict.
        """
        if grid is None:
            grid = self.components_grid
        return dict(zip(*np.unique(grid, return_counts=True)))

    def _set_roots(self):
        """
        Returns a list of root nodes (absolute outsides).
        """
        return list((node for node, in_degree in self.containment_graph.in_degree() if in_degree == 0))

    def _set_leaves(self):
        """
        Returns a list of leave nodes (deepest level of containment).
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

    def print_containment(self, nodes=False):
        """
        Prints the selected nodes as a containment graph.

        By default prints the complete graph.
        """
        if nodes is not False:
            print_dag_structure(self.containment_graph.subgraph(nodes), self.voxel_counts)
        else:
            print_dag_structure(self.containment_graph, self.voxel_counts)
        return ''

    def draw(self, nodes=False):
        if nodes is False:
            nx.draw_networkx(self.containment_graph)
        else:
            nx.draw_networkx(self.containment_graph.subgraph(nodes))
        plt.show()


if __name__ == '__main__':
    from gen_data import create_3d_boolean_grid
    #@profile
    def test_1():
        print()
        print('PERIODIC TEST CASE')
        shape = [160, 160, 40]
        freqs = [8,8,8]
        boolean_grid = create_3d_boolean_grid(shape, res=freqs)
        slab = False
        containment = VoxelContainment(boolean_grid, slab=slab, nbox=False, counts=True, verbose=False, draw_graphs=False, write_structures=False)
        print(containment)
        print(containment.component_ranks)
    
    #@profile
    def test_2():
        print()
        print('NON PERIODIC TEST CASE')
        shape = [320, 320, 80]
        freqs = [8,8,8]
        boolean_grid = create_3d_boolean_grid(shape, res=freqs)
        slab = True
        containment = VoxelContainment(boolean_grid, slab=slab, write_structures=False)
        print(containment)
        print(containment.component_ranks)
        
    #test_1()
    test_2()
    
    
    
