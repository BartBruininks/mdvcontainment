#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:14:26 2023

@author: bart

# The Final containment algorithm
1. Read MD data
2. Convert to voxels
3. Obtain bridges
4. Obtain rank
5. Obtain pairs
6. Create neighbor graph
7. Create containment graph
"""

## Imports
import shlex
import subprocess
import time
import numpy as np
import networkx as nx
import MDAnalysis as mda
#import matplotlib
#import matplotlib.pyplot as plt
from scipy import ndimage
from mdvwhole import whole as mdvw
from pyvis import network as pvnet
from mdvoxelsegmentation.clustering import blur
#from PIL import Image
#from PIL import ImageOps
import cv2
import os
# useful for showing the final conainment graph in browser
import webbrowser


## Generic helper functions
def fix_bfactor_per_residue(atomgroup):
    """
    Sets the b-factor of a residue to the most prevalent non-zero value per residue.
    
    !TODO This is really slow as it is making a unique call per residue building the matrix,
      much quicker could be if I could somehow group the residues, however, I am not completely
      sure how to do this, as even residues with the same name, might not be the same. On the
      other hand, if the GRO/PDB is runnable, the resname should refer to identical residues,
      as the tpr makes use of that info. Anway, for now this is too slow to be used for resiude
      bfactor correction.
    """
    universe = atomgroup.universe
    for residue in atomgroup.residues:
        ids = residue.atoms.ix
        counts = dict(zip(*np.unique(universe.atoms.tempfactors[ids], return_counts=True)))
        try:
            counts.pop(0)
        except KeyError:
            pass
        dominant_id = max(counts, key=counts.get)
        residue.atoms.tempfactors = dominant_id
    return


def gen_test_110_plane_with_inclusions(x=11, y=11, z=11):
    """
    Returns a 110 diagonal plane (rank 2) with two objects in the void
    one of them is crossing PBC and is not periodic the other 
    is not crossing the PBC and it not periodic (both rank 0). The
    space should also be rank 2.
    """
    occupancy_array = np.zeros((x,y,z))
    for idx, plane in enumerate(occupancy_array):
        plane[idx] = 1
        plane[idx-1] = 1
    occupancy_array[0,-1,:] = 1
    occupancy_array[x//2,0,z//2] = 1
    occupancy_array[x//2,0,(z//2) + 1] = 1
    occupancy_array[x//2,y//10,0] = 1
    occupancy_array[x//2,y//10,-1] = 1
    print(f'A PCD with {int(np.sum(occupancy_array))} points.')
    return occupancy_array


def write_test_gro(occupancy_array, x=11, y=11, z=11, scaling=5, filename='test_box.gro'):
    """
    Write an occupancy matrix to a gro.
    """
    n_residues = int(np.sum(occupancy_array))
    n_atoms = n_residues #* 3

    # create resindex list.
    resindices = range(n_residues)
    assert len(resindices) == n_atoms

    # all molecules belong to 1 segment.
    segindices = [0] * n_residues

    # create the Universe.
    u = mda.Universe.empty(n_atoms,
                             n_residues=n_residues,
                             atom_resindex=resindices,
                             residue_segindex=segindices,
                             trajectory=True) # necessary for adding coordinates.
    u.atoms.positions = np.vstack(np.where(occupancy_array)).T*scaling
    # Place the atoms off grid.
    u.atoms.positions += scaling/2
    # Add PBC description.
    u.dimensions = np.array([x*scaling, y*scaling, z*scaling, 90, 90, 90], dtype='int32')
    u.atoms.write(filename)
    return

    
def create_voxels(atomgroup, resolution):
    """
    Returns 3 voxel objects:
    
    all_voxels, voxels, inv_voxels. Where voxels containt alls the atoms in the atomgroup, all_voxels
    contains all atoms in the universe belonging to the atomgroup. Inv_voxels contains all atoms which 
    are in the universe of the atomgroup, but not in the atomgroup.
    """
    # Getting the all voxel mask for later indexin when we want to get void particles.
    all_voxels = mdvw.Voxels(atomgroup.universe.atoms, resolution=resolution)
    # Getting the selected voxels
    voxels = mdvw.Voxels(atomgroup, resolution=resolution)
    # Getting the inverted voxels (this might be an expensive way to do it as we only have to invert the grid).
    inv_voxels = mdvw.Voxels(atomgroup, resolution=resolution)
    inv_voxels.grid = ~inv_voxels.grid
    return all_voxels, voxels, inv_voxels


def blur_voxels(voxels, inv_voxels, blur_amount=1):
    """
    Dilates and erodes once in place.
    """
    # Possible dilation and erosion to remove small holes for CG data.
    for i in range(blur_amount):
        voxels.grid = blur(voxels.grid, voxels.nbox, 1)
    for i in range(blur_amount):
        inv_voxels.grid = blur(~voxels.grid, voxels.nbox, 1)
    voxels.grid = ~inv_voxels.grid
    return voxels, inv_voxels


def plot_graph(containment_graph, key='value', sizes=None, 
                           name='containment.html', height='800px', 
                           width='800px', directed=True, images=None, scale=10):
    """
    Draws the containment graph using pvnet.
    
    The nodes and edges are red and blue for the postive and negative labels
    respectively. Edges are colored by their container and point toward the 
    containee.

    Parameters
    ----------
    containment_graph : networkx.classes.digraph.DiGraph
        The containment graph.
    sizes : dict, optional
        The amount of elements in each label. The default is None.
    name : TYPE, optional
        DESCRIPTION. The default is 'containment.html'.
    height : int, optional
        The height of the graph. The default is '800px'.
    width : int, optional
        The width of the graph. The default is '800px'.

    Returns
    -------
    Graph displayer.

    """
    # Instantiate the graph.
    net = pvnet.Network(notebook=True, directed=directed, height=height, 
                        width=width)
    # Add the nodes from the containment graph and set their sie and color.
    for node in containment_graph.nodes:
        # size by label size
        if sizes is not None:
            size = (int(np.log(sizes[node])) + 1)*scale
        else:
            size = 1*scale
        if int(node) >= 0:
            if images is not None:
                net.add_node(int(node), label=f'{node}', color='blue', size=size, shape='image', image =images[node])
            else:
                net.add_node(int(node), label=f'{node}', color='blue', size=size)
        elif int(node < 0):
            if images is not None:
                net.add_node(int(node), label=f'{node}', color='red', size=size, shape='image', image =images[node])
            else:
                net.add_node(int(node), label=f'{node}', color='red', size=size)
    # Add the edges in a directed manner (container -> containee).
    for edge in containment_graph.edges(data=True):
        # Do not draw edge annotation if False is given.
        if key is None:
            net.add_edge(int(edge[0]), int(edge[1]))
        else:
            net.add_edge(int(edge[0]), int(edge[1]), label=str(edge[2][key]))
    # Set physics for graph rendering.
    opts = '''
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -300,
              "centralGravity": 0.11,
              "springLength": 100,
              "springConstant": 0.09,
              "avoidOverlap": 1
            },
            "minVelocity": 3.75,
            "solver": "forceAtlas2Based",
            "timestep": 0.05
          }
        }
    '''
    net.set_options(opts)
    return net.show(name)


def label_voxels(voxels, inv_voxels, structure=np.ones((3,3,3))):
    """
    Labels the voxels in place without PBC.
    """
    voxels.grid, unique_labels = ndimage.label(voxels.grid, structure=np.ones((3,3,3)))
    inv_voxels.grid, inv_unique_labels = ndimage.label(inv_voxels.grid, structure=np.ones((3,3,3)))
    inv_voxels.grid *= -1
    return voxels, inv_voxels


def create_edge_mask(array):
    """
    Returns as boolean mask with only the edge (2d)/surfaces (3d) set to true. The mask is of equal size 
    as the input array.
    """
    mask = np.zeros(array.shape, dtype=bool)
    assert 1 < len(mask.shape) < 4, 'Only 2d and 3d input arrays are suported!'
    if len(mask.shape) == 2:
        mask[ :, 0] = True
        mask[ :,-1] = True
        mask[ 0, :] = True
        mask[-1, :] = True
    elif len(mask.shape) == 3:
        mask[ :, :, 0] = True
        mask[ :, :,-1] = True
        mask[ :, 0, :] = True
        mask[ :,-1, :] = True
        mask[ 0, :, :] = True
        mask[-1, :, :] = True    
    return mask


def pbc_wrap(positions, nbox):
    """
    returns the divider and the indices, wrapped into the triclinic box.
    """
    div, mod = np.divmod(positions, nbox.diagonal())
    return div, np.mod(mod - div@nbox, nbox.diagonal())


def get_all_pairs_update(label_array, nbox, mask=None, return_counts=False, 
                         return_bridges=False, no_zero_shift=True, no_zero=True):
    """
    Returns all contact pairs in periodic 2d or 3d array. The mask can be used to restrict the function
    on only true values. The mask should be of equal dimensions as the label array.

    Parameters
    ----------
    array : numpy.ndarray(int64)
        The input label array.

    Returns
    -------
    dict: 
        set of all neighbors to a given label
    
    """
    # get the unique values and their locations.
    if mask is None:
        labels_indices_dict = ndimage.value_indices(label_array)
    else:
        assert label_array.shape == mask.shape
        none_value = 0
        temp_array = label_array.copy()
        temp_array[~mask] = none_value
        labels_indices_dict = ndimage.value_indices(temp_array)
        # Remove 0 if it is present
        try:
            labels_indices_dict.pop(none_value)
        except KeyError:
            pass
    shape = label_array.shape
    # Check the 4 2d cases (and their inverse).
    if len(shape) == 2:
        shifts = np.array([
            # One sided
            (-1, 0),
            ( 0,-1),
            (-1,-1),
            (-1, 1),
            # Opposite
            #( 1, 0),
            #( 0, 1),
            #( 1, 1),
            #( 1,-1)
        ])
    # Check the 13 3d cases.
    else:
        shifts = np.array([
            # pseudo 2d
            (-1, 0, 0),
            ( 0,-1, 0),
            (-1,-1, 0),
            (-1, 1, 0),
            # pseudo 2d opposite
            #( 1, 0, 0),
            #( 0, 1, 0),
            #( 1, 1, 0),
            #( 1,-1, 0),
            # pseudo 2d one down
            (-1, 0,-1),
            ( 0,-1,-1),
            (-1,-1,-1),
            (-1, 1,-1),
            # inv pseudo 2d one down
            ( 1, 0,-1),
            ( 0, 1,-1),
            ( 1, 1,-1),
            ( 1,-1,-1),
            # unity one down
            ( 0, 0,-1),
            # pseudo 2d one up
            #(-1, 0, 1),
            #( 0,-1, 1),
            #(-1,-1, 1),
            #(-1, 1, 1),
            # inv pseudo 2d one up
            #( 1, 0, 1),
            #( 0, 1, 1),
            #( 1, 1, 1),
            #( 1,-1, 1),
            # unity one up
            #( 0, 0, 1),
        ])
    # This is for obtaining the pairs.
    contacts_dict = dict()
    contacts_counts_dict = dict()
    # This is for obtaining the brirdes.
    bridges_dict = dict() # shift: label : labels
    for label, indices in labels_indices_dict.items():
        contacts = list()
        #for dim, size in enumerate(label_array.shape): 
        for shift in shifts:
            active_dims = shift.astype(bool)
            these_indices = np.array(indices).T
            # Here we actually change the dim indices with a value. To allow for 
            #  non-cubic boxes, we have to look at the potential off axis shifts
            #  and also apply them.
            shifted_div, shifted_indices = pbc_wrap(these_indices + shift, nbox)
            new_contacts = label_array[tuple(shifted_indices.T)].tolist()
            contacts += new_contacts
            # Checking the bridges and adding to the dict.
            if return_bridges:
                # Handling the bridges allocation for one side.
                bridges = np.zeros((len(these_indices), 2+len(nbox.diagonal())), dtype=int)
                bridges[:, 0] = label
                bridges[:, 1] = new_contacts
                bridges[:, 2:] = shifted_div
                # Inverting the bridge labels and shift for the inverse direction.
                inv_bridges = np.copy(bridges)
                inv_bridges[:, [0,1]] = inv_bridges[:, [1,0]]
                inv_bridges[:,2:] *= -1
                bridges = np.vstack([bridges, inv_bridges])
                for bridge in bridges:
                    if no_zero and 0 in bridge[:2]:
                        continue
                    # Remove the inbox bridges. This might actually not be needed anymore w
                    #  when we only have the onside check and boundaries as described below.
                    if no_zero_shift:
                        if np.sum(bridge[2:]) == 0:
                            continue
                    # Due to the inherant symmetry in a contact we can only check
                    #  half the edges/edge_planes and half the nieghbours, then 
                    #  We can simple invert the contact by swapping the label indices
                    #  and inverting the div. This would reduce the amount of checks
                    #  and on top makes the hash calls for the dictionary less,
                    #  probably resulting in a large performance impact.
                    try:
                        bridges_dict[tuple(bridge)] += 1
                    except KeyError:
                        bridges_dict[tuple(bridge)] = 1
        # Check if we have to return the counts as well. To run np.unique
        #  is quite expensive so obtaining the count is really up to the usecase.
        if return_counts:
            contacts, contacts_counts = np.unique(contacts, return_counts=True)
            contacts_counts_dict[label] = contacts_counts
        else:
            contacts = np.array(list(set(contacts)))
        contacts_dict[label] = contacts
    # Writing the ouput as a dictionary collection.
    output = dict()
    output['contacts'] = contacts_dict
    if return_counts:
        output['contact_counts'] = contacts_counts_dict
    if return_bridges:
        output['bridges'] = bridges_dict
    return output


def bridges2graph(bridges, nodes):
    """
    Returns the bridges graph using the first two indices as the labels and the following elements as the value.
    """
    graph = nx.MultiDiGraph()
    for node in nodes:
        if node == 0:
            continue
        graph.add_node(int(node))
    for bridge in bridges.keys():
        if 0 in bridge[:2]:
            continue
        #G.add_edge(labels[a], labels[b], label=str(rev_shift), value=int(bridges[a,b,s]), cost=rev_shift)
        graph.add_edge(int(bridge[0]), int(bridge[1]), label=str(bridge[2:]), value=bridges[bridge], cost=bridge[2:])
    return graph


def get_bridges(voxels):
    """
    Returns the bridges graph for the voxels.
    """
    unique_labels_names = np.unique(voxels.grid)
    bridges = Bridges(bridges2graph(
        get_all_pairs_update(
            voxels.grid, voxels.nbox, 
            mask=create_edge_mask(voxels.grid), 
            return_bridges=True
        )['bridges'], unique_labels_names))
    return bridges


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


def get_rank(graph, key='value'):
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


def get_subgraph_ranks(bridges):
    """
    Returns a dictionary with the sorted subgraph nodes as keys and the ranks as values.
    The covered labels are also returned.
    """
    ranks = {}
    covered_labels = set()
    subgraphs = [bridges.graph.subgraph(c).copy() for c in nx.connected_components(nx.Graph(bridges.graph))]
    for idx, subgraph in enumerate(subgraphs):
        rank = get_rank(subgraph, key='cost')
        nodes = tuple(sorted(subgraph.nodes))
        for node in nodes:
            covered_labels.add(node)
        ranks[nodes] = rank
    return ranks, covered_labels


def add_zero_ranks(ranks, covered_labels, unique_labels):
    """
    Returns the rank of all subgraphs (also the ones which are not in the bridges graph).
    
    The ranks are in a dictionary with the sorted subgraphs as keys and the rank as values.
    """
    for label in unique_labels:
        if label != 0 and label not in covered_labels:
            ranks[tuple([label])] = 0
            covered_labels.add(label)
    return ranks


def get_all_ranks(bridges, unique_labels):
    """
    Returns all ranks of the current selection (real or inv).
    
    The ranks are in a dictionary with the sorted subgraphs as keys and the rank as values.
    """
    ranks, covered_labels = get_subgraph_ranks(bridges)
    ranks = add_zero_ranks(ranks, covered_labels, unique_labels)
    return ranks


def get_combined_ranks(label_array, bridges, inv_label_array, inv_bridges):
    """
    Returns the ranks of both the densities and inverted densities.
    """
    ranks = get_all_ranks(bridges, np.unique(label_array))
    inv_ranks = get_all_ranks(inv_bridges, np.unique(inv_label_array))
    all_ranks = {**ranks, **inv_ranks}
    return all_ranks


def get_combined_label_array(label_array, inv_label_array):
    """
    Returns the combined label array and its unique elements.
    """
    combined_label_array = label_array + inv_label_array
    combined_labels = np.unique(combined_label_array)
    return combined_label_array, combined_labels


def generate_subgraphs(linked_labels, no_zero=True, labels=False):
    """
    Returns a list of subgraphs from linked labels.

    Parameters
    ----------
    linked_labels : numpy.ndarray(int64)
        Pairs of linked labels (Nx2).
    no_zero : bool, optional
        Do not include pairs with a 0 in it. The default is True.
    labels : numpy.ndarray(int64), optional
        All labels, this might be bigger than the labels occuring in the
        linked pairs. The default is False.

    Returns
    -------
    list(set(int64))
        A list of subgraphs.

    """
    # check if there are linked labels at all.
    if not linked_labels.size:
        return [set()]
    graph = nx.Graph()
    if labels is False:
        graph.add_nodes_from(np.unique(linked_labels))
    else:
        graph.add_nodes_from(labels)
    graph.add_edges_from(linked_labels)
    # Remove all links to the non-segmented.
    if no_zero:
        if 0 in graph.nodes:
            graph.remove_node(0)
    #TODO Maybe it is better to return a list of tuples here. I can sort
    # them and then I can be sure that I can use them as keys in a dict.
    subgraphs = list(nx.connected_components(graph))
    return subgraphs


def relabel_labels(array, subgraphs):
    """
    Relabels the array (in place) using the subgraphs.
    
    0 is forced to always be mapped to 0, as it has a special 
    meaning (being nothing).

    Parameters
    ----------
    array : numpy.ndarray(int64)
        A labeled array.
    subgraphs : list(set())
        The linked labels.
    inplace : bool, optional
        Relabels the array in place. The default is False.

    Returns
    -------
    array : numpy.ndarray(int64)
        The relabeled array.
    relabel_dict : dict(int : set(int64))
        A dict with the new labels as keys and the old labels as values.

    """
    out_array = array.copy()
    # Obtaining all unique labels.
    all_labels = set(np.unique(array))
    # Discarding the 0 label.
    try:
        all_labels.remove(0)
    # If it is not there that is ok.
    except KeyError:
        pass
    # Start new labeling from 1.
    new_label = 1
    relabel_dict = dict()
    for subgraph in subgraphs:
        # Dealing with an empy entry.
        if len(subgraph) > 0:
            # No relabeling of 0 can occur.
            if 0 in subgraph:
                assert len(subgraph) == 1, 'The zero containing subgraph should always be of length 1.'
                continue
            # Relabeling the individual labels in the subgraph.
            for label in subgraph:
                all_labels.remove(label)
                out_array[array == label] = new_label
            relabel_dict[new_label] = subgraph
            new_label += 1
    # Making sure that labels which are not in a subgraph are still relabeled.
    for label in all_labels:
        out_array[array == label] = new_label
        relabel_dict[new_label] = set([label])
        # This labeling increments even further (no reset!).
        new_label += 1
    return out_array, relabel_dict


def pairs_dict2array(pairs_dict, data_type='all'):
    """
    Returns the sorted and unique pair array.
    
    The data type can be used to sort through only 'density' or 'inv_density' or 'all'
    """
    contacts = pairs_dict['contacts']
    all_pairs = []
    for label in contacts:
        for target_label in contacts[label]:
            if (label < 0 or target_label < 0) and data_type == 'density':
                continue
            elif (label > 0 or target_label > 0) and data_type == 'inv_density':
                continue
            # No self identity
            if label == target_label:
                continue
            all_pairs.append((label, target_label))
    # Convert the list into an array.
    all_pairs = np.array(all_pairs)
    # Sort the pairs
    all_pairs = np.array([sorted(pair) for pair in all_pairs])
    # Filter for unique pairs.
    all_pairs = np.unique(all_pairs, axis=0)
    return all_pairs


def get_pairs(combined_label_array, nbox):
    """
    Returns the pairs arrays for all, density and inv_density.
    """
    all_pairs_dict = get_all_pairs_update(combined_label_array, nbox)
    all_pairs = pairs_dict2array(all_pairs_dict, 'all')
    density_pairs = pairs_dict2array(all_pairs_dict, 'density')
    inv_density_pairs = pairs_dict2array(all_pairs_dict, 'inv_density')
    return all_pairs, density_pairs, inv_density_pairs


def combine_relabel_dicts(relabel_dict, inv_relabel_dict):
    """
    Returns the two way combined relabal dict where the sign indicates the 
    density or void labeling.
    
    The dict can be indexes by either the object id, or the subgraph (object)
    tuple.

    Parameters
    ----------
    relabel_dict : dict(int : set(int64))
        A dict with the labels as keys and the subgraphs as values.
    inv_relabel_dict : dict(int : set(int64))
        A dict with the labels as keys and the subgraphs as values.

    Returns
    -------
    combined_relabel_dict : dict(int/tuple(int64) : set(int64)/int64)
        A 2way dict with the labels as keys and the subgraphs as values and 
        vice versa.

    """
    combined_relabel_dict = dict()
    for label in relabel_dict:
        combined_relabel_dict[label] = relabel_dict[label]
        # TODO I HAVE TO SORT THIS THIS IS STUPID I SHOULD NOT BE USING SETS
        combined_relabel_dict[
            tuple(sorted(tuple(relabel_dict[label])))] = label
    for label in inv_relabel_dict:
        combined_relabel_dict[-label] = inv_relabel_dict[label]
        # TODO THIS SORT IS STUPID, I SHOULD NOT BE USING SETS
        combined_relabel_dict[
            tuple(sorted(tuple(inv_relabel_dict[label])))] = -label
    return combined_relabel_dict


def create_periodic_label_dict(relabeled_labels, combined_relabel_dict, 
                               all_ranks):
    """
    Returns a dict for each object id specifying if its intrinsically periodic.

    Parameters
    ----------
    relabeled_labels : numpy.ndarray(int32)
        All object labels.
    combined_relabel_dict : dict(int/tuple(int64) : set(int64)/int64)
        A 2way dict with the labels as keys and the subgraphs as values and 
        vice versa.
    complete_spanning_subgraphs : list(set(int32))
        All objects spanning at least one dimension.

    Returns
    -------
    periodic_label_dict : dict(int64 : bool)
        Labels as keys and a bool for being intrinsically periodic or not.

    """
    periodic_label_dict = dict()
    for label in relabeled_labels:
        periodic_label_dict[label] = False
    for subgraph in all_ranks:
        # Sort the subgraph for tuple indexing.
        subgraph = tuple(sorted(list(subgraph)))
        # TODO THIS IS STUPID I SHOULD NOT NEED THIS SORT AND USE TUPLES FROM THE START
        current_label = combined_relabel_dict[tuple(subgraph)]
        periodic_label_dict[current_label] = all_ranks[subgraph]
    return periodic_label_dict


def relabel_pairs(all_pairs, combined_relabel_dict):
    new_pairs = np.copy(all_pairs)
    for relabel in combined_relabel_dict:
        if type(relabel) != int:
            continue
        for label in combined_relabel_dict[relabel]:
            new_pairs[all_pairs == label] = relabel
    new_pairs = np.unique(new_pairs, axis=0)
    # Removing self entries.
    temp_pairs = []
    for pair in new_pairs:
        if pair[0] == pair[1]:
            continue
        else:
            temp_pairs.append(pair)
    new_pairs = np.array(temp_pairs)
    return new_pairs


def generate_contact_graph(relabeled_labels, periodic_labels, all_pairs, combined_relabel_dict):
    """
    Creates the contacts graph.

    Parameters
    ----------
    relabeled_labels : numpy.ndarray(int32)
        All object labels.
    periodic_labels : numpy.ndarray(int32)
        All subgraph labels.
    relabeled_combined_label_array : numpy.ndarray(int64)
        The relabeled segmentation array.

    Returns
    -------
    contact_graph : networkx.classes.graph.Graph
        A graph containing all contacts, all nodes are annotated to be
        periodic or not.

    """    
    contact_graph = nx.Graph()

    for label in relabeled_labels:
        contact_graph.add_node(label, rank=periodic_labels[label], 
                               label=label)

    # TODO This is indeed a slow step!!!
    # Here we calculate all pairs for the relabaled array again, this might
    #  be overly expensive and we could actually get all the edges by combining
    #  the edges of the subgraphs for the objects, however, for now that would
    #  be a performance enhancement which I do not care about.
    contacts = relabel_pairs(all_pairs, combined_relabel_dict)
    # Adding all the contacts
    for pair in contacts:
        contact_graph.add_edge(int(pair[0]), int(pair[1]))
    return contact_graph


def generate_containment_graph(contact_graph):
    """
    Returns the containment_graph using the contact graph 
    (annotated with periodic labels).

    Parameters
    ----------
    contact_graph : networkx.classes.graph.Graph
        A graph containing all contacts, all nodes are annotated to be
        periodic or not.

    Returns
    -------
    containment_graph : networkx.classes.digraph.DiGraph
        A graph with the containment information encoded as the directionality.
        The directionality goes from container -> containee.

    """
    contact_graph = contact_graph.copy()
    containment_graph = nx.DiGraph()
    degrees = contact_graph.degree()
    nodes = list(contact_graph.nodes())
    for node in nodes:
        containment_graph.add_node(int(node), label=str(node))
    # Check if something changed
    #  This is a recursive approach where nodes with
    #  one edge are removed and pointed to from their 
    #  neighbour. This is enough in combination with the 
    #  fact if a node is intrinsically periodic to 
    #  direct the containment.
    current_rank = 0
    while current_rank < 4:
        degrees = dict(contact_graph.degree())
        old_degrees = None
        while old_degrees != degrees:
            handled_nodes = []
            for node, values in contact_graph.nodes(data=True):
                # If the node is periodic we are not interested.
                if values['rank'] > current_rank:
                    continue
                # If the node only has one edge it is a candidate to be contained
                #  we only have to find the partner and check its degree to be not 1.
                elif degrees[node] == 1:
                    edge = list(contact_graph.edges([node]))[0]
                    for i in edge:
                        if i != node:
                            target = i
                    # Making sure our linked partner is not also a degree 1 node
                    #  this is to prevent circular definitions in equal rank situations.
                    #  If the target rank has a higher rank than the current node, it does
                    #  not have to have a higher degree.
                    node_rank = contact_graph.nodes(data=True)[node]['rank']
                    target_rank = contact_graph.nodes(data=True)[target]['rank']
                    if degrees[target] > 1 or node_rank < target_rank:
                        containment_graph.add_edge(int(target), int(node))
                        # Add the node to the list for removal for 
                        #  the next iteration.
                        handled_nodes.append(node)
            # Remove all nodes that were processed.
            for node in handled_nodes:
                contact_graph.remove_node(node)
            old_degrees = degrees
            degrees = dict(contact_graph.degree())
        current_rank += 1
    return containment_graph


def create_corners(universe):
    """
    Creates a GRO with the corners of the box for autocentering.
    """
    n_residues = 8
    n_atoms = 8
    box_points = np.zeros((8,3))
    dimensions = Periodic(universe.dimensions).pbc_matrix
    box_points[0, :] = np.array((0, 0, 0))@dimensions
    box_points[1, :] = np.array((0, 0, 1))@dimensions
    box_points[2, :] = np.array((0, 1, 0))@dimensions
    box_points[3, :] = np.array((1, 0, 0))@dimensions
    box_points[4, :] = np.array((0, 1, 1))@dimensions
    box_points[5, :] = np.array((1, 1, 0))@dimensions
    box_points[6, :] = np.array((1, 0, 1))@dimensions
    box_points[7, :] = np.array((1, 1, 1))@dimensions
    #print('box points', box_points)
    all_positions = np.vstack([universe.atoms.positions, box_points])
    
    min_values = np.min(all_positions, axis=0)
    max_values = np.max(all_positions, axis=0)

    # create resindex list.
    resindices = range(n_residues)
    assert len(resindices) == n_atoms
    #print("resindices:", resindices[:10])

    # all water molecules belong to 1 segment.
    segindices = [0] * n_residues
    #print("segindices:", segindices[:10])

    # create the Universe.
    sol = mda.Universe.empty(n_atoms,
                             n_residues=n_residues,
                             atom_resindex=resindices,
                             residue_segindex=segindices,
                             trajectory=True) # necessary for adding coordinates
    positions = np.zeros((8, 3))
    sol.dimensions = universe.dimensions
    positions[0, :] = min_values[0], min_values[1], min_values[2]
    positions[1, :] = min_values[0], min_values[1], max_values[2]
    positions[2, :] = min_values[0], max_values[1], min_values[2]
    positions[3, :] = min_values[0], max_values[1], max_values[2]
    positions[4, :] = max_values[0], max_values[1], max_values[2]
    positions[5, :] = max_values[0], max_values[1], min_values[2]
    positions[6, :] = max_values[0], min_values[1], max_values[2]
    positions[7, :] = max_values[0], min_values[1], min_values[2]
    
    sol.atoms.positions = positions
    sol.atoms.write('box_corners.gro')
    return


def make_neighbor_graph(all_pairs):
    """
    Returns a nx.Graph with all neighbor connectivity.
    """
    neighbor_graph = nx.Graph()
    neighbor_graph.add_edges_from(all_pairs)
    return neighbor_graph


def plot_neighor_graph(neighbor_graph, combined_label_array):
    """
    Plots the neighbor graph.
    """
    labels, counts = np.unique(combined_label_array, return_counts=True)
    counts = counts / np.min(counts)
    combined_sizes = dict(zip(labels, counts))
    return plot_graph(neighbor_graph, key=None, directed=False, name='neighbor.html', sizes=combined_sizes)


def plot_contact_graph(contact_graph, relabeled_combined_label_array):
    """
    Plots the contacts graph.
    """
    labels, counts = np.unique(relabeled_combined_label_array, return_counts=True)
    counts = counts / np.min(counts)
    relabeled_sizes = dict(zip(labels, counts))
    return plot_graph(contact_graph, key=None, directed=False, name='contact.html', sizes=relabeled_sizes)


def plot_containment_graph(containment_graph, relabeled_combined_label_array, images=None):
    """
    Plots the containment graph.
    """
    labels, counts = np.unique(relabeled_combined_label_array, return_counts=True)
    counts = counts / np.min(counts)
    relabeled_sizes = dict(zip(labels, counts))
    if images is None:
        return plot_graph(containment_graph, key=None, directed=True, name='containment.html', sizes=relabeled_sizes)
    else:
        return plot_graph(containment_graph, key=None, directed=True, name='containment_img.html', images=images, scale=40, width='1000px', height='1000px')
    
    
def crop_image(filename):
    img = cv2.imread(filename) # Read in the image and convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255*(gray < 230).astype(np.uint8) # To invert the text to white.
    coords = cv2.findNonZero(gray) # Find all non-zero points (text).
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box.
    rect = img[y:y+h, x:x+w] # Crop the image - note we do this on the original image.
    cv2.imwrite(filename, rect) # Save the image.


def write_containment(universe, all_voxels, relabeled_combined_label_array, containment_graph, prefix, ftype):
    """
    Returns a dict with filenames of the rendered images. 
    
    Writes the containment gro files and renders.
    """
    # Settting the correct boundaries.
    create_corners(universe)
    # Setting containment status.
    containers = list((node for node, out_degree in containment_graph.out_degree() if out_degree > 0))
    label_images = np.unique(relabeled_combined_label_array)
    # I have to check if an atomgroup is empty and if it is not render it and remove it from the graph?
    for target_container in list(containment_graph.nodes()):
        print(f'\n\nContainer {target_container}\n------------------------')
        # We can use MDA logic from here on to get all the residues etc. We will not be able to check the 
        #  composition of this container as the empty space is literally empty. The solvent was removed.
        #  However, the logic is the same and you can just put in a negative number if the solvent is there.
        target_voxel_indices = np.array(list(zip(*np.where(relabeled_combined_label_array == target_container))))
        target_atomgroup_indices = all_voxels.get_voxels(target_voxel_indices).ix#.residues.atoms
        target_atomgroup = universe.atoms[target_atomgroup_indices] 
        print(dict(zip(*np.unique(target_atomgroup.resnames, return_counts=True))))

        # Sometimes we end up with empy voxels due to the dilation.
        if len(target_atomgroup.atoms) == 0:
            continue
        target_name = f'{prefix}{target_container}.gro'
        #out_name = f'whole_test_container_{target_container}.gro'
        target_atomgroup.write(target_name) # Save the coordinates.
        time.sleep(1)
        print(f'I wrote {target_name}')
        #!mdvwhole -f $target_name -x $target_name -o $out_name -wa True -sel 'all'
        # This allows for rendering outsides different than insides.
        if target_container in containers:
            #!vmd -m $target_name box_corners.gro -e render_outside_command.vmd
            subprocess.call(shlex.split(f'vmd -m {target_name} box_corners.gro -e render_outside_command.vmd'))
        else:
            #!vmd -m $target_name box_corners.gro -e render_command.vmd
            subprocess.call(shlex.split(f'vmd -m {target_name} box_corners.gro -e render_command.vmd'))
        render_name = f'{prefix}{target_container}.{ftype}'
        #!mv text_render.png $render_name
        subprocess.call(shlex.split(f'mv text_render.png {render_name}'))
        
        print('\n\nRENDER NAME\n\n', render_name)
        try:
            # Sometimes we cannot crop the image.
            crop_image(render_name)
        except:
            pass
    # Returning the images.
    images = {}
    path = os.getcwd()
    for label in np.unique(relabeled_combined_label_array):
        image = f'{prefix}{label}.{ftype}'
        images[label] = f'{path}/{image}'
    return plot_containment_graph(containment_graph, relabeled_combined_label_array, images)


def print_container(all_voxels, relabeled_combined_label_array, target_container):
    print(f'\n\nContainer {target_container}\n------------------------')
    # We can use MDA logic from here on to get all the residues etc. We will not be able to check the 
    #  composition of this container as the empty space is literally empty. The solvent was removed.
    #  However, the logic is the same and you can just put in a negative number if the solvent is there.
    target_voxel_indices = np.array(list(zip(*np.where(relabeled_combined_label_array == target_container))))
    target_atomgroup = all_voxels.get_voxels(target_voxel_indices)#.residues.atoms
    print(dict(zip(*np.unique(target_atomgroup.resnames, return_counts=True))))
    
    
def get_containers(atomgroup, resolution=0.5, blur_amount=1, plotting=False):
    """
    Returns all_voxels, voxels, inv_voxels and the containment graph.
    """
    output_dict = {}
    print(f'Creating voxel masks with a resolution of {resolution}...')
    # Creating the voxels objects.
    all_voxels, voxels, inv_voxels = create_voxels(atomgroup, resolution)
    print(f'Blurring voxel masks with {blur_amount}...')
    # Blurring the density and anti desntiy voxels.
    voxels, inv_voxels = blur_voxels(voxels, inv_voxels, blur_amount)
    print(f'Non PBC-labeling...')
    # Non PBC-voxel connected component labeling (in place).
    voxels, inv_voxels = label_voxels(voxels, inv_voxels, structure=np.ones((3,3,3)))
    
    output_dict['all_voxels'] = all_voxels
    output_dict['voxels'] = voxels
    output_dict['inv_voxels'] = inv_voxels
    
    print(f'Obtaining bridges...')
    # Obtain the bridges due to PBC (expensive).
    bridges = get_bridges(voxels)
    inv_bridges = get_bridges(inv_voxels)
    print(f'Calculating the ranks...')
    # Calculating the rank for the sorting of periodic opbjects.
    all_ranks = get_combined_ranks(voxels.grid, bridges, inv_voxels.grid, inv_bridges)
    
    output_dict['bridges'] = bridges
    output_dict['inv_bridges'] = inv_bridges
    output_dict['all_ranks'] = all_ranks
    
    print(f'The ranks are {all_ranks}')
    # Combining the density and inv_density labels.
    combined_label_array, combined_labels = get_combined_label_array(voxels.grid, inv_voxels.grid)
    print(f'Calculating the pairs...')
    # Calculting all label contacts in the combined array.
    all_pairs, density_pairs, inv_density_pairs = get_pairs(combined_label_array, voxels.nbox)
    print(f'Relabeling taking PBC into account...')
    # Combining fragments into objects
    label_subgraphs = generate_subgraphs(density_pairs)
    relabeled_label_array, relabel_dict = relabel_labels(voxels.grid, label_subgraphs)
    inv_label_subgraphs = generate_subgraphs(inv_density_pairs)
    relabeled_inv_label_array, inv_relabel_dict = relabel_labels(inv_voxels.grid, inv_label_subgraphs)
    # Relabel the combined label array.
    relabeled_combined_label_array = relabeled_label_array - relabeled_inv_label_array
    combined_relabel_dict = combine_relabel_dicts(relabel_dict, inv_relabel_dict)
    #print(f'_t combined relable dict {combined_relabel_dict}')
    relabeled_labels = np.unique(relabeled_combined_label_array)
    periodic_labels = create_periodic_label_dict(relabeled_labels, combined_relabel_dict, all_ranks)
    
    output_dict['combined_label_array'] = combined_label_array
    output_dict['all_pairs'] = all_pairs
    output_dict['density_pairs'] = density_pairs
    output_dict['inv_density_pairs'] = inv_density_pairs
    output_dict['label_subgraphs'] = label_subgraphs
    output_dict['relabeled_label_array'] = relabeled_label_array
    output_dict['relabel_dict'] = relabel_dict
    output_dict['inv_label_subgraphs'] = inv_label_subgraphs
    output_dict['relabeled_inv_label_array'] = relabeled_inv_label_array
    output_dict['relabeled_combined_label_array'] = relabeled_combined_label_array
    output_dict['combined_relabel_dict'] = combined_relabel_dict
    output_dict['inv_relabel_dict'] = inv_relabel_dict
    output_dict['relabeled_labels'] = relabeled_labels
    output_dict['periodic_labels'] = periodic_labels
    
    print(f'Creating graphs...')
    # Writing the final graphs.
    neighbor_graph = make_neighbor_graph(all_pairs)
    contact_graph = generate_contact_graph(relabeled_labels, periodic_labels, all_pairs, combined_relabel_dict)
    
    output_dict['neighbor_graph'] = neighbor_graph
    output_dict['contact_graph'] = contact_graph
    
    #print(f'_t Contact graph ranks\n {contact_graph.nodes(data=True)}')
    containment_graph = generate_containment_graph(contact_graph)
    # Plotting the CC bridges graphs and saving them.
    if plotting:
        print(f'Plotting...')
        plot_graph(bridges.graph, key='label', name='density_bridges.html')
        plot_graph(inv_bridges.graph, key='label', name='inv_density_bridges.html')
        plot_neighor_graph(neighbor_graph, combined_label_array)
        plot_contact_graph(contact_graph, relabeled_combined_label_array)
        plot_containment_graph(containment_graph, relabeled_combined_label_array)

    
    output_dict['containment_graph'] = containment_graph
    return output_dict


def _get_container_atomgroups(containment_output, targets=None, verbose=False):
    """
    Returns a list of  atomgroups of the containers. If the targets is set to None all containers will be returned.
    """
    containment_graph = containment_output['containment_graph']
    relabeled_combined_label_array = containment_output['relabeled_combined_label_array']
    all_voxels = containment_output['all_voxels']
    universe = all_voxels.atomgroup.universe
    label_images = np.unique(relabeled_combined_label_array)
    atomgroups = {}
    if targets is None:
        targets = np.unique(relabeled_combined_label_array)
    for target_container in targets:
        if verbose:
            print(f'\n\nContainer {target_container}\n------------------------')
        # We can use MDA logic from here on to get all the residues etc. We will not be able to check the 
        #  composition of this container as the empty space is literally empty. The solvent was removed.
        #  However, the logic is the same and you can just put in a negative number if the solvent is there.
        target_voxel_indices = np.array(list(zip(*np.where(relabeled_combined_label_array == target_container))))
        target_atomgroup_indices = all_voxels.get_voxels(target_voxel_indices).ix#.residues.atoms
        target_atomgroup = universe.atoms[target_atomgroup_indices] 
        atomgroups[target_container] = target_atomgroup
        if verbose:
            print(dict(zip(*np.unique(target_atomgroup.resnames, return_counts=True))))
    return atomgroups


def _get_atomgroup_sizes(container_atomgroups):
    """
    Returns a dict with the node labels and the atomgroup sizes
    """
    sizes = {}
    for node in container_atomgroups.keys():
        sizes[node] = len(container_atomgroups[node])
    return sizes


def _get_normalized_sizes(sizes):
    """
    Returns the normalized sizes (0-1).
    """
    keys = list(sizes.keys())
    values = list(sizes.values())
    value_array = np.array(values, dtype=float)
    value_array /= np.max(value_array)
    values = list(value_array)
    return dict(zip(keys, values))


def _get_single_depth(containment_graph, node, depth=0):
    """
    Returns the depth of the node.
    
    This is a prettty naive approach as we traverse the tree for every node, 
    if the tree gets large this is not great. However, this does allow of querrying 
    one node at a time. It could be better to do all nodes at once starting from the 
    nodes which do not have an in_degree.
    """
    ins = list(containment_graph.in_edges(node))
    if len(ins):
        depth += 1
        node = ins[0][0]
        _get_single_depth(containment_graph, node, depth)
    return depth


def _get_all_depths(containment_graph, depth_dict={}, depth=0):
    """
    Returns the depth of all nodes in the containment tree.
    
    A copy of the tree of interest should be given.
    """
    # Initialize the recursion with a copy as we will
    #  remove nodes from the graph.
    if depth == 0:
        containment_graph = containment_graph.copy()
    nodes = set(containment_graph.nodes())
    try:
        roots = set(np.array(containment_graph.in_edges())[:, 1]) ^ nodes
    # This is the out, as there can be no roots left in the graph, meaning
    #  we are done.
    except IndexError:
        # Adding the leftover nodes to the dict.
        for node in nodes:
            depth_dict[node] = depth
        roots = False
    if roots:
        # Process the current roots and remove them.
        for root in roots:
            depth_dict[root] = depth
        containment_graph.remove_nodes_from(roots)
        depth += 1
        # Recursion.
        _get_all_depths(containment_graph, depth_dict, depth)
    return depth_dict


## Generic helper classes
class Periodic():
    """
    Createse periodic system descriptions in common formats.
    
    Boundary condition styles can be gmx (gromacs), mda (mdanalysis), matrix and freud.
    Freud uses an upper triangle convention of the linalg representation, matrix uses the
    bottom triangle convention. These can be easily converted by transforming the matrix.
    
    Atrributes
    -----------------------------------------------------------------------------------
    Periodic.pbc_gmx : np.array([xx, yy, zz, xy, xz, yx, yz, zx, zy], dtype=float) 
    Periodic.pbc_mda: np.array([xx, yy, zz, alpha, beta, gamma], dtype=float) 
    Periodic.pbc_matrix: np.array([xx, xy, xz],[xy, yy, yz],[xz, yz, zz], dtype=float) 
    Periodic.pbc_freud : np.array([xx, xy, xz],[xy, yy, yz],[xz, yz, zz], dtype=float)
    """
    def __init__(self, boundary_conditions):
        """
        Initialize a periodic descriptor by one of the representations.
        """
        self._convert_boundary_conditions(boundary_conditions)
            
    def _convert_boundary_conditions(self, boundary_conditions):
        """
        Sets all the different PBC representation given a valid representation.
        """
        boundary_conditions = np.asarray(boundary_conditions, dtype='float32')
        # Check for MDA style.
        if boundary_conditions.shape == (6,):
            #print('mda')
            self.pbc_mda = boundary_conditions.astype('float32')
            self.pbc_matrix = self._mda2matrix(*self.pbc_mda)
            self.pbc_freud = self.pbc_matrix.T
            self.pbc_gmx = self._matrix2gmx(self.pbc_matrix)
        # Check for gromacs style.
        elif boundary_conditions.shape == (9,):
            #print('gmx')
            self.pbc_gmx = boundary_conditions.astype('float32')
            self.pbc_matrix = self._gmx2matrix(self.pbc_gmx)
            self.pbc_freud = self.pbc_matrix.T
            self.pbc_mda = self._matrix2mda(self.pbc_matrix)
        # Check the triangle convention.
        elif boundary_conditions.shape == (3,3):
            # Check lower triangle (matrix).
            if boundary_conditions[0,2] == 0:
                #print('matrix')
                self.pbc_matrix = boundary_conditions.astype('float32')
                self.pbc_mda = self._matrix2mda(self.pbc_matrix)
                self.pbc_gmx = self._matrix2gmx(self.pbc_matrix)
                self.pbc_freud = self.pbc_matrix.T
            # Check upper triangle (freud).
            elif boundary_conditions[2,0] == 0:
                #print('freud')
                self.pbc_freud = boundary_conditions.astype('float32')
                self.pbc_matrix = self.pbc_freud.T
                self.pbc_mda = self._matrix2mda(self.pbc_matrix)
                self.pbc_gmx = self._matrix2gmx(self.pbc_matrix)
        else:
            raise Warning("The PBC format is not recognized from the specified boundary conditions and cannot be processed. Use help(Periodic)")
            
            
    def _mda2matrix(self, x, y, z, alpha=90, beta=90, gamma=90):
        """Convert mdanalysis fromat to matrix format"""
        input_array = np.array([x,y,z,alpha,beta,gamma]).astype('float64')
        x,y,z,alpha,beta,gamma = input_array
        cosa = np.cos( np.pi * alpha / 180 )
        cosb = np.cos( np.pi * beta / 180 )
        cosg = np.cos( np.pi * gamma / 180 )
        sing = np.sin( np.pi * gamma / 180 )
        zx = z * cosb
        zy = z * ( cosa - cosb * cosg ) / sing
        zz = np.sqrt( z**2 - zx**2 - zy**2 )
        return np.array([x, 0, 0, y * cosg, y * sing, 0, zx, zy, zz]).reshape((3,3)).astype('float32')
    
    def _gmx2matrix(self, pbc_gmx):
        """Convert gromacs format to matrix format"""
        pbc = np.zeros((3,3))
        pbc[0,0] = pbc_gmx[0]
        pbc[1,1] = pbc_gmx[1]
        pbc[2,2] = pbc_gmx[2]
        pbc[0,1] = pbc_gmx[3] 
        pbc[0,2] = pbc_gmx[4]
        pbc[0,1] = pbc_gmx[5]
        pbc[0,2] = pbc_gmx[6]
        pbc[2,0] = pbc_gmx[7]
        pbc[2,1] = pbc_gmx[8]
        return pbc.astype('float32')
    
    def _matrix2mda(self, pbc_matrix):
        """
        Convert matrix format to mdanalysis format.
        
        This code was taken from MDAnalysis.
        """
        #TODO THIS IS NOT FINISHED!
        x = pbc_matrix[0,:].astype('float64')
        y = pbc_matrix[1,:].astype('float64')
        z = pbc_matrix[2,:].astype('float64')
        
        lx = np.linalg.norm(x)
        ly = np.linalg.norm(y)
        lz = np.linalg.norm(z)
        alpha = np.rad2deg(np.arccos(np.dot(y, z) / (ly * lz)))
        beta = np.rad2deg(np.arccos(np.dot(x, z) / (lx * lz)))
        gamma = np.rad2deg(np.arccos(np.dot(x, y) / (lx * ly)))
        box = np.array([lx, ly, lz, alpha, beta, gamma], dtype=np.float32)
        return box.astype('float32')
    
    def _matrix2gmx(self, pbc_matrix):
        """Convert matrix format to gromacs format"""
        pbc = np.zeros(9)
        pbc[0] = pbc_matrix[0,0]
        pbc[1] = pbc_matrix[1,1]
        pbc[2] = pbc_matrix[2,2]
        pbc[3] = pbc_matrix[0,1]
        pbc[4] = pbc_matrix[0,2]
        pbc[5] = pbc_matrix[0,1]
        pbc[6] = pbc_matrix[0,2]
        pbc[7] = pbc_matrix[2,0]
        pbc[8] = pbc_matrix[2,1]
        return pbc.astype('float32')
    

class Bridges():
    """
    A stupid wrapper class.
    """
    def __init__(self, bridges_graph):
        self.graph = bridges_graph
        

## API Classes
class Containers():
    """
    Creating and handling PCD density containers using a voxel framework.
    
    All units are converted to nm (SI) except for the MDAnalysis objects which use Angstrom,
    as is default in MDAnalysis.
    """
    def __init__(self, atomgroup, resolution=0.5, blur_amount=1, plotting=False, rendering=False, images=False):
        # Instantiating atributes.
        self._atomgroup = atomgroup
        self._resolution = resolution
        self._blur_amount = blur_amount
        self._plotting = plotting
        self._rendering = rendering
        self._images = images
        # Running the containment logic. This data should always remain raw!
        self.data = get_containers(self._atomgroup, self._resolution, self._blur_amount, self._plotting)
        # Obtaining the real used resolution which should be close to the target resolution.
        #  This resoltuion and volume are used for voxel based volume estimation.
        self._real_resolution = (self._atomgroup.dimensions[:3] / self.data['relabeled_combined_label_array'].shape)*0.1
        self._voxel_volume = np.prod(self._real_resolution)
        # Creating and expanding the containment graph (this is a copy).
        self.containment_graph = self._annotate_containment_graph()
        self._raw_containment_graph = self.containment_graph.copy()
        # Final prints
        print('Done!')
        
    def __str__(self):
        nnodes = len(self.containment_graph.nodes())
        return f'{nnodes} containers generated from an atomgroup with {len(self._atomgroup)} atoms, a voxel size of {self._resolution}, and a blur of {self._blur_amount}.'
        
    def reset(self):
        """
        Resets the containment graph from the original.
        """
        self.containment_graph = self._raw_containment_graph
    
    def get_components(self):
        """
        Returns the components in the containment graph.
        """
        return list(self.containment_graph.nodes())
    
    def _annotate_containment_graph(self):
        """
        Returns the containment graph with extra values from the output_dict of get_containers.

        extra values:
        - atomgroup
        - size
        - depth
        - color (normalized size)
        """
        print('Annotating the containment graph...')
        containment_graph = self.data['containment_graph'].copy()
        # Add atomgroups to containers
        atomgroups = _get_container_atomgroups(self.data)
        nx.set_node_attributes(containment_graph, atomgroups, name="atoms")
        # Add atomgroup sizes explicitly
        sizes = _get_atomgroup_sizes(atomgroups)
        nx.set_node_attributes(containment_graph, sizes, name="size")
        # Add the normalized sizes as a color value (0-1).
        normalized_sizes = _get_normalized_sizes(sizes)
        nx.set_node_attributes(containment_graph, normalized_sizes, name="color")
        # Explicitly add the depth of the node in the tree.
        depths = _get_all_depths(containment_graph)
        nx.set_node_attributes(containment_graph, depths, name="depth")
        return containment_graph
    
    def collapse_small_nodes(self, cutoff=50, key='size', self_loops=False, in_place=False):
        """
        Returns a collapsed containment graph, where all nodes smaller than the cutoff 
        are merged with their upstream container.
        """
        containment_graph = self.containment_graph.copy()
        sizes_dict = nx.get_node_attributes(self.containment_graph, key)
        for node in sizes_dict:
            if sizes_dict[node] < cutoff:
                print(f'Size check passed {sizes_dict[node]} < {cutoff}')
                try:
                    parent = list(containment_graph.in_edges(node))[0][0]
                except TypeError:
                    continue
                # Collapsing the graph.
                print('Collapsing!')
                containment_graph = nx.contracted_nodes(containment_graph, parent, node, self_loops)
        if in_place:
            self.containment_graph = containment_graph
        return containment_graph
    
    
    def get_downstream_nodes(self, targets, nodes=[], init=True):
        """
        Returns the node list from the selected node towards the leaves including self.
        
        There is something strange going on, if I do not pass the init argument, I get
        stacking results if I run this function multiple times. I have no idea why that is,
        as the default value is [] and it should be instantiated with an empty input.
        """
        if init:
            nodes = []
        new_targets = []
        for target in targets:
            nodes.append(target)
            temp_targets = [out_edge[1] for out_edge in list(self.containment_graph.out_edges(target))]
            if len(temp_targets) > 0:
                new_targets += [x for x in temp_targets]
        if len(new_targets) > 0:
            self.get_downstream_nodes(new_targets, nodes, init=False)
        return nodes
    
    def get_upstream_nodes(self, targets, nodes=[], init=True):
        """
        Returns the node list from the selected node towards the leaves including self.
        
        There is something strange going on, if I do not pass the init argument, I get
        stacking results if I run this function multiple times. I have no idea why that is,
        as the default value is [] and it should be instantiated with an empty input.
        """
        if init:
            nodes = []
        new_targets = []
        for target in targets:
            nodes.append(target)
            temp_targets = [in_edge[0] for in_edge in list(self.containment_graph.in_edges(target))]
            if len(temp_targets) > 0:
                new_targets += [x for x in temp_targets]
        if len(new_targets) > 0:
            self.get_upstream_nodes(new_targets, nodes, init=False)
        return nodes
    
    def get_volume(self, nodes):
        """
        Returns the voxel volume of the nodes.
        """
        volume = 0
        counts_dict = dict(zip(*np.unique(self.data['relabeled_combined_label_array'], return_counts=True)))
        for node in nodes:
            volume += counts_dict[node]
        volume *= self._voxel_volume
        return volume
    
    def get_atomgroup_from_nodes(self, nodes, b_factor=True, residue=True):
        """
        Takes a node list and returns the merged atomgroup.
        
        The node ids can be written to the b-factor (default). The b-factor is taken
        from the voxel mask, therefore a residue can lie in different masks and have
        multiple b-factors. If residues is True, the most prevalent non-zero b-factor
        is picked per residue. This also returns complete residues even if they are only
        partially in the atomgroup.
        """
        # Instantiate the b_factors if they are not present
        if b_factor:
            universe = self._atomgroup.universe
            try:
                self._atomgroup.tempfactors
            except AttributeError:
                universe.add_TopologyAttr(
                    mda.core.topologyattrs.Tempfactors(np.zeros(len(universe.atoms))))
                
        atomgroup = self.containment_graph.nodes(data=True)[nodes[0]]['atoms']
        if b_factor:
            atomgroup.tempfactors = nodes[0]
        for node in nodes[1:]:
            target_atomgroup = self.containment_graph.nodes(data=True)[node]['atoms']
            atomgroup = atomgroup.union(target_atomgroup)
            target_atomgroup.tempfactors = node
        if residue:
            atomgroup = atomgroup.residues.atoms
            fix_bfactor_per_residue(atomgroup)
        return atomgroup
    
    def get_root_nodes(self):
        """
        Returns the root nodes of the containment graph.
        """
        return [node for node in self.containment_graph.nodes if 
                self.containment_graph.in_degree(node) == 0]
    
    
    def get_leaf_nodes(self):
        """
        Returns the leaf nodes of the containment graph.
        """
        return [node for node in self.containment_graph.nodes if 
                self.containment_graph.out_degree(node) == 0]
    
    def render(self, prefix='test_container_', ftype='png'):
        """
        Renders the cotainment graph nodes.
        """
        write_containment(
            self._atomgroup.universe, 
            self.data['all_voxels'], 
            self.data['relabeled_combined_label_array'],
            self.containment_graph,
            prefix,
            ftype,
        )
        
    def load_renders(self, prefix='test_container_', ftype='png'):
        """
        Loads images which can be used for plotting the containment graph.
        """
        # Returning the images.
        self._images = {}
        path = os.getcwd()
        for label in self.containment_graph.nodes():
            image = f'{prefix}{label}.{ftype}'
            self._images[label] = f'{path}/{image}'
        
    def plot(
        self,
        key='size',
        size_label='color', 
        name='containment.html', 
        height='800px', 
        width='800px', 
        directed=True,  
        scale=50,
    ):
        """
        Draws the containment graph using pvnet.

        The nodes and edges are red and blue for the postive and negative labels
        respectively. Edges are colored by their container and point toward the 
        containee.

        Parameters
        ----------
        containment_graph : networkx.classes.digraph.DiGraph
            The containment graph.
        sizes : dict, optional
            The amount of elements in each label. The default is None.
        name : TYPE, optional
            DESCRIPTION. The default is 'containment.html'.
        height : int, optional
            The height of the graph. The default is '800px'.
        width : int, optional
            The width of the graph. The default is '800px'.

        Returns
        -------
        Graph displayer.

        """
        # Instantiate the graph.
        net = pvnet.Network(notebook=True, directed=directed, height=height, 
                            width=width)
        
        images = self._images

        containment_graph = self.containment_graph
        # Add the nodes from the containment graph and set their sie and color.
        for node in containment_graph.nodes:
            if not images:
                size = containment_graph.nodes(data=True)[node][size_label] * scale
            if int(node) >= 0:
                if images:
                    net.add_node(int(node), label=f'{node}', color='blue', size=scale, shape='image', image =images[node])
                else:
                    net.add_node(int(node), label=f'{node}', color='blue', size=size)
            elif int(node < 0):
                if images:
                    net.add_node(int(node), label=f'{node}', color='red', size=scale, shape='image', image =images[node])
                else:
                    net.add_node(int(node), label=f'{node}', color='red', size=size)
        # Add the edges in a directed manner (container -> containee).
        for edge in containment_graph.edges(data=True):
            net.add_edge(int(edge[0]), int(edge[1]))
        # Set physics for graph rendering.
        opts = '''
            var options = {
              "physics": {
                "forceAtlas2Based": {
                  "gravitationalConstant": -300,
                  "centralGravity": 0.11,
                  "springLength": 100,
                  "springConstant": 0.09,
                  "avoidOverlap": 1
                },
                "minVelocity": 3.75,
                "solver": "forceAtlas2Based",
                "timestep": 0.05
              }
            }
        '''
        net.set_options(opts)
        return net.show(name)
    
    def _create_containment_dictionary(self):
        """
        Returns the containment in a flat dictionary form.
    
        e.g.
        -4 : -4, 1, -2
        """
        containment_dictionary = {}
        for node in self.get_components():
            containment_dictionary[node] = self.get_downstream_nodes([node])
        return containment_dictionary

    def _write_tcl_containment(self, containment_dictionary, fname='containment.vmd'):
        """
        Writes the containment dictionary as VMD selection macros.
        """
        with open(fname, 'w') as f:
            for container in containment_dictionary.keys():
                containment_string = ''.join([f'\'{x}\' ' for x in containment_dictionary[container]])
                if container >= 0 :
                    f.write(f"atomselect macro cont_p{container} \"beta {containment_string}\"\n")
                else:
                    f.write(f"atomselect macro cont_n{abs(container)} \"beta {containment_string}\"\n")

    def _write_tcl_visualization(self, containment_dictionary, fname='containment.vmd'):
        """
        Appends the QuickSurf transparent selection style to the VMD file.
        """
        with open(fname, 'a') as f:
            f.write(f'\nmol delrep 0 0\n')
            for idx, container in enumerate(containment_dictionary.keys()):
                f.write(f'mol addrep 0\n')
                if container >= 0:
                    f.write(f'mol modselect {idx} 0 cont_p{container}\n')
                else:
                    f.write(f'mol modselect {idx} 0 cont_n{abs(container)}\n')
                f.write(f'mol modstyle {idx} 0 QuickSurf 2.500000 0.500000 1.000000 1.000000\n')
                f.write(f'mol modmaterial {idx} 0 Transparent\n')
                f.write(f'mol modcolor {idx} 0 ColorID {idx}\n')    


    def write_components(self, 
                         fname_struc='containment.pdb', 
                         fname_vmd='containment.vmd',
                         add_style=True,
                         residue=False):
        """
        Writes the complete structure as a PDB with the components annotated
        in the beta field. Also adds an additional VMD TCL file which sets
        some basic rendering settings (style) and adds the containment hierarchy
        to the selection syntax.
        
        i.e.
        $vmd cont_p1 and not cont_n1
        
        Where p stands for positive and n for negative, as VMD does not 
        allow for the dash sign in variable names.
        
        Residue can be set to True, to take the dominant component per residue,
        this is pretty slow in MDA and usually not required. Turning this off
        sets the component ID per atom.

        Returns
        -------
        None.

        """
        # Write the system file with the beta factor.
        atomgroup = self.get_atomgroup_from_nodes(
            self.get_components(),
            b_factor=True,
            residue=residue)
        atomgroup.write(fname_struc)
        # Bake the containment dictionary
        containment_dict = self._create_containment_dictionary()
        # Write the dicitonary to TCL
        self._write_tcl_containment(containment_dict, fname_vmd)
        # Write the visualization
        self._write_tcl_visualization(containment_dict, fname_vmd)
    
        
def get_atomgroup_composition(atomgroup, molar=False):
    """
    Returns the composition of component of atomgroup as a dict 
    {resname : count}.
    
    Returns
    -------
    composition : dict
        The resnames are keys and the count of the resname in the atoms
        is the amount (i.e. multiple atoms in a residue count double).
        Unless molar is set to True, which will result in each residue
        only being counter once.
    """
    if molar:
        composition = dict(zip(
            *np.unique(atomgroup.residues.resnames, return_counts=True)))
    else:
        composition = dict(zip(
            *np.unique(atomgroup.resnames, return_counts=True)))
    return composition