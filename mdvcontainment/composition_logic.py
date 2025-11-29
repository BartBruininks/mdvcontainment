import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def get_compositions(containment, nodes=False, mode='resnames'):
    """
    Returns the compositions dict.

    * 'resname' can be used to get the residuenames count for all atoms,
    * 'names' uses the atomnames per atom in a the atomgroup.
    * 'molar' counts resnames per residue.
    """
    compositions_dict = {}
    # Iterate over all components
    if nodes is False:
        nodes = containment.voxel_containment.nodes
    for node in nodes:
        # Obtain the per component atomgroup
        atomgroup = containment.get_atomgroup_from_nodes([node])
        # Add the composition to the label dict
        if mode == 'resnames':
            selector = atomgroup.resnames # can be changed to names as well
        elif mode == 'names':
            selector = atomgroup.names # can be changed to names as well
        elif mode == 'molar':
            selector = atomgroup.residues.resnames # can be changed to names as well
        else:
            raise "Please specify a valic mode ('resnames', 'names' or 'molar')"
        composition = dict(zip(*np.unique(selector, return_counts=True)))
        compositions_dict[node] = composition
    return compositions_dict

def combine_dicts(dictionaries):
    """
    Returns the combined counts in the dictionaries.
    """
    # Combine the dictionaries
    combined_counter = Counter()
    for dictionary in dictionaries.values():
        combined_counter.update(dictionary)
    # Convert back to a regular dictionary if needed
    return dict(combined_counter)

def get_unique_labels(universe, mode='resnames'):
    """
    Returns the set of unique labels in the compositions dict.
    """
    # Create a consistent color map for labels
    if mode == 'resnames':
        unique_labels = set(universe.atoms.resnames)
    elif mode == 'names':
        unique_labels = set(universe.atoms.names)
    elif mode == 'molar':
        unique_labels = set(universe.residues.resnames) 
    else:
        raise "Please specify a valic mode ('resnames', 'names' or 'molar')"
    return unique_labels

def get_color_mapping(unique_labels):
    """
    Returns a color map, mapping each label to a well separated color.
    """
    # Set the color map
    colors = plt.cm.tab20c(range(len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    return color_map

def plot_pie_chart(data, ax, color_map, cutoff=10):
    labels = list(data.keys())
    sizes = list(data.values())
    colors = [color_map[label] for label in labels]

    # Calculate percentages
    total = sum(sizes)
    percentages = [size / total * 100 for size in sizes]

    # Filter labels and percentages for the pie chart display
    display_labels = [label if pct >= cutoff else '' for label, pct in zip(labels, percentages)]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=display_labels, colors=colors, autopct=lambda pct: f'{pct:.1f}%' if pct >= cutoff else '', startangle=140
    )

    # Customize text properties for visibility
    for text in autotexts:
        text.set_color('white')

    # Find the maximum label length for alignment
    max_label_length = 0
    for label in labels:
        if len(label) > max_label_length:
            max_label_length = len(label)

    # Creat the aligned legend labels and percentages
    legend_labels = [f'{label}{" "*(max_label_length-len(label))} {pct:5.1f}%' for label, pct in zip(labels, percentages)]

     # Add a legend with a monospaced font
    font_properties = {'family': 'monospace'}
    ax.legend(wedges, legend_labels, title="Labels", loc="center left", bbox_to_anchor=(1.0, 0, 0.5, 1), prop=font_properties)
