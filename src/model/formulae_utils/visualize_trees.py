import matplotlib.pyplot as plt
import networkx as nx
import torch
from networkx.drawing.nx_pydot import graphviz_layout
from graphviz import Digraph



def plot_graph(edge_index, labels=None, node_size=500, font_size=12):
    """
    Plots a graph from an edge_index and an optional dictionary of node labels.

    Args:
        edge_index (torch.Tensor): Tensor of shape [2, num_edges] representing edges.
        labels (dict): Dictionary of labels for nodes {node_index: label}.
        node_size (int): Size of the nodes in the plot.
        font_size (int): Font size for node labels.
    """
    # Convert edge_index to a list of edges
    edges = edge_index.t().tolist()

    # Create a NetworkX graph
    G = nx.DiGraph()
    G.add_edges_from(edges)
    # print(G.nodes)

    # If no labels are provided, use default node indices as labels
    if labels is None:
        labels = {i: i for i in range(max(edge_index.max().item() + 1, len(G.nodes)))}

    for label in labels.keys():
        if label not in G.nodes:
            G.add_node(label)

    # Draw the graph
    pos = nx.spring_layout(G)  # Positions for nodes
    # print(pos.keys())
    # print(labels.keys())
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=node_size, font_size=font_size, node_color="skyblue",
            edge_color="gray")
    plt.title("Graph Visualization")
    plt.show()


def plot_forest(edge_index, x, roots, label_dict=None, title="Batched Trees"):
    """
    Plots multiple rooted trees from a batched graph.

    Parameters:
    - edge_index: torch.Tensor of shape [2, num_edges]
    - x: torch.Tensor of shape [num_nodes, ...]
    - roots: list of root node indices (one per tree)
    - labels: Optional list/dict of node labels (or None to use x[i].item())
    - title: Global title for the plot
    """

    edges = edge_index.t().tolist()
    G = nx.DiGraph()
    G.add_nodes_from(list(range(len(x))))
    G.add_edges_from(edges)

    # Default labels from x if none provided
    if label_dict is None:
        labels = {i: str(x[i].item()) for i in range(x.size(0))}
    else:
        labels = {i: label_dict[emb.item()] for i, emb in enumerate(x)}

    # Find connected components (assumes one tree per component)
    components = list(nx.weakly_connected_components(G))
    subgraphs = [G.subgraph(c).copy() for c in components]

    # Plot each tree separately
    num_trees = len(subgraphs)
    fig, axs = plt.subplots(1, num_trees, figsize=(5 * num_trees, 4))

    if num_trees == 1:
        axs = [axs]

    for i, (subg, ax) in enumerate(zip(subgraphs, axs)):
        try:
            pos = graphviz_layout(subg, prog="dot", root=roots[i].item())
        except:
            pos = nx.spring_layout(subg, seed=42)

        node_labels = {n: labels[n] for n in subg.nodes}
        nx.draw(subg, pos, ax=ax,
                labels=node_labels,
                with_labels=True,
                arrows=True,
                node_size=800,
                node_color='lightblue',
                font_size=10)
        ax.set_title(f"Tree rooted at node {roots[i]}")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_forest_two_batches(edge_index, x, roots, titles, label_dict=None, title="Batched Trees"):
    """
    Plots rooted trees from two batched graphs in a 2-row grid.

    Parameters:
    - edge_index: tuple of two torch.Tensors [2, num_edges]
    - x: tuple of two torch.Tensors [num_nodes, ...]
    - roots: tuple of two lists of root indices
    - label_dict: optional dict mapping x[i].item() to label string
    - title: main plot title
    """

    def build_graph(edges, features):
        G = nx.DiGraph()
        G.add_nodes_from(list(range(len(features))))
        G.add_edges_from(edges)
        return G

    def get_labels(features, label_dict):
        if label_dict is None:
            return {i: str(features[i].item()) for i in range(features.size(0))}
        else:
            return {i: label_dict[feat.item()] for i, feat in enumerate(features)}

    def make_subgraphs(G):
        comps = list(nx.weakly_connected_components(G))
        return [G.subgraph(c).copy() for c in comps]

    edge_index1, edge_index2 = edge_index
    x1, x2 = x
    roots1, roots2 = roots
    titles1, titles2 = titles

    G1 = build_graph(edge_index1.t().tolist(), x1)
    G2 = build_graph(edge_index2.t().tolist(), x2)

    labels1 = get_labels(x1, label_dict)
    labels2 = get_labels(x2, label_dict)

    subgraphs1 = make_subgraphs(G1)
    subgraphs2 = make_subgraphs(G2)

    num_trees = len(roots1)
    fig, axs = plt.subplots(2, num_trees, figsize=(5 * num_trees, 8))
    if num_trees == 1:
        axs = [[axs[0]], [axs[1]]]

    # Plot top row (first batch)
    for i, (subg, ax) in enumerate(zip(subgraphs1, axs[0])):
        pos = graphviz_layout(subg, prog="dot", root=roots1[i].item())

        for k in pos:
            pos[k] = (pos[k][0], -pos[k][1])

        node_labels = {n: labels1[n] for n in subg.nodes}
        nx.draw(subg, pos, ax=ax,
                labels=node_labels,
                with_labels=True,
                arrows=True,
                node_size=4800,
                node_color='lightgreen',
                font_size=20,
                )

        cur_title = rf"$A_{i + 1}^+$"
        ax.set_title("Reach " + cur_title + " = " + titles1[i])

    # Plot bottom row (second batch)
    for i, (subg, ax) in enumerate(zip(subgraphs2, axs[1])):
        # try:
        pos = graphviz_layout(subg, prog="dot", root=roots2[i].item())

        for k in pos:
            pos[k] = (pos[k][0], -pos[k][1])
        # print("A")
        # except:
        #     pos = nx.spring_layout(subg, seed=42)
        node_labels = {n: labels2[n] for n in subg.nodes}
        nx.draw(subg, pos, ax=ax,
                labels=node_labels,
                with_labels=True,
                arrows=True,
                node_size=4800,
                node_color='lightcoral',
                font_size=20)

        cur_title = rf"$A_{i+1}^-$"
        ax.set_title("Avoid " + cur_title + " = " + titles2[i])

    plt.suptitle(title, fontsize=40, fontweight='bold')
    plt.tight_layout()
    plt.show()






if __name__ == "__main__":
    # Example usage
    # edge_index = torch.tensor([[0, 1, 2, 3, 3], [1, 2, 3, 0, 1]])  # [2, num_edges]
    # labels = {0: "A", 1: "B", 2: "C", 3: "D"}  # Node labels
    # plot_graph(edge_index, labels)

    dot = Digraph()

    dot.node('A', 'LTL Task')
    dot.node('B', 'Instructions Sequence')
    dot.node('C', 'Sequence Embedding')
    dot.node('D', 'Environment Observation')
    dot.node('E', 'Final Observation')
    dot.node('F', 'Action')

    # dot.edges(['AB'])
    dot.edge('A', 'B', label=' Büchi Automaton')
    dot.edge('C', 'E')
    dot.edge('B', 'C', label='< <FONT COLOR="red">DeepSets</FONT> + RNN >')
    dot.edge('D', 'E', label="MLP")
    dot.edge('E', 'F', label=" MLP")

    dot.render('flowchart', view=True)  # saves and opens the file
