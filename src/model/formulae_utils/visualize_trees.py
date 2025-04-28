import matplotlib.pyplot as plt
import networkx as nx
import torch


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


if __name__ == "__main__":
    # Example usage
    edge_index = torch.tensor([[0, 1, 2, 3, 3], [1, 2, 3, 0, 1]])  # [2, num_edges]
    labels = {0: "A", 1: "B", 2: "C", 3: "D"}  # Node labels
    plot_graph(edge_index, labels)
