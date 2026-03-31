import random
import networkx as nx
import numpy as np

from LLM_Utils import (
    save_graph,
    minimal_graph_shower,
    normalize_opinions,
    compute_graph_statistics,
    assign_community_opinions,
)
import matplotlib.pyplot as plt


def generate_ssbm(
    stubborn_mu=0.65,
    stubborn_sigma=0.20,
    activeness_alpha=0.6,
    activeness_beta=3.0,
    seed=42,
    verbose = False,
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    num_communities = 4
    nodes_per_comm = 10
    sizes = [nodes_per_comm] * num_communities

    p = 0.5
    q = 0.01

    probs = [[p if i == j else q for j in range(num_communities)]
            for i in range(num_communities)]

    G = nx.stochastic_block_model(sizes, probs, seed=seed)

    means = {0: 0.1, 1: 0.1, 2: -0.1, 3: -0.1}
    var = 0.001
    std = np.sqrt(var)

    opinions = {}
    for node in G.nodes():
        comm = G.nodes[node]['block']
        val = np.random.normal(means[comm], std)
        val = np.clip(val, -1.0, 1.0)
        opinions[node] = float(val)

    if verbose:
        node_colors = [G.nodes[n]['block'] for n in G.nodes()]

        pos = nx.spring_layout(G, seed=seed)

        labels = {n: f"{opinions[n]:.2f}" for n in G.nodes()}

        plt.figure(figsize=(6, 6), dpi=100)
        nx.draw(
            G,
            pos,
            node_color=node_colors,
            cmap=plt.cm.Set2,
            node_size=200,
            edge_color='gray',
            width=0.5,
            with_labels=False
        )

        nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)

        plt.title("SBM with opinions")
        plt.tight_layout()
        plt.show()

    n = len(G.nodes())
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['opinion'] = opinions[node]
        G.nodes[node]["stubbornness"] = 0.5
        G.nodes[node]["activeness"] = 0.5 

    return G


def force_connected(G, seed=42):
    random.seed(seed)
    components = list(nx.connected_components(G))
    if len(components) <= 1:
        return G

    components = [list(c) for c in components]
    for i in range(len(components) - 1):
        u = random.choice(components[i])
        v = random.choice(components[i + 1])
        G.add_edge(u, v)

    return G


def Main_Runner(
    save_graph_flag=False,
    draw_flag=False,
    graph_path="",
    random_seed=42,
):
    G = generate_ssbm(seed=random_seed)
    G = force_connected(G, seed=random_seed)

    if save_graph_flag:
        save_graph(G, graph_path)
        
    return G