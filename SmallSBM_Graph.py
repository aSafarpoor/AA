import random
import networkx as nx
import numpy as np

from LLM_Utils import save_graph


def generate_ssbm(
    stubborn_mu=0.65,
    stubborn_sigma=0.20,
    activeness_alpha=0.6,
    activeness_beta=3.0,
    seed=42,
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    num_communities = 4
    nodes_per_comm = 10
    sizes = [nodes_per_comm] * num_communities

    p = 0.5
    q = 0.01

    probs = [
        [p if i == j else q for j in range(num_communities)]
        for i in range(num_communities)
    ]

    G0 = nx.stochastic_block_model(sizes, probs, seed=seed)

    block_attr = nx.get_node_attributes(G0, "block")

    edges = list(G0.edges())

    G = nx.DiGraph()
    G.add_nodes_from(G0.nodes())
    G.add_edges_from(edges)
    G.add_edges_from([(v, u) for u, v in edges])

    # Restore block attribute
    nx.set_node_attributes(G, block_attr, "block")

    means = {0: 0.1, 1: 0.1, 2: -0.1, 3: -0.1}
    std = np.sqrt(0.001)

    opinions = {}
    for node in G.nodes():
        comm = G.nodes[node]["block"]
        val = np.random.normal(means[comm], std)
        val = np.clip(val, -1.0, 1.0)
        opinions[node] = float(val)

    nodes = list(G.nodes())

    for node in nodes:
        G.nodes[node]["opinion"] = opinions[node]
        G.nodes[node]["stubbornness"] = 0.5
        G.nodes[node]["activeness"] = 0.5

    return G


def force_connected(G, seed=42):
    random.seed(seed)

    components = list(nx.weakly_connected_components(G))
    if len(components) <= 1:
        return G

    components = [list(c) for c in components]

    for i in range(len(components) - 1):
        u = random.choice(components[i])
        v = random.choice(components[i + 1])

        G.add_edge(u, v)
        G.add_edge(v, u)

    return G


def Main_Runner(
    save_graph_flag=False,
    graph_path="",
    random_seed=42,
):
    G = generate_ssbm(seed=random_seed)
    G = force_connected(G, seed=random_seed)

    if save_graph_flag:
        save_graph(G, graph_path)

    return G