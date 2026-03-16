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


def generate_sbm(
    n_communities=5,
    community_size=10,
    p_in=0.30,
    p_out=0.03,
    stubborn_mu=0.65,
    stubborn_sigma=0.20,
    activeness_alpha=0.6,
    activeness_beta=3.0,
    seed=42,
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    sizes = [community_size] * n_communities
    n = n_communities * community_size


    probs = [[p_out for _ in range(n_communities)]
             for _ in range(n_communities)]
    for i in range(n_communities):
        probs[i][i] = p_in

    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    G = nx.Graph(G)  
    
    stubbornness = np.random.normal(stubborn_mu, stubborn_sigma, n)
    stubbornness = np.clip(stubbornness, 0.0, 1.0)

    activeness = np.random.beta(activeness_alpha, activeness_beta, n)

    for i, node in enumerate(G.nodes()):
        G.nodes[node]["stubbornness"] = float(stubbornness[i])
        G.nodes[node]["activeness"] = float(activeness[i])

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
    G = generate_sbm(seed=random_seed)

    G = force_connected(G, seed=random_seed)

    opinions = assign_community_opinions(G, seed=random_seed)

    for n in G.nodes():
        G.nodes[n]["opinion"] = opinions[n]

    normalize_opinions(G)
    compute_graph_statistics(G)

    if draw_flag:
        minimal_graph_shower(G)

    if save_graph_flag:
        save_graph(G, graph_path)

    return G