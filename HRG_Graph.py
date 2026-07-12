import random
import networkit as nk
import networkx as nx
import numpy as np

from LLM_Utils import (
    save_graph,
    normalize_opinions,
    compute_graph_statistics,
    assign_community_opinions,
)


def generate_hrg(
    n=50,
    avg_degree=4.5,
    gamma=6.0,
    temperature=0.30,
    stubborn_mu=0.65,
    stubborn_sigma=0.20,
    activeness_alpha=0.6,
    activeness_beta=3.0,
    seed=42,
):
    if seed is not None:
        nk.setSeed(seed, True)
        np.random.seed(seed)
        random.seed(seed)

    generator = nk.generators.HyperbolicGenerator(
        n=n,
        k=avg_degree,
        gamma=gamma,
        T=temperature,
    )
    G_nk = generator.generate()
    G_nx = nk.nxadapter.nk2nx(G_nk)

    edges = list(G_nx.edges())

    G = nx.DiGraph()
    G.add_nodes_from(G_nx.nodes())
    G.add_edges_from(edges)
    G.add_edges_from([(v, u) for u, v in edges])

    stubbornness = np.random.normal(
        stubborn_mu,
        stubborn_sigma,
        n,
    )
    stubbornness = np.clip(stubbornness, 0.0, 1.0)

    activeness = np.random.beta(
        activeness_alpha,
        activeness_beta,
        n,
    )

    nodes = list(G.nodes())
    for i, node in enumerate(nodes):
        G.nodes[node]["stubbornness"] = float(stubbornness[i])
        G.nodes[node]["activeness"] = float(activeness[i])

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
    G = generate_hrg(
        seed=random_seed,
        n=50,
        avg_degree=4,
        gamma=5,
        temperature=0.25,
        stubborn_mu=0.65,
        stubborn_sigma=0.20,
        activeness_alpha=0.6,
        activeness_beta=3.0,
    )

    G = force_connected(G, seed=random_seed)

    opinions = assign_community_opinions(
        G,
        seed=random_seed,
    )

    nodes = list(G.nodes())
    for node in nodes:
        G.nodes[node]["opinion"] = opinions[node]

    normalize_opinions(G)
    compute_graph_statistics(G)

    if save_graph_flag:
        save_graph(G, graph_path)

    return G