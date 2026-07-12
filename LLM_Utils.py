import pickle
import networkx as nx
import numpy as np
from networkx.algorithms import community

def save_graph(graph, path):
    with open(path, "wb") as f:
        pickle.dump(graph, f)


def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_small_directed_graph(save_graph_flag=False, graph_path=""):

    G = nx.DiGraph()

    num_cliques = 9
    clique_size = 5
    total_nodes = num_cliques * clique_size

    G.add_nodes_from(range(total_nodes))

    for c in range(num_cliques):
        start = c * clique_size
        nodes = list(range(start, start + clique_size))
        for i in nodes:
            for j in nodes:
                if i != j:
                    G.add_edge(i, j)

    stubbornness_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    uniform_opinions = [1.0, 0.2, 0.0, -0.2, -1.0]

    for c in range(5):
        start = c * clique_size
        opinion = uniform_opinions[c]
        for node, s in zip(range(start, start + clique_size), stubbornness_vals):
            G.nodes[node]["opinion"] = opinion
            G.nodes[node]["stubbornness"] = s
            G.nodes[node]["activeness"] = 0.2

    start = 5 * clique_size
    mixed = [1.0, 0.2, 0.2, 0.2, 0.2]
    for node, op, s in zip(range(start, start + clique_size), mixed, stubbornness_vals):
        G.nodes[node]["opinion"] = op
        G.nodes[node]["stubbornness"] = s
        G.nodes[node]["activeness"] = 1.0 if op == 1.0 else 0.0

    start = 6 * clique_size
    mixed = [-1.0, -0.2, -0.2, -0.2, -0.2]
    for node, op, s in zip(range(start, start + clique_size), mixed, stubbornness_vals):
        G.nodes[node]["opinion"] = op
        G.nodes[node]["stubbornness"] = s
        G.nodes[node]["activeness"] = 1.0 if op == -1.0 else 0.0

    start = 7 * clique_size
    mixed = [1.0, -0.2, -0.2, -0.2, -0.2]
    for node, op, s in zip(range(start, start + clique_size), mixed, stubbornness_vals):
        G.nodes[node]["opinion"] = op
        G.nodes[node]["stubbornness"] = s
        G.nodes[node]["activeness"] = 1.0 if op == 1.0 else 0.0

    start = 8 * clique_size
    mixed = [-1.0, 0.2, 0.2, 0.2, 0.2]
    for node, op, s in zip(range(start, start + clique_size), mixed, stubbornness_vals):
        G.nodes[node]["opinion"] = op
        G.nodes[node]["stubbornness"] = s
        G.nodes[node]["activeness"] = 1.0 if op == -1.0 else 0.0

    if save_graph_flag:
        save_graph(G, graph_path)

    return G

def variance_computer(opinion_history):
    nodes = sorted(opinion_history.keys())
    min_len = min(len(opinion_history[n]) for n in nodes)
    opinion_matrix = np.array(
        [opinion_history[n][:min_len] for n in nodes]
    )
    return np.var(opinion_matrix, axis=0)

def normalize_opinions(G):
    vals = np.array([G.nodes[n]["opinion"] for n in G.nodes()])
    vals = vals - vals.mean()
    max_abs = np.max(np.abs(vals))
    if max_abs > 0:
        vals = vals / max_abs
    nodes = list(G.nodes())
    for node, v in zip(nodes, vals):
        G.nodes[node]["opinion"] = float(np.clip(v, -1.0, 1.0))


def compute_graph_statistics(G):
    G_u = G.to_undirected()
    n = G_u.number_of_nodes()
    m = G_u.number_of_edges()

    components = list(nx.connected_components(G_u))
    gcc = max(components, key=len)

    avg_degree = 2 * m / n
    avg_clustering = nx.average_clustering(G_u)
    assortativity = nx.degree_assortativity_coefficient(G_u)

    gcc_sub = G_u.subgraph(gcc)
    avg_shortest_path = nx.average_shortest_path_length(gcc_sub)
    diameter = nx.diameter(gcc_sub)

    comms = list(community.greedy_modularity_communities(G_u))
    modularity = community.modularity(G_u, comms)

    opinions = np.array([G.nodes[n]["opinion"] for n in G.nodes()])

    print("Graph statistics")
    print(f"Nodes: {n}")
    print(f"Edges: {m}")
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Connected: {nx.is_connected(G_u)}")
    print(f"GCC size: {len(gcc)} ({len(gcc)/n:.2%})")
    print(f"Average clustering coefficient: {avg_clustering:.3f}")
    print(f"Average shortest path length: {avg_shortest_path:.3f}")
    print(f"Diameter: {diameter}")
    print(f"Degree assortativity: {assortativity:.3f}")
    print(f"Modularity (greedy): {modularity:.3f}")
    print(f"Number of communities: {len(comms)}")
    print(f"Opinion mean: {opinions.mean():.3f}")
    print(f"Opinion min: {opinions.min():.3f}")
    print(f"Opinion max: {opinions.max():.3f}")
    print(f"Opinion variance: {opinions.var():.3f}")


def assign_community_opinions(G, seed=42):
    np.random.seed(seed)
    communities = list(community.greedy_modularity_communities(G))
    k = len(communities)

    means = np.random.normal(0.0, 0.6, k)
    means = np.clip(means, -1.0, 1.0)

    opinions = {}
    for cid, comm in enumerate(communities):
        mu = means[cid]
        vals = np.random.normal(mu, 0.35, len(comm))
        vals = np.clip(vals, -1.0, 1.0)
        for n, v in zip(comm, vals):
            opinions[n] = float(v)

    return opinions


def build_My_FB(
    save_graph_flag,
    graph_path,
    random_seed,
    stubborn_mu=0.65,
    stubborn_sigma=0.20,
    activeness_alpha=0.6,
    activeness_beta=3.0,
):
    if random_seed > 9:
        raise ValueError(
            f"random seed is {random_seed} should be less than 10"
        )

    np.random.seed(random_seed)

    with open(f"FB_Edges/FB_{random_seed}_edges.pkl", "rb") as f:
        edges = pickle.load(f)

    G = nx.DiGraph()
    G.add_edges_from(edges)
    G.add_edges_from([(v, u) for u, v in edges])

    opinions = assign_community_opinions(G, seed=random_seed)

    nodes = list(G.nodes())
    for node in nodes:
        G.nodes[node]["opinion"] = opinions[node]

    normalize_opinions(G)

    n = len(nodes)

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

    for i, node in enumerate(nodes):
        G.nodes[node]["stubbornness"] = float(stubbornness[i])
        G.nodes[node]["activeness"] = float(activeness[i])

    compute_graph_statistics(G)

    if save_graph_flag:
        save_graph(G, graph_path)

    return G


def top_k_degree_nodes(G: nx.DiGraph, k: int):
    if k <= 0:
        return []
    if k > G.number_of_nodes():
        k = G.number_of_nodes()
    sorted_nodes = sorted(
        G.out_degree,
        key=lambda x: x[1],
        reverse=True,
    )
    return [node for node, _ in sorted_nodes[:k]]


def top_k_betweenness_nodes(
    G: nx.DiGraph,
    k: int,
    normalized: bool = True,
):
    if k <= 0:
        return []
    if k > G.number_of_nodes():
        k = G.number_of_nodes()

    bet = nx.betweenness_centrality(
        G,
        normalized=normalized,
        endpoints=False,
    )
    sorted_nodes = sorted(
        bet,
        key=bet.get,
        reverse=True,
    )
    return sorted_nodes[:k]


def rescaler(g):
    opinions = [
        data["opinion"]
        for _, data in g.nodes(data=True)
        if "opinion" in data
    ]

    if len(opinions) == 0:
        return g

    min_op = np.min(opinions)
    max_op = np.max(opinions)

    if max_op > min_op:
        for _, data in g.nodes(data=True):
            if "opinion" in data:
                x = data["opinion"]
                data["opinion"] = 2 * (x - min_op) / (max_op - min_op) - 1
    else:
        for _, data in g.nodes(data=True):
            if "opinion" in data:
                data["opinion"] = 0.0

    return g


def build_My_RedditTwitter(
    save_graph_flag,
    graph_path,
    random_seed,
    graph_type,
    stubborn_mu=0.65,
    stubborn_sigma=0.20,
    activeness_alpha=0.4,
    activeness_beta=3.0,
):
    if random_seed > 9:
        raise ValueError(
            f"random seed is {random_seed} should be less than 10"
        )

    np.random.seed(random_seed)

    with open(
        f"RedditTwitter/{graph_type.lower()}_{random_seed}_nx_Graph.pkl",
        "rb",
    ) as f:
        G = pickle.load(f)

    if not G.is_directed():
        G = G.to_directed()

    if graph_type.lower() == "twitter":
        G = rescaler(G)

    nodes = list(G.nodes())
    n = len(nodes)

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

    for i, node in enumerate(nodes):
        G.nodes[node]["stubbornness"] = float(stubbornness[i])
        G.nodes[node]["activeness"] = float(activeness[i])

    compute_graph_statistics(G)

    if save_graph_flag:
        save_graph(G, graph_path)

    return G