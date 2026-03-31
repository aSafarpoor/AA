import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.colors as colors
from networkx.algorithms import community
from matplotlib.patches import FancyArrowPatch
# ================================================================
# Graph I/O
# ================================================================

def save_graph(graph, path):
    with open(path, "wb") as f:
        pickle.dump(graph, f)


def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ================================================================
# Small controlled graph for opinion dynamics
# ================================================================

def build_small_directed_graph(save_graph_flag=False, graph_path=""):

    G = nx.Graph()

    num_cliques = 9
    clique_size = 5
    total_nodes = num_cliques * clique_size

    G.add_nodes_from(range(total_nodes))

    # -------------------------------
    # Build disconnected cliques
    # -------------------------------
    for c in range(num_cliques):
        start = c * clique_size
        nodes = range(start, start + clique_size)
        for i in nodes:
            for j in nodes:
                if i != j:
                    G.add_edge(i, j)

    # -------------------------------
    # Stubbornness pattern (size = 5)
    # -------------------------------
    stubbornness_vals = [0.0, 0.25, 0.5, 0.75, 1.0]

    # -------------------------------
    # First 5 cliques: uniform opinions
    # -------------------------------
    uniform_opinions = [1.0, 0.2, 0.0, -0.2, -1.0]

    for c in range(5):
        start = c * clique_size
        opinion = uniform_opinions[c]

        for i, s in zip(range(start, start + clique_size), stubbornness_vals):
            G.nodes[i]["opinion"] = opinion
            G.nodes[i]["stubbornness"] = s
            G.nodes[i]["activeness"] = 0.2

    # ----------------------------------------
    # Clique 6: one strong positive, others mild positive
    # ----------------------------------------
    start = 5 * clique_size
    mixed = [1.0, 0.2, 0.2, 0.2, 0.2]

    for i, op, s in zip(range(start, start + clique_size), mixed, stubbornness_vals):
        G.nodes[i]["opinion"] = op
        G.nodes[i]["stubbornness"] = s
        G.nodes[i]["activeness"] = 1.0 if op == 1.0 else 0.0

    # ----------------------------------------
    # Clique 7: one strong negative, others mild negative
    # ----------------------------------------
    start = 6 * clique_size
    mixed = [-1.0, -0.2, -0.2, -0.2, -0.2]

    for i, op, s in zip(range(start, start + clique_size), mixed, stubbornness_vals):
        G.nodes[i]["opinion"] = op
        G.nodes[i]["stubbornness"] = s
        G.nodes[i]["activeness"] = 1.0 if op == -1.0 else 0.0

    # ----------------------------------------
    # Clique 8: one strong positive, others mild negative
    # ----------------------------------------
    start = 7 * clique_size
    mixed = [1.0, -0.2, -0.2, -0.2, -0.2]

    for i, op, s in zip(range(start, start + clique_size), mixed, stubbornness_vals):
        G.nodes[i]["opinion"] = op
        G.nodes[i]["stubbornness"] = s
        G.nodes[i]["activeness"] = 1.0 if op == 1.0 else 0.0

    # ----------------------------------------
    # Clique 9: one strong negative, others mild positive
    # ----------------------------------------
    start = 8 * clique_size
    mixed = [-1.0, 0.2, 0.2, 0.2, 0.2]

    for i, op, s in zip(range(start, start + clique_size), mixed, stubbornness_vals):
        G.nodes[i]["opinion"] = op
        G.nodes[i]["stubbornness"] = s
        G.nodes[i]["activeness"] = 1.0 if op == -1.0 else 0.0

    # ----------------------------------------
    if save_graph_flag:
        print(f"Saving graph to {graph_path}")
        save_graph(G, graph_path)

    return G


# ================================================================
# Statistics
# ================================================================

def variance_computer(opinion_history):
    nodes = sorted(opinion_history.keys())
    min_len = min(len(opinion_history[n]) for n in nodes)
    opinion_matrix = np.array([opinion_history[n][:min_len] for n in nodes])
    return np.var(opinion_matrix, axis=0)


# ================================================================
# Visualization
# ================================================================

def plot_opinion_evolution(opinion_history, stubbornness, fig_name, topic=""):

    nodes = sorted(opinion_history.keys())
    min_len = min(len(opinion_history[n]) for n in nodes)

    opinion_matrix = np.array([opinion_history[n][:min_len] for n in nodes])
    time_steps = np.arange(min_len)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

  

    cmap = LinearSegmentedColormap.from_list(
        "blue_red", [(0, 0, 1), (1, 0, 0)], N=256
    )

    for i, node in enumerate(nodes):
        s = min(1.0, max(0.0, stubbornness[node]))
        ax1.plot(
            time_steps,
            opinion_matrix[i],
            color=cmap(s),
            linewidth=1,
            alpha=0.7,
        )

    ax1.set_ylabel("Opinion")
    ax1.set_title(f"Opinion Evolution: {topic}")
    ax1.set_ylim(-1, 1)
    ax1.grid(True, alpha=0.3, linestyle=":")

    sm = cm.ScalarMappable(
        cmap=cmap,
        norm=colors.Normalize(vmin=0, vmax=1),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax1, label="Stubbornness")

    variance = np.var(opinion_matrix, axis=0)
    ax2.plot(time_steps, variance, linewidth=2)
    ax2.fill_between(time_steps, 0, variance, alpha=0.2)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Variance")
    ax2.set_title("Opinion Polarization")
    ax2.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()
    plt.savefig(fig_name, dpi=300, bbox_inches="tight")
    plt.close()

    return variance



# ================================================================
# General Graph-related Function and assigning opinions
# ================================================================

def minimal_graph_shower(G, pos=None, return_pos=False): 
    if pos is None:
        pos = nx.spring_layout(G, seed=42, k=0.15)

    fig, ax = plt.subplots()


    byr = LinearSegmentedColormap.from_list(
        "byr",
        ["#1116ac", "#f6c431", "#b2182b"],  # blue → yellow → red
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=25,
        node_color=[G.nodes[n]["opinion"] for n in G.nodes()],
        cmap=byr,
        vmin=-1,
        vmax=1,
        alpha=1,
        ax=ax,
    )

    for u, v in G.edges():
        patch = FancyArrowPatch(
            pos[u],
            pos[v],
            connectionstyle="arc3,rad=0.15",
            arrowstyle="-",
            linewidth=0.4,
            color="0.45",
            alpha=0.55,
        )
        ax.add_patch(patch)

    ax.set_axis_off()
    plt.show()

    if return_pos:
        return pos

def normalize_opinions(G): 
    vals = np.array([G.nodes[n]["opinion"] for n in G.nodes()])

    vals = vals - vals.mean()

    max_abs = np.max(np.abs(vals))
    if max_abs > 0:
        vals = vals / max_abs

    for n, v in zip(G.nodes(), vals):
        G.nodes[n]["opinion"] = float(np.clip(v, -1.0, 1.0))


def compute_graph_statistics(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()

    components = list(nx.connected_components(G))
    gcc = max(components, key=len)

    avg_degree = 2 * m / n
    avg_clustering = nx.average_clustering(G)
    assortativity = nx.degree_assortativity_coefficient(G)

    gcc_sub = G.subgraph(gcc)
    avg_shortest_path = nx.average_shortest_path_length(gcc_sub)
    diameter = nx.diameter(gcc_sub)

    comms = list(community.greedy_modularity_communities(G))
    modularity = community.modularity(G, comms)

    opinions = np.array([G.nodes[n]["opinion"] for n in G.nodes()])

    print("Graph statistics")
    print(f"Nodes: {n}")
    print(f"Edges: {m}")
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Connected: {nx.is_connected(G)}")
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

# ================================================================
# Read FB subgraph and assign opinions
# ================================================================


def build_My_FB(save_graph_flag, graph_path,random_seed, draw_flag = False, stubborn_mu=0.65, stubborn_sigma=0.20, activeness_alpha=0.6, activeness_beta=3.0):  

    if random_seed > 9:
        raise ValueError(f"random seed is {random_seed} should be less than 10")
    
    with open(f'FB_Edges/FB_{random_seed}_edges.pkl', "rb") as f:
        edges = pickle.load(f)
    G = nx.Graph()
    G.add_edges_from(edges)

    opinions = assign_community_opinions(G, seed=random_seed)
    for n in G.nodes():
        G.nodes[n]["opinion"] = opinions[n]
    normalize_opinions(G)

    stubbornness = np.random.normal(stubborn_mu, stubborn_sigma, n)
    stubbornness = np.clip(stubbornness, 0.0, 1.0)
    activeness = np.random.beta(activeness_alpha, activeness_beta, n)

    for i, node in enumerate(G.nodes()):
        G.nodes[node]["stubbornness"] = float(stubbornness[i])
        G.nodes[node]["activeness"] = float(activeness[i])


    compute_graph_statistics(G)

    if draw_flag:
        pos = minimal_graph_shower(G, return_pos=True)

    if save_graph_flag:
        save_graph(G, graph_path)

    return G

# ================================================================
# Top k cev=ntrality functions
# ================================================================

def top_k_degree_nodes(G: nx.Graph, k: int):

    if k <= 0:
        return []
    if k > G.number_of_nodes():
        k = G.number_of_nodes()

    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)

    return [node for node, _ in sorted_nodes[:k]]

def top_k_betweenness_nodes(G: nx.Graph, k: int, normalized: bool = True):
  
    if k <= 0:
        return []
    if k > G.number_of_nodes():
        k = G.number_of_nodes()   

    bet = nx.betweenness_centrality(G, normalized=normalized)

    sorted_nodes = sorted(bet, key=bet.get, reverse=True)

    return sorted_nodes[:k]


# ================================================================
# Read build_My_RedditTwitter subgraph and assign opinions
# ================================================================
def build_My_RedditTwitter(save_graph_flag, graph_path,random_seed, graph_type, draw_flag = False, stubborn_mu=0.65, stubborn_sigma=0.20, activeness_alpha=0.4, activeness_beta=3.0):  

    if random_seed > 9:
        raise ValueError(f"random seed is {random_seed} should be less than 10")
    
    with open(f'RedditTwitter/{graph_type.lower()}_{random_seed}_nx_Graph.pkl', "rb") as f:
        G = pickle.load(f)
     
    n = len(G.nodes)

    stubbornness = np.random.normal(stubborn_mu, stubborn_sigma, n)
    stubbornness = np.clip(stubbornness, 0.0, 1.0)
    activeness = np.random.beta(activeness_alpha, activeness_beta, n) # activeness_alpha=0.6, activeness_beta=3.0

    for i, node in enumerate(G.nodes()):
        G.nodes[node]["stubbornness"] = float(stubbornness[i])
        G.nodes[node]["activeness"] = float(activeness[i])


    compute_graph_statistics(G)

    if draw_flag:
        pos = minimal_graph_shower(G, return_pos=True)

    if save_graph_flag:
        save_graph(G, graph_path)

    return G


