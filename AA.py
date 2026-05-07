import random
import networkx as nx
import community.community_louvain as community_louvain


def top_k_degree_nodes(G: nx.DiGraph, k: int):
    if k <= 0 or k > G.number_of_nodes():
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
    if k <= 0 or k > G.number_of_nodes():
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


def select_adversarial_nodes(
    graph,
    k,
    change_value=1.0,
    bias_if_odd="pos",
):
    partition = community_louvain.best_partition(
        graph.to_undirected()
    )

    communities = {}
    for node, cid in partition.items():
        communities.setdefault(cid, []).append(node)

    community_info = []
    for cid, nodes in communities.items():
        opinions = [graph.nodes[n]["opinion"] for n in nodes]
        mean_o = sum(opinions) / len(opinions)
        size = len(nodes)
        priority = (1 - abs(mean_o)) * size
        community_info.append((cid, size, mean_o, priority))

    community_mean_dict = {
        cid: mean_o for cid, _, mean_o, _ in community_info
    }

    pos_comms = [info for info in community_info if info[2] >= 0]
    neg_comms = [info for info in community_info if info[2] <= 0]

    k_pos = k // 2
    k_neg = k // 2
    if k % 2 != 0:
        if bias_if_odd == "pos":
            k_pos += 1
        else:
            k_neg += 1

    def allocate_alternating(
        comms_pos, comms_neg, slots_pos, slots_neg
    ):
        pos_sorted = sorted(
            comms_pos, key=lambda x: x[3], reverse=True
        )
        neg_sorted = sorted(
            comms_neg, key=lambda x: x[3], reverse=True
        )

        if len(pos_sorted) == 0 and len(neg_sorted) == 0:
            return {}, {}

        pos_alloc = {cid: 0 for cid, _, _, _ in pos_sorted}
        neg_alloc = {cid: 0 for cid, _, _, _ in neg_sorted}

        pos_idx = 0
        neg_idx = 0
        pos_picked = 0
        neg_picked = 0
        turn_pos = True

        while pos_picked < slots_pos or neg_picked < slots_neg:
            if (
                turn_pos
                and pos_picked < slots_pos
                and len(pos_sorted) > 0
            ):
                cid = pos_sorted[pos_idx][0]
                pos_alloc[cid] += 1
                pos_picked += 1
                pos_idx = (pos_idx + 1) % len(pos_sorted)

            elif (
                not turn_pos
                and neg_picked < slots_neg
                and len(neg_sorted) > 0
            ):
                cid = neg_sorted[neg_idx][0]
                neg_alloc[cid] += 1
                neg_picked += 1
                neg_idx = (neg_idx + 1) % len(neg_sorted)

            if (
                turn_pos
                and (pos_picked >= slots_pos or len(pos_sorted) == 0)
                and neg_picked < slots_neg
            ):
                turn_pos = False
                continue

            if (
                not turn_pos
                and (neg_picked >= slots_neg or len(neg_sorted) == 0)
                and pos_picked < slots_pos
            ):
                turn_pos = True
                continue

            turn_pos = not turn_pos

        pos_alloc = {
            cid: s for cid, s in pos_alloc.items() if s > 0
        }
        neg_alloc = {
            cid: s for cid, s in neg_alloc.items() if s > 0
        }

        return pos_alloc, neg_alloc

    pos_alloc, neg_alloc = allocate_alternating(
        pos_comms, neg_comms, k_pos, k_neg
    )

    selected = []
    selected_set = set()

    def pick_from_community(
        community_nodes, n_pick, community_mean
    ):
        candidates = []
        for node in community_nodes:
            deg = graph.out_degree(node)
            opinion = graph.nodes[node]["opinion"]
            if opinion * community_mean >= 0:
                candidates.append((node, deg, abs(opinion)))

        candidates.sort(key=lambda x: (-x[1], x[2]))

        picked = []
        for node, _, _ in candidates:
            if node not in selected_set and len(picked) < n_pick:
                picked.append(node)
                selected_set.add(node)

        return picked

    for cid, slots in pos_alloc.items():
        selected.extend(
            pick_from_community(
                communities[cid],
                slots,
                community_mean_dict[cid],
            )
        )

    for cid, slots in neg_alloc.items():
        selected.extend(
            pick_from_community(
                communities[cid],
                slots,
                community_mean_dict[cid],
            )
        )

    if len(selected) < k:
        remaining = [
            n for n in graph.nodes if n not in selected_set
        ]
        remaining.sort(
            key=lambda n: graph.out_degree(n), reverse=True
        )
        needed = k - len(selected)
        selected.extend(remaining[:needed])

    return selected


def select_node_AA(
    graph,
    AA_level,
    AA_type,
    AA_k,
    random_seed=42,
):
    random.seed(random_seed)

    if AA_level.startswith("grid2"):
        delta1 = float(AA_level.split("_")[1])
        delta2 = float(AA_level.split("_")[2])
        AAs = select_adversarial_nodes(graph, AA_k)

        for node in AAs:
            o_node = graph.nodes[node]["opinion"]
            if o_node > 0:
                graph.nodes[node]["opinion"] = 1
            elif o_node < 0:
                graph.nodes[node]["opinion"] = -1
            else:
                graph.nodes[node]["opinion"] = random.choice(
                    [-1.0, 1.0]
                )

            graph.nodes[node]["activeness"] = delta1
            graph.nodes[node]["stubbornness"] = delta2

    elif AA_level.startswith("grid"):
        delta = float(AA_level.split("_")[1])
        AAs = select_adversarial_nodes(graph, AA_k)

        for node in AAs:
            o_node = graph.nodes[node]["opinion"]
            if o_node > 0:
                graph.nodes[node]["opinion"] = 1
            elif o_node < 0:
                graph.nodes[node]["opinion"] = -1
            else:
                graph.nodes[node]["opinion"] = random.choice(
                    [-1.0, 1.0]
                )

            graph.nodes[node]["activeness"] = delta
            graph.nodes[node]["stubbornness"] = 1.0

    else:
        if AA_level == "nope":
            change_value = 0.1
        elif AA_level == "weak":
            change_value = 0.5
        else:
            change_value = 1.0

        if AA_type == "None":
            AAs = []
        elif AA_type == "random":
            AAs = random.sample(
                list(graph.nodes()),
                min(AA_k, graph.number_of_nodes()),
            )
        elif AA_type == "betweenness":
            AAs = top_k_betweenness_nodes(graph, AA_k)
        elif AA_type == "degree":
            AAs = top_k_degree_nodes(graph, AA_k)
        elif AA_type == "greedy":
            AAs = select_adversarial_nodes(graph, AA_k)
        else:
            raise ValueError(
                f"Unknown AA_type: {AA_type}"
            )

        for node in AAs:
            o_node = graph.nodes[node]["opinion"]
            if o_node > 0:
                graph.nodes[node]["opinion"] = 1
            elif o_node < 0:
                graph.nodes[node]["opinion"] = -1
            else:
                graph.nodes[node]["opinion"] = random.choice(
                    [-1.0, 1.0]
                )

            graph.nodes[node]["activeness"] = min(
                1.0,
                graph.nodes[node]["activeness"] + change_value,
            )
            graph.nodes[node]["stubbornness"] = 1.0

    return graph, AAs[:]