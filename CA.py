import random
import networkx as nx
import community.community_louvain as community_louvain


def top_k_degree_nodes(G: nx.DiGraph, k: int, volunteers: list):
    candidates = [v for v in volunteers if v in G]
    if k <= 0 or k > len(candidates):
        k = len(candidates)
    sorted_nodes = sorted(
        [(v, G.out_degree(v)) for v in candidates],
        key=lambda x: x[1],
        reverse=True,
    )
    return [node for node, _ in sorted_nodes[:k]]


def top_k_betweenness_nodes(G: nx.DiGraph, k: int, volunteers: list):
    candidates = [v for v in volunteers if v in G]
    if k <= 0 or k > len(candidates):
        k = len(candidates)
    bet = nx.betweenness_centrality(
        G, normalized=True, endpoints=False
    )
    sorted_nodes = sorted(
        candidates, key=lambda v: bet[v], reverse=True
    )
    return sorted_nodes[:k]


def select_nodes_by_community_outdegree(
    G: nx.DiGraph, k: int, volunteers: list, random_seed=42
):
    candidates = [v for v in volunteers if v in G]
    if k <= 0 or k > len(candidates):
        k = len(candidates)

    partition = community_louvain.best_partition(
        G.to_undirected()
    )

    communities = {}
    for v in candidates:
        cid = partition[v]
        communities.setdefault(cid, []).append(v)

    community_info = []
    for cid, nodes in communities.items():
        opinions = [G.nodes[n]["opinion"] for n in nodes]
        mean_o = sum(opinions) / len(opinions)
        size = len(nodes)
        priority = (1 - abs(mean_o)) * size
        community_info.append((cid, size, mean_o, priority))

    pos_comms = [info for info in community_info if info[2] >= 0]
    neg_comms = [info for info in community_info if info[2] < 0]

    k_pos = k // 2
    k_neg = k // 2
    if k % 2 != 0:
        k_pos += 1

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

    def pick_from_community(community_nodes, n_pick):
        comm_sorted = sorted(
            community_nodes,
            key=lambda v: G.out_degree(v),
            reverse=True,
        )
        picked = []
        for v in comm_sorted:
            if v not in selected_set and len(picked) < n_pick:
                picked.append(v)
                selected_set.add(v)
        return picked

    for cid, slots in pos_alloc.items():
        selected.extend(
            pick_from_community(communities[cid], slots)
        )

    for cid, slots in neg_alloc.items():
        selected.extend(
            pick_from_community(communities[cid], slots)
        )

    if len(selected) < k:
        remaining = [
            v for v in candidates if v not in selected_set
        ]
        remaining.sort(
            key=lambda v: G.out_degree(v), reverse=True
        )
        needed = k - len(selected)
        selected.extend(remaining[:needed])

    return selected


def select_node_CA(
    graph,
    volunteers,
    CA_param,
    CA_k,
    random_seed=42,
):
    if CA_k > len(volunteers) or CA_k < 0:
        raise ValueError(
            f"{CA_k} > {len(volunteers)} or Negative !!!"
        )

    random.seed(random_seed)

    if CA_k == 0:
        return graph, []

    CAs = []

    if CA_param == "random":
        CAs = random.sample(volunteers, CA_k)

    elif CA_param == "degree":
        CAs = top_k_degree_nodes(
            graph, CA_k, volunteers
        )

    elif CA_param == "betweenness":
        CAs = top_k_betweenness_nodes(
            graph, CA_k, volunteers
        )

    elif CA_param == "greedy":
        CAs = select_nodes_by_community_outdegree(
            graph, CA_k, volunteers, random_seed
        )

    for node in CAs:
        graph.nodes[node]["opinion"] = 0
        graph.nodes[node]["activeness"] = 1

    return graph, CAs[:]