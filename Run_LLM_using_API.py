import concurrent.futures
import os
import pickle
import random
import re
import threading
import time
from datetime import datetime
import networkx as nx
import numpy as np
import requests
from tqdm import tqdm
# HRG_Graph depends on `networkit`, which is only needed for synthetic HRG
# graphs. Import lazily so the rest of the framework (Reddit/Twitter/SBM/etc.)
# works even in environments without networkit installed.
from SBM_Graph import Main_Runner as Main_Runner_SBM
from SmallSBM_Graph import Main_Runner as Main_Runner_SmallSBM
from AA import select_node_AA
from CA import select_node_CA
from LLM_Utils import build_small_directed_graph, build_My_FB, load_graph, variance_computer, build_My_RedditTwitter
from Prompts import Prompt_General


def volunteers_selection(whole_bag, ratio_of_volunteers, random_seed):
    random.seed(random_seed)
    k = int(ratio_of_volunteers * len(whole_bag))
    volunteers = random.sample(whole_bag, k)
    return volunteers


def add_random_in_edges(G: nx.DiGraph, new_edges_number: int, seed: int=42):
    random.seed(seed)
    nodes = list(G.nodes())
    for v in nodes:
        candidates = [u for u in nodes if u!= v and (not G.has_edge(u, v))]
        if not candidates:
            continue
        else:
            k = min(new_edges_number, len(candidates))
            chosen = random.sample(candidates, k)
            for u in chosen:
                G.add_edge(u, v)
    return G


def boost_activeness(G: nx.DiGraph, delta_a: float):
    for node in G.nodes():
        a = G.nodes[node].get('activeness', 0.0)
        G.nodes[node]['activeness'] = min(1.0, a + delta_a)
    return G


class Simulation:
    def __init__(self, graph_type, api_key, AA_type, AA_k, AA_level, CA_param, CA_type, CA_k, random_seed=0, load_graph_from_file=False, save_graph_flag=False, graph_path='graph.pkl'):
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.random_seed = random_seed
        self.api_key = api_key
        self.request_lock = threading.Lock()
        self.last_request_time = time.time()
        self.request_count = 0
        self.base_url = 'https://api.openai.com/v1/chat/completions'
        self.headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
        self.model = 'gpt-4.1-mini'
        self.post_history = []
        self.update_history = []
        self.CA_param = CA_param
        self.CA_type = CA_type
        self.CA_k = CA_k
        self.AA_type = AA_type
        self.AA_k = AA_k
        self.AA_level = AA_level
        self.delta_a = CA_k
        self.new_edges_number = CA_k
        if load_graph_from_file and os.path.exists(graph_path):
            self.graph = load_graph(graph_path)

        elif graph_type == "small":
            self.graph = build_small_directed_graph(
                save_graph_flag,
                graph_path
            )

        elif graph_type == "hrg":
            from HRG_Graph import Main_Runner as Main_Runner_HRG
            self.graph = Main_Runner_HRG(
                save_graph_flag,
                graph_path=graph_path,
                random_seed=random_seed
            )

        elif graph_type == "sbm":
            self.graph = Main_Runner_SBM(
                save_graph_flag,
                graph_path=graph_path,
                random_seed=random_seed
            )

        elif graph_type == "smallsbm":
            self.graph = Main_Runner_SmallSBM(
                save_graph_flag,
                graph_path=graph_path,
                random_seed=random_seed
            )

        elif graph_type == "FB":
            self.graph = build_My_FB(
                save_graph_flag,
                graph_path,
                random_seed
            )

        elif graph_type.lower() in ["reddit", "twitter"]:
            self.graph = build_My_RedditTwitter(
                save_graph_flag,
                graph_path=graph_path,
                random_seed=random_seed,
                graph_type=graph_type
            )

        else:
            raise ValueError(
                "graph_type must be one of: "
                "'small', 'hrg', 'sbm', "
                "'smallsbm', 'FB', 'reddit', or 'twitter'"
            )
        
        self.graph, self.AA_nodes = select_node_AA(self.graph, AA_level, AA_type, AA_k, random_seed=random_seed)
        if not self.graph.is_directed():
            self.graph = self.graph.to_directed()
        if CA_type == 'Broadening_social_ties':
            self.graph = add_random_in_edges(self.graph, self.new_edges_number, seed=random_seed)
        elif CA_type == 'Distributed_activity_boost':
            self.graph = boost_activeness(self.graph, self.delta_a)
        volunteers = volunteers_selection(list(set(self.graph.nodes) - set(self.AA_nodes)), ratio_of_volunteers=0.2, random_seed=random_seed)
        if CA_type in ['moderator', 'contrarian']:
            # Reactive mitigation: select the moderator set M from the pool.
            # CA_param = selection strategy, CA_type = neutral/contrarian behaviour.
            self.graph, self.CA_nodes = select_node_CA(
                self.graph, volunteers, CA_param, CA_type, int(CA_k),
                random_seed=random_seed,
            )
        else:
            self.CA_nodes = []
    def _call_api(self, prompt, temperature, max_tokens):
        with self.request_lock:
            now = time.time()
            dt = now - self.last_request_time
            if dt < 0.5:
                time.sleep(0.5 - dt)
            self.last_request_time = time.time()
            self.request_count += 1
            if self.request_count % 20 == 0:
                time.sleep(2.0)
        payload = {'model': self.model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': temperature, 'max_tokens': max_tokens}
        try:
            r = requests.post(self.base_url, headers=self.headers, json=payload, timeout=60)
            if r.status_code == 200:
                data = r.json()
                return data['choices'][0]['message']['content'].strip()
            else:
                print(f'API ERROR {r.status_code}: {r.text[:200]}')
                return ''
        except Exception as e:
            print(f'API CALL FAILED: {e}')
            return ''
    def generate_post(self, opinion, topic):
        percent = int((opinion + 1) * 50)
        if percent < 20:
            intensity = "strongly opposed"

        elif percent < 45:
            intensity = "moderately opposed"

        elif percent < 55:
            intensity = "neutral or undecided"

        elif percent < 80:
            intensity = "moderately in favor"

        else:
            intensity = "strongly in favor"

        prompt = Prompt_General.Prompt_Post(topic=topic, percent=percent, intensity_level=intensity)
        return self._call_api(prompt, temperature=0.7, max_tokens=120)
    

    def _process_target(self, target, post, topic):
        old_opinion = self.graph.nodes[target]['opinion']
        stubbornness = self.graph.nodes[target]['stubbornness']
        percent = int((old_opinion + 1) * 50)
        prompt = Prompt_General.Prompt_Influence(topic=topic, percent=percent, stubbornness=round(stubbornness, 2), post=post)
        response = self._call_api(prompt, temperature=0.2, max_tokens=16)
        m = re.search('\\d+', response)
        if not m:
            print('PARSE FAILED:', repr(response))
            return (target, percent, old_opinion, old_opinion)
        else:
            score = int(m.group())
            score = max(0, min(100, score))
            new_opinion = (score - 50) / 50
            new_opinion = max((-1.0), min(1.0, new_opinion))
            return (target, score, new_opinion, old_opinion)
        

    def compute_contrarian_opinion(self, node):
        """Contrarian moderator opinion (paper Sec. 5.1):

            o_v^t = -sign( sum_{u in Gamma_v^+} o_u^{t-1} ),  and 0 if the sum is 0.

        The moderator reacts in the opposite direction to the dominant opinion
        of its outgoing neighbourhood, using their opinions from the previous
        step (i.e. the current stored values before this step's update).
        """
        successors = list(self.graph.successors(node))
        if len(successors) == 0:
            return 0
        s = sum(self.graph.nodes[n]['opinion'] for n in successors)
        if s > 0:
            return -1
        elif s < 0:
            return 1
        else:
            return 0
            

    def run(self, topic, iterations=10, max_workers=2, random_seed=0):
        post_bag = []
        np.random.seed(random_seed)
        nodes = list(self.graph.nodes())
        opinion_history = {n: [round(self.graph.nodes[n]['opinion'], 2)] for n in nodes}
        activeness = [self.graph.nodes[n]['activeness'] for n in nodes]
        probs = np.array(activeness) / sum(activeness)
        if self.CA_type == 'Active_cross_checking_from_zero':
            zero_bag = [self.generate_post(0, topic) for i in range(10)]
        for iteration in tqdm(range(iterations), desc=f'Simulating {topic}'):
            active_node = np.random.choice(nodes, p=probs)
            if active_node in self.CA_nodes and self.CA_type == 'moderator':
                # Neutral moderator: always generates neutral (opinion 0) content.
                self.graph.nodes[active_node]['opinion'] = 0
                post = self.generate_post(0, topic)
            elif active_node in self.CA_nodes and self.CA_type == 'contrarian':
                # Contrarian moderator: opinion = opposite of dominant local opinion,
                # recomputed each step from neighbours' current (previous-step) opinions.
                contrarian_value = self.compute_contrarian_opinion(active_node)
                self.graph.nodes[active_node]['opinion'] = contrarian_value
                post = self.generate_post(contrarian_value, topic)
            else:
                post = self.generate_post(self.graph.nodes[active_node]['opinion'], topic)
            if not post:
                for n in nodes:
                    if len(opinion_history[n]) == iteration + 1:
                        opinion_history[n].append(opinion_history[n][(-1)])
            else:
                if self.CA_type == 'Active_cross_checking_from_feeds':
                    post_bag.append(post)
                self.post_history.append((active_node, post))
                neighbors = list(self.graph.successors(active_node))
                if self.CA_type == 'Resistance_extreme_content' and random.random() > 0.5 and (abs(self.graph.nodes[active_node]['opinion']) > 0.5):
                            neighbors = [neighbor for neighbor in neighbors if random.random() < 0.5]
                if self.AA_level in ['strong', 'grid']:
                    successors = [node for node in neighbors if node not in self.AA_nodes and node not in self.CA_nodes]
                else:
                    successors = [node for node in neighbors if node not in self.CA_nodes]
                results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    if self.CA_type == 'Active_cross_checking_from_feeds' and len(post_bag) > 1 and (random.random() > 0.5):
                        post_from_bag = random.choice(post_bag[:(-1)])
                        futures = [executor.submit(self._process_target, target=t, post=f'post from your friend {post}  and also see this random post {post_from_bag}', topic=topic) for t in successors]
                    elif self.CA_type == 'Active_cross_checking_from_zero' and len(zero_bag) > 0 and (random.random() > 0.5):
                        post_from_zero_bag = random.choice(zero_bag)
                        futures = [executor.submit(self._process_target, target=t, post=f'post from your friend {post}  and also see this random post {post_from_zero_bag}', topic=topic) for t in successors]
                    else:
                        futures = [executor.submit(self._process_target, target=t, post=post, topic=topic) for t in successors]
                    for f in concurrent.futures.as_completed(futures):
                        try:
                            results.append(f.result())
                        except Exception as e:
                            print('TARGET FAILED:', e)
                for node, score, new_op, _ in results:
                    self.graph.nodes[node]['opinion'] = new_op
                    opinion_history[node].append(round(new_op, 2))
                    self.update_history.append((node, score, new_op, iteration))
                for n in nodes:
                    if len(opinion_history[n]) == iteration + 1:
                        opinion_history[n].append(opinion_history[n][(-1)])
                if (iteration + 1) % 100 == 0:
                    current_values = [opinion_history[n][(-1)] for n in nodes]
                    current_variance = np.var(current_values)
                    N = len(current_values)
                    pos = sum((1 for v in current_values if v >= 0.25))
                    neg = sum((1 for v in current_values if v <= (-0.25)))
                    neu = N - pos - neg
                    pos_ratio = pos / N
                    neg_ratio = neg / N
                    neu_ratio = neu / N
                    print(f'Iteration {iteration + 1}: Var={current_variance:.4f} | Pos={pos_ratio:.3f} Neg={neg_ratio:.3f} Neu={neu_ratio:.3f}', flush=True)
        stubbornness = {n: round(self.graph.nodes[n]['stubbornness'], 2) for n in nodes}
        final_opinions = {n: opinion_history[n][(-1)] for n in nodes}
        return (opinion_history, stubbornness, final_opinions, self.post_history, self.update_history)
    

def main(graph_type, topic, iterations, API_KEY, AA_type, AA_k, AA_level, CA_type, CA_k, CA_param, load_graph=True, save_graph_flag=False, graph_path='some_graph.pkl', max_workers=2, out_dir='results', random_seed=0):
    sim = Simulation(graph_type=graph_type, api_key=API_KEY, load_graph_from_file=load_graph, save_graph_flag=save_graph_flag, random_seed=random_seed, graph_path=graph_path, AA_type=AA_type, AA_k=AA_k, AA_level=AA_level, CA_type=CA_type, CA_k=CA_k, CA_param=CA_param)
    opinion_history, stubbornness, final_opinions, post_history, update_history = sim.run(topic, iterations=iterations, max_workers=max_workers)
    variance = variance_computer(opinion_history)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f'seed_{random_seed}_{ts}.pkl')
    with open(path, 'wb') as f:
        pickle.dump({'opinion_history': opinion_history, 'stubbornness': stubbornness, 'variance': variance, 'final_opinions': final_opinions, 'post_history': post_history, 'update_history': update_history}, f)
    for i, j in opinion_history.items():
        print(f'{i} s={stubbornness[i]}: {j[0]} -> {j[(-1)]}')
    print(f'Saved: {path}')
    return (opinion_history, stubbornness, variance)