import concurrent.futures
import os
import pickle
import random
import re
import threading
import time
from datetime import datetime

import numpy as np
import requests
from tqdm import tqdm

from HRG_Graph import Main_Runner as Main_Runner_HRG
from SBM_Graph import Main_Runner as Main_Runner_SBM
from AA import select_node_AA
from LLM_Utils import (
    build_small_directed_graph,
    build_My_FB,
    load_graph,
    variance_computer
)
from Prompts import Prompt_General


class Simulation:
    def __init__(
        self,
        graph_type,
        api_key,
        AA_type,
        AA_k,
        AA_level,
        random_seed=0,
        load_graph_from_file=False,
        save_graph_flag=False,
        graph_path="graph.pkl",
    ):

        random.seed(random_seed)
        np.random.seed(random_seed)
        self.random_seed = random_seed
        self.api_key = api_key
        self.request_lock = threading.Lock()
        self.last_request_time = time.time()
        self.request_count = 0

        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.model = "gpt-4.1-mini"  # "gpt-4.1-mini" # "gpt-4o"

        self.post_history = []
        self.update_history = []

        self.AA_type = (AA_type,)
        self.AA_k = (AA_k,)
        self.AA_level = (AA_level,)

        if load_graph_from_file and os.path.exists(graph_path):
            self.graph = load_graph(graph_path)
        else:
            if graph_type == "small":
                self.graph = build_small_directed_graph(save_graph_flag, graph_path)
            elif graph_type == "hrg":
                self.graph = Main_Runner_HRG(
                    save_graph_flag, graph_path=graph_path, random_seed=random_seed
                )
            elif graph_type == "sbm":
                self.graph = Main_Runner_SBM(
                    save_graph_flag, graph_path=graph_path, random_seed=random_seed
                )
            elif graph_type == "FB":
                self.graph = build_My_FB(save_graph_flag, graph_path, random_seed)
            else:
                raise ValueError("graph_type must be 'small' or 'hrg'")

        ##############################################
        ##################### AA #####################
        ##############################################
        self.graph,self.AA_nodes = select_node_AA(self.graph,AA_level,AA_type,AA_k,random_seed = random_seed) 

    
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

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            r = requests.post(
                self.base_url, headers=self.headers, json=payload, timeout=60
            )
            if r.status_code == 200:
                data = r.json()
                return data["choices"][0]["message"]["content"].strip()
            print(f"API ERROR {r.status_code}: {r.text[:200]}")
            return ""
        except Exception as e:
            print(f"API CALL FAILED: {e}")
            return ""

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

        prompt = Prompt_General.Prompt_Post(
            topic=topic,
            percent=percent,
            intensity_level=intensity,
        )

        return self._call_api(prompt, temperature=0.7, max_tokens=120)

    def _process_target(self, target, post, topic, friendship_flag):
        old_opinion = self.graph.nodes[target]["opinion"]
        stubbornness = self.graph.nodes[target]["stubbornness"]
        percent = int((old_opinion + 1) * 50)

        prompt = Prompt_General.Prompt_Influence(
            topic=topic,
            percent=percent,
            stubbornness=round(stubbornness, 2),
            post=post,
            is_friend_flag=friendship_flag,
        )

        response = self._call_api(prompt, temperature=0.2, max_tokens=16)
        m = re.search(r"\d+", response)

        if not m:
            print("PARSE FAILED:", repr(response))
            return target, percent, old_opinion, old_opinion

        score = int(m.group())
        score = max(0, min(100, score))
        new_opinion = (score - 50) / 50
        new_opinion = max(-1.0, min(1.0, new_opinion))

        return target, score, new_opinion, old_opinion

    def run(
        self,
        topic,
        iterations=10,
        max_workers=2,
        random_seed=0,
    ):

        np.random.seed(random_seed)
        nodes = list(self.graph.nodes())
        opinion_history = {n: [round(self.graph.nodes[n]["opinion"], 2)] for n in nodes}

        activeness = [self.graph.nodes[n]["activeness"] for n in nodes]
        probs = np.array(activeness) / sum(activeness)

        for iteration in tqdm(range(iterations), desc=f"Simulating {topic}"):
            active_node = np.random.choice(nodes, p=probs)
            post = self.generate_post(self.graph.nodes[active_node]["opinion"], topic)

            if not post:
                for n in nodes:
                    if len(opinion_history[n]) == iteration + 1:
                        opinion_history[n].append(opinion_history[n][-1])
                continue

            self.post_history.append((active_node, post))

            neighbors = list(self.graph.neighbors(active_node))
            if self.AA_level == 'strong':
                successors= [node for node in neighbors if node not in self.AA_nodes]
            else:
                successors= [node for node in neighbors]

            results = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = [
                    executor.submit(
                        self._process_target,
                        target=t,
                        post=post,
                        topic=topic,
                        friendship_flag=self.graph.has_edge(active_node, t),
                    )
                    for t in successors
                ]
                for f in concurrent.futures.as_completed(futures):
                    try:
                        results.append(f.result())
                    except Exception as e:
                        print("TARGET FAILED:", e)

            for node, score, new_op, _ in results:
                self.graph.nodes[node]["opinion"] = new_op
                opinion_history[node].append(round(new_op, 2))
                self.update_history.append((node, score, new_op, iteration))

            for n in nodes:
                if len(opinion_history[n]) == iteration + 1:
                    opinion_history[n].append(opinion_history[n][-1])

            if (iteration + 1) % 100 == 0:
                current_values = [opinion_history[n][-1] for n in nodes]

                # variance
                current_variance = np.var(current_values)

                # counts
                N = len(current_values)
                pos = sum(1 for v in current_values if v >= 0.25)
                neg = sum(1 for v in current_values if v <= -0.25)
                neu = N - pos - neg

                pos_ratio = pos / N
                neg_ratio = neg / N
                neu_ratio = neu / N

                print(
                    f"Iteration {iteration+1}: "
                    f"Var={current_variance:.4f} | "
                    f"Pos={pos_ratio:.3f} "
                    f"Neg={neg_ratio:.3f} "
                    f"Neu={neu_ratio:.3f}",
                    flush=True,
                )

        stubbornness = {n: round(self.graph.nodes[n]["stubbornness"], 2) for n in nodes}
        final_opinions = {n: opinion_history[n][-1] for n in nodes}

        return (
            opinion_history,
            stubbornness,
            final_opinions,
            self.post_history,
            self.update_history,
        )


def main(
    graph_type,
    topic,
    iterations,
    API_KEY,
    AA_type,
    AA_k,
    AA_level,
    load_graph=True,
    save_graph_flag=False,
    graph_path="hrg_graph.pkl",
    max_workers=2,
    out_dir="results",
    random_seed=0,
):

    sim = Simulation(
        graph_type=graph_type,
        api_key=API_KEY,
        load_graph_from_file=load_graph,
        save_graph_flag=save_graph_flag,
        random_seed=random_seed,
        graph_path=graph_path,
        AA_type=AA_type,
        AA_k=AA_k,
        AA_level=AA_level,
    )

    (
        opinion_history,
        stubbornness,
        final_opinions,
        post_history,
        update_history,
    ) = sim.run(topic, iterations=iterations, max_workers=max_workers)

    variance = variance_computer(opinion_history)

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"simulation_results_k_{AA_k}_type_{AA_type}_level_{AA_level}_seed_{random_seed}_{ts}.pkl")

    with open(path, "wb") as f:
        pickle.dump(
            {
                "opinion_history": opinion_history,
                "stubbornness": stubbornness,
                "variance": variance,
                "final_opinions": final_opinions,
                "post_history": post_history,
                "update_history": update_history,
            },
            f,
        )

    for i, j in opinion_history.items():
        print(f"{i} s={stubbornness[i]}: {j[0]} -> {j[-1]}")

    print(f"Saved: {path}")
    return opinion_history, stubbornness, variance
