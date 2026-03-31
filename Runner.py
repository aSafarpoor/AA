import argparse
import os

from Run_LLM_using_API import main


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--graph_type", type=str, default="hrg") # small, hrg, FB, sbm, smallsbm
    parser.add_argument("--topic", type=str, default="Support Remote Work") # "Support Remote Work" # Using AI tools in academic assignments # Support Group Work Assignments

    parser.add_argument("--iterations", type=int, default=10)

    parser.add_argument(
        "--AA_type",
        type=str,
        default="None",
        choices=["None", "random", "degree", "betweenness", "greedy"],
        help="Type of adversarial attack."
    )

    def non_negative_int(value):
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError("AA_k must be an integer >= 0.")
        return ivalue

    parser.add_argument(
        "--AA_k",
        type=non_negative_int,
        default=0,
        help="Number of nodes to attack (integer >= 0)."
    )

    parser.add_argument(
        "--AA_level",
        type=str,
        default="weak",
        choices=["weak", "medium", "strong"],
        help="Attack intensity level."
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--load_graph", action="store_true")
    parser.add_argument("--save_graph", action="store_true")
    parser.add_argument("--out_dir", type=str, default="results-ERS")
    
    return parser.parse_args() 


if __name__ == "__main__":

    args = parse_args()

    API_KEY = os.environ.get("OPENAI_API_KEY")
    if API_KEY is None:
        raise RuntimeError("OPENAI_API_KEY not set")
    
    random_seed = 0
    
    
    for random_seed in range(2):
        for AA_type in ['None']:
            main(random_seed= random_seed,
                graph_type=args.graph_type,
                topic=args.topic,
                iterations=150,#args.iterations,
                API_KEY=API_KEY,
                load_graph=args.load_graph,
                save_graph_flag=args.save_graph,
                graph_path="some_graph.pkl",
                out_dir=f"{args.out_dir}-{args.graph_type}-{AA_type}-{0}",
                AA_type = AA_type,#args.AA_type,
                AA_k = 0, # args.AA_k,
                AA_level = args.AA_level,
            )
        for AA_type in ["random", "degree", "betweenness", "greedy"]:
            for k in [2,3,4]:
                main(random_seed= random_seed,
                    graph_type=args.graph_type,
                    topic=args.topic,
                    iterations=150,#args.iterations,
                    API_KEY=API_KEY,
                    load_graph=args.load_graph,
                    save_graph_flag=args.save_graph,
                    graph_path="some_graph.pkl",
                    out_dir=f"{args.out_dir}-{args.graph_type}-{AA_type}-{k}",
                    AA_type = AA_type,#args.AA_type,
                    AA_k = k, # args.AA_k,
                    AA_level = args.AA_level,
                )

    # for random_seed in range(2):
    #     out_dir = (
    #         f"{args.out_dir}-grid1"
    #     )
    #     for k in [1,2,3,4,5,6]:
    #         for new_activness in [0.125,0.25,0.375,0.5,0.625,0.75,0.875,1]:
    #             main(random_seed= random_seed,
    #                 graph_type='smallsbm',#args.graph_type,
    #                 topic=args.topic,
    #                 iterations=80,#args.iterations,
    #                 API_KEY=API_KEY,
    #                 load_graph=args.load_graph,
    #                 save_graph_flag=args.save_graph,
    #                 graph_path="some_graph.pkl",
    #                 out_dir=out_dir,
    #                 AA_type = "greedy",#args.AA_type,
    #                 AA_k = k,#args.AA_k,
    #                 AA_level = f"grid_{new_activness}" #args.AA_level,
    #             )
                
    # for random_seed in range(2):
    #     out_dir = (
    #         f"{args.out_dir}-grid2"
    #     )
    #     for k in [4]:
    #       for new_stubborness in [0.125,0.25,0.375,0.5,0.625,0.75,0.875,1]:
    #         for new_activness in [0.125,0.25,0.375,0.5,0.625,0.75,0.875,1]:
    #             main(random_seed= random_seed,
    #                 graph_type='smallsbm',#args.graph_type,
    #                 topic=args.topic,
    #                 iterations=80,#args.iterations,
    #                 API_KEY=API_KEY,
    #                 load_graph=args.load_graph,
    #                 save_graph_flag=args.save_graph,
    #                 graph_path="some_graph.pkl",
    #                 out_dir=out_dir,
    #                 AA_type = "greedy",#args.AA_type,
    #                 AA_k = k,#args.AA_k,
    #                 AA_level = f"grid2_{new_activness}_{new_stubborness}" #args.AA_level,
    #             )