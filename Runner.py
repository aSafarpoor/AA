import argparse
import os

from Run_LLM_using_API import main


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--graph_type", type=str, default="hrg") # small, hrg, FB, sbm
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
    
   
    out_dir = (
        f"{args.out_dir}-"
    )
  
    
    # for random_seed in range(5):
    #     for k in [0]:
    #         for  typeee in ["None"]:
    #             main(random_seed= random_seed,
    #                 graph_type=args.graph_type,
    #                 topic=args.topic,
    #                 iterations=args.iterations,
    #                 API_KEY=API_KEY,
    #                 load_graph=args.load_graph,
    #                 save_graph_flag=args.save_graph,
    #                 graph_path="hrg_graph.pkl",
    #                 out_dir=out_dir,
    #                 AA_type = typeee, #args.AA_type,
    #                 AA_k = k, #args.AA_k,
    #                 AA_level = args.AA_level,
    #             )

    for random_seed in range(2):
      for k in [2,4]:
        for  typeee in ["random", "degree", "betweenness", "greedy"]:
          main(random_seed= random_seed,
              graph_type=args.graph_type,
              topic=args.topic,
              iterations=args.iterations,
              API_KEY=API_KEY,
              load_graph=args.load_graph,
              save_graph_flag=args.save_graph,
              graph_path="hrg_graph.pkl",
              out_dir=out_dir,
              AA_type = typeee, #args.AA_type,
              AA_k = k, #args.AA_k,
              AA_level = 'weak' # args.AA_level,
          )
