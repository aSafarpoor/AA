import argparse
import os

from Run_LLM_using_API import main


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--graph_type", type=str, default="twitter")
    parser.add_argument("--topic", type=str, default="Support Remote Work")
    parser.add_argument("--iterations", type=int, default=10)

    parser.add_argument(
        "--AA_type",
        type=str,
        default="None",
        choices=["None", "random", "degree", "betweenness", "greedy"],
    )

    parser.add_argument(
        "--CA_type",
        type=str,
        default="moderator",
        choices=[
            "moderator",
            "contrarian",
            "Broadening_social_ties",
            "Distributed_activity_boost",
            "Active_cross_checking_from_feeds",
            "Active_cross_checking_from_zero",
            "Resistance_extreme_content",
        ],
    )

    parser.add_argument(
        "--CA_param",
        type=str,
        default="random",
        choices=["random", "degree", "betweenness", "greedy", "None"],
    )

    parser.add_argument(
        "--AA_level",
        type=str,
        default="weak",
        choices=["weak", "strong"],
    )

    def non_negative_int(value):
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError("Must be >= 0")
        return ivalue

    parser.add_argument("--AA_k", type=non_negative_int, default=0)
    parser.add_argument("--CA_k", type=float, default=0)

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

    random_seeds = [args.random_seed]

    # no attack
    for random_seed in random_seeds:
        main(
            random_seed=random_seed,
            graph_type=args.graph_type,
            topic=args.topic,
            iterations=args.iterations,
            API_KEY=API_KEY,
            load_graph=False,
            save_graph_flag=False,
            graph_path="some_graph.pkl",
            out_dir=f"{args.out_dir}-{args.graph_type}-None-0-strong-None-0",
            AA_type="None",
            AA_k=0,
            AA_level="strong",
            CA_type="None",
            CA_k=0,
            CA_param="None",
        )

    # only attack
    for random_seed in random_seeds:
        for AA_level in ["weak", "strong"]:
            for AA_type in ["random", "degree", "betweenness", "greedy"]:
                for AA_k in [2, 4, 6]:
                    main(
                        random_seed=random_seed,
                        graph_type=args.graph_type,
                        topic=args.topic,
                        iterations=args.iterations,
                        API_KEY=API_KEY,
                        load_graph=False,
                        save_graph_flag=False,
                        graph_path="some_graph.pkl",
                        out_dir=f"{args.out_dir}-{args.graph_type}-{AA_type}-{AA_k}-{AA_level}-None-0",
                        AA_type=AA_type,
                        AA_k=AA_k,
                        AA_level=AA_level,
                        CA_type="None",
                        CA_k=0,
                        CA_param="None",
                    )

    # CA or both
    for random_seed in random_seeds:
        for CA_type in [
            "moderator",
            "contrarian",
            "Broadening_social_ties",
            "Distributed_activity_boost",
            "Active_cross_checking_from_feeds",
            "Active_cross_checking_from_zero",
            "Resistance_extreme_content",
        ]:

            for AA_type in ["random", "degree", "betweenness", "greedy"]:
                for AA_level in ["weak", "strong"]:
                    for AA_k in [0, 2, 4, 6]:

                        if AA_k == 0 and not (
                            AA_level == "strong" and AA_type == "random"
                        ):
                            continue

                        if CA_type == "Distributed_activity_boost":
                            for CA_k in [0.2, 0.5, 0.8]:
                                main(
                                    random_seed=random_seed,
                                    graph_type=args.graph_type,
                                    topic=args.topic,
                                    iterations=args.iterations,
                                    API_KEY=API_KEY,
                                    load_graph=False,
                                    save_graph_flag=False,
                                    graph_path="some_graph.pkl",
                                    out_dir=f"{args.out_dir}-{args.graph_type}-{AA_type}-{AA_k}-{AA_level}-{CA_type}-None-{CA_k}",
                                    AA_type=AA_type,
                                    AA_k=AA_k,
                                    AA_level=AA_level,
                                    CA_type=CA_type,
                                    CA_k=CA_k,
                                    CA_param="None",
                                )

                        elif CA_type == "Broadening_social_ties":
                            for CA_k in [1, 2, 3]:
                                main(
                                    random_seed=random_seed,
                                    graph_type=args.graph_type,
                                    topic=args.topic,
                                    iterations=args.iterations,
                                    API_KEY=API_KEY,
                                    load_graph=False,
                                    save_graph_flag=False,
                                    graph_path="some_graph.pkl",
                                    out_dir=f"{args.out_dir}-{args.graph_type}-{AA_type}-{AA_k}-{AA_level}-{CA_type}-None-{CA_k}",
                                    AA_type=AA_type,
                                    AA_k=AA_k,
                                    AA_level=AA_level,
                                    CA_type=CA_type,
                                    CA_k=CA_k,
                                    CA_param="None",
                                )

                        elif CA_type in ["moderator", "contrarian"]:
                            for CA_param in ["random", "degree", "betweenness", "greedy"]:
                                for CA_k in [2, 3, 4]:
                                    main(
                                        random_seed=random_seed,
                                        graph_type=args.graph_type,
                                        topic=args.topic,
                                        iterations=args.iterations,
                                        API_KEY=API_KEY,
                                        load_graph=False,
                                        save_graph_flag=False,
                                        graph_path="some_graph.pkl",
                                        out_dir=f"{args.out_dir}-{args.graph_type}-{AA_type}-{AA_k}-{AA_level}-{CA_type}-{CA_param}-{CA_k}",
                                        AA_type=AA_type,
                                        AA_k=AA_k,
                                        AA_level=AA_level,
                                        CA_type=CA_type,
                                        CA_k=CA_k,
                                        CA_param=CA_param,
                                    )

                        else:
                            main(
                                random_seed=random_seed,
                                graph_type=args.graph_type,
                                topic=args.topic,
                                iterations=args.iterations,
                                API_KEY=API_KEY,
                                load_graph=False,
                                save_graph_flag=False,
                                graph_path="some_graph.pkl",
                                out_dir=f"{args.out_dir}-{args.graph_type}-{AA_type}-{AA_k}-{AA_level}-{CA_type}-None-0",
                                AA_type=AA_type,
                                AA_k=AA_k,
                                AA_level=AA_level,
                                CA_type=CA_type,
                                CA_k=0,
                                CA_param="None",
                            )
        


#############################
############Greedy GRID###########
#############################
'''
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
'''