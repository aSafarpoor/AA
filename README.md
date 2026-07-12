# Opinion Polarization in LLM-Based Social Networks — Simulation Code

Reference implementation for *"Opinion Polarization in LLM-Based Social Networks:
Manipulation and Mitigation."* The framework simulates a directed social network
of LLM agents that exchange natural-language posts and update their opinions, and
evaluates adversarial **manipulation** strategies and **mitigation** mechanisms.

## Requirements

- Python 3.9+
- `networkx`, `numpy`, `requests`, `tqdm`, `python-louvain` (`community`)
- `networkit` — only required for the `hrg` graph type
- An OpenAI API key (or other companies related APIs) in the `OPENAI_API_KEY` environment variable

The current simulation calls the OpenAI Chat Completions API
(`https://api.openai.com/v1/chat/completions`, model `gpt-4.1-mini`) for both
post generation and opinion updates.

## Files

| File | Role |
|------|------|
| `Runner.py` | It's the Runner :), sweeps over manipulation/mitigation settings. |
| `Run_LLM_using_API.py` | Core simulation (`Simulation` class + `main()`): LLM calls and the opinion-dynamics loop. |
| `Prompts.py` | Post-generation and response-generation prompts (paper Appendix A). |
| `AA.py` | Manipulator (adversarial agent) selection: random / degree / betweenness / community-aware greedy. |
| `CA.py` | Reactive moderator selection (same four strategies, restricted to the moderator pool). |
| `HRG_Graph.py` | Synthetic Hyperbolic Random Graph generator (requires `networkit`). |
| `SBM_Graph.py`, `SmallSBM_Graph.py` | Stochastic block-model generators. |
| `LLM_Utils.py` | Graph builders (Reddit/Twitter/Facebook), opinion initialisation, metrics. |

## Terminology

- **AA — adversarial agents** (manipulators, the set `A` in the paper).
  `--AA_type` selection strategy; `--AA_k` budget `k_A`; `--AA_level`
  persistence (`weak` = susceptible, delta = 0.5; `strong` = persistent, delta = 1).
- **CA — counter agents / mitigations** (`--CA_type`).
  Reactive: `moderator` (neutral) and `contrarian`.
  Proactive: `Broadening_social_ties`, `Distributed_activity_boost`,
  `Active_cross_checking_from_feeds`, `Active_cross_checking_from_zero`,
  `Resistance_extreme_content`.
  `--CA_param` is the moderator selection strategy; `--CA_k` is the moderator
  budget `k_M` (or the proactive intensity parameter, depending on `CA_type`).

## Model summary

Opinions live in `[-1, 1]` internally and are mapped to an integer percentage
`[0, 100]` (`0 = against`, `100 = favor`) when shown to the model, which the
paper found to be the most stable representation. Each step: a node is activated
with probability proportional to its activeness, it generates a post via the LLM,
the post is broadcast to its outgoing neighbours, and each recipient updates its
opinion via the LLM conditioned on its previous opinion and stubbornness.

Manipulators are initialised to an extreme opinion (+/-1) aligned with their
initial leaning, and their activeness and stubbornness are increased by delta
(`min(1, x + delta)`). Persistent manipulators become fully stubborn and remain
fixed; susceptible manipulators can still update over time.

Reactive moderators (activeness 1) are drawn from a moderator pool
(`M_pool = V \ A`, 20% of non-manipulator nodes). Neutral moderators hold a
fixed opinion of 0; contrarian moderators post the opposite of their
neighbourhood's dominant opinion, `o_v = -sign(sum_{u in Gamma+} o_u)`.

Two objective measures are reported (paper Sec. 3.2):
polarization = variance of opinions, extremization = mean absolute opinion.

## Running

```bash
export OPENAI_API_KEY=sk-...
python Runner.py --graph_type twitter --topic "Support Remote Work" --iterations 2000
```

`Runner.py` runs three blocks per seed: (1) a no-manipulation baseline,
(2) manipulation-only sweeps, and (3) mitigation (and manipulation+mitigation)
sweeps. Results are pickled per run under `results-*` directories, each
containing `opinion_history`, `stubbornness`, `variance`, `final_opinions`,
`post_history`, and `update_history`.

## Datasets
- Synthetic graphs (`hrg`, `sbm`, `smallsbm`, `small`) are generated in code.
