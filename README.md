# Opinion Polarization in LLM-Based Social Networks — Simulation Code

Reference implementation for *"Opinion Polarization in LLM-Based Social Networks:
Manipulation and Mitigation."* The framework simulates a directed social network
of LLM agents that exchange natural-language posts and update their opinions,
and evaluates adversarial **manipulation** strategies and **mitigation** mechanisms.

> The code in this repository was cleaned and commented with the assistance of
> large language models. All simulation logic was reviewed to match the
> methodology described in the paper.

## Files

| File | Role |
|------|------|
| `Runner.py` | CLI entry point; sweeps over manipulation/mitigation settings. |
| `Run_LLM_using_API.py` | Core simulation (`Simulation` class + `main()`); LLM calls, opinion dynamics loop. |
| `Prompts.py` | Post-generation and response-generation prompts (paper Appendix A). |
| `AA.py` | Adversarial (manipulator) node selection: random / degree / betweenness / community-aware greedy. |
| `CA.py` | Reactive moderator selection (same four strategies, restricted to the moderator pool). |
| `HRG_Graph.py` | Synthetic Hyperbolic Random Graph generator (requires `networkit`). |
| `SBM_Graph.py`, `SmallSBM_Graph.py` | Stochastic block-model generators (optional / sanity + grid experiments). |
| `LLM_Utils.py` | Graph builders (Reddit/Twitter/Facebook), opinion init, metrics, drawing. |

## Naming: `AA` vs `CA`

- **AA = Adversarial Agents** (manipulators, set `A` in the paper). Controlled by
  `--AA_type` (selection strategy), `--AA_k` (budget `k_A`), `--AA_level`
  (`weak` = susceptible, `delta=0.5`; `strong` = persistent, `delta=1`).
- **CA = Counter Agents / mitigations** (`--CA_type`). Reactive: `moderator`
  (neutral) and `contrarian`. Proactive: `Broadening_social_ties`,
  `Distributed_activity_boost`, `Active_cross_checking_from_feeds`,
  `Active_cross_checking_from_zero`, `Resistance_extreme_content`.
  `--CA_param` is the moderator selection strategy; `--CA_k` is the moderator
  budget `k_M` (or the proactive intensity parameter, depending on `CA_type`).

## Running

```bash
export OPENAI_API_KEY=sk-...
python Runner.py --graph_type twitter --topic "Support Remote Work" --iterations 2000
```

`Runner.py` runs three blocks per seed: (1) no-manipulation baseline,
(2) manipulation-only sweeps, (3) mitigation (and manipulation+mitigation) sweeps.
Results are pickled per run under `results-*` directories.

## Opinion representation

Opinions live in `[-1, 1]` internally and are mapped to an integer percentage
`[0, 100]` when shown to the LLM (`0 = against`, `100 = favor`), which the paper
found to be the most stable representation.

## Changes made while producing this corrected version

These fixes bring the code in line with the paper's methodology and remove
leftover experimental scaffolding:

1. **`select_node_CA` crash fixed.** The function is now defined with a
   `CA_type` argument matching the call site (previously a `TypeError`, which
   broke every reactive-moderator run).
2. **Contrarian moderators implemented** per the paper
   (`o_v^t = -sign(Σ_{u∈Γ⁺_v} o_u)`, 0 on ties). Previously only the neutral
   `moderator` behaviour was wired into the loop.
3. **Neutral moderators** now explicitly post neutral (opinion 0) content when
   active and stay pinned at 0.
4. **Removed leftover "friendship" experiment** (`is_friend_flag`,
   `responsiveness`) that was not part of the paper.
5. **Missing `minimal_graph_shower`** added to `LLM_Utils.py` (its absence made
   `SBM_Graph.py` and `HRG_Graph.py` fail at import).
6. **Reproducibility:** `build_My_FB` and `build_My_RedditTwitter` now seed
   `np.random` before sampling stubbornness/activeness.
7. **Lazy HRG import** so non-HRG graph types work without `networkit`.
8. **Prompts** aligned with Appendix A.
9. Removed the dead commented-out grid-search block from `Runner.py`.
10. Removed all `draw_flag` / `verbose` plotting parameters and the matplotlib
    dependency from the graph builders (visualisation was not part of the
    experimental pipeline).
11. Removed a stray `pass` at the top of the `Simulation` class body.

## One discrepancy to be aware of (code vs. paper text)

The paper (Sec. 4.2) states δ = 0.5 for **susceptible** manipulators and δ = 1
for **persistent** ones, applied to *both* activeness and stubbornness. In the
code (`AA.py`), the **activeness** boost matches this (weak → +0.5, strong →
+1.0), but **stubbornness is set to 1.0 for both** weak and strong manipulators.

This is the original behaviour, not an artefact of cleanup, and it is
**behaviourally consistent with the paper's results** because the persistent vs.
susceptible distinction is actually enforced structurally in the simulation loop:

- **Persistent (strong):** manipulator nodes are *excluded from the recipient
  set*, so they never receive posts and never update — they stay pinned at ±1.
- **Susceptible (weak):** manipulator nodes *remain in the recipient set*, so
  they receive posts and can drift back toward the baseline over time.

So susceptibility is determined by whether a manipulator is updated at all, not
by its stubbornness value. If you intended susceptible manipulators to also have
stubbornness = 0.5 (matching the δ text literally), change the final
`graph.nodes[node]["stubbornness"] = 1.0` line in `select_node_AA` to use
`change_value`. Left as-is to preserve the original results.

## Verification performed

The corrected code was tested (with a mocked LLM API, no network calls):

- All 7 graph types build and simulate: `small`, `smallsbm`, `sbm`, `hrg`,
  `reddit`, `twitter`, `FB`.
- All 5 manipulator strategies × {weak, susceptible / strong, persistent} ×
  budgets run; persistent manipulators verified to stay fixed at ±1.
- All reactive (neutral, contrarian) and proactive mitigations run end to end.
- Contrarian sign convention unit-tested against the paper equation
  `o_v = -sign(Σ neighbours)`.
- History-length invariant checked (each node's trajectory grows by exactly one
  entry per iteration; no double-append).
- Reproducibility confirmed (same seed → identical trajectories with
  `max_workers=1`).
- `Runner.py`'s full sweep issues 850 valid `main()` calls with no bad kwargs.

## Notes / optional components (not used for the main paper results)

- `SBM_Graph.py`, `SmallSBM_Graph.py`, `build_small_directed_graph`, and
  `build_My_FB` are optional graph sources. The paper's reported results use
  **HRG** (synthetic) plus the **Reddit** and **Twitter** real-world datasets.
- `AA_level` values `nope`, `grid`, `grid2` in `AA.py` are grid-search hooks and
  are not exercised by the default `Runner.py`.
- Real-world / FB datasets are loaded from `RedditTwitter/` and `FB_Edges/`
  pickle files (not included here) and expect `random_seed <= 9`.
