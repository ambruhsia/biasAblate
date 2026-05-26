"""
Insert reviewer-response experiment cells into layer-trials.ipynb.
Replaces the two empty cells at positions 43-44 with a markdown header + sensitivity grid,
then appends baseline comparisons, per-metric deltas, bootstrap CIs, and runtime profiling.
"""
import json, uuid, copy, sys

NB_PATH = r"c:\Users\rajpu\OneDrive\Desktop\mitigating bias from llms\bias-mitigation-in-llms\layer-trials.ipynb"

def make_code_cell(source_str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_str.split("\n")]
    }

def make_md_cell(source_str):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": [line + "\n" for line in source_str.split("\n")]
    }

# ═══════════════════════════════════════════════════════════════════
# CELL CONTENTS
# ═══════════════════════════════════════════════════════════════════

MD_HEADER = """\
# Reviewer Response: Sensitivity Analysis, Baselines & Statistical Rigor

The following cells address the key reviewer concerns:
1. **Hybrid weight sensitivity** — α, β, γ grid sweep
2. **Baseline comparisons** — random, utility-only, bias-only, perplexity-only
3. **Per-metric delta table** with win/loss/tie counts
4. **Bootstrap confidence intervals** over prompt pairs
5. **Runtime/memory profiling** for all 12 scoring strategies"""

SENSITIVITY_GRID = r'''# ═══════════════════════════════════════════════════════════════════════
# 17.  HYBRID WEIGHT SENSITIVITY ANALYSIS  (α, β, γ grid sweep)
# ═══════════════════════════════════════════════════════════════════════
#
# Sweeps a grid of (α, β, γ) values and for each:
#   1. Recomputes hybrid scores with those weights
#   2. Finds the top prune candidate (lowest hybrid score, non-boundary)
#   3. Looks up that layer's ablation results from the existing run
#   4. Reports the resulting PPL ratio and fairness improvements
#
# This answers: "How sensitive are the results to the hybrid weights?"
# ═══════════════════════════════════════════════════════════════════════

import itertools

def run_sensitivity_analysis(analysis_results, model_name,
                              alpha_range=(0.5, 1.0, 1.5),
                              beta_range=(0.25, 0.5, 0.75),
                              gamma_range=(0.5, 1.0, 1.5, 2.0),
                              exclude_ablation_variants=(False, True)):
    """
    Grid sweep over hybrid weights, reusing the cached strategy scores.

    Parameters
    ----------
    analysis_results : dict — output of run_layer_by_layer_analysis()
    alpha_range, beta_range, gamma_range : tuples of values to sweep
    exclude_ablation_variants : tuple of bool — test with/without bias_attribution

    Returns
    -------
    list[dict] — one row per (α, β, γ, exclude_ablation) combination
    """
    strategy_scores = analysis_results["strategy_scores"]
    baseline        = analysis_results["baseline"]
    ablation_results = analysis_results["ablation_results"]
    n_layers        = analysis_results["n_layers"]

    abl_by_layer = {r["layer_removed"]: r for r in ablation_results}

    utility_keys    = ["l1_norm", "l2_norm", "taylor_importance",
                       "fisher_information", "mean_activation", "gradient_magnitude"]
    redundancy_keys = ["activation_variance", "representational_entropy"]
    bias_keys_full  = ["bias_attribution", "demographic_embedding",
                       "bias_prompt_sensitivity", "hybrid_bias_gradient"]

    def _safe_mean(keys, layer_idx, invert=False):
        vals = []
        for k in keys:
            if k in strategy_scores and layer_idx < len(strategy_scores[k]):
                v = float(strategy_scores[k][layer_idx])
                if np.isfinite(v):
                    vals.append(1.0 - v if invert else v)
        return float(np.mean(vals)) if vals else 0.0

    rows = []
    for excl_abl in exclude_ablation_variants:
        bias_keys = [k for k in bias_keys_full if not (excl_abl and k == "bias_attribution")]

        for alpha, beta, gamma in itertools.product(alpha_range, beta_range, gamma_range):
            hybrid = []
            for i in range(n_layers):
                util  = _safe_mean(utility_keys, i)
                redun = _safe_mean(redundancy_keys, i, invert=True)
                bias  = _safe_mean(bias_keys, i)
                hybrid.append(alpha * util + beta * redun - gamma * bias)

            arr = np.array(hybrid)
            mn, mx = arr.min(), arr.max()
            if abs(mx - mn) > 1e-12:
                hybrid_norm = ((arr - mn) / (mx - mn)).tolist()
            else:
                hybrid_norm = [0.5] * n_layers

            protected = {0, n_layers - 1}
            ranked = sorted(enumerate(hybrid_norm), key=lambda x: x[1])
            candidate = None
            for idx, score in ranked:
                if idx not in protected:
                    candidate = idx
                    break

            if candidate is None or candidate not in abl_by_layer:
                continue

            abl = abl_by_layer[candidate]
            gate = strict_acceptance_gate(abl, baseline)

            rows.append({
                "alpha": alpha, "beta": beta, "gamma": gamma,
                "exclude_ablation": excl_abl,
                "best_candidate": candidate,
                "hybrid_score": hybrid_norm[candidate],
                "ppl_ratio": gate["ppl_ratio"],
                "fairness_improved": gate["fairness_improvements"],
                "accepted": gate["accepted"],
                "tvd": abl.get("counterfactual_tvd", float("nan")),
                "pll_corrected": abl.get("pll_stereotype_corrected", float("nan")),
            })

    # ── Print results ─────────────────────────────────────────────────
    print(f"\n{'═'*90}")
    print(f"  HYBRID WEIGHT SENSITIVITY ANALYSIS — {model_name}")
    print(f"  Grid: α∈{alpha_range}  β∈{beta_range}  γ∈{gamma_range}")
    print(f"{'═'*90}")
    print(f"  {'α':>4} {'β':>5} {'γ':>5} {'ExclAbl':>8} {'Layer':>6} {'PPL×':>6} "
          f"{'#Fair↑':>7} {'Accept':>7} {'TVD':>7} {'PLL→.5':>7}")
    print("  " + "─" * 82)

    for r in rows:
        print(f"  {r['alpha']:>4.1f} {r['beta']:>5.2f} {r['gamma']:>5.1f} "
              f"{'Yes' if r['exclude_ablation'] else 'No':>8} "
              f"L{r['best_candidate']:>4} {r['ppl_ratio']:>6.3f} "
              f"{r['fairness_improved']:>7} {'✓' if r['accepted'] else '✗':>7} "
              f"{r['tvd']:>7.4f} {r['pll_corrected']:>7.4f}")

    accepted_rows = [r for r in rows if r["accepted"]]
    unique_layers = set(r["best_candidate"] for r in rows)
    accepted_layers = set(r["best_candidate"] for r in accepted_rows)

    print(f"\n  Summary:")
    print(f"    Total combinations : {len(rows)}")
    print(f"    Accepted combos    : {len(accepted_rows)} ({100*len(accepted_rows)/max(1,len(rows)):.0f}%)")
    print(f"    Unique layers selected : {sorted(unique_layers)}")
    print(f"    Layers that pass gate  : {sorted(accepted_layers)}")

    high_gamma = [r for r in rows if r["gamma"] > r["alpha"]]
    low_gamma  = [r for r in rows if r["gamma"] <= r["alpha"]]
    if high_gamma and low_gamma:
        avg_fair_high = np.mean([r["fairness_improved"] for r in high_gamma])
        avg_fair_low  = np.mean([r["fairness_improved"] for r in low_gamma])
        print(f"    Avg fairness improvements (γ>α): {avg_fair_high:.2f}")
        print(f"    Avg fairness improvements (γ≤α): {avg_fair_low:.2f}")
        print(f"    → γ>α {'does' if avg_fair_high > avg_fair_low else 'does NOT'} "
              f"consistently favour fairer layers")

    print(f"{'═'*90}")
    return rows


# ── Run on available results ──────────────────────────────────────────
sensitivity_results = {}
if "analysis_results_tinyllama" in dir():
    print("\n━━━ TinyLlama-1.1B ━━━")
    sensitivity_results["tinyllama"] = run_sensitivity_analysis(
        analysis_results_tinyllama, "TinyLlama-1.1B")
if "analysis_results_qwen" in dir():
    print("\n━━━ Qwen2.5-0.5B ━━━")
    sensitivity_results["qwen"] = run_sensitivity_analysis(
        analysis_results_qwen, "Qwen2.5-0.5B")'''

BASELINE_COMPARISONS = r'''# ═══════════════════════════════════════════════════════════════════════
# 18.  BASELINE COMPARISONS — Random, Utility-Only, Bias-Only vs Hybrid
# ═══════════════════════════════════════════════════════════════════════
#
# Compares the full hybrid ranking (α=1.0, β=0.5, γ=1.5) against:
#   1. Random layer removal       — average over N random non-boundary layers
#   2. Utility-only ranking       — α=1.0, β=0.5, γ=0.0 (no fairness signal)
#   3. Bias-only ranking          — α=0.0, β=0.0, γ=1.5 (only fairness signal)
#   4. Perplexity-only baseline   — rank by PPL impact from ablation results
#
# All baselines use the same acceptance gate.
# ═══════════════════════════════════════════════════════════════════════

import random as _rng

def run_baseline_comparisons(analysis_results, model_name, n_random_trials=20, rand_seed=42):
    """Compare hybrid ranking against simpler baselines."""
    strategy_scores  = analysis_results["strategy_scores"]
    baseline_metrics = analysis_results["baseline"]
    ablation_results = analysis_results["ablation_results"]
    hybrid_scores    = analysis_results["hybrid_scores"]
    n_layers         = analysis_results["n_layers"]

    abl_by_layer = {r["layer_removed"]: r for r in ablation_results}
    protected    = {0, n_layers - 1}
    ablatable    = [i for i in abl_by_layer if i not in protected]

    utility_keys    = ["l1_norm", "l2_norm", "taylor_importance",
                       "fisher_information", "mean_activation", "gradient_magnitude"]
    redundancy_keys = ["activation_variance", "representational_entropy"]
    bias_keys       = ["bias_attribution", "demographic_embedding",
                       "bias_prompt_sensitivity", "hybrid_bias_gradient"]

    def _recompute_hybrid(alpha, beta, gamma, bkeys=None):
        bkeys = bkeys or bias_keys
        scores = []
        for i in range(n_layers):
            u_vals = [strategy_scores[k][i] for k in utility_keys
                      if k in strategy_scores and i < len(strategy_scores[k])]
            r_vals = [1.0 - strategy_scores[k][i] for k in redundancy_keys
                      if k in strategy_scores and i < len(strategy_scores[k])]
            b_vals = [strategy_scores[k][i] for k in bkeys
                      if k in strategy_scores and i < len(strategy_scores[k])]
            u = float(np.mean(u_vals)) if u_vals else 0.0
            r = float(np.mean(r_vals)) if r_vals else 0.5
            b = float(np.mean(b_vals)) if b_vals else 0.0
            scores.append(alpha * u + beta * r - gamma * b)
        return scores

    def _pick_best(scores):
        ranked = sorted(enumerate(scores), key=lambda x: x[1])
        for idx, _ in ranked:
            if idx not in protected and idx in abl_by_layer:
                return idx
        return None

    def _eval_layer(layer_idx):
        abl = abl_by_layer[layer_idx]
        gate = strict_acceptance_gate(abl, baseline_metrics)
        target = compute_observed_target(abl, baseline_metrics)
        return {
            "layer": layer_idx,
            "accepted": gate["accepted"],
            "ppl_ratio": gate["ppl_ratio"],
            "fairness_improved": gate["fairness_improvements"],
            "observed_target": target,
            "tvd": abl.get("counterfactual_tvd", float("nan")),
            "pll_corrected": abl.get("pll_stereotype_corrected", float("nan")),
        }

    results = {}

    # 1. Full hybrid
    hybrid_layer = _pick_best([-h for h in hybrid_scores])
    if hybrid_layer is not None:
        results["hybrid (α=1.0, β=0.5, γ=1.5)"] = _eval_layer(hybrid_layer)

    # 2. Utility-only
    util_scores = _recompute_hybrid(1.0, 0.5, 0.0)
    util_layer = _pick_best(util_scores)
    if util_layer is not None:
        results["utility-only (γ=0)"] = _eval_layer(util_layer)

    # 3. Bias-only
    bias_scores = _recompute_hybrid(0.0, 0.0, 1.5)
    bias_layer = _pick_best(bias_scores)
    if bias_layer is not None:
        results["bias-only (α=β=0)"] = _eval_layer(bias_layer)

    # 4. Perplexity-only
    ppl_by_layer = {}
    for i in ablatable:
        abl = abl_by_layer[i]
        ppl_ratio = abl.get("perplexity_neutral", 1e9) / max(
            baseline_metrics.get("perplexity_neutral", 1.0), 1e-9)
        ppl_by_layer[i] = ppl_ratio
    if ppl_by_layer:
        ppl_best = min(ppl_by_layer, key=ppl_by_layer.get)
        results["perplexity-only"] = _eval_layer(ppl_best)

    # 5. Random
    _rng.seed(rand_seed)
    random_scores = [_eval_layer(_rng.choice(ablatable)) for _ in range(n_random_trials)]
    avg_random = {
        "layer": "random",
        "accepted": sum(1 for r in random_scores if r["accepted"]) / len(random_scores),
        "ppl_ratio": float(np.mean([r["ppl_ratio"] for r in random_scores])),
        "fairness_improved": float(np.mean([r["fairness_improved"] for r in random_scores])),
        "observed_target": float(np.mean([r["observed_target"] for r in random_scores])),
        "tvd": float(np.mean([r["tvd"] for r in random_scores])),
        "pll_corrected": float(np.mean([r["pll_corrected"] for r in random_scores])),
    }
    results[f"random (avg of {n_random_trials})"] = avg_random

    # 6. Hybrid without ablation score (circularity-free)
    clean_bias_keys = [k for k in bias_keys if k != "bias_attribution"]
    clean_scores = _recompute_hybrid(1.0, 0.5, 1.5, bkeys=clean_bias_keys)
    clean_layer = _pick_best(clean_scores)
    if clean_layer is not None:
        results["hybrid-no-ablation (circ-free)"] = _eval_layer(clean_layer)

    # ── Print comparison table ────────────────────────────────────────
    print(f"\n{'═'*95}")
    print(f"  BASELINE COMPARISON — {model_name}")
    print(f"{'═'*95}")
    print(f"  {'Method':<40} {'Layer':>6} {'PPL×':>6} {'#Fair↑':>7} "
          f"{'Accept':>7} {'Target':>8} {'TVD':>7}")
    print("  " + "─" * 87)

    for method, r in results.items():
        layer_str = f"L{r['layer']}" if isinstance(r['layer'], int) else r['layer']
        acc_str = f"{'✓' if r['accepted'] else '✗'}" if isinstance(r['accepted'], bool) \
                  else f"{r['accepted']:.0%}"
        print(f"  {method:<40} {layer_str:>6} {r['ppl_ratio']:>6.3f} "
              f"{r['fairness_improved']:>7.1f} {acc_str:>7} "
              f"{r['observed_target']:>8.4f} {r['tvd']:>7.4f}")

    print(f"{'═'*95}")

    # ── Pareto dominance check ────────────────────────────────────────
    hybrid_res = results.get("hybrid (α=1.0, β=0.5, γ=1.5)")
    if hybrid_res:
        print(f"\n  Pareto dominance (hybrid vs baselines):")
        for method, r in results.items():
            if method == "hybrid (α=1.0, β=0.5, γ=1.5)":
                continue
            h_target = hybrid_res["observed_target"]
            r_target = r["observed_target"]
            h_ppl    = hybrid_res["ppl_ratio"]
            r_ppl    = r["ppl_ratio"]
            if h_target >= r_target and h_ppl <= r_ppl:
                dom = "hybrid DOMINATES"
            elif r_target >= h_target and r_ppl <= h_ppl:
                dom = "baseline DOMINATES"
            else:
                dom = "neither dominates (trade-off)"
            print(f"    vs {method:<38} → {dom}")

    return results


# ── Run on available results ──────────────────────────────────────────
baseline_comparison_results = {}
if "analysis_results_tinyllama" in dir():
    print("\n━━━ TinyLlama-1.1B ━━━")
    baseline_comparison_results["tinyllama"] = run_baseline_comparisons(
        analysis_results_tinyllama, "TinyLlama-1.1B")
if "analysis_results_qwen" in dir():
    print("\n━━━ Qwen2.5-0.5B ━━━")
    baseline_comparison_results["qwen"] = run_baseline_comparisons(
        analysis_results_qwen, "Qwen2.5-0.5B")'''

PER_METRIC_DELTAS = r'''# ═══════════════════════════════════════════════════════════════════════
# 19.  PER-METRIC DELTA TABLE — Win/Loss/Tie Counts
# ═══════════════════════════════════════════════════════════════════════
#
# For EVERY ablated layer (not just the best candidate):
#   • Compute delta = baseline_metric - ablated_metric for each bias metric
#   • delta > 0 → "Win" (bias decreased)
#   • delta < -ε → "Loss" (bias increased)
#   • otherwise → "Tie"
#
# Aggregates: % of layers that reduce each metric, plus mean delta.
# ═══════════════════════════════════════════════════════════════════════

def compute_per_metric_deltas(analysis_results, model_name, epsilon=0.005):
    """Per-metric win/loss/tie analysis across all ablated layers."""
    baseline = analysis_results["baseline"]
    ablation_results = analysis_results["ablation_results"]

    fairness_metrics = [
        "pll_stereotype_score",
        "pll_stereotype_corrected",
        "counterfactual_tvd",
        "demographic_parity_gap",
        "toxicity_propensity",
    ]

    rows = []
    for metric in fairness_metrics:
        baseline_val = baseline.get(metric)
        if baseline_val is None:
            continue

        wins, losses, ties = 0, 0, 0
        deltas = []
        for abl in ablation_results:
            abl_val = abl.get(metric)
            if abl_val is None:
                continue
            delta = baseline_val - abl_val  # positive = metric decreased = good
            deltas.append(delta)
            if delta > epsilon:
                wins += 1
            elif delta < -epsilon:
                losses += 1
            else:
                ties += 1

        total = wins + losses + ties
        rows.append({
            "metric": metric,
            "baseline": baseline_val,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "total": total,
            "win_pct": 100 * wins / max(total, 1),
            "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
            "std_delta": float(np.std(deltas)) if deltas else 0.0,
            "best_delta": float(max(deltas)) if deltas else 0.0,
            "worst_delta": float(min(deltas)) if deltas else 0.0,
        })

    # ── Print table ───────────────────────────────────────────────────
    print(f"\n{'═'*100}")
    print(f"  PER-METRIC DELTA ANALYSIS — {model_name}  (ε={epsilon})")
    print(f"{'═'*100}")
    print(f"  {'Metric':<30} {'Base':>6} {'Win':>5} {'Loss':>5} {'Tie':>5} "
          f"{'Win%':>6} {'μ(Δ)':>8} {'σ(Δ)':>8} {'Best Δ':>8} {'Worst Δ':>8}")
    print("  " + "─" * 93)

    for r in rows:
        print(f"  {r['metric']:<30} {r['baseline']:>6.4f} {r['wins']:>5} "
              f"{r['losses']:>5} {r['ties']:>5} {r['win_pct']:>5.0f}% "
              f"{r['mean_delta']:>8.4f} {r['std_delta']:>8.4f} "
              f"{r['best_delta']:>8.4f} {r['worst_delta']:>8.4f}")

    # Overall summary
    avg_win_pct = np.mean([r["win_pct"] for r in rows])
    print(f"\n  Average win rate across metrics: {avg_win_pct:.1f}%")

    any_metric_where_most_wins = any(r["win_pct"] > 50 for r in rows)
    print(f"  ≥1 metric where majority of ablations improve bias: "
          f"{'Yes' if any_metric_where_most_wins else 'No'}")

    print(f"{'═'*100}")
    return rows


# ── Run ───────────────────────────────────────────────────────────────
per_metric_results = {}
if "analysis_results_tinyllama" in dir():
    print("\n━━━ TinyLlama-1.1B ━━━")
    per_metric_results["tinyllama"] = compute_per_metric_deltas(
        analysis_results_tinyllama, "TinyLlama-1.1B")
if "analysis_results_qwen" in dir():
    print("\n━━━ Qwen2.5-0.5B ━━━")
    per_metric_results["qwen"] = compute_per_metric_deltas(
        analysis_results_qwen, "Qwen2.5-0.5B")'''

BOOTSTRAP_CIS = r'''# ═══════════════════════════════════════════════════════════════════════
# 20.  BOOTSTRAP CONFIDENCE INTERVALS — Bias Metric Stability
# ═══════════════════════════════════════════════════════════════════════
#
# Resamples the 54 counterfactual prompt pairs (with replacement) ×1000
# to produce 95% confidence intervals for each bias metric, both:
#   - At BASELINE (no ablation)
#   - After ablation of the best candidate layer
#
# This answers: "Are the bias metric differences statistically significant?"
# ═══════════════════════════════════════════════════════════════════════

def bootstrap_bias_metrics(model, tokenizer, device, model_name,
                            ablation_layer=None, n_bootstrap=1000,
                            confidence=0.95, seed=42):
    """
    Bootstrap 95% CIs for bias metrics over counterfactual prompt pairs.

    Parameters
    ----------
    model, tokenizer, device : standard model inputs
    model_name : str
    ablation_layer : int or None — if set, ablate this layer before measuring
    n_bootstrap : int — number of bootstrap resamples
    confidence : float — confidence level for interval
    seed : int — random seed for reproducibility

    Returns
    -------
    dict mapping metric_name → {"mean": float, "ci_low": float, "ci_high": float}
    """
    rng = np.random.RandomState(seed)

    # Get the prompt pairs for all categories
    all_pairs = []
    for category, prompts in EXPANDED_BIAS_PROMPTS.items():
        for tmpl in prompts:
            # Build both stereotyped and anti-stereotyped versions
            all_pairs.append((category, tmpl))

    n_pairs = len(all_pairs)
    print(f"  Bootstrap: {n_bootstrap} resamples of {n_pairs} prompt pairs "
          f"({'baseline' if ablation_layer is None else f'ablating L{ablation_layer}'})")

    # Set up ablation if needed
    registry = ModelRegistry()
    layers = registry.get_layer_module_list(model, model_name)
    saved_layer = None
    if ablation_layer is not None:
        saved_layer = layers[ablation_layer]
        layers[ablation_layer] = _SkipLayer()

    try:
        # Collect per-pair scores for each metric
        pair_scores = {
            "pll_stereotype_score": [],
            "counterfactual_tvd": [],
        }

        for category, tmpl in all_pairs:
            # Measure PLL ratio for this single pair
            stereo_text = tmpl.replace("[MASK]", "he")
            anti_text = tmpl.replace("[MASK]", "she")

            with torch.no_grad():
                stereo_ids = tokenizer(stereo_text, return_tensors="pt").to(device)
                anti_ids = tokenizer(anti_text, return_tensors="pt").to(device)

                s_out = model(**stereo_ids, labels=stereo_ids["input_ids"])
                a_out = model(**anti_ids, labels=anti_ids["input_ids"])

                s_pll = -s_out.loss.item()
                a_pll = -a_out.loss.item()

            # PLL stereotype: how much model prefers stereotyped version
            total = abs(s_pll) + abs(a_pll)
            if total > 1e-12:
                bias_score = abs(s_pll) / total
            else:
                bias_score = 0.5
            pair_scores["pll_stereotype_score"].append(bias_score)

            # TVD proxy: absolute difference in PLL
            pair_scores["counterfactual_tvd"].append(abs(s_pll - a_pll))

        # Bootstrap resampling
        results = {}
        for metric, scores in pair_scores.items():
            scores_arr = np.array(scores)
            boot_means = []
            for _ in range(n_bootstrap):
                sample = rng.choice(scores_arr, size=len(scores_arr), replace=True)
                boot_means.append(sample.mean())

            boot_means = np.array(boot_means)
            alpha_half = (1 - confidence) / 2
            ci_low = np.percentile(boot_means, 100 * alpha_half)
            ci_high = np.percentile(boot_means, 100 * (1 - alpha_half))

            results[metric] = {
                "mean": float(scores_arr.mean()),
                "std": float(scores_arr.std()),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "n_pairs": len(scores_arr),
            }

    finally:
        # Restore layer
        if saved_layer is not None:
            layers[ablation_layer] = saved_layer

    return results


def run_bootstrap_analysis(model, tokenizer, device, model_name, best_layer):
    """Run bootstrap CIs for both baseline and ablated model, print comparison."""
    print(f"\n{'═'*85}")
    print(f"  BOOTSTRAP CONFIDENCE INTERVALS — {model_name}")
    print(f"  1000 resamples, 95% CI, comparing baseline vs L{best_layer} ablation")
    print(f"{'═'*85}")

    baseline_ci = bootstrap_bias_metrics(model, tokenizer, device, model_name,
                                          ablation_layer=None)
    ablated_ci  = bootstrap_bias_metrics(model, tokenizer, device, model_name,
                                          ablation_layer=best_layer)

    print(f"\n  {'Metric':<30} {'Condition':<12} {'Mean':>7} {'95% CI':>20} {'n':>5}")
    print("  " + "─" * 78)

    for metric in baseline_ci:
        b = baseline_ci[metric]
        a = ablated_ci[metric]
        print(f"  {metric:<30} {'baseline':<12} {b['mean']:>7.4f} "
              f"[{b['ci_low']:.4f}, {b['ci_high']:.4f}] {b['n_pairs']:>5}")
        print(f"  {'':<30} {'ablated':<12} {a['mean']:>7.4f} "
              f"[{a['ci_low']:.4f}, {a['ci_high']:.4f}] {a['n_pairs']:>5}")

        # Check overlap
        if a['ci_high'] < b['ci_low']:
            sig = "SIGNIFICANT reduction (CIs don't overlap)"
        elif a['ci_low'] > b['ci_high']:
            sig = "SIGNIFICANT increase (CIs don't overlap)"
        else:
            sig = "Not significant (CIs overlap)"
        print(f"  {'':<30} {'→':<12} {sig}")
        print()

    print(f"{'═'*85}")
    return {"baseline": baseline_ci, "ablated": ablated_ci}


# ── Run (requires model to be loaded) ─────────────────────────────────
# bootstrap_results = run_bootstrap_analysis(
#     model, tokenizer, device, current_model_name, best_layer=BEST_LAYER_IDX)
print("Bootstrap CI cell ready. Uncomment and set best_layer to run.")'''

RUNTIME_PROFILING = r'''# ═══════════════════════════════════════════════════════════════════════
# 21.  RUNTIME & MEMORY PROFILING — Per-Strategy Costs
# ═══════════════════════════════════════════════════════════════════════
#
# Times each of the 12 importance scoring strategies, reports:
#   - Wall-clock time per strategy
#   - Peak GPU memory increment
#   - Total pipeline time
#   - Extrapolation to 7B parameters
#
# This answers: "What is the computational overhead of the full pipeline?"
# ═══════════════════════════════════════════════════════════════════════

import time

def profile_scoring_strategies(model, model_name, tokenizer, device,
                                n_repeats=3):
    """
    Profile wall-clock time and GPU memory for each scoring strategy.

    Parameters
    ----------
    model : transformer model
    model_name : str
    tokenizer : tokenizer
    device : torch.device
    n_repeats : int — average over this many repetitions

    Returns
    -------
    dict mapping strategy_name → {"time_s": float, "mem_mb": float}
    """
    registry = ModelRegistry()
    layers = registry.get_layer_module_list(model, model_name)
    n_layers = len(layers)

    # Strategy implementations
    strategy_names = [
        "l1_norm", "l2_norm", "gradient_magnitude", "fisher_information",
        "taylor_importance", "mean_activation", "activation_variance",
        "representational_entropy", "bias_prompt_sensitivity",
        "bias_attribution", "demographic_embedding", "hybrid_bias_gradient"
    ]

    results = {}
    total_time = 0.0

    print(f"\n{'═'*75}")
    print(f"  RUNTIME PROFILING — {model_name}")
    print(f"  {n_layers} layers, device={device}, averaging over {n_repeats} runs")
    print(f"{'═'*75}")
    print(f"  {'Strategy':<35} {'Time (s)':>10} {'GPU ΔMB':>10} {'Per-layer':>12}")
    print("  " + "─" * 68)

    scorer = LayerImportanceScorer(model, model_name, tokenizer, device)

    for strat in strategy_names:
        if not hasattr(scorer, strat):
            print(f"  {strat:<35} {'N/A':>10} {'N/A':>10}")
            continue

        times = []
        mem_deltas = []
        for _ in range(n_repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated()

            t0 = time.perf_counter()
            try:
                getattr(scorer, strat)()
            except Exception as e:
                print(f"  {strat:<35} ERROR: {str(e)[:30]}")
                break
            t1 = time.perf_counter()

            times.append(t1 - t0)

            if device.type == "cuda":
                torch.cuda.synchronize()
                mem_after = torch.cuda.peak_memory_allocated()
                mem_deltas.append((mem_after - mem_before) / 1024**2)
        else:
            avg_time = float(np.mean(times))
            avg_mem = float(np.mean(mem_deltas)) if mem_deltas else 0.0
            per_layer = avg_time / n_layers

            results[strat] = {
                "time_s": avg_time,
                "mem_mb": avg_mem,
                "per_layer_s": per_layer,
            }
            total_time += avg_time

            print(f"  {strat:<35} {avg_time:>9.2f}s {avg_mem:>9.1f}MB "
                  f"{per_layer:>10.3f}s/L")

    print("  " + "─" * 68)
    print(f"  {'TOTAL':<35} {total_time:>9.2f}s")

    # Extrapolation to 7B
    model_params = sum(p.numel() for p in model.parameters()) / 1e9
    if model_params > 0:
        scale_factor = 7.0 / model_params
        est_7b = total_time * scale_factor
        print(f"\n  Model size: {model_params:.2f}B parameters")
        print(f"  Estimated time for 7B model: {est_7b:.0f}s ({est_7b/60:.1f} min) "
              f"[linear extrapolation ×{scale_factor:.1f}]")

    # Category breakdown
    norm_strats = ["l1_norm", "l2_norm"]
    grad_strats = ["gradient_magnitude", "fisher_information", "taylor_importance", "hybrid_bias_gradient"]
    act_strats  = ["mean_activation", "activation_variance", "representational_entropy"]
    bias_strats = ["bias_prompt_sensitivity", "bias_attribution", "demographic_embedding"]

    for label, group in [("Norm-based", norm_strats), ("Gradient-based", grad_strats),
                         ("Activation-based", act_strats), ("Fairness-aware", bias_strats)]:
        group_time = sum(results.get(s, {}).get("time_s", 0) for s in group)
        pct = 100 * group_time / max(total_time, 1e-9)
        print(f"  {label:<20}: {group_time:>7.2f}s ({pct:>4.1f}%)")

    print(f"{'═'*75}")
    return results


# ── Run (requires model to be loaded) ─────────────────────────────────
# profiling_results = profile_scoring_strategies(model, current_model_name, tokenizer, device)
print("Runtime profiling cell ready. Uncomment to run.")'''

# ═══════════════════════════════════════════════════════════════════
# BUILD AND SAVE
# ═══════════════════════════════════════════════════════════════════

def main():
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]
    print(f"Before: {len(cells)} cells")

    # Find the two empty cells at positions 43 and 44 (0-indexed)
    # and replace them, then append new cells
    
    # Remove last 2 empty cells
    while cells and not "".join(cells[-1].get("source", [])).strip():
        removed = cells.pop()
        print(f"  Removed empty cell id={removed.get('id', '?')}")

    # Build new cells
    new_cells = [
        make_md_cell(MD_HEADER),
        make_code_cell(SENSITIVITY_GRID),
        make_code_cell(BASELINE_COMPARISONS),
        make_code_cell(PER_METRIC_DELTAS),
        make_code_cell(BOOTSTRAP_CIS),
        make_code_cell(RUNTIME_PROFILING),
    ]

    cells.extend(new_cells)
    nb["cells"] = cells

    # Fix trailing newlines in source arrays
    for cell in new_cells:
        src = cell["source"]
        if src and src[-1].endswith("\n"):
            src[-1] = src[-1].rstrip("\n")

    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"After: {len(cells)} cells")
    print("Done! New cells appended to layer-trials.ipynb")

if __name__ == "__main__":
    main()
