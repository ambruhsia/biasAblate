import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('layer-trials.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Helper: store source as list of lines (each ending with \n except last)
def to_lines(s):
    return s.splitlines(keepends=True)

# ── Cell 64: Bootstrap analysis ──────────────────────────────────────────────
cell64 = (
'# Bootstrap analysis (paired bootstrap over prompt pairs)\n'
'import json\n'
'import numpy as np\n'
'\n'
'def paired_bootstrap_summary(baseline_per_prompt, pruned_per_prompt, metric_key, n_boot=5000, seed=0):\n'
'    """Paired bootstrap test. baseline/pruned are lists of per-prompt metric dicts."""\n'
'    rng   = np.random.default_rng(seed)\n'
'    base  = np.array([d[metric_key] for d in baseline_per_prompt])\n'
'    pruned = np.array([d[metric_key] for d in pruned_per_prompt])\n'
'    diffs  = pruned - base\n'
'    n      = len(diffs)\n'
'    boots  = np.array([rng.choice(diffs, n).mean() for _ in range(n_boot)])\n'
'    ci     = np.percentile(boots, [2.5, 97.5])\n'
'    mean   = boots.mean()\n'
'    pval   = 2 * min((boots >= 0).mean(), (boots <= 0).mean())\n'
'    return {"mean_diff": float(mean), "ci": [float(ci[0]), float(ci[1])], "pval": float(pval)}\n'
'\n'
'print("paired_bootstrap_summary defined.")\n'
'print("Usage after evaluate_per_prompt:")\n'
"print('  result = paired_bootstrap_summary(baseline_data, pruned_data, \"cf_tvd\")')\n"
"print('  print(result)  # prints mean_diff, 95% CI, p-value')\n"
'\n'
'# Quick demo: use aggregate metrics from existing analysis results\n'
'if "analysis_results_tinyllama" in dir() and analysis_results_tinyllama:\n'
'    print()\n'
'    print("Demo: ablation-level bootstrap over TinyLlama pll_stereotype_score")\n'
'    bsl_val = analysis_results_tinyllama["baseline"]["pll_stereotype_score"]\n'
'    pruned_vals = [r["pll_stereotype_score"] for r in analysis_results_tinyllama["ablation_results"]]\n'
'    base_list   = [{"pll_stereotype_score": bsl_val}] * len(pruned_vals)\n'
'    pruned_list = [{"pll_stereotype_score": v}        for v in pruned_vals]\n'
'    r = paired_bootstrap_summary(base_list, pruned_list, "pll_stereotype_score")\n'
'    print(f"  mean_diff={r[\"mean_diff\"]:+.4f}  95%CI=[{r[\"ci\"][0]:+.4f}, {r[\"ci\"][1]:+.4f}]  p={r[\"pval\"]:.4f}")\n'
)

# ── Cell 65: Sensitivity sweep ────────────────────────────────────────────────
cell65 = (
'# Sensitivity sweep for hybrid weights (alpha, beta, gamma)\n'
'# Uses per-layer ablation metrics from analysis_results_tinyllama (no file I/O).\n'
'import json, itertools\n'
'import numpy as np\n'
'\n'
'def run_hybrid_sensitivity(analysis_results, model_name="model"):\n'
'    ablations = analysis_results.get("ablation_results", [])\n'
'    if not ablations:\n'
'        print("No ablation results."); return\n'
'    bsl = analysis_results["baseline"]\n'
'    layer_ids  = [r["layer_removed"] for r in ablations]\n'
'    ppl_ratios = np.array([r["ppl_ratio"] for r in ablations])\n'
'    utility    = 1.0 - np.clip(ppl_ratios - 1.0, 0, 1)\n'
'    redundancy = np.full(len(ablations), 0.5)\n'
'    pll_vals   = np.array([r["pll_stereotype_score"] for r in ablations])\n'
'    bias       = pll_vals\n'
'    alphas = [0.5, 1.0, 1.5]\n'
'    betas  = [0.0, 0.5, 1.0]\n'
'    gammas = [0.5, 1.5, 2.5]\n'
'    print(f"Hybrid weight sensitivity sweep -- {model_name}")\n'
'    print(f"  {len(layer_ids)} layers | baseline pll_stereotype_score={bsl[\"pll_stereotype_score\"]:.4f}")\n'
'    print()\n'
'    print(f"  {\'alpha\':>6} {\'beta\':>6} {\'gamma\':>6}  top-3 candidates")\n'
'    print("  " + "-"*50)\n'
'    rows = []\n'
'    for a, b, g in itertools.product(alphas, betas, gammas):\n'
'        raw = a * utility + b * redundancy - g * bias\n'
'        mn, mx = raw.min(), raw.max()\n'
'        norm = (raw - mn) / (mx - mn + 1e-9)\n'
'        top3_idx = np.argsort(norm)[-3:][::-1]\n'
'        top3 = [layer_ids[i] for i in top3_idx]\n'
'        rows.append({"alpha": a, "beta": b, "gamma": g, "top3": top3})\n'
'        print(f"  {a:6.1f} {b:6.1f} {g:6.1f}  L{top3[0]}, L{top3[1]}, L{top3[2]}")\n'
'    print()\n'
'    print("--- JSON (copy-paste) ---")\n'
'    print(json.dumps(rows, indent=2))\n'
'    return rows\n'
'\n'
'if "analysis_results_tinyllama" in dir() and analysis_results_tinyllama:\n'
'    run_hybrid_sensitivity(analysis_results_tinyllama, "TinyLlama-1.1B")\n'
'elif "analysis_results_qwen" in dir() and analysis_results_qwen:\n'
'    run_hybrid_sensitivity(analysis_results_qwen, "Qwen2.5-0.5B")\n'
'else:\n'
'    print("Run TinyLlama/Qwen analysis first (cells 28/30), then re-run this cell.")\n'
)

# ── Cell 66: Multi-layer ablation ─────────────────────────────────────────────
cell66 = (
'# Multi-layer ablation driver (greedy small-k)\n'
'# Results printed to output -- copy-paste for paper.\n'
'import torch, json\n'
'\n'
'@torch.no_grad()\n'
'def greedy_multi_layer_ablation(model, tokenizer, model_name, registry, evaluate_fn, k=2):\n'
'    """Greedy multi-layer ablation: iteratively remove the layer that most reduces CF-TVD.\n'
'\n'
'    Returns: (selected_layers, results_list)\n'
'    """\n'
'    layers   = registry.get_layer_module_list(model, model_name)\n'
'    n        = len(layers)\n'
'    selected = []\n'
'    results  = []\n'
'\n'
'    class _SkipLayer(torch.nn.Module):\n'
'        def forward(self, hidden_states, *args, **kwargs):\n'
'            return hidden_states\n'
'\n'
'    baseline = evaluate_fn(model, tokenizer, device)\n'
'    print(f"Baseline: tvd={baseline.get(\"counterfactual_tvd\", float(\"nan\")):.4f}  "\n'
'          f"pll={baseline.get(\"pll_stereotype_score\", float(\"nan\")):.4f}")\n'
'\n'
'    for step in range(k):\n'
'        best_delta = None\n'
'        best_idx   = None\n'
'        best_eval  = None\n'
'        for i in range(n):\n'
'            if i in selected:\n'
'                continue\n'
'            saved = layers[i]\n'
'            try:\n'
'                layers[i] = _SkipLayer().to(device)\n'
'                evals = evaluate_fn(model, tokenizer, device)\n'
'            finally:\n'
'                layers[i] = saved\n'
'            delta = baseline.get("counterfactual_tvd", 0.0) - evals.get("counterfactual_tvd", 0.0)\n'
'            if best_delta is None or delta > best_delta:\n'
'                best_delta = delta\n'
'                best_idx   = i\n'
'                best_eval  = evals\n'
'        if best_idx is None:\n'
'            break\n'
'        layers[best_idx] = _SkipLayer().to(device)\n'
'        selected.append(best_idx)\n'
'        results.append({"step": step + 1, "layer": int(best_idx),\n'
'                        "delta_tvd": float(best_delta), "eval": best_eval})\n'
'        baseline = best_eval\n'
'        print(f"  Step {step+1}: removed L{best_idx}  "\n'
'              f"delta_tvd={best_delta:+.4f}  "\n'
'              f"new_tvd={best_eval.get(\"counterfactual_tvd\", float(\"nan\")):.4f}")\n'
'\n'
'    print("\\nSelected layers:", selected)\n'
'    print("\\n--- Results JSON (copy-paste) ---")\n'
'    print(json.dumps({"selected": selected, "results": results}, indent=2))\n'
'    return selected, results\n'
'\n'
'print("greedy_multi_layer_ablation defined.")\n'
'print("Call: selected, results = greedy_multi_layer_ablation(model, tokenizer, model_name, registry, evaluate_fn)")\n'
)

# ── Cell 67: Post-LoRA comparison ─────────────────────────────────────────────
cell67 = (
'# Post-LoRA comparison: pruned vs LoRA-recovered using lora_results in memory\n'
'import json\n'
'\n'
'def compare_lora_results_inline(lora_results):\n'
'    """Print side-by-side baseline/pruned/recovered from lora_results dict."""\n'
'    bsl   = lora_results.get("baseline_eval",  {})\n'
'    prnd  = lora_results.get("pruned_eval",    {})\n'
'    recov = lora_results.get("recovered_eval", {})\n'
'    keys  = ["pll_stereotype_score", "pll_stereotype_corrected",\n'
'             "counterfactual_tvd",   "toxicity_propensity", "perplexity_neutral"]\n'
'    print("LoRA recovery comparison (baseline -> pruned -> LoRA-recovered):")\n'
'    print(f"  {\'Metric\':<32} {\'Baseline\':>10} {\'Pruned\':>10} {\'Recovered\':>12}")\n'
'    print("  " + "-"*68)\n'
'    for k in keys:\n'
'        bv = bsl.get(k, float("nan"))\n'
'        pv = prnd.get(k, float("nan"))\n'
'        rv = recov.get(k, float("nan"))\n'
'        print(f"  {k:<32} {bv:>10.4f} {pv:>10.4f} {rv:>12.4f}")\n'
'    rr = lora_results.get("recovery_ratio", float("nan"))\n'
'    print(f"\\n  Recovery ratio (LoRA): {rr:.4f}")\n'
'    out = {"baseline": bsl, "pruned": prnd, "recovered": recov, "recovery_ratio": rr}\n'
'    print("\\n--- JSON (copy-paste) ---")\n'
'    print(json.dumps(out, indent=2))\n'
'    return out\n'
'\n'
'if "lora_results" in dir() and lora_results:\n'
'    compare_lora_results_inline(lora_results)\n'
'else:\n'
'    print("lora_results not in memory.")\n'
'    print("Run the Validation Framework cell (cell 42) first to generate lora_results.")\n'
)

cells[64]['source'] = to_lines(cell64)
cells[65]['source'] = to_lines(cell65)
cells[66]['source'] = to_lines(cell66)
cells[67]['source'] = to_lines(cell67)

for i in [64, 65, 66, 67]:
    cells[i]['outputs'] = []
    # Verify syntax
    try:
        compile(''.join(cells[i]['source']), f'<cell{i}>', 'exec')
        print(f'Cell {i}: OK')
    except SyntaxError as e:
        print(f'Cell {i}: SyntaxError at line {e.lineno}: {e.msg}')
        src_lines = ''.join(cells[i]['source']).split('\n')
        for ln in range(max(0, e.lineno-3), min(len(src_lines), e.lineno+2)):
            print(f'  {ln+1}: {repr(src_lines[ln])}')

with open('layer-trials.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Saved.')
