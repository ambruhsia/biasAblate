import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('layer-trials.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

def to_lines(s):
    return s.splitlines(keepends=True)

# ── Cell 62 ──────────────────────────────────────────────────────────────────
cell62 = (
'# Export counterfactual prompt pairs for reproducibility\n'
'import json\n'
'\n'
'candidates = ["EXPANDED_BIAS_PROMPTS", "PROMPT_PAIRS", "prompt_pairs",\n'
'              "bias_prompts", "prompt_pairs_list", "prompts"]\n'
'pp, pp_name = None, None\n'
'for name in candidates:\n'
'    if name in globals() and globals()[name]:\n'
'        pp, pp_name = globals()[name], name\n'
'        break\n'
'\n'
'if pp is None:\n'
'    print("No prompt variable found -- ensure EXPANDED_BIAS_PROMPTS is defined (run Cell 21).")\n'
'else:\n'
'    flat = [[cat, a, b] for cat, pairs in pp.items() for a, b in pairs] \\\n'
'           if isinstance(pp, dict) else pp\n'
'    n_cats = len(pp) if isinstance(pp, dict) else "?"\n'
'    print(f"{pp_name} -- {len(flat)} counterfactual pairs across {n_cats} categories")\n'
'    print()\n'
'    for i, (cat, a, b) in enumerate(flat, 1):\n'
'        print(f"  [{i:2d}] [{cat}]")\n'
'        print(f"        A: {a!r}")\n'
'        print(f"        B: {b!r}")\n'
'    print()\n'
'    print("--- JSON (copy-paste) ---")\n'
'    print(json.dumps(flat, ensure_ascii=False, indent=2))\n'
)

# ── Cell 63 ──────────────────────────────────────────────────────────────────
cell63 = (
'# Per-prompt bias metric evaluation\n'
'import json\n'
'\n'
'def evaluate_per_prompt(model, tokenizer, device, bias_prompts=None):\n'
'    """Evaluate bias metrics for each prompt pair individually.\n'
'    Requires model to be loaded. Returns list of dicts.\n'
'    """\n'
'    bias_prompts = bias_prompts or EXPANDED_BIAS_PROMPTS\n'
'    results = []\n'
'    for cat, pairs in bias_prompts.items():\n'
'        for a, b in pairs:\n'
'            suite = BiasTestSuite(model, tokenizer, device, bias_prompts={cat: [(a, b)]})\n'
'            results.append({\n'
'                "category": cat, "prompt_a": a, "prompt_b": b,\n'
'                "pll_raw":   suite.measure_pll_stereotype_score(),\n'
'                "pll_corr":  suite.measure_pll_stereotype_score_corrected(),\n'
'                "cf_tvd":    suite.measure_counterfactual_tvd(),\n'
'                "stereo_kl": suite.measure_stereotype_score(),\n'
'                "toxicity":  suite.measure_toxicity_propensity(),\n'
'                "ppl":       suite.measure_perplexity_neutral(),\n'
'            })\n'
'    return results\n'
'\n'
'print("evaluate_per_prompt defined (call with loaded model + tokenizer).")\n'
'print("  Example: per_prompt_data = evaluate_per_prompt(model, tokenizer, device)")\n'
"print('  Then print: json.dumps(per_prompt_data, indent=2)')\n"
)

# ── Cell 64 ──────────────────────────────────────────────────────────────────
cell64 = (
'# Bootstrap analysis (paired bootstrap over prompt pairs)\n'
'import json\n'
'import numpy as np\n'
'\n'
'def paired_bootstrap_summary(baseline_per_prompt, pruned_per_prompt, metric_key, n_boot=5000, seed=0):\n'
'    """Paired bootstrap test. baseline/pruned are lists of per-prompt metric dicts."""\n'
'    rng    = np.random.default_rng(seed)\n'
'    base   = np.array([d[metric_key] for d in baseline_per_prompt])\n'
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
"print('Usage: result = paired_bootstrap_summary(baseline_data, pruned_data, \"cf_tvd\")')\n"
"print('       print(result)  # mean_diff, 95% CI, p-value')\n"
'\n'
'# Quick demo using existing TinyLlama analysis results\n'
'if "analysis_results_tinyllama" in dir() and analysis_results_tinyllama:\n'
'    print()\n'
'    print("Demo: bootstrap over TinyLlama pll_stereotype_score (per-ablation)")\n'
'    bsl_val     = analysis_results_tinyllama["baseline"]["pll_stereotype_score"]\n'
'    pruned_vals = [r["pll_stereotype_score"]\n'
'                   for r in analysis_results_tinyllama["ablation_results"]]\n'
'    base_list   = [{"pll_stereotype_score": bsl_val}] * len(pruned_vals)\n'
'    pruned_list = [{"pll_stereotype_score": v} for v in pruned_vals]\n'
'    r = paired_bootstrap_summary(base_list, pruned_list, "pll_stereotype_score")\n'
'    md   = r["mean_diff"]\n'
'    ci0  = r["ci"][0]\n'
'    ci1  = r["ci"][1]\n'
'    pval = r["pval"]\n'
'    print(f"  mean_diff={md:+.4f}  95%CI=[{ci0:+.4f}, {ci1:+.4f}]  p={pval:.4f}")\n'
)

# ── Cell 65 ──────────────────────────────────────────────────────────────────
cell65 = (
'# Sensitivity sweep for hybrid weights (alpha, beta, gamma)\n'
'# Uses per-layer ablation metrics from analysis_results -- no file I/O.\n'
'import json, itertools\n'
'import numpy as np\n'
'\n'
'def run_hybrid_sensitivity(analysis_results, model_name="model"):\n'
'    ablations = analysis_results.get("ablation_results", [])\n'
'    if not ablations:\n'
'        print("No ablation results."); return\n'
'    bsl        = analysis_results["baseline"]\n'
'    layer_ids  = [r["layer_removed"] for r in ablations]\n'
'    ppl_ratios = np.array([r["ppl_ratio"] for r in ablations])\n'
'    utility    = 1.0 - np.clip(ppl_ratios - 1.0, 0, 1)\n'
'    redundancy = np.full(len(ablations), 0.5)\n'
'    bias       = np.array([r["pll_stereotype_score"] for r in ablations])\n'
'    alphas = [0.5, 1.0, 1.5]\n'
'    betas  = [0.0, 0.5, 1.0]\n'
'    gammas = [0.5, 1.5, 2.5]\n'
'    bsl_pll = bsl["pll_stereotype_score"]\n'
'    print(f"Hybrid weight sensitivity sweep -- {model_name}")\n'
'    print(f"  {len(layer_ids)} layers | baseline pll_stereotype_score={bsl_pll:.4f}")\n'
'    print()\n'
'    header = f"  {\'alpha\':>6} {\'beta\':>6} {\'gamma\':>6}  top-3 candidates"\n'
'    print(header)\n'
'    print("  " + "-"*50)\n'
'    rows = []\n'
'    for a, b, g in itertools.product(alphas, betas, gammas):\n'
'        raw  = a * utility + b * redundancy - g * bias\n'
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

# ── Cell 66 ──────────────────────────────────────────────────────────────────
cell66 = (
'# Multi-layer ablation driver (greedy small-k)\n'
'# Results printed to output cell -- copy-paste for paper.\n'
'import torch, json\n'
'\n'
'@torch.no_grad()\n'
'def greedy_multi_layer_ablation(model, tokenizer, model_name, registry, evaluate_fn, k=2):\n'
'    """Greedy multi-layer ablation: iteratively remove the layer that most reduces CF-TVD.\n'
'    Returns: (selected_layers, results_list) -- results printed to output.\n'
'    """\n'
'    layers   = registry.get_layer_module_list(model, model_name)\n'
'    n        = len(layers)\n'
'    selected = []\n'
'    results  = []\n'
'\n'
'    class _SkipLayer(torch.nn.Module):\n'
'        def forward(self, hidden_states, *a, **kw):\n'
'            return hidden_states\n'
'\n'
'    baseline = evaluate_fn(model, tokenizer, device)\n'
'    bsl_tvd  = baseline.get("counterfactual_tvd", float("nan"))\n'
'    bsl_pll  = baseline.get("pll_stereotype_score", float("nan"))\n'
'    print(f"Baseline: tvd={bsl_tvd:.4f}  pll={bsl_pll:.4f}")\n'
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
'                best_delta, best_idx, best_eval = delta, i, evals\n'
'        if best_idx is None:\n'
'            break\n'
'        layers[best_idx] = _SkipLayer().to(device)\n'
'        selected.append(best_idx)\n'
'        results.append({"step": step + 1, "layer": int(best_idx),\n'
'                        "delta_tvd": float(best_delta), "eval": best_eval})\n'
'        baseline   = best_eval\n'
'        new_tvd    = best_eval.get("counterfactual_tvd", float("nan"))\n'
'        print(f"  Step {step+1}: removed L{best_idx}  "\n'
'              f"delta_tvd={best_delta:+.4f}  new_tvd={new_tvd:.4f}")\n'
'\n'
'    print("\\nSelected layers:", selected)\n'
'    print("\\n--- Results JSON (copy-paste) ---")\n'
'    print(json.dumps({"selected": selected, "results": results}, indent=2))\n'
'    return selected, results\n'
'\n'
'print("greedy_multi_layer_ablation defined.")\n'
'print("Call: selected, results = greedy_multi_layer_ablation(model, tokenizer, model_name, registry, evaluate_fn)")\n'
)

# ── Cell 67 ──────────────────────────────────────────────────────────────────
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
'    hdr = f"  {\'Metric\':<32} {\'Baseline\':>10} {\'Pruned\':>10} {\'Recovered\':>12}"\n'
'    print(hdr)\n'
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

for idx, src in [(62, cell62), (63, cell63), (64, cell64),
                 (65, cell65), (66, cell66), (67, cell67)]:
    # Force cell type to 'code'
    cells[idx]['cell_type'] = 'code'
    cells[idx]['source']    = to_lines(src)
    cells[idx]['outputs']   = []
    if 'metadata' not in cells[idx]:
        cells[idx]['metadata'] = {}
    # Verify syntax
    try:
        compile(src, f'<cell{idx}>', 'exec')
        print(f'Cell {idx}: OK (code)')
    except SyntaxError as e:
        print(f'Cell {idx}: SyntaxError line {e.lineno}: {e.msg}')
        lines = src.split('\n')
        for ln in range(max(0, e.lineno-3), min(len(lines), e.lineno+2)):
            print(f'  {ln+1}: {repr(lines[ln])}')

with open('layer-trials.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Saved.')
