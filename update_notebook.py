import json


def to_lines(text: str):
    return [line + "\n" for line in text.strip("\n").split("\n")]


PATCH_CELL_SOURCE = to_lines(
    """
# Architecture-agnostic ablation patch + robust hybrid scoring + safe runner

def _create_ablated_model_arch_agnostic(self, layer_to_remove):
    import copy
    pruned = copy.deepcopy(self.model)

    module_list = self.registry.get_layer_module_list(pruned, self.model_name)
    layers = list(module_list)

    if layer_to_remove < 0 or layer_to_remove >= len(layers):
        raise IndexError(f"layer_to_remove {layer_to_remove} out of range for {len(layers)} layers")

    del layers[layer_to_remove]
    new_layers = torch.nn.ModuleList(layers)

    if hasattr(pruned, 'model') and hasattr(pruned.model, 'layers'):
        pruned.model.layers = new_layers
        if hasattr(pruned, 'config') and hasattr(pruned.config, 'num_hidden_layers'):
            pruned.config.num_hidden_layers = len(new_layers)
    elif hasattr(pruned, 'model') and hasattr(pruned.model, 'decoder') and hasattr(pruned.model.decoder, 'layers'):
        pruned.model.decoder.layers = new_layers
        if hasattr(pruned, 'config') and hasattr(pruned.config, 'num_hidden_layers'):
            pruned.config.num_hidden_layers = len(new_layers)
    elif hasattr(pruned, 'transformer') and hasattr(pruned.transformer, 'h'):
        pruned.transformer.h = new_layers
        if hasattr(pruned, 'config') and hasattr(pruned.config, 'n_layer'):
            pruned.config.n_layer = len(new_layers)
    else:
        raise ValueError(f"Unsupported architecture for ablation (model_type={getattr(pruned.config, 'model_type', 'unknown')})")

    pruned.eval()
    return pruned


@torch.no_grad()
def _bias_attribution_score_qwen_safe(self):
    base_stereo = measure_stereotype_score(self.model, self.tokenizer, self.device)["overall"]
    scores = []

    class _SafeSkipLayer(nn.Module):
        def __init__(self, template_layer):
            super().__init__()
            for attr in ("attention_type", "layer_idx", "is_sliding"):
                if hasattr(template_layer, attr):
                    setattr(self, attr, getattr(template_layer, attr))

        def forward(self, hidden_states, *args, **kwargs):
            return hidden_states

    for i in range(self.n):
        saved = self.layers[i]
        try:
            self.layers[i] = _SafeSkipLayer(saved).to(self.device)
            stereo_without = measure_stereotype_score(self.model, self.tokenizer, self.device)["overall"]
        finally:
            self.layers[i] = saved
        scores.append(-(stereo_without - base_stereo))

    return self._normalize(scores)


def _compute_hybrid_importance_robust(self):
    if self._strategy_scores is None:
        self.compute_all_strategies()

    S = self._strategy_scores
    utility_keys = ["l1_norm", "l2_norm", "taylor_importance", "fisher_information", "mean_activation", "gradient_magnitude"]
    redundancy_keys = ["activation_variance", "representational_entropy"]
    bias_keys = ["bias_attribution", "demographic_embedding", "bias_prompt_sensitivity", "hybrid_bias_gradient"]

    def _safe_vals(keys, idx):
        vals = []
        for k in keys:
            if k in S and idx < len(S[k]):
                v = float(S[k][idx])
                if np.isfinite(v):
                    vals.append(v)
        return vals

    def _safe_mean(vals, default=0.0):
        return float(np.mean(vals)) if vals else float(default)

    raw = []
    rows = []
    for i in range(self.n_layers):
        util = _safe_mean(_safe_vals(utility_keys, i), 0.0)
        redun = _safe_mean([1.0 - x for x in _safe_vals(redundancy_keys, i)], 0.5)
        bias = _safe_mean(_safe_vals(bias_keys, i), 0.0)
        score = self.alpha * util + self.beta * redun - self.gamma * bias
        if not np.isfinite(score):
            score = 0.0
        rows.append((i, util, redun, bias, float(score)))
        raw.append(float(score))

    arr = np.array(raw, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
    mn, mx = float(np.min(arr)), float(np.max(arr))
    self._hybrid_scores = [0.5] * len(arr) if (not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < 1e-12) else ((arr - mn) / (mx - mn)).tolist()

    print("\nHybrid score trace (per layer):")
    print("  Layer |  utility  redundancy    bias    raw_hybrid   norm_hybrid")
    print("  " + "-" * 66)
    for (i, util, redun, bias, raw_score), norm in zip(rows, self._hybrid_scores):
        print(f"  L{i:>2d}   |  {util:>7.4f}    {redun:>7.4f}   {bias:>7.4f}    {raw_score:>9.4f}     {norm:>9.4f}")

    print(f"  Finite raw hybrid scores: {int(np.isfinite(np.array(raw, dtype=np.float64)).sum())}/{self.n_layers}")
    return self._hybrid_scores


def _find_least_important_layers_robust(self, percentile=30, protect_boundaries=True):
    if self._hybrid_scores is None:
        self.compute_hybrid_importance()

    scores = np.array(self._hybrid_scores, dtype=np.float64)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    ranking = sorted(enumerate(scores.tolist()), key=lambda x: x[1])
    protected = {0, self.n_layers - 1} if protect_boundaries else set()
    threshold = float(np.percentile(scores, percentile))
    candidates = [idx for idx, s in ranking if s <= threshold and idx not in protected]

    print(f"\n{'═'*65}")
    print(f"  Layer Importance Ranking — {self.model_name}")
    print(f"  Hybrid weights: α={self.alpha}, β={self.beta}, γ={self.gamma}")
    print(f"{'═'*65}")
    print(f"  {'Layer':<8} {'Hybrid Score':<14} {'Status'}")
    print(f"  {'─'*50}")
    for idx, score in ranking:
        if idx in protected:
            status = "🛡️ PROTECTED"
        elif idx in set(candidates):
            status = "✂️ PRUNE CANDIDATE"
        else:
            status = "  keep"
        print(f"  Layer {idx:<4} {score:<14.6f} {status}")
    print(f"{'═'*65}")
    print(f"  Threshold (p{percentile}): {threshold:.6f}")
    print(f"  Prune candidates: {candidates}")
    print(f"  Protected layers: {sorted(protected)}")

    return {"ranking": ranking, "candidates": candidates, "threshold": threshold, "protected": protected}


LayerImportanceScorer.bias_attribution_score = _bias_attribution_score_qwen_safe
LayerImportanceFinder._create_ablated_model = _create_ablated_model_arch_agnostic
LayerImportanceFinder.compute_hybrid_importance = _compute_hybrid_importance_robust
LayerImportanceFinder.find_least_important_layers = _find_least_important_layers_robust


def run_analysis_for_model(model_key, alpha=1.0, beta=0.5, gamma=1.5, dtype=None, overflow_safe=True):
    clean_key = model_key.strip()
    if dtype is None:
        dtype = torch.float32 if (overflow_safe and clean_key.startswith("qwen")) else torch.float16
    print(f"\nRunning {clean_key} with dtype={dtype} (overflow_safe={overflow_safe})")
    m, t = registry.load(clean_key, dtype=dtype)
    try:
        f = LayerImportanceFinder(m, clean_key, t, device, registry, alpha=alpha, beta=beta, gamma=gamma)
        return f.run_layer_by_layer_analysis()
    finally:
        registry.unload(clean_key, dtype=dtype)
        gc.collect()
        torch.cuda.empty_cache()


print("Patched LayerImportanceScorer/LayerImportanceFinder for Qwen-safe bias attribution and robust hybrid scoring")
print("Hybrid score trace will now print per-layer values during analysis")
print("Qwen runs default to float32 via run_analysis_for_model(..., overflow_safe=True)")
"""
)


QWEN_RUN_SOURCE = to_lines(
    """
# Run Qwen analysis with robust hybrid scoring + overflow-safe dtype
analysis_results_qwen = run_analysis_for_model(
    'qwen2.5-0.5b',
    alpha=1.0, beta=0.5, gamma=1.5,
    overflow_safe=True,
    dtype=torch.float32,
)
print('Qwen analysis complete')
"""
)


with open("layer-trials.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

patched = 0
for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    cid = cell.get("id")
    if cid == "#VSC-1f50a035":
        cell["source"] = PATCH_CELL_SOURCE
        patched += 1
    elif cid == "#VSC-c4ff94a3":
        cell["source"] = QWEN_RUN_SOURCE
        patched += 1

with open("layer-trials.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=4)

print(f"Updated notebook cells: {patched}")
