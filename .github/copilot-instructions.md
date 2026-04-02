# Bias Mitigation in LLMs — Workspace Instructions

## Project Overview

This workspace investigates **which transformer layers encode social biases** in language models and whether removing them reduces bias while maintaining model utility.

**Core Methodology**: Single-layer ablation experiments + 12 importance scoring strategies + hybrid fairness-aware scoring.

**Goal**: Identify layers for pruning to create debiased LLMs (targeting ~50% stereotype preference instead of 0–100% skewed).

---

## Architecture & File Organization

### Primary Notebooks
- **[layer-trials.ipynb](../layer-trials.ipynb)** — Production notebook. Complete framework: model loading, bias evaluation, layer importance scoring (12 strategies), hybrid scoring, heatmap visualization. **Start here for new experiments.**
- **[trials.ipynb](../trials.ipynb)** — Original exploration notebook. Reference for initial bias metric development and ablation techniques.
- **[Untitled-1.ipynb](../Untitled-1.ipynb)** — Scratch notebook for ad-hoc analysis.

### Utility Scripts
- **[patch_qwen_cell.py](../patch_qwen_cell.py)** — Patches `_bias_attribution_score()` for Qwen2 architecture compatibility (adds required `attention_type`, `layer_idx` attributes).
- **[update_notebook.py](../update_notebook.py)** — Programmatically updates notebook cells to inject architecture-specific patches.

### Outputs
- **[bias_analysis_report.txt](../bias_analysis_report.txt)** — Results log from completed experiments.
- **[layer-trials.txt](../layer-trials.txt)** — Timestamped outputs from layer-trials notebook execution.

---

## Supported Models

10 registered models in `ModelRegistry` (lines ~50–150 in layer-trials.ipynb):
- **GPT-2** (124M)
- **OPT** (125M, 350M)
- **TinyLlama** (1.1B)
- **Llama-3.2** (1B)
- **Phi** (1.3B, 2.7B)
- **Qwen2.5** (0.5B, 1.5B)

Model layers accessed via `registry.get_layer_module_list(model, model_name)` — abstracts architecture differences:
- GPT-2: `model.transformer.h`
- OPT: `model.model.decoder.layers`
- Llama/Phi/Qwen: `model.model.layers`

---

## Code Conventions & Patterns

### Function Naming
- `measure_*()` — Bias evaluation (e.g., `measure_stereotype_score`, `measure_frequency_corrected_pll`)
- `compute_*()` — Utility metrics (e.g., `compute_perplexity`)
- `plot_*()` — Visualization (e.g., `plot_layer_importance_heatmap`)
- `_*()` — Internal/helper functions
- Strategy scores: `{method}_score` (e.g., `bias_attribution_score`, `mean_activation_magnitude`)

### Code Organization (within notebooks)
Notebooks follow ASCII-delimited sections:
```python
═══════════════════════════════════════════════════════════════════════
# 1. MODEL REGISTRY — Load/cache/manage models
═══════════════════════════════════════════════════════════════════════
# 2. BIAS PROMPTS — Counterfactual templates (8 demographic categories)
═══════════════════════════════════════════════════════════════════════
# 3. BIAS MEASUREMENT FUNCTIONS — Core metrics (PLL, TVD, toxicity, perplexity)
═══════════════════════════════════════════════════════════════════════
# ... (3–6 more sections per notebook)
```

### Key Execution Patterns
- **All forward passes**: Use `@torch.no_grad()` decorator to prevent gradient accumulation
- **Gradient-based scores** (Fisher info, gradient magnitude): Explicitly call `.backward()` then `model.zero_grad()`
- **Activation capture**: Register temporary hooks via `register_forward_hook()`, clean up after use
- **Normalization**: Use `_normalize()` function to scale scores to [0, 1] range
- **Resource cleanup**: Call `model.unload()` after use to free GPU/CPU memory

### Layer Ablation (Passthrough Pattern)
```python
# Save original layer
saved_layer = layers[i]

# Replace with silent module
layers[i] = _SkipLayer()

# Measure behavior without layer
score = measure_stereotype_score(model, tokenizer, device)

# Restore
layers[i] = saved_layer
```

---

## Bias Metrics Reference

| Metric | Purpose | Range | Interpretation |
|--------|---------|-------|-----------------|
| **PLL Stereotype Score** | Pronoun likelihood bias | [0, 1] | 0.5 = unbiased; 1.0 = full bias |
| **Frequency-Corrected PLL** | PLL adjusted for token priors | [0, 1] | Removes confounds from token frequency |
| **Demographic Parity Gap** | Prediction divergence between groups | [0, 1] | 0 = equal; 1 = maximally different |
| **Counterfactual TVD** | Total variation distance of pronouns | [0, 1] | 0 = uniform; 1 = maximally skewed |
| **Toxicity Propensity** | Toxic word continuation rate | [0, 1] | 0 = safe; 1 = highly toxic |
| **Perplexity (Neutral)** | Loss on neutral text (utility check) | [1, ∞) | Lower = better utility |

---

## Hybrid Importance Scoring Formula

$$S_{\text{hybrid}} = \alpha \cdot \overline{S_{\text{utility}}} + \beta \cdot \overline{(1 - S_{\text{redundancy}})} - \gamma \cdot \overline{S_{\text{bias}}}$$

**Tunable weights** (Cell ~25, layer-trials.ipynb):
- α = 1.0 (utility contribution weight)
- β = 0.5 (redundancy suppression)
- γ = 1.5 (bias penalty — layers encoding bias are demoted)

**Rationale**: Layers that hurt model utility or encode high bias are ranked lower; layers driving redundancy are ranked higher.

---

## Common Tasks

### Load a Model
```python
registry = ModelRegistry()
model, tokenizer = registry.load("llama-3.2-1b")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### Measure Baseline Bias
```python
baseline = full_bias_evaluation(model, tokenizer, device)
# Returns: {metric_name: score_per_category, ...}
```

### Score Layer Importance (All 12 Strategies)
```python
scorer = LayerImportanceScorer(model, "llama-3.2-1b", tokenizer, device)
importance_scores = scorer.all_strategies()
# Returns: {strategy_name: [scores_per_layer], ...}
```

### Ablate & Measure Single Layer
```python
layers = registry.get_layer_module_list(model, "llama-3.2-1b")
saved = layers[5]
layers[5] = _SkipLayer()
ablated_bias = measure_stereotype_score(model, tokenizer, device)
layers[5] = saved
delta = baseline - ablated_bias  # Importance of layer 5
```

### Visualize Layer Importance
```python
plot_layer_importance_heatmap(model, "llama-3.2-1b", tokenizer, device)
# Saves PNG of strategies × layers heatmap
```

### Cleanup
```python
registry.unload("llama-3.2-1b")
torch.cuda.empty_cache()
```

---

## Known Issues & Workarounds

### 1. Qwen2 Architecture Incompatibility
**Issue**: `_bias_attribution_score()` fails on Qwen2 layers because skip-layer module lacks `attention_type` and `layer_idx` attributes.

**Workaround**:
1. Run [patch_qwen_cell.py](../patch_qwen_cell.py) to generate patch code
2. Use [update_notebook.py](../update_notebook.py) to inject patch into layer-trials.ipynb
3. Re-run ablation cells

**Details**: See [patch_qwen_cell.py](../patch_qwen_cell.py) for full patch logic.

### 2. Docker DNS Resolution in Notebooks
**Issue**: Notebook kernel runs in Docker isolation; DNS may fail ("Temporary failure in name resolution") even if host terminal works.

**Checks**:
```python
import socket
socket.gethostbyname("huggingface.co")  # Test from kernel
```

**Fix**: Update Docker's `/etc/resolv.conf` nameserver (typically to 8.8.8.8 or 1.1.1.1).

**Reference**: [/memories/repo/notebook-networking.md](/memories/repo/notebook-networking.md)

### 3. GPU Memory Pressure
**Issue**: Running 10+ models in sequence exhausts GPU memory.

**Prevention**:
- Call `registry.unload(model_name)` between experiments
- Use `torch.cuda.empty_cache()` after unload
- Monitor via `torch.cuda.memory_allocated()` in notebooks

---

## Development Patterns

### Adding a New Model
1. Add model name to `ModelRegistry.MODELS` dict (layer-trials.ipynb, Cell ~3)
2. Specify layer access path (e.g., `"model.model.layers"` for Llama)
3. Test with `registry.load(new_model_name)`

### Adding a New Bias Metric
1. Implement `measure_{metric_name}()` in Section 3 (Bias Measurement Functions)
2. Add to `full_bias_evaluation()` return dict
3. Update `bias_analysis_report.txt` with metric explanation

### Adding a New Importance Strategy
1. Implement strategy method in `LayerImportanceScorer` class
2. Add method name to `self.strategies` list (in `__init__`)
3. Test with `scorer.all_strategies()`

### Patching Architecture-Specific Code
1. Write fix in standalone Python script (e.g., [patch_qwen_cell.py](../patch_qwen_cell.py))
2. Use [update_notebook.py](../update_notebook.py) to inject into notebooks programmatically
3. Document fix location and rationale in comments

---

## Execution Recommendations

### New to the Project?
1. Start with [README.md](../README.md) for overview
2. Open [layer-trials.ipynb](../layer-trials.ipynb)
3. Run cells 1–5 to load a small model (TinyLlama or Phi-1.3B)
4. Run cell 8 to compute baseline bias
5. Run cell 15 to compute layer importance scores
6. Run cell 17 to visualize heatmap

### Running Experiments
- **Quick test** (~2 min): Phi-1.3B, single metric
- **Medium experiment** (~15 min): Llama-3.2-1b, all 6 metrics, 10 strategies
- **Full evaluation** (~45 min): All 10 models, all metrics, all strategies

### Notebook Execution Order
1. Cell 1–2: Imports & setup
2. Cell 3–5: ModelRegistry, load model
3. Cell 6–7: BIAS PROMPTS section
4. Cell 8–12: BIAS MEASUREMENT section → baseline evaluation
5. Cell 13–16: LAYER IMPORTANCE SCORING → compute all 12 strategies
6. Cell 17–19: VISUALIZATION → heatmaps & sensitivity curves
7. Cell 20+: Results aggregation & reporting

---

## AI Agent Guidance

When assisting with this project:

1. **Model Loading**: Always call `registry.unload()` after experiments to free resources
2. **Experiment Isolation**: Run models sequentially, not in parallel, to avoid GPU OOM
3. **Bias Metric Validation**: Cross-reference new metrics against papers in [README.md](../README.md)
4. **Architecture Compatibility**: Test patches on Qwen, newer phi models; they often require custom skip-layer logic
5. **Layer Naming**: Use numeric indices (0-indexed) consistently; document layer purpose in comments
6. **Gradient Cleanup**: Always clear gradients after `backward()` to prevent cascading errors
7. **Reproducibility**: Log model checksums, random seeds, and metric computation dates to `bias_analysis_report.txt`

### Error Triage
- **AttributeError on `.layers`**: Model not registered in `ModelRegistry` or layer path incorrect
- **Shape mismatches in bias scoring**: Tokenizer mismatch with model or batch size issue
- **OOM errors**: Unload previous models or reduce batch size
- **DNS errors in notebooks**: Check Docker nameserver resolution (not a code issue)

---

## Related Documentation

- **Project Overview**: [README.md](../README.md)
- **Bias Metrics**: Section 3 of [layer-trials.ipynb](../layer-trials.ipynb)
- **Model Registry**: Section 1 of [layer-trials.ipynb](../layer-trials.ipynb)
- **Results**: [bias_analysis_report.txt](../bias_analysis_report.txt)
- **Docker Networking Notes**: [/memories/repo/notebook-networking.md](/memories/repo/notebook-networking.md)

---

**Last Updated**: April 2, 2026  
**Maintainer**: Research Team  
**Status**: Active Research
