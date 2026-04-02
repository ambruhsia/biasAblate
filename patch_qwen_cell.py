import json

NB_PATH = "layer-trials.ipynb"
TARGET_CELL_ID = "f4c520ca"

NEW_SOURCE = [
    "# Patch _SkipLayer for Qwen2 compatibility, then run analysis\n",
    "# Qwen2Model.forward reads decoder_layer.attention_type on each layer;\n",
    "# our skip layer must carry that attribute.\n",
    "\n",
    "import torch.nn as nn\n",
    "\n",
    "class _SafeSkipLayer(nn.Module):\n",
    "    \"\"\"Pass-through that copies arch-specific attrs from the original layer.\"\"\"\n",
    "    def __init__(self, template_layer):\n",
    "        super().__init__()\n",
    "        for attr in ('attention_type', 'layer_idx', 'is_sliding'):\n",
    "            if hasattr(template_layer, attr):\n",
    "                setattr(self, attr, getattr(template_layer, attr))\n",
    "\n",
    "    def forward(self, hidden_states, *args, **kwargs):\n",
    "        return hidden_states\n",
    "\n",
    "@torch.no_grad()\n",
    "def _bias_attribution_score_safe(self):\n",
    "    base_stereo = measure_stereotype_score(self.model, self.tokenizer, self.device)[\"overall\"]\n",
    "    scores = []\n",
    "    for i in range(self.n):\n",
    "        saved = self.layers[i]\n",
    "        try:\n",
    "            self.layers[i] = _SafeSkipLayer(saved).to(self.device)\n",
    "            stereo_without = measure_stereotype_score(self.model, self.tokenizer, self.device)[\"overall\"]\n",
    "        finally:\n",
    "            self.layers[i] = saved\n",
    "        delta = stereo_without - base_stereo\n",
    "        scores.append(-delta)\n",
    "    return self._normalize(scores)\n",
    "\n",
    "LayerImportanceScorer.bias_attribution_score = _bias_attribution_score_safe\n",
    "print(\"Patched bias_attribution_score with _SafeSkipLayer (Qwen2-safe)\")\n",
    "\n",
    "# Run Qwen analysis\n",
    "analysis_results_qwen = run_analysis_for_model(\n",
    "    'qwen2.5-0.5b',\n",
    "    alpha=1.0, beta=0.5, gamma=1.5,\n",
    ")\n",
    "print('Qwen analysis complete')",
]

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

patched = 0
for cell in nb["cells"]:
    if cell.get("id") == TARGET_CELL_ID:
        cell["source"] = NEW_SOURCE
        # Clear old outputs/execution count so VS Code reloads cleanly
        cell["outputs"] = []
        cell["execution_count"] = None
        patched += 1
        break

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Patched {patched} cell(s)")
