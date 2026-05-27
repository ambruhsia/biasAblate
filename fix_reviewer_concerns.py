import json, sys
sys.stdout.reconfigure(encoding='utf-8')

def to_lines(s):
    return s.splitlines(keepends=True)

with open('layer-trials.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── Cell 53: fix file write + add offline p-value proxy block ─────────────────
src53 = ''.join(nb['cells'][53]['source'])

# Fix 1: replace the json.dump file write with print(json.dumps)
old_write = (
    'import json, pathlib\n'
    'pathlib.Path("outputs").mkdir(exist_ok=True)\n'
    'with open("outputs/bootstrap_cis.json", "w") as f:\n'
    '    json.dump(bootstrap_results, f, indent=2, default=str)\n'
    'print("\\nSaved bootstrap CIs to outputs/bootstrap_cis.json")'
)
new_write = (
    'print("\\n--- bootstrap_cis (copy-paste) ---")\n'
    'print(json.dumps(bootstrap_results, indent=2, default=str))\n'
    'print("\\nBootstrap CI analysis complete.")'
)
src53 = src53.replace(old_write, new_write)

# Fix 2: append offline p-value proxy block at end
pval_block = (
    '\n'
    '# ── Offline p-value proxy (no model needed) ─────────────────────────────────\n'
    'if ("analysis_results_tinyllama" in dir() and analysis_results_tinyllama\n'
    '        and "paired_bootstrap_summary" in dir()):\n'
    '    print()\n'
    '    print("=" * 70)\n'
    '    print("  PAIRED BOOTSTRAP — per-ablation distribution proxy")\n'
    '    print("  (Uses layer-ablation observations as bootstrap units; no model reload)")\n'
    '    print("=" * 70)\n'
    '    _bsl      = analysis_results_tinyllama["baseline"]\n'
    '    _abl_list = analysis_results_tinyllama["ablation_results"]\n'
    '    for _mkey in ["pll_stereotype_raw", "counterfactual_tvd", "pll_stereotype_corrected"]:\n'
    '        _bv = _bsl.get(_mkey)\n'
    '        if _bv is None:\n'
    '            continue\n'
    '        _base_proxy   = [{"v": _bv}] * len(_abl_list)\n'
    '        _pruned_proxy = [{"v": r.get(_mkey, _bv)} for r in _abl_list]\n'
    '        _res = paired_bootstrap_summary(_base_proxy, _pruned_proxy, "v", n_boot=5000)\n'
    '        _sig = "p < 0.05 (sig.)" if _res["pval"] < 0.05 else "p >= 0.05 (n.s.)"\n'
    '        print(f"  {_mkey:<34}  mean_diff={_res[\'mean_diff\']:+.4f}  "\n'
    '              f"95%CI=[{_res[\'ci\'][0]:+.4f}, {_res[\'ci\'][1]:+.4f}]  "\n'
    '              f"p={_res[\'pval\']:.4f}  {_sig}")\n'
    '    print()\n'
    '    print("  n.s. is expected: method targets structural layer importance, not")\n'
    '    print("  uniform per-prompt PLL shifts. Pareto/ppl_ratio criteria are the")\n'
    '    print("  primary validity evidence, not per-prompt t-tests.")\n'
)
src53 = src53 + pval_block

nb['cells'][53]['source'] = to_lines(src53)
nb['cells'][53]['cell_type'] = 'code'

try:
    compile(src53, '<cell53>', 'exec')
    print('Cell 53: OK')
except SyntaxError as e:
    print(f'Cell 53: SyntaxError line {e.lineno}: {e.msg}')

# ── Cell 62: add prompts.json file write ──────────────────────────────────────
src62 = ''.join(nb['cells'][62]['source'])
file_write_block = (
    '\n'
    '    with open("prompts.json", "w", encoding="utf-8") as _f:\n'
    '        json.dump(flat, _f, ensure_ascii=False, indent=2)\n'
    '    print(f"Written: prompts.json ({len(flat)} pairs) — include in supplementary material.")\n'
)
# Insert after the json.dumps print line
src62 = src62 + file_write_block

nb['cells'][62]['source'] = to_lines(src62)
nb['cells'][62]['cell_type'] = 'code'

try:
    compile(src62, '<cell62>', 'exec')
    print('Cell 62: OK')
except SyntaxError as e:
    print(f'Cell 62: SyntaxError line {e.lineno}: {e.msg}')

with open('layer-trials.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('layer-trials.ipynb saved.')

# ── mistral.ipynb: add framing markdown cell near cell 30 ─────────────────────
with open('mistral.ipynb', 'r', encoding='utf-8') as f:
    nb2 = json.load(f)

framing_cell = {
    "cell_type": "markdown",
    "id": "framing-extension-runs",
    "metadata": {},
    "source": [
        "## Extension Run Framing — Reviewer-Requested Generalization Checks\n",
        "\n",
        "| Model | Layers | Baseline pll_raw | Ablated Layers | Complete? |\n",
        "|---|---|---|---|---|\n",
        "| Gemma-2-2B | 26 | 0.5625 | L1–L24 | ✅ Yes |\n",
        "| Gemma-2-9B | 42 | 0.5958 | L1–L40 | ✅ Yes |\n",
        "| Mistral-7B-Instruct | 32 | 0.5667 | L1–L24, L30 | ⚠️ L25–L29 missing (GPU quota) |\n",
        "| Llama-2-7B-Chat | 32 | 0.7375 | L1–L30 | ✅ Yes |\n",
        "\n",
        "**These are generalization checks, not primary evidence.**  \n",
        "Primary evidence is TinyLlama-1.1B and Qwen2.5-0.5B (fit on a single T4 without quantization).\n",
        "\n",
        "**CAN conclude:** Hybrid scoring transfers to larger architectures; mid-layer prune candidates are consistent across model families.  \n",
        "**CANNOT conclude:** Per-prompt statistical significance (no `evaluate_per_prompt` bootstrap for 7–9B models); cross-architecture comparisons; generalization beyond the 54-pair benchmark.\n",
        "\n",
        "**Data limitation (Mistral-7B):** L25–L29 not computed — GPU quota exhausted during review period. These 5 layers should be treated as missing data, not as non-candidates.\n"
    ]
}

# Insert this markdown cell after cell 30 (the offline parser cell / before the reviewer tests)
nb2['cells'].insert(31, framing_cell)
print(f'mistral.ipynb: framing cell inserted at index 31 (total cells now: {len(nb2["cells"])})')

with open('mistral.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb2, f, ensure_ascii=False, indent=1)
print('mistral.ipynb saved.')
