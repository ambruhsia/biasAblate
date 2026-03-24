import json

with open("layer-trials.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "class LayerImportanceFinder:" in source:
            print("Found LayerImportanceFinder class.")
            
            # 1. Update find_least_important_layers return statement
            old_return = '''        return {
            "ranking": ranking,
            "candidates": candidates,
            "protected": sorted(protected),
            "threshold": threshold,
        }'''
            new_return = '''        return {
            "ranking": ranking,
            "threshold": threshold,
            "protected": protected
        }'''
            source = source.replace(old_return, new_return)
            
            # 2. Update run_full_analysis -> run_layer_by_layer_analysis
            old_run = '''    # ══════════════════════════════════════════════════════════════════
    # D.  RUN FULL ANALYSIS — Prune at 10 ratios, evaluate each
    # ══════════════════════════════════════════════════════════════════

    def run_full_analysis(self, ratios=None, verbose=True):
        """
        Complete analysis pipeline for one model:
          1. Compute all 12 strategy scores
          2. Compute hybrid importance
          3. Identify least-important layers
          4. Prune at each of 10 ratios (5%–50%)
          5. Evaluate bias after each prune using BiasTestSuite
          6. Return structured results

        Args:
            ratios:  list of floats (default: EXTENDED_PRUNE_RATIOS)
            verbose: print progress

        Returns:
            dict with:
              - 'model_name', 'n_layers'
              - 'strategy_scores': dict[strategy -> list[float]]
              - 'hybrid_scores': list[float]
              - 'layer_ranking': from find_least_important_layers()
              - 'baseline': bias metrics before pruning
              - 'pruning_results': list of dicts, one per ratio
        """
        ratios = ratios or EXTENDED_PRUNE_RATIOS

        # Step 1-3: Importance computation
        if verbose:
            print(f"\\n{'━'*65}")
            print(f"  FULL ANALYSIS: {self.model_name} ({self.n_layers} layers)")
            print(f"{'━'*65}")

        self.compute_all_strategies()
        self.compute_hybrid_importance()
        layer_info = self.find_least_important_layers()

        # Step 4: Baseline evaluation
        if verbose:
            print(f"\\n── Baseline (no pruning) ──")
        suite = BiasTestSuite(self.model, self.tokenizer, self.device)
        baseline = suite.run_full_evaluation()
        baseline["ratio"] = 0.0
        baseline["layers_removed"] = []
        baseline["layers_remaining"] = self.n_layers
        if verbose:
            self._print_metrics(baseline, "Baseline")

        # Step 5-6: Prune at each ratio and evaluate
        pruning_results = []
        for ratio in ratios:
            pct = int(ratio * 100)
            if verbose:
                print(f"\\n── Pruning {pct}% of layers ──")

            pruned, removed = prune_model(
                self.model, self.model_name,
                self._hybrid_scores, ratio
            )
            remaining = len(self.registry.get_layer_module_list(
                pruned, self.model_name
            ))

            if verbose:
                print(f"  Removed: {removed}  |  Remaining: {remaining}")

            p_suite = BiasTestSuite(pruned, self.tokenizer, self.device)
            evals = p_suite.run_full_evaluation()
            evals["ratio"] = ratio
            evals["layers_removed"] = removed
            evals["layers_remaining"] = remaining

            if verbose:
                self._print_metrics(evals, f"{pct}% pruned")

            pruning_results.append(evals)

            del pruned
            gc.collect()
            torch.cuda.empty_cache()

        self._analysis_results = {
            "model_name": self.model_name,
            "n_layers": self.n_layers,
            "hybrid_weights": {
                "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma
            },
            "strategy_scores": self._strategy_scores,
            "hybrid_scores": self._hybrid_scores,
            "layer_ranking": layer_info,
            "baseline": baseline,
            "pruning_results": pruning_results,
        }
        return self._analysis_results'''
            new_run = '''    # ══════════════════════════════════════════════════════════════════
    # D.  RUN LAYER-BY-LAYER ANALYSIS
    # ══════════════════════════════════════════════════════════════════

    def run_layer_by_layer_analysis(self, protect_boundaries=True, verbose=True):
        """
        Test pruning each layer individually (single-layer ablation).
        """
        self.compute_all_strategies()
        self.compute_hybrid_importance()

        if verbose:
            print(f"\\n{'━'*65}")
            print(f"  SINGLE-LAYER ABLATION ANALYSIS: {self.model_name} ({self.n_layers} layers)")
            print(f"{'━'*65}")

        # 1. Baseline Evaluation (No Pruning)
        if verbose:
            print(f"\\n── Baseline (no pruning) ──")
        suite = BiasTestSuite(self.model, self.tokenizer, self.device)
        baseline = suite.run_full_evaluation()
        baseline["layer_removed"] = None
        if verbose:
            self._print_metrics(baseline, "Baseline")

        # 2. Iterate through each layer and ablate
        ablation_results = []

        layer_info = self.find_least_important_layers(protect_boundaries=protect_boundaries)
        protected_layers = layer_info["protected"]

        for layer_idx in range(self.n_layers):
            if layer_idx in protected_layers:
                if verbose:
                    print(f"\\n── Skipping Layer {layer_idx} (Protected) ──")
                continue

            if verbose:
                print(f"\\n── Ablating Layer {layer_idx} ──")

            # Create a model with ONLY this layer removed
            pruned_model = self._create_ablated_model(layer_idx)

            # Evaluate the pruned model
            p_suite = BiasTestSuite(pruned_model, self.tokenizer, self.device)
            evals = p_suite.run_full_evaluation()
            evals["layer_removed"] = layer_idx
            evals["hybrid_score"] = self._hybrid_scores[layer_idx]

            if verbose:
                self._print_metrics(evals, f"L{layer_idx} pruned")

            ablation_results.append(evals)

            # Cleanup memory
            del pruned_model
            gc.collect()
            torch.cuda.empty_cache()

        self._analysis_results = {
            "model_name": self.model_name,
            "n_layers": self.n_layers,
            "hybrid_weights": {
                "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma
            },
            "hybrid_scores": self._hybrid_scores,
            "baseline": baseline,
            "ablation_results": ablation_results,
        }

        return self._analysis_results

    def _create_ablated_model(self, layer_to_remove):
        """
        Creates a completely new copy of the model with one layer missing,
        so we don't permanently alter the base model during the loop.
        """
        import copy
        # Standard shallow/deep copy trick for huggingface models
        pruned = copy.deepcopy(self.model)

        module_list = self.registry.get_layer_module_list(pruned, self.model_name)
        # Convert ModuleList to normal python list to manipulate
        layers = list(module_list)

        # Remove the specific layer
        del layers[layer_to_remove]

        # Re-assign back to the model
        if self.model_name in ["llama-2-7b", "llama-3.2-1b", "llama-3.2-3b"]:
            pruned.model.layers = torch.nn.ModuleList(layers)
            pruned.config.num_hidden_layers = len(layers)
        elif self.model_name in ["gpt2", "gpt-neo-125m"]:
            pruned.transformer.h = torch.nn.ModuleList(layers)
            pruned.config.n_layer = len(layers)
        elif "opt" in self.model_name:
            pruned.model.decoder.layers = torch.nn.ModuleList(layers)
            pruned.config.num_hidden_layers = len(layers)
        else:
            raise ValueError(f"Unsupported architecture for ablation: {self.model_name}")

        return pruned'''
            if old_run in source:
                print("Replaced run_full_analysis")
                source = source.replace(old_run, new_run)
            else:
                print("Warning: old_run not found")
                
            # 3. Update analyze_all_models
            old_analyze = '''    @classmethod
    def analyze_all_models(cls, reg, device, model_names=None,
                           ratios=None, alpha=None, beta=None, gamma=None):
        """
        Run run_full_analysis() on each model in the registry.
        Loads → analyzes → unloads each model to conserve memory.

        Args:
            reg:         ModelRegistry instance
            device:      torch device
            model_names: list of keys (default: all non-gated models)
            ratios:      pruning ratios (default: EXTENDED_PRUNE_RATIOS)

        Returns:
            dict[model_name -> analysis_results]
        """
        if model_names is None:
            model_names = [
                k for k, v in ModelRegistry.MODELS.items()
                if not v["gated"]
            ]

        all_results = {}
        for name in model_names:
            print(f"\\n{'╔'+'═'*63+'╗'}")
            print(f"  ANALYZING MODEL: {name}")
            print(f"{'╚'+'═'*63+'╝'}")

            model, tokenizer = reg.load(name)
            finder = cls(model, name, tokenizer, device, reg,
                         alpha=alpha, beta=beta, gamma=gamma)
            results = finder.run_full_analysis(ratios=ratios)
            all_results[name] = results

            # Show pruning candidates for this model
            finder.show_pruning_candidates()

            # Unload to free memory
            reg.unload(name)
            gc.collect()
            torch.cuda.empty_cache()

        # Cross-model summary
        cls._print_cross_model_summary(all_results)
        return all_results'''
            new_analyze = '''    @classmethod
    def analyze_all_models(cls, reg, device, model_names=None,
                           alpha=None, beta=None, gamma=None):
        """
        Run run_layer_by_layer_analysis() on each model in the registry.
        Loads → analyzes → unloads each model to conserve memory.

        Args:
            reg:         ModelRegistry instance
            device:      torch device
            model_names: list of keys (default: all non-gated models)

        Returns:
            dict[model_name -> analysis_results]
        """
        if model_names is None:
            model_names = [
                k for k, v in ModelRegistry.MODELS.items()
                if not v["gated"]
            ]

        all_results = {}
        for name in model_names:
            print(f"\\n{'╔'+'═'*63+'╗'}")
            print(f"  ANALYZING MODEL: {name}")
            print(f"{'╚'+'═'*63+'╝'}")

            model, tokenizer = reg.load(name)
            finder = cls(model, name, tokenizer, device, reg,
                         alpha=alpha, beta=beta, gamma=gamma)
            results = finder.run_layer_by_layer_analysis()
            all_results[name] = results

            # Unload to free memory
            reg.unload(name)
            gc.collect()
            torch.cuda.empty_cache()

        # Cross-model summary
        cls._print_cross_model_summary(all_results)
        return all_results'''
            if old_analyze in source:
                print("Replaced analyze_all_models")
                source = source.replace(old_analyze, new_analyze)
            else:
                print("Warning: old_analyze not found")
                
            # 4. Updates for plotting methods
            old_plots = '''    def plot_pruning_sweep(self):
        """Bias metrics vs pruning % for all 10 ratios."""
        if self._analysis_results is None:
            raise RuntimeError("Run run_full_analysis() first.")

        base = self._analysis_results["baseline"]
        prs = self._analysis_results["pruning_results"]

        metrics = [
            ("stereotype_overall", "Stereotype Score (↓ = better)"),
            ("demographic_parity_gap", "Demographic Parity Gap (↓ = better)"),
            ("toxicity_propensity", "Toxicity Propensity (↓ = better)"),
            ("perplexity_neutral", "Perplexity — neutral (↓ = better)"),
            ("profession_gender_bias", "Profession-Gender Bias (↓ = better)"),
            ("demographic_ppl_variance", "Demographic PPL Variance (↓ = better)"),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f"Bias Metrics vs Pruning % — {self.model_name} "
                     f"(Hybrid Scoring)", fontsize=14, fontweight="bold")

        for ax, (metric, title) in zip(axes.flat, metrics):
            xs = [r["ratio"] * 100 for r in prs]
            ys = [r.get(metric, 0) for r in prs]
            ax.plot(xs, ys, "o-", color="steelblue", markersize=6,
                    linewidth=2, label="hybrid pruning")
            ax.axhline(y=base.get(metric, 0), color="red", linestyle="--",
                        linewidth=1.5, label="baseline")
            ax.set_xlabel("Layers Removed (%)")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"pruning_sweep_{self.model_name}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved {fname}")

    @staticmethod
    def plot_cross_model_comparison(all_results):
        """Side-by-side comparison of pruning outcomes across models."""
        if not all_results:
            print("No results to plot.")
            return

        model_names = list(all_results.keys())
        metrics = ["stereotype_overall", "demographic_parity_gap",
                    "toxicity_propensity", "perplexity_neutral"]

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle("Cross-Model Pruning Comparison (Hybrid Scoring)",
                     fontsize=14, fontweight="bold")

        for ax, metric in zip(axes.flat, metrics):
            for mname in model_names:
                res = all_results[mname]
                base_val = res["baseline"].get(metric, 0)
                xs = [r["ratio"] * 100 for r in res["pruning_results"]]
                ys = [r.get(metric, 0) for r in res["pruning_results"]]
                ax.plot(xs, ys, "o-", markersize=4, label=mname)
                ax.axhline(y=base_val, linestyle=":", alpha=0.3)

            ax.set_xlabel("Layers Removed (%)")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(metric.replace("_", " ").title())
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("cross_model_comparison.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved cross_model_comparison.png")'''
            new_plots = '''    def plot_single_layer_ablation_impact(self):
        """
        Plot the impact on bias metrics when each individual layer is removed.
        Shows absolute metric values per layer ablated, with baseline as dashed line.
        """
        if self._analysis_results is None:
            raise RuntimeError("Run run_layer_by_layer_analysis() first.")

        res = self._analysis_results
        base = res["baseline"]
        ablations = res["ablation_results"]

        metrics = [
            ("stereotype_overall", "Stereotype Score (↓ = better)"),
            ("demographic_parity_gap", "Demographic Parity Gap (↓ = better)"),
            ("toxicity_propensity", "Toxicity Propensity (↓ = better)"),
            ("perplexity_neutral", "Perplexity — neutral (↓ = better)"),
            ("profession_gender_bias", "Profession-Gender Bias (↓ = better)"),
            ("demographic_ppl_variance", "Demographic PPL Var (↓ = better)")
        ]

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f"Single-Layer Ablation Impact — {self.model_name}", fontsize=16, fontweight="bold")

        layers = [r["layer_removed"] for r in ablations]

        for ax, (metric, title) in zip(axes.flat, metrics):
            base_val = base.get(metric, 0)
            vals = [r.get(metric, 0) for r in ablations]

            ax.bar(layers, vals, color="skyblue", edgecolor="black", alpha=0.7)
            ax.axhline(y=base_val, color="red", linestyle="--", linewidth=2, label=f"Baseline: {base_val:.4f}")
            
            ax.set_xticks(layers)
            ax.set_xlabel("Layer Removed")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(title)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fname = f"ablation_impact_{self.model_name}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved {fname}")

    @staticmethod
    def plot_cross_model_comparison(all_results):
        """
        Side-by-side comparison of the BEST single-layer ablation outcome across models.
        """
        if not all_results:
            print("No results to plot.")
            return

        model_names = list(all_results.keys())
        metrics = ["stereotype_overall", "demographic_parity_gap"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Best Single-Layer Ablation Comparison", fontsize=14, fontweight="bold")

        for ax, metric in zip(axes, metrics):
            baselines = []
            bests = []
            best_layers = []
            
            for mname in model_names:
                res = all_results[mname]
                base_val = res["baseline"].get(metric, 0)
                baselines.append(base_val)
                
                # Find best ablation for this metric
                ablations = res["ablation_results"]
                best_ablation = min(ablations, key=lambda r: r.get(metric, float('inf')))
                bests.append(best_ablation.get(metric, 0))
                best_layers.append(f"L{best_ablation['layer_removed']}")

            x = np.arange(len(model_names))
            width = 0.35

            ax.bar(x - width/2, baselines, width, label='Baseline', color='lightgray')
            rects = ax.bar(x + width/2, bests, width, label='Best Ablation', color='steelblue')

            # Label the bars with the layer removed
            for i, rect in enumerate(rects):
                height = rect.get_height()
                ax.annotate(best_layers[i],
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(metric.replace("_", " ").title())
            ax.set_xticks(x)
            ax.set_xticklabels(model_names)
            ax.legend()

        plt.tight_layout()
        plt.savefig("cross_model_ablation_comparison.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved cross_model_ablation_comparison.png")'''
            if old_plots in source:
                print("Replaced plot methods")
                source = source.replace(old_plots, new_plots)
            else:
                print("Warning: old_plots not found")

            # 5. Export results update
            old_export = '''    def export_results(self, filename=None):
        """Save analysis results to JSON + print summary report."""
        if self._analysis_results is None:
            raise RuntimeError("Run run_full_analysis() first.")

        res = self._analysis_results
        fname = filename or f"layer_importance_{self.model_name}.json"

        # Make JSON-serializable (remove _detail sub-dicts)
        export = {
            "model_name": res["model_name"],
            "n_layers": res["n_layers"],
            "hybrid_weights": res["hybrid_weights"],
            "hybrid_scores": res["hybrid_scores"],
            "layer_ranking": {
                "candidates": res["layer_ranking"]["candidates"],
                "protected": res["layer_ranking"]["protected"],
                "threshold": res["layer_ranking"]["threshold"],
            },
            "baseline": {k: v for k, v in res["baseline"].items()
                         if not k.startswith("_detail")},
            "pruning_results": [
                {k: v for k, v in r.items() if not k.startswith("_detail")}
                for r in res["pruning_results"]
            ],
        }

        with open(fname, "w") as f:
            json.dump(export, f, indent=2, default=str)
        print(f"Results saved to {fname}")

        # ── Text summary report ──────────────────────────────────────
        report_name = fname.replace(".json", "_report.txt")
        lines = []
        lines.append(f"{'='*65}")
        lines.append(f"  LAYER IMPORTANCE ANALYSIS REPORT")
        lines.append(f"  Model: {res['model_name']}  ({res['n_layers']} layers)")
        lines.append(f"  Hybrid weights: α={self.alpha}, β={self.beta}, "
                     f"γ={self.gamma}")
        lines.append(f"{'='*65}")
        lines.append("")

        # Baseline
        lines.append("BASELINE (no pruning):")
        for k, v in res["baseline"].items():
            if not k.startswith("_detail") and k not in (
                "ratio", "layers_removed", "layers_remaining"):
                lines.append(f"  {k}: {v}")
        lines.append("")

        # Layer ranking
        lines.append("LAYER IMPORTANCE RANKING (ascending):")
        for idx, score in res["layer_ranking"]["ranking"]:
            marker = " ← PRUNE" if idx in set(
                res["layer_ranking"]["candidates"]) else ""
            lines.append(f"  Layer {idx:>2}: {score:.6f}{marker}")
        lines.append(f"\\nPrune candidates: {res['layer_ranking']['candidates']}")
        lines.append(f"Protected layers: {res['layer_ranking']['protected']}")
        lines.append("")

        # Pruning sweep results
        lines.append("PRUNING SWEEP RESULTS:")
        lines.append(f"  {'Ratio':<8} {'Stereotype':<12} {'Parity':<12} "
                     f"{'Toxicity':<12} {'PPL':<12} {'Removed'}")
        lines.append(f"  {'─'*70}")
        for r in res["pruning_results"]:
            lines.append(
                f"  {r['ratio']:<8.0%} "
                f"{r.get('stereotype_overall', 0):<12.6f} "
                f"{r.get('demographic_parity_gap', 0):<12.6f} "
                f"{r.get('toxicity_propensity', 0):<12.6f} "
                f"{r.get('perplexity_neutral', 0):<12.2f} "
                f"{r.get('layers_removed', [])}"
            )
        lines.append("")

        # Best configuration
        best = min(res["pruning_results"],
                   key=lambda r: r.get("stereotype_overall", float("inf")))
        lines.append(f"BEST CONFIGURATION (lowest stereotype score):")
        lines.append(f"  Ratio: {best['ratio']:.0%}")
        lines.append(f"  Layers removed: {best.get('layers_removed', [])}")
        lines.append(f"  Stereotype: {best.get('stereotype_overall', 0):.6f}")
        lines.append(f"  Parity gap: {best.get('demographic_parity_gap', 0):.6f}")
        lines.append(f"  Perplexity: {best.get('perplexity_neutral', 0):.2f}")

        report = "\\n".join(lines)
        with open(report_name, "w") as f:
            f.write(report)
        print(f"Report saved to {report_name}")
        print(report)

    # ══════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _normalize(scores):
        """Min-max normalize to [0, 1]."""
        mn, mx = min(scores), max(scores)
        if mx - mn < 1e-12:
            return [0.5] * len(scores)
        return [(s - mn) / (mx - mn) for s in scores]

    @staticmethod
    def _print_metrics(m, label):
        """Print a compact metrics summary line."""
        print(f"  [{label}]  stereo={m.get('stereotype_overall', 0):.4f}  "
              f"parity={m.get('demographic_parity_gap', 0):.4f}  "
              f"toxic={m.get('toxicity_propensity', 0):.4f}  "
              f"ppl={m.get('perplexity_neutral', 0):.2f}  "
              f"prof_bias={m.get('profession_gender_bias', 0):.4f}  "
              f"demo_var={m.get('demographic_ppl_variance', 0):.4f}")

    @staticmethod
    def _print_cross_model_summary(all_results):
        """Print a cross-model comparison table."""
        print(f"\\n{'═'*80}")
        print(f"  CROSS-MODEL SUMMARY")
        print(f"{'═'*80}")
        print(f"  {'Model':<20} {'Layers':<8} {'Prune Candidates':<20} "
              f"{'Best Ratio':<12} {'Best Stereo'}")
        print(f"  {'─'*75}")
        for name, res in all_results.items():
            cands = res["layer_ranking"]["candidates"]
            best = min(res["pruning_results"],
                       key=lambda r: r.get("stereotype_overall", float("inf")))
            print(f"  {name:<20} {res['n_layers']:<8} "
                  f"{str(cands):<20} "
                  f"{best['ratio']:<12.0%} "
                  f"{best.get('stereotype_overall', 0):.6f}")
        print(f"{'═'*80}")'''
            new_export = '''    def export_results(self, filename=None):
        """Save analysis results to JSON + print summary report."""
        if self._analysis_results is None:
            raise RuntimeError("Run run_layer_by_layer_analysis() first.")

        res = self._analysis_results
        fname = filename or f"layer_ablation_{self.model_name}.json"

        # Make JSON-serializable (remove _detail sub-dicts)
        export = {
            "model_name": res["model_name"],
            "n_layers": res["n_layers"],
            "hybrid_weights": res["hybrid_weights"],
            "hybrid_scores": res["hybrid_scores"],
            "baseline": {k: v for k, v in res["baseline"].items()
                         if not k.startswith("_detail")},
            "ablation_results": [
                {k: v for k, v in r.items() if not k.startswith("_detail")}
                for r in res["ablation_results"]
            ],
        }

        with open(fname, "w") as f:
            json.dump(export, f, indent=2, default=str)
        print(f"Results saved to {fname}")

        # ── Text summary report ──────────────────────────────────────
        report_name = fname.replace(".json", "_report.txt")
        lines = []
        lines.append(f"{'='*65}")
        lines.append(f"  SINGLE-LAYER ABLATION REPORT")
        lines.append(f"  Model: {res['model_name']}  ({res['n_layers']} layers)")
        lines.append(f"  Hybrid weights: α={self.alpha}, β={self.beta}, "
                     f"γ={self.gamma}")
        lines.append(f"{'='*65}")
        lines.append("")

        # Baseline
        lines.append("BASELINE (no pruning):")
        for k, v in res["baseline"].items():
            if not k.startswith("_detail") and k != "layer_removed":
                lines.append(f"  {k}: {v}")
        lines.append("")

        # Pruning sweep results
        lines.append("ABLATION RESULTS (One Layer Removed at a Time):")
        lines.append(f"  {'Removed':<8} {'Hybrid':<10} {'Stereotype':<12} {'Parity':<12} "
                     f"{'Toxicity':<12} {'PPL'}")
        lines.append(f"  {'─'*70}")
        for r in res["ablation_results"]:
            lines.append(
                f"  L{r['layer_removed']:<7} "
                f"{r.get('hybrid_score', 0):<10.4f} "
                f"{r.get('stereotype_overall', 0):<12.6f} "
                f"{r.get('demographic_parity_gap', 0):<12.6f} "
                f"{r.get('toxicity_propensity', 0):<12.6f} "
                f"{r.get('perplexity_neutral', 0):<12.2f}"
            )
        lines.append("")

        # Best configuration
        best = min(res["ablation_results"],
                   key=lambda r: r.get("stereotype_overall", float("inf")))
        lines.append(f"BEST ABLATION (lowest stereotype score):")
        lines.append(f"  Layer removed: L{best.get('layer_removed')}")
        lines.append(f"  Hybrid Score: {best.get('hybrid_score', 0):.4f}")
        lines.append(f"  Stereotype: {best.get('stereotype_overall', 0):.6f}")
        lines.append(f"  Parity gap: {best.get('demographic_parity_gap', 0):.6f}")
        lines.append(f"  Perplexity: {best.get('perplexity_neutral', 0):.2f}")

        report = "\\n".join(lines)
        with open(report_name, "w") as f:
            f.write(report)
        print(f"Report saved to {report_name}")
        print(report)

    # ══════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _normalize(scores):
        """Min-max normalize to [0, 1]."""
        mn, mx = min(scores), max(scores)
        if mx - mn < 1e-12:
            return [0.5] * len(scores)
        return [(s - mn) / (mx - mn) for s in scores]

    @staticmethod
    def _print_metrics(m, label):
        """Print a compact metrics summary line."""
        print(f"  [{label}]  stereo={m.get('stereotype_overall', 0):.4f}  "
              f"parity={m.get('demographic_parity_gap', 0):.4f}  "
              f"toxic={m.get('toxicity_propensity', 0):.4f}  "
              f"ppl={m.get('perplexity_neutral', 0):.2f}  "
              f"prof_bias={m.get('profession_gender_bias', 0):.4f}  "
              f"demo_var={m.get('demographic_ppl_variance', 0):.4f}")

    @staticmethod
    def _print_cross_model_summary(all_results):
        """Print a cross-model comparison table for single ablation."""
        print(f"\\n{'═'*80}")
        print(f"  CROSS-MODEL SUMMARY (BEST ABLATION)")
        print(f"{'═'*80}")
        print(f"  {'Model':<20} {'Layers':<8} "
              f"{'Best Layer':<12} {'Best Stereo'}")
        print(f"  {'─'*75}")
        for name, res in all_results.items():
            best = min(res["ablation_results"],
                       key=lambda r: r.get("stereotype_overall", float("inf")))
            print(f"  {name:<20} {res['n_layers']:<8} "
                  f"L{best.get('layer_removed', 'N/A'):<11} "
                  f"{best.get('stereotype_overall', 0):.6f}")
        print(f"{'═'*80}")'''
            if old_export in source:
                print("Replaced export_results")
                source = source.replace(old_export, new_export)
            else:
                print("Warning: old_export not found")
                
        # Split source back to list of strings
        cell["source"] = [source]

# We also need to update the execution cell at the end of the notebook!
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "finder.run_full_analysis" in source:
            print("Found analysis execution cell")
            source = source.replace("finder.run_full_analysis()", "finder.run_layer_by_layer_analysis()")
            source = source.replace("finder.plot_pruning_sweep()", "finder.plot_single_layer_ablation_impact()")
            cell["source"] = [source]

with open("layer-trials.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    
print("Notebook updated successfully.")
