#!/usr/bin/env python3
"""Re-run analysis/plotting from saved JSON results for exp_059."""
import os
import json
import sys

# Re-import and monkey-patch to replot from saved data
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import scripts.exp_059_large_n_chain_length as exp059

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "results", "exp_059")

# Load per-problem results from main run output
# We need to rebuild the all_results structure from the output
# Since the JSON was only saved for Qwen (before the crash), let me load from stdout

# Actually, let's just re-run the analysis function with the data
# We need to save the raw per-problem data. Let me check if we can extract from
# the background task output.

# Better approach: re-run just the plotting code
import numpy as np

def replot_from_output():
    """Manually create the results and plots from the output we captured."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Data from captured output
    models_data = [
        {
            'model': 'Qwen3-4B-Base',
            'n_valid': 20,
            'strategy_acc': {
                'recent': {'n': 20, 'c': 20, 'a': 100.0},
                'k_norm': {'n': 20, 'c': 19, 'a': 95.0},
                'random': {'n': 20, 'c': 16, 'a': 80.0},
                'hybrid_50_50': {'n': 20, 'c': 20, 'a': 100.0},
                'hybrid_70_30': {'n': 20, 'c': 20, 'a': 100.0},
            },
            'length_data': {
                'short': {
                    'n': 6,
                    'recent': (6, 6), 'k_norm': (5, 6), 'random': (4, 6),
                    'hybrid_50_50': (6, 6), 'hybrid_70_30': (6, 6),
                },
                'medium': {
                    'n': 14,
                    'recent': (14, 14), 'k_norm': (14, 14), 'random': (12, 14),
                    'hybrid_50_50': (14, 14), 'hybrid_70_30': (14, 14),
                },
            },
        },
        {
            'model': 'Llama-3.1-8B-Instruct',
            'n_valid': 40,
            'strategy_acc': {
                'recent': {'n': 40, 'c': 40, 'a': 100.0},
                'k_norm': {'n': 40, 'c': 35, 'a': 87.5},
                'random': {'n': 40, 'c': 35, 'a': 87.5},
                'hybrid_50_50': {'n': 40, 'c': 40, 'a': 100.0},
                'hybrid_70_30': {'n': 40, 'c': 40, 'a': 100.0},
            },
            'length_data': {},  # Will fill from Llama JSON if available
        },
    ]

    STRATEGIES = ['recent', 'k_norm', 'random', 'hybrid_50_50', 'hybrid_70_30']
    colors = ['#2ecc71', '#3498db', '#95a5a6', '#e67e22', '#e74c3c']

    def wilson_ci(n_success, n_total, z=1.96):
        if n_total == 0:
            return (0.0, 0.0)
        p = n_success / n_total
        denom = 1 + z**2 / n_total
        center = (p + z**2 / (2 * n_total)) / denom
        spread = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
        return (max(0, center - spread), min(1, center + spread))

    # Try to load Llama JSON if saved
    llama_json = os.path.join(RESULTS_DIR, 'results_Llama-3.1-8B-Instruct.json')
    if os.path.exists(llama_json):
        with open(llama_json) as f:
            llama_data = json.load(f)
        # Extract chain-length breakdown
        if 'length_distribution' in llama_data:
            models_data[1]['length_data'] = llama_data['length_distribution']

    for md in models_data:
        model_name = md['model']
        n_valid = md['n_valid']

        # Figure 1: Overall accuracy bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(STRATEGIES))
        accs = [md['strategy_acc'][s]['a'] for s in STRATEGIES]

        ci_data = []
        for s in STRATEGIES:
            sa = md['strategy_acc'][s]
            lo, hi = wilson_ci(sa['c'], sa['n'])
            ci_data.append((lo * 100, hi * 100))

        errors_lo = [max(0, a - l) for a, (l, h) in zip(accs, ci_data)]
        errors_hi = [max(0, h - a) for a, (l, h) in zip(accs, ci_data)]

        bars = ax.bar(x, accs, yerr=[errors_lo, errors_hi], capsize=5,
                      color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in STRATEGIES], fontsize=9)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{model_name} — KV Eviction at 33% Budget (Masking)\nn={n_valid} problems')
        ax.set_ylim(0, 115)
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)

        for bar, acc_val, s in zip(bars, accs, STRATEGIES):
            sa = md['strategy_acc'][s]
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    f'{acc_val:.1f}%\n({sa["c"]}/{sa["n"]})', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        fig_path = os.path.join(RESULTS_DIR, f'overall_accuracy_{model_name}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fig_path}")

        # Figure 2: Chain-length breakdown (if available)
        ld = md['length_data']
        if ld and len(ld) >= 2:
            bucket_names = [b for b in ['short', 'medium', 'long'] if b in ld]
            fig, axes = plt.subplots(1, len(bucket_names), figsize=(5*len(bucket_names), 6),
                                     sharey=True)
            if len(bucket_names) == 1:
                axes = [axes]

            for ax_idx, bucket_name in enumerate(bucket_names):
                ax = axes[ax_idx]
                bd = ld[bucket_name]
                n_bucket = bd['n'] if isinstance(bd.get('n'), int) else bd.get('n', 0)

                accs_b = []
                err_lo_b = []
                err_hi_b = []
                for s in STRATEGIES:
                    if isinstance(bd.get(s), tuple):
                        nc, nt = bd[s]
                    elif isinstance(bd.get('strategies'), dict) and s in bd['strategies']:
                        sd = bd['strategies'][s]
                        nc, nt = sd.get('n_correct', 0), sd.get('n_total', n_bucket)
                    else:
                        nc, nt = 0, n_bucket
                    acc = nc / nt * 100 if nt > 0 else 0
                    lo, hi = wilson_ci(nc, nt)
                    accs_b.append(acc)
                    err_lo_b.append(max(0, acc - lo*100))
                    err_hi_b.append(max(0, hi*100 - acc))

                ax.bar(np.arange(len(STRATEGIES)), accs_b,
                       yerr=[err_lo_b, err_hi_b], capsize=4,
                       color=colors, edgecolor='black', linewidth=0.5)
                ax.set_xticks(np.arange(len(STRATEGIES)))
                ax.set_xticklabels([s.replace('_', '\n') for s in STRATEGIES], fontsize=7)
                ax.set_title(f'{bucket_name} (n={n_bucket})')
                ax.set_ylim(0, 115)
                ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
                if ax_idx == 0:
                    ax.set_ylabel('Accuracy (%)')

            plt.suptitle(f'{model_name} — Strategy × Chain Length at 33% Budget', fontsize=12)
            plt.tight_layout()
            fig_path = os.path.join(RESULTS_DIR, f'chain_length_{model_name}.png')
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {fig_path}")

    # Figure 3: Cross-model comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax_idx, md in enumerate(models_data):
        ax = axes[ax_idx]
        model_name = md['model']
        n_valid = md['n_valid']
        x = np.arange(len(STRATEGIES))
        accs = [md['strategy_acc'][s]['a'] for s in STRATEGIES]

        ci_data = []
        for s in STRATEGIES:
            sa = md['strategy_acc'][s]
            lo, hi = wilson_ci(sa['c'], sa['n'])
            ci_data.append((lo * 100, hi * 100))

        errors_lo = [max(0, a - l) for a, (l, h) in zip(accs, ci_data)]
        errors_hi = [max(0, h - a) for a, (l, h) in zip(accs, ci_data)]

        bars = ax.bar(x, accs, yerr=[errors_lo, errors_hi], capsize=5,
                      color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in STRATEGIES], fontsize=9)
        ax.set_title(f'{model_name} (n={n_valid})')
        ax.set_ylim(0, 115)
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel('Accuracy (%)')

        for bar, acc_val, s in zip(bars, accs, STRATEGIES):
            sa = md['strategy_acc'][s]
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    f'{acc_val:.1f}%\n({sa["c"]}/{sa["n"]})', ha='center', va='bottom', fontsize=8)

    plt.suptitle('KV Eviction at 33% Budget (Masking) — Cross-Model Comparison', fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'cross_model_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    replot_from_output()
