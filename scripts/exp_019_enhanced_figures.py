#!/usr/bin/env python3
"""Generate enhanced figures for exp_019 double dissociation analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results/exp_019"

with open(f"{RESULTS_DIR}/summary.json") as f:
    summary = json.load(f)

agg = summary['aggregated']

# Figure 1: Position vs Selectivity — the key comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: Answer accuracy by strategy at 5%
strategies_5 = ['pos_early', 'selac', 'q3_selac', 'q3_seltc', 'seltc', 'random', 'pos_late']
labels_5 = ['PosEarly', 'SelAC', 'Q3+AC', 'Q3+TC', 'SelTC', 'Random', 'PosLate']
positions_5 = [agg[f'{s}_5pct']['mean_position'] for s in strategies_5]
acc_5 = [agg[f'{s}_5pct']['accuracy'] * 100 for s in strategies_5]
txt_5 = [agg[f'{s}_5pct']['text_accuracy'] * 100 for s in strategies_5]

colors = ['#c0392b', '#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#95a5a6', '#2ecc71']

ax = axes[0]
ax.scatter(positions_5, acc_5, c=colors, s=200, zorder=5, edgecolors='black', linewidth=1.5)
for i, label in enumerate(labels_5):
    ax.annotate(label, (positions_5[i], acc_5[i]), textcoords="offset points",
                xytext=(0, 12), ha='center', fontsize=8, fontweight='bold')
ax.set_xlabel('Mean Position (0=early, 1=late)')
ax.set_ylabel('Answer Accuracy (%)')
ax.set_title('A. Position → Accuracy (5% noise)')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-5, 55)
ax.grid(True, alpha=0.3)

# Panel B: Text accuracy by strategy at 5%
ax = axes[1]
ax.scatter(positions_5, txt_5, c=colors, s=200, zorder=5, edgecolors='black', linewidth=1.5)
for i, label in enumerate(labels_5):
    ax.annotate(label, (positions_5[i], txt_5[i]), textcoords="offset points",
                xytext=(0, 12), ha='center', fontsize=8, fontweight='bold')
ax.set_xlabel('Mean Position (0=early, 1=late)')
ax.set_ylabel('Text Prediction Accuracy (%)')
ax.set_title('B. Position → Text Quality (5% noise)')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-5, 105)
ax.grid(True, alpha=0.3)

# Panel C: Position-controlled comparison (Q3 and Q4)
ax = axes[2]
q_strats = ['q3_selac', 'q3_seltc', 'q4_selac', 'q4_seltc']
q_labels = ['Q3+AC', 'Q3+TC', 'Q4+AC', 'Q4+TC']
q_colors = ['#e74c3c', '#3498db', '#e74c3c', '#3498db']
q_markers = ['o', 'o', 's', 's']

for nf_label, nf_pct in [('5%', '5pct'), ('10%', '10pct')]:
    for i, s in enumerate(q_strats):
        key = f'{s}_{nf_pct}'
        a = agg[key]['accuracy_drop'] * 100
        t = agg[key]['text_drop'] * 100
        marker = q_markers[i]
        alpha = 1.0 if nf_pct == '5pct' else 0.5
        label = f'{q_labels[i]} ({nf_label})' if nf_pct == '5pct' else None
        ax.scatter(t, a, c=q_colors[i], s=150, marker=marker, alpha=alpha,
                  edgecolors='black', linewidth=1, zorder=5, label=label)
        if nf_pct == '5pct':
            ax.annotate(q_labels[i], (t, a), textcoords="offset points",
                       xytext=(8, 5), ha='left', fontsize=8)

ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Equal impact')
ax.set_xlabel('Text Drop (%)')
ax.set_ylabel('Accuracy Drop (%)')
ax.set_title('C. Double Dissociation Test\n(AC=red, TC=blue, Q3=circle, Q4=square)')
ax.set_xlim(-5, 50)
ax.set_ylim(65, 105)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7, loc='lower right')

plt.suptitle('Exp 019: Double Dissociation — No Selectivity Effect, Position Dominates (n=17)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/position_vs_selectivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved position_vs_selectivity.png")

# Figure 2: Positional dissociation — the real finding
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for nf_label, nf_pct in [('5%', '5pct'), ('10%', '10pct')]:
    strats = ['pos_early', 'random', 'selac', 'seltc', 'pos_late']
    labels = ['PosEarly', 'Random', 'SelAC', 'SelTC', 'PosLate']
    a_drops = [agg[f'{s}_{nf_pct}']['accuracy_drop'] * 100 for s in strats]
    t_drops = [agg[f'{s}_{nf_pct}']['text_drop'] * 100 for s in strats]
    positions = [agg[f'{s}_{nf_pct}']['mean_position'] for s in strats]

    ax = ax1 if nf_pct == '5pct' else ax2
    x = np.arange(len(strats))
    w = 0.35
    bars1 = ax.bar(x - w/2, a_drops, w, label='Accuracy Drop', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + w/2, t_drops, w, label='Text Drop', color='#3498db', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}\n(pos={p:.2f})' for l, p in zip(labels, positions)],
                       fontsize=8, rotation=0)
    ax.set_ylabel('Drop from Clean (%)')
    ax.set_title(f'{nf_label} Noise: Accuracy vs Text Impact')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.2, axis='y')

    # Add dissociation annotations
    for i, s in enumerate(strats):
        d = agg[f'{s}_{nf_pct}']['dissociation'] * 100
        ax.annotate(f'Δ={d:.0f}%', (x[i], max(a_drops[i], t_drops[i]) + 2),
                   ha='center', fontsize=7, style='italic', color='green' if d > 50 else 'gray')

plt.suptitle('Positional Dissociation: Late Positions Hurt Accuracy >> Text',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/positional_dissociation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved positional_dissociation.png")

# Figure 3: Within-quartile null result
fig, ax = plt.subplots(figsize=(10, 6))
quartiles = ['Q3', 'Q4']
for qi, q in enumerate(['q3', 'q4']):
    for nfi, (nf_label, nf_pct) in enumerate([('5%', '5pct'), ('10%', '10pct')]):
        ac_acc = agg[f'{q}_selac_{nf_pct}']['accuracy'] * 100
        tc_acc = agg[f'{q}_seltc_{nf_pct}']['accuracy'] * 100
        ac_txt = agg[f'{q}_selac_{nf_pct}']['text_accuracy'] * 100
        tc_txt = agg[f'{q}_seltc_{nf_pct}']['text_accuracy'] * 100

        x_base = qi * 3 + nfi * 1.2
        w = 0.25
        ax.bar(x_base - w*1.5, ac_acc, w, color='#e74c3c', alpha=0.8,
               label='AC Accuracy' if qi == 0 and nfi == 0 else None)
        ax.bar(x_base - w*0.5, tc_acc, w, color='#c0392b', alpha=0.5,
               label='TC Accuracy' if qi == 0 and nfi == 0 else None)
        ax.bar(x_base + w*0.5, ac_txt, w, color='#3498db', alpha=0.8,
               label='AC Text' if qi == 0 and nfi == 0 else None)
        ax.bar(x_base + w*1.5, tc_txt, w, color='#2980b9', alpha=0.5,
               label='TC Text' if qi == 0 and nfi == 0 else None)

        ax.text(x_base, -6, f'{quartiles[qi]} {nf_label}', ha='center', fontsize=9, fontweight='bold')

ax.set_ylabel('Metric (%)')
ax.set_title('Within-Quartile Comparison: AC vs TC Selectivity\n'
             '(If spatial dissociation existed: AC accuracy < TC accuracy, AC text > TC text)',
             fontsize=11)
ax.set_xticks([])
ax.set_ylim(-10, 105)
ax.legend(fontsize=9, ncol=4, loc='upper right')
ax.grid(True, alpha=0.2, axis='y')
ax.axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/within_quartile_null.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved within_quartile_null.png")

print("\nAll figures generated.")
