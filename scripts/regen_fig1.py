#!/usr/bin/env python3
"""Regenerate the missing KV decomposition figure for exp_024."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('results/exp_024/summary.json') as f:
    summary = json.load(f)
agg = summary['aggregated']

kv_conds = [
    ('dir_kv_late', 'Dir K+V'),
    ('dir_k_late', 'Dir K'),
    ('dir_v_late', 'Dir V'),
    ('mag_kv_10_late', 'Mag K+V'),
    ('mag_k_10_late', 'Mag K'),
    ('mag_v_10_late', 'Mag V'),
]
kv_names = [n for k, n in kv_conds if k in agg]
kv_keys = [k for k, _ in kv_conds if k in agg]
kv_accs = [agg[k]['accuracy'] * 100 for k in kv_keys]
kv_txts = [agg[k]['text_accuracy'] * 100 for k in kv_keys]
kv_colors = ['#e74c3c', '#ff9999', '#ff9999', '#3498db', '#99ccff', '#99ccff']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(kv_names))

bars1 = ax1.bar(x, kv_accs, 0.6, color=kv_colors, alpha=0.85)
ax1.set_ylabel('Answer Accuracy (%)', fontsize=12)
ax1.set_title('Llama K vs V: Accuracy (Late 5%)', fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(kv_names, rotation=30, ha='right', fontsize=10)
ax1.set_ylim(0, 110)
ax1.grid(True, alpha=0.2, axis='y')
for b, v in zip(bars1, kv_accs):
    ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
            f'{v:.0f}%', ha='center', va='bottom', fontsize=9)

bars2 = ax2.bar(x, kv_txts, 0.6, color=kv_colors, alpha=0.85)
ax2.set_ylabel('Text Prediction Accuracy (%)', fontsize=12)
ax2.set_title('Llama K vs V: Text Quality (Late 5%)', fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels(kv_names, rotation=30, ha='right', fontsize=10)
ax2.set_ylim(0, 110)
ax2.grid(True, alpha=0.2, axis='y')
for b, v in zip(bars2, kv_txts):
    ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
            f'{v:.0f}%', ha='center', va='bottom', fontsize=9)

plt.suptitle('Exp 024: K vs V Decomposition on Llama-3.1-8B (Late 5%)', fontsize=14)
plt.tight_layout()
plt.savefig('results/exp_024/llama_kv_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()
print('Figure generated successfully')
