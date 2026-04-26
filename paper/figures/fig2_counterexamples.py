"""
Figure 2: Four counterexamples unified by σ/π decomposition
==============================================================
(a) N2F4 N-N bond (no π, ε=0.69)
(b) Cyclopropane C-C (Walsh σ-aromaticity, ε=0.84)
(c) BH3 B-H (no occupied π MO, ε=0.28)
(d) CHF3 C-F (σ asymmetry, ε=0.21)

Each panel: stacked bar showing δ_σ, δ_π, δ_total
"""
import sys, os, numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings; warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from sigma_pi_decomposition import analyze_decomp

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 100,
})

# 4 counterexamples
cases = [
    ("(a) N$_2$F$_4$ N–N bond\n(no $\\pi$ bond, $\\varepsilon = 0.69$)",
     "N 0 0.706 0; N 0 -0.706 0; F 1.059 1.227 0; F -1.059 1.227 0; F 1.059 -1.227 0; F -1.059 -1.227 0",
     [0,0.706,0],[0,-0.706,0]),
    ("(b) Cyclopropane C–C\n(Walsh $\\sigma$-aromaticity, $\\varepsilon = 0.84$)",
     "C 0 0.866 0; C -0.750 -0.433 0; C 0.750 -0.433 0; H 0 1.480 0.890; H 0 1.480 -0.890; H -1.282 -0.740 0.890; H -1.282 -0.740 -0.890; H 1.282 -0.740 0.890; H 1.282 -0.740 -0.890",
     [0,0.866,0],[-0.750,-0.433,0]),
    ("(c) BH$_3$ B–H\n(no occupied $\\pi$ MO, $\\varepsilon = 0.28$)",
     "B 0 0 0; H 1.193 0 0; H -0.596 1.032 0; H -0.596 -1.032 0",
     [0,0,0],[1.193,0,0]),
    ("(d) CHF$_3$ C–F\n($\\sigma$ asymmetry, $\\varepsilon = 0.21$)",
     "C 0 0 0; F 0 0 1.336; F 1.260 0 -0.445; F -0.630 1.092 -0.445; H 0 0 -1.085",
     [0,0,0],[0,0,1.336]),
]

# Run and collect
results = []
for label, atoms, A, B in cases:
    print(f"Running: {label[:30]}...")
    r = analyze_decomp(label, atoms, A, B)
    results.append(r)
    print(f"  ε={r['eps']:.4f}, δ_σ={r['delta_sigma_num']:+.4f}, δ_π={r['delta_pi_num']:+.4f}")

# Build figure: 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes = axes.flatten()

for ax, (label, _, _, _), r in zip(axes, cases, results):
    eps = r['eps']
    dsg = r['delta_sigma_num']
    dpi = r['delta_pi_num']
    dtot = r['delta_bader']
    
    # Vertical stacked bar showing how σ + π = total
    x_pos = ['$\\delta_\\sigma$', '$\\delta_\\pi$', '$\\delta_{\\rm tot}$']
    values = [dsg, dpi, dtot]
    colors = ['steelblue', 'indianred', '#2ca02c']
    
    bars = ax.bar(x_pos, values, color=colors, edgecolor='black', linewidth=0.8, width=0.6)
    
    # Value labels (skip near-zero)
    for bar, v in zip(bars, values):
        if abs(v) < 0.005:
            label_text = '~0'
        else:
            label_text = f'{v:+.3f}'
        h = bar.get_height()
        offset = 0.04 if h >= 0 else -0.05
        ax.text(bar.get_x() + bar.get_width()/2, h + offset,
                label_text, ha='center', fontsize=10, fontweight='bold')
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(label, fontsize=10)
    ax.set_ylabel('Hessian eigenvalue diff. (a.u.)')
    ax.grid(True, linestyle=':', axis='y', alpha=0.5)
    
    # Unified y-limits (-0.45 to 1.0) so panels are visually comparable
    ax.set_ylim(-0.45, 1.0)

fig.suptitle(r'Four high-$\varepsilon$ systems lacking conventional $\pi$ bonds, '
             r'unified by $\sigma$/$\pi$ Hessian decomposition (HF/6-31G*)',
             fontsize=11.5, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('fig2_counterexamples.pdf', bbox_inches='tight')
plt.savefig('fig2_counterexamples.png', bbox_inches='tight', dpi=200)
print("\nSaved fig2_counterexamples.pdf and .png")
