"""
Figure 3: Degenerate (ε ≈ 0) systems showing exact σ/π cancellation
=====================================================================
Bar chart with mirrored σ/π contributions for cylindrical bonds:
- Acetylene (HCCH)
- N2
- Ethane
- Methane
- CO2
- Ethylene (for contrast: ε > 0)
"""
import sys, os, numpy as np
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from sigma_pi_decomposition import analyze_decomp

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 9,
    'figure.dpi': 100,
})

# Selected systems (degenerate + 1 non-degenerate for contrast)
cases = [
    ("HCCH C$\\equiv$C", 
     "C 0 0 -0.601; C 0 0 0.601; H 0 0 -1.667; H 0 0 1.667",
     [0,0,-0.601],[0,0,0.601]),
    ("N$_2$",
     "N 0 0 -0.549; N 0 0 0.549",
     [0,0,-0.549],[0,0,0.549]),
    ("HCN",
     "H 0 0 -1.657; C 0 0 -0.597; N 0 0 0.563",
     [0,0,-0.597],[0,0,0.563]),
    ("Ethane C-C",
     "C 0 0 -0.764; C 0 0 0.764; H 0 1.018 -1.160; H 0.882 -0.509 -1.160; H -0.882 -0.509 -1.160; H 0 -1.018 1.160; H -0.882 0.509 1.160; H 0.882 0.509 1.160",
     [0,0,-0.764],[0,0,0.764]),
    ("CH$_4$ C-H",
     "C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; H 0.629 -0.629 -0.629; H -0.629 0.629 -0.629",
     [0,0,0],[0.629,0.629,0.629]),
    ("CO$_2$ C=O",
     "C 0 0 0; O 0 0 1.161; O 0 0 -1.161",
     [0,0,0],[0,0,1.161]),
    # Non-degenerate for contrast
    ("Ethylene C=C\n($\\varepsilon = 0.45$)",
     "C 0 0 -0.667; C 0 0 0.667; H 0.921 0 -1.241; H -0.921 0 -1.241; H 0.921 0 1.241; H -0.921 0 1.241",
     [0,0,-0.667],[0,0,0.667]),
]

# Run all
results = []
for label, atoms, A, B in cases:
    print(f"Running {label[:20]}...")
    r = analyze_decomp(label, atoms, A, B)
    results.append(r)

print("Building figure...")

fig, ax = plt.subplots(figsize=(10, 5))

n = len(cases)
x = np.arange(n)
width = 0.35

dsg = [r['delta_sigma_num'] for r in results]
dpi = [r['delta_pi_num'] for r in results]
dtot = [r['delta_bader'] for r in results]

bars1 = ax.bar(x - width/2, dsg, width, label=r'$\delta_\sigma$',
               color='steelblue', edgecolor='black', linewidth=0.6)
bars2 = ax.bar(x + width/2, dpi, width, label=r'$\delta_\pi$',
               color='indianred', edgecolor='black', linewidth=0.6)

# Plot total as a marker on top
ax.scatter(x, dtot, s=80, color='black', marker='D', zorder=5, label=r'$\delta_{\rm tot}$')

# Value labels for δ_total only (since σ and π are mirrored)
for i, r in enumerate(results):
    eps_val = r['eps']
    label = f'$\\varepsilon = {eps_val:.2f}$'
    # Place label above the σ bar (always positive)
    y_above = max(dsg[i], 0.02) + 0.05
    if eps_val < 0.005:
        ax.text(i, y_above, label, ha='center', fontsize=8.5, color='dimgray')
    else:
        ax.text(i, y_above, label, ha='center', fontsize=8.5, color='black', fontweight='bold')

# Removed cancellation arrows (visual noise)

ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([r['name'] for r in results], rotation=15, ha='right', fontsize=9)
ax.set_ylabel(r'Hessian eigenvalue contribution (a.u.)')
ax.set_title(r'$\sigma$/$\pi$ contributions in degenerate ($\varepsilon \approx 0$) systems vs. ethylene',
             fontsize=11)
ax.legend(loc='upper right')
ax.set_ylim(-1.2, 1.4)
ax.grid(True, linestyle=':', axis='y', alpha=0.5)

# Add interpretive annotation
ax.text(0.02, 0.05, r'Cylindrical bonds: $|\delta_\sigma| = |\delta_\pi|$, '
                  r'$\delta_{\rm tot} = 0$',
        transform=ax.transAxes, fontsize=9, va='bottom', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='goldenrod', alpha=0.85))

plt.tight_layout()
plt.savefig('fig3_degenerate.pdf', bbox_inches='tight')
plt.savefig('fig3_degenerate.png', bbox_inches='tight', dpi=200)
print("Saved fig3_degenerate.pdf and .png")
