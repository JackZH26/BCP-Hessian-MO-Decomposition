"""
Figure 4: Robustness scatter — paired delta_sigma vs delta_pi
==================================================================
Across methods, basis sets, and MO localization schemes,
shows that all data points cluster in the (δ_σ > 0, δ_π < 0) quadrant
for ε > 0 systems.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings; warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 100,
})

# Data extracted from v6-v13 results (HF/6-31G*, cc-pVTZ, B3LYP/def2-TZVP, CCSD/cc-pVDZ, opt-geom, LMO)
# Format: (compound, method/basis, delta_sigma, delta_pi)
# Only ε > 0 systems (degenerate excluded)

data = [
    # HF/6-31G*
    ("Ethylene", "HF/6-31G*", +0.388, -0.148),
    ("N2H2", "HF/6-31G*", +0.622, -0.442),
    ("H2CO", "HF/6-31G*", +0.768, -0.669),
    ("Acetone", "HF/6-31G*", +0.698, -0.629),
    ("Cyclopropane", "HF/6-31G*", +0.316, -0.094),
    ("Allene", "HF/6-31G*", +0.571, -0.309),
    ("Benzene", "HF/6-31G*", +0.178, -0.048),
    ("Pyridine", "HF/6-31G*", +0.177, -0.044),
    ("Furan", "HF/6-31G*", +0.219, -0.040),
    ("N2F4", "HF/6-31G*", +0.474, -0.103),
    ("CHF=CHF", "HF/6-31G*", +0.467, -0.095),
    ("CHF3", "HF/6-31G*", +0.430, -0.301),
    ("Naphthalene 0-1", "HF/6-31G*", +0.206, -0.105),
    ("Anthracene 1-2", "HF/6-31G*", +0.125, -0.090),
    ("Phenanthrene 9-10", "HF/6-31G*", +0.184, -0.091),
    ("Pyrazine", "HF/6-31G*", +0.161, -0.000),  # σ-only
    ("Furan", "HF/6-31G*", +0.219, -0.040),
    ("Thiophene", "HF/6-31G*", +0.133, -0.028),
    ("Acrolein", "HF/6-31G*", +0.259, -0.057),
    
    # HF/cc-pVTZ
    ("Ethylene", "HF/cc-pVTZ", +0.417, -0.167),
    ("N2H2", "HF/cc-pVTZ", +0.682, -0.465),
    ("H2CO", "HF/cc-pVTZ", +0.786, -0.661),
    ("Acetone", "HF/cc-pVTZ", +0.715, -0.611),
    ("Cyclopropane", "HF/cc-pVTZ", +0.335, -0.104),
    ("Allene", "HF/cc-pVTZ", +0.593, -0.319),
    ("Benzene", "HF/cc-pVTZ", +0.191, -0.054),
    ("Pyridine", "HF/cc-pVTZ", +0.186, -0.046),
    ("Furan", "HF/cc-pVTZ", +0.224, -0.039),
    ("N2F4", "HF/cc-pVTZ", +0.445, -0.087),
    ("CHF=CHF", "HF/cc-pVTZ", +0.489, -0.103),
    
    # B3LYP/def2-TZVP
    ("Ethylene", "B3LYP/def2-TZVP", +0.383, -0.183),
    ("N2H2", "B3LYP/def2-TZVP", +0.641, -0.480),
    ("H2CO", "B3LYP/def2-TZVP", +0.803, -0.721),
    ("Acetone", "B3LYP/def2-TZVP", +0.744, -0.661),
    ("Cyclopropane", "B3LYP/def2-TZVP", +0.310, -0.115),
    ("Allene", "B3LYP/def2-TZVP", +0.418, -0.208),
    ("Benzene", "B3LYP/def2-TZVP", +0.174, -0.061),
    ("Pyridine", "B3LYP/def2-TZVP", +0.169, -0.054),
    ("Furan", "B3LYP/def2-TZVP", +0.207, -0.061),
    ("N2F4", "B3LYP/def2-TZVP", +0.410, -0.096),
    ("CHF=CHF", "B3LYP/def2-TZVP", +0.450, -0.133),
    
    # CCSD/cc-pVDZ
    ("Ethylene", "CCSD/cc-pVDZ", +0.380, -0.161),
    ("N2H2", "CCSD/cc-pVDZ", +0.630, -0.448),
    ("H2CO", "CCSD/cc-pVDZ", +0.739, -0.678),
    ("Cyclopropane", "CCSD/cc-pVDZ", +0.318, -0.102),
    ("Benzene", "CCSD/cc-pVDZ", +0.174, -0.054),
    ("N2F4", "CCSD/cc-pVDZ", +0.468, -0.108),
    
    # Boys LMO (HF/6-31G*)
    ("Ethylene", "Boys", +0.388, -0.148),
    ("N2H2", "Boys", +0.619, -0.438),
    ("H2CO", "Boys", +0.742, -0.643),
    ("Cyclopropane", "Boys", +0.316, -0.094),
    ("Allene", "Boys", +0.586, -0.324),
    ("Benzene", "Boys", +0.248, -0.118),
    ("Furan", "Boys", +0.276, -0.097),
    ("N2F4", "Boys", +0.442, -0.071),
    ("CHF=CHF", "Boys", +0.455, -0.083),
    
    # Pipek-Mezey LMO
    ("Ethylene", "PM", +0.388, -0.148),
    ("N2H2", "PM", +0.621, -0.441),
    ("H2CO", "PM", +0.744, -0.646),
    ("Cyclopropane", "PM", +0.316, -0.094),
    ("Benzene", "PM", +0.247, -0.117),
    ("Furan", "PM", +0.276, -0.097),
    ("N2F4", "PM", +0.423, -0.052),
    
    # IBO LMO
    ("Ethylene", "IBO", +0.374, -0.134),
    ("N2H2", "IBO", +0.622, -0.441),
    ("H2CO", "IBO", +0.748, -0.649),
    ("Cyclopropane", "IBO", +0.292, -0.070),
    ("Benzene", "IBO", +0.246, -0.116),
    ("Furan", "IBO", +0.275, -0.096),
    ("N2F4", "IBO", +0.421, -0.050),
]

# Category colors
method_colors = {
    "HF/6-31G*": "#1f77b4",
    "HF/cc-pVTZ": "#ff7f0e",
    "B3LYP/def2-TZVP": "#2ca02c",
    "CCSD/cc-pVDZ": "#d62728",
    "Boys": "#9467bd",
    "PM": "#8c564b",
    "IBO": "#e377c2",
}

method_markers = {
    "HF/6-31G*": "o",
    "HF/cc-pVTZ": "s",
    "B3LYP/def2-TZVP": "D",
    "CCSD/cc-pVDZ": "^",
    "Boys": "v",
    "PM": "p",
    "IBO": "*",
}

print(f"Total points: {len(data)}")

fig, ax = plt.subplots(figsize=(9, 7))

# Highlight the (δ_σ > 0, δ_π < 0) quadrant
ax.add_patch(Rectangle((0, -0.8), 1.0, 0.8, color='#e8f5e9', alpha=0.5, zorder=0))
ax.text(0.85, -0.78, 'Sign-rule quadrant\n($\\delta_\\sigma > 0$, $\\delta_\\pi \\leq 0$)',
        ha='right', va='bottom', fontsize=9, color='#2e7d32', fontweight='bold')

# Plot all points
seen_methods = set()
for compound, method, dsg, dpi in data:
    color = method_colors.get(method, 'gray')
    marker = method_markers.get(method, 'o')
    label = method if method not in seen_methods else None
    seen_methods.add(method)
    ax.scatter(dsg, dpi, color=color, marker=marker, s=70,
               alpha=0.75, edgecolors='black', linewidths=0.5, label=label)

# Reference lines
ax.axhline(0, color='black', linewidth=0.7)
ax.axvline(0, color='black', linewidth=0.7)

# Sign-rule line
xx = np.linspace(0, 0.9, 50)
ax.plot(xx, -xx*0.5, '--', color='gray', linewidth=0.8, alpha=0.6,
        label=r'$\delta_\pi = -0.5 \delta_\sigma$ (typical)')

ax.set_xlabel(r'$\delta_\sigma$ (a.u.)')
ax.set_ylabel(r'$\delta_\pi$ (a.u.)')
ax.set_xlim(-0.05, 1.0)
ax.set_ylim(-0.75, 0.05)
ax.set_title(r'Sign-rule robustness across methods, basis sets, and MO localization schemes ' + '\n' +
             r'(' + str(len(data)) + r' data points across $\sim$15 compounds)',
             fontsize=11)
ax.grid(True, linestyle=':', alpha=0.4)
ax.legend(loc='upper right', ncol=2, framealpha=0.9, fontsize=9)

# Statistics annotation
in_quadrant = sum(1 for _, _, dsg, dpi in data if dsg > 0 and dpi <= 0)
ax.text(0.6, 0.45, f'{in_quadrant}/{len(data)} points satisfy\n'
                   r'$\delta_\sigma > 0$ and $\delta_\pi \leq 0$' + '\n(100%)',
        transform=ax.transAxes, fontsize=11, va='top', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='goldenrod', alpha=0.95))

plt.tight_layout()
plt.savefig('fig4_robustness_scatter.pdf', bbox_inches='tight')
plt.savefig('fig4_robustness_scatter.png', bbox_inches='tight', dpi=200)
print(f"In sign-rule quadrant: {in_quadrant}/{len(data)} = {100*in_quadrant/len(data):.1f}%")
print("Saved fig4_robustness_scatter.pdf and .png")
