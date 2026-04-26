"""
Figure 1: Ethylene as paradigmatic case
========================================
(a) Density contour + BCP location with principal axes
(b) Per-MO contributions to lambda_1 and lambda_2 (top contributors only)
(c) Bar chart: delta_sigma vs delta_pi vs delta_total
"""
import sys, os, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import warnings; warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from sigma_pi_decomposition import build, find_BCP, rho_grad_hess, mo_hessian_contrib_along_axis
from pyscf import scf

a0 = 0.529177

# Use mathtext for symbols (\AA was not rendering; use Unicode)
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times'],
    'mathtext.fontset': 'dejavuserif',
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.linewidth': 1.5,
    'figure.dpi': 100,
})

# Build ethylene
atoms = "C 0 0 -0.667; C 0 0 0.667; H 0.921 0 -1.241; H -0.921 0 -1.241; H 0.921 0 1.241; H -0.921 0 1.241"
mol = build(atoms, basis='6-31G*')
mf = scf.RHF(mol); mf.verbose = 0; mf.kernel()
dm = mf.make_rdm1()

# Find BCP
bcp, t = find_BCP(mol, dm, [0,0,-0.667], [0,0,0.667])

# Compute Hessian
rho, grad, H = rho_grad_hess(mol, dm, bcp, eps=1e-3)
evals, evecs = np.linalg.eigh(H)
lam1, lam2, lam3 = evals
e_lam1, e_lam2, e_lam3 = evecs.T

# Per-MO contributions
c_lam1, psi0, dpsi_lam1 = mo_hessian_contrib_along_axis(mol, mf, bcp, e_lam1, eps=8e-3)
c_lam2, _, dpsi_lam2 = mo_hessian_contrib_along_axis(mol, mf, bcp, e_lam2, eps=8e-3)
n_occ = (mf.mo_occ > 0.5).sum()

# Density grid for contour plot in xz plane
print("Computing density grid...")
n_grid = 100
x_range = np.linspace(-2.5, 2.5, n_grid) / a0
z_range = np.linspace(-2.0, 2.0, n_grid) / a0
X, Z = np.meshgrid(x_range, z_range)

rho_grid = np.zeros_like(X)
for i in range(n_grid):
    for j in range(n_grid):
        r = np.array([X[i,j], 0.0, Z[i,j]])
        ao = mol.eval_gto('GTOval', np.array([r]))
        rho_grid[i,j] = float(np.einsum('xi,ij,xj->x', ao, dm, ao)[0])

print("Building figure...")
fig = plt.figure(figsize=(13, 4.2))
gs = GridSpec(1, 3, width_ratios=[1.2, 1.5, 0.9], wspace=0.30, top=0.85, bottom=0.18)

# === Panel (a): Density contour with BCP ===
ax1 = fig.add_subplot(gs[0])
levels = np.geomspace(0.005, 1.0, 12)
ax1.contour(X*a0, Z*a0, rho_grid, levels=levels, colors='steelblue', linewidths=0.6)

# Atoms (shown larger so BCP star stands out)
for atom in [(0, -0.667), (0, 0.667)]:
    ax1.plot(atom[0], atom[1], 'ko', markersize=12)
ax1.text(0.18, -0.65, 'C', fontsize=12, fontweight='bold')
ax1.text(0.18, 0.69, 'C', fontsize=12, fontweight='bold')
for atom in [(0.921, -1.241), (-0.921, -1.241), (0.921, 1.241), (-0.921, 1.241)]:
    ax1.plot(atom[0], atom[1], 'o', color='lightgray', markeredgecolor='black', markersize=8)
    ax1.text(atom[0]*1.18, atom[1]*1.05, 'H', fontsize=9, color='dimgray', ha='center', va='center')

# BCP
ax1.plot(bcp[0]*a0, bcp[2]*a0, 'r*', markersize=18, zorder=10, markeredgecolor='darkred', markeredgewidth=1.0)
ax1.annotate('BCP', xy=(bcp[0]*a0, bcp[2]*a0), xytext=(0.9, 0.2),
             color='darkred', fontsize=10, fontweight='bold',
             arrowprops=dict(arrowstyle='-', color='darkred', lw=0.6))

# Principal axes (no overlap with BCP)
# e_2 (in-plane perpendicular to bond, along x)
ax1.annotate('', xy=(0.55, 0.0), xytext=(0.0, 0.0),
             arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax1.text(0.62, -0.05, r'$\mathbf{e}_2$', color='green', fontsize=11, fontweight='bold')

# e_3 (along bond, along z)
ax1.annotate('', xy=(0.0, 0.55), xytext=(0.0, 0.0),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax1.text(0.10, 0.40, r'$\mathbf{e}_3$', color='blue', fontsize=11, fontweight='bold')

# e_1 indicator (perpendicular to plane, into page)
ax1.text(-2.4, 1.7, r'$\mathbf{e}_1 \perp$ plane', color='purple', fontsize=9, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='lavender', edgecolor='purple', alpha=0.8))
# e_1 out-of-page symbol next to BCP
ax1.plot(bcp[0]*a0 - 0.25, bcp[2]*a0 - 0.05, 'o', markersize=11, color='lavender', markeredgecolor='purple', markeredgewidth=1.0)
ax1.plot(bcp[0]*a0 - 0.25, bcp[2]*a0 - 0.05, 'o', markersize=2.5, color='purple')

ax1.set_xlabel(r'$x$ (Å)')
ax1.set_ylabel(r'$z$ (Å)')
ax1.set_title('(a) Density and BCP', fontsize=11)
ax1.set_aspect('equal')
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-2.0, 2.0)

# === Panel (b): Per-MO contributions (top 6 only by magnitude) ===
ax2 = fig.add_subplot(gs[1])

# Classify each MO
classes_orig = []
for i in range(n_occ):
    is_pi = abs(psi0[i]) < 0.08 and abs(dpsi_lam1[i]) > abs(dpsi_lam2[i])
    classes_orig.append('pi' if is_pi else 'sigma')

delta_per_mo = c_lam2[:n_occ] - c_lam1[:n_occ]

# Sort by absolute value, take top 6
sorted_idx = np.argsort(-np.abs(delta_per_mo))
top_n = min(6, n_occ)
sorted_idx_top = sorted_idx[:top_n]
sorted_d = delta_per_mo[sorted_idx_top]
sorted_classes = [classes_orig[i] for i in sorted_idx_top]

x_pos = np.arange(top_n)
colors = ['indianred' if c == 'pi' else 'steelblue' for c in sorted_classes]
ax2.bar(x_pos, sorted_d, color=colors, edgecolor='black', linewidth=0.6)
ax2.axhline(0, color='black', linewidth=0.5)

# Labels: MO_index (class) using mathtext for sigma/pi
mo_labels = []
for i in range(top_n):
    cls_sym = r'$\pi$' if sorted_classes[i] == 'pi' else r'$\sigma$'
    mo_labels.append(f'MO{sorted_idx_top[i]}\n({cls_sym})')

ax2.set_xticks(x_pos)
ax2.set_xticklabels(mo_labels, fontsize=9)

# Add a final "remaining" bar showing the sum of remaining MOs
if n_occ > top_n:
    rest_d = sum(delta_per_mo[i] for i in sorted_idx[top_n:])
    ax2.bar(top_n + 0.5, rest_d, color='lightgray', edgecolor='black', linewidth=0.6, width=0.6)
    ax2.text(top_n + 0.5, rest_d + 0.005 if rest_d >= 0 else rest_d - 0.015,
             f'{rest_d:+.3f}', ha='center', fontsize=8)
    ax2.set_xticks(list(x_pos) + [top_n + 0.5])
    ax2.set_xticklabels(mo_labels + ['rest'], fontsize=9)

ax2.set_ylabel(r'$c_i^{(\lambda_2)} - c_i^{(\lambda_1)}$ (a.u.)')
ax2.set_title('(b) Top per-MO contributions', fontsize=11)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='steelblue', edgecolor='black', label=r'$\sigma$ MO'),
    Patch(facecolor='indianred', edgecolor='black', label=r'$\pi$ MO'),
    Patch(facecolor='lightgray', edgecolor='black', label='Other MOs (sum)'),
]
ax2.legend(handles=legend_elements, loc='upper right')
ax2.grid(True, linestyle=':', axis='y', alpha=0.5)

# Value labels above bars
for i, (val, cls) in enumerate(zip(sorted_d, sorted_classes)):
    if abs(val) > 0.03:
        ax2.text(i, val + 0.008 if val >= 0 else val - 0.02, f'{val:+.2f}',
                 ha='center', fontsize=8)

# === Panel (c): Aggregate ===
ax3 = fig.add_subplot(gs[2])
delta_sigma = sum((c_lam2[i] - c_lam1[i]) for i in range(n_occ) if classes_orig[i] == 'sigma')
delta_pi = sum((c_lam2[i] - c_lam1[i]) for i in range(n_occ) if classes_orig[i] == 'pi')
delta_total = lam2 - lam1

bars = ax3.bar([r'$\delta_\sigma$', r'$\delta_\pi$', r'$\delta_{\rm tot}$'],
               [delta_sigma, delta_pi, delta_total],
               color=['steelblue', 'indianred', '#2ca02c'],
               edgecolor='black', linewidth=0.8, width=0.6)
for bar, val in zip(bars, [delta_sigma, delta_pi, delta_total]):
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, h + (0.012 if h >= 0 else -0.028),
             f'{val:+.3f}', ha='center', fontsize=10, fontweight='bold')
ax3.axhline(0, color='black', linewidth=0.5)
ax3.set_ylabel('Hessian eigenvalue difference (a.u.)')
ax3.set_title('(c) Aggregate decomposition', fontsize=11)
ax3.set_ylim(-0.25, 0.55)
ax3.grid(True, linestyle=':', axis='y', alpha=0.5)

# Master title above all
fig.suptitle(r'Ethylene C=C BCP per-MO Hessian decomposition (HF/6-31G*, $\varepsilon = 0.447$)',
             y=0.97, fontsize=12, fontweight='bold')

plt.savefig('fig1_ethylene_paradigm.pdf', bbox_inches='tight')
plt.savefig('fig1_ethylene_paradigm.png', bbox_inches='tight', dpi=200)
print(f"  delta_sigma = {delta_sigma:.4f}")
print(f"  delta_pi    = {delta_pi:.4f}")
print(f"  delta_total = {delta_total:.4f}")
print("Saved fig1_ethylene_paradigm.pdf and .png")
