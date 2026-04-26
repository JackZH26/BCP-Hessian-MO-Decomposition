"""
v12：B3LYP/cc-pVTZ 几何优化后再分解
====================================
验证手填几何（用于 v6/v7/v8/v9）vs B3LYP 优化几何的差异
6 个关键分子用 B3LYP/cc-pVTZ 优化，再做 σ/π 分解
"""
import sys, numpy as np
sys.path.insert(0, '/data/.openclaw/workspace/research/experiments/cbt/2026-04-26/loop_1')
from sigma_pi_decomposition import find_BCP, rho_grad_hess, mo_hessian_contrib_along_axis, classify_MO
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize
import warnings; warnings.filterwarnings('ignore')

def analyze_optimized(name, atoms, A_idx, B_idx, basis='cc-pVTZ', xc='B3LYP'):
    """A_idx, B_idx are atom indices (0-based) in the input"""
    mol = gto.Mole()
    mol.atom = atoms; mol.basis = basis; mol.verbose = 0
    mol.build()
    mf = dft.RKS(mol); mf.xc = xc; mf.verbose = 0
    
    print(f"  Optimizing geometry...")
    mol_eq = optimize(mf, maxsteps=50)
    
    # Re-run SCF on optimized geometry
    mf2 = dft.RKS(mol_eq); mf2.xc = xc; mf2.verbose = 0; mf2.kernel()
    dm = mf2.make_rdm1()
    
    # Get atom coords (in Bohr, convert to Angstrom for find_BCP)
    A_ang = mol_eq.atom_coord(A_idx) * 0.529177
    B_ang = mol_eq.atom_coord(B_idx) * 0.529177
    
    bcp, t = find_BCP(mol_eq, dm, A_ang.tolist(), B_ang.tolist())
    rho, grad, H = rho_grad_hess(mol_eq, dm, bcp, eps=1e-3)
    evals, evecs = np.linalg.eigh(H)
    lam1, lam2 = evals[0], evals[1]
    delta_bader = lam2 - lam1
    eps_val = lam1/lam2 - 1 if abs(lam2) > 1e-6 else 0
    e_lam1 = evecs[:, 0]; e_lam2 = evecs[:, 1]
    
    c_lam1, psi0, dpsi_lam1 = mo_hessian_contrib_along_axis(mol_eq, mf2, bcp, e_lam1, eps=8e-3)
    c_lam2, _, dpsi_lam2 = mo_hessian_contrib_along_axis(mol_eq, mf2, bcp, e_lam2, eps=8e-3)
    
    classes = []
    for i in range(len(mf2.mo_occ)):
        if mf2.mo_occ[i] < 0.5:
            classes.append('virtual')
        else:
            cl = classify_MO(psi0[i], dpsi_lam1[i], dpsi_lam2[i], 0, threshold_psi=0.08)
            classes.append(cl)
    
    lam1_pi = sum(c_lam1[i] for i in range(len(classes)) if classes[i]=='pi')
    lam1_sigma = sum(c_lam1[i] for i in range(len(classes)) if classes[i]=='sigma')
    lam2_pi = sum(c_lam2[i] for i in range(len(classes)) if classes[i]=='pi')
    lam2_sigma = sum(c_lam2[i] for i in range(len(classes)) if classes[i]=='sigma')
    
    return {'name': name, 'eps': eps_val, 'delta_bader': delta_bader,
            'delta_pi_num': lam2_pi-lam1_pi, 'delta_sigma_num': lam2_sigma-lam1_sigma}

KEY = [
    ("Ethylene", "C 0 0 -0.667; C 0 0 0.667; H 0.921 0 -1.241; H -0.921 0 -1.241; H 0.921 0 1.241; H -0.921 0 1.241", 0, 1),
    ("Cyclopropane", "C 0 0.866 0; C -0.750 -0.433 0; C 0.750 -0.433 0; H 0 1.480 0.890; H 0 1.480 -0.890; H -1.282 -0.740 0.890; H -1.282 -0.740 -0.890; H 1.282 -0.740 0.890; H 1.282 -0.740 -0.890", 0, 1),
    ("Benzene", "C 1.395 0 0; C 0.698 1.208 0; C -0.698 1.208 0; C -1.395 0 0; C -0.698 -1.208 0; C 0.698 -1.208 0; H 2.479 0 0; H 1.240 2.148 0; H -1.240 2.148 0; H -2.479 0 0; H -1.240 -2.148 0; H 1.240 -2.148 0", 0, 1),
    ("N2F4", "N 0 0.706 0; N 0 -0.706 0; F 1.059 1.227 0; F -1.059 1.227 0; F 1.059 -1.227 0; F -1.059 -1.227 0", 0, 1),
    ("Allene", "C 0 0 0; C 0 0 1.309; C 0 0 -1.309; H 0.926 0 1.839; H -0.926 0 1.839; H 0 0.926 -1.839; H 0 -0.926 -1.839", 0, 1),
    ("BH3", "B 0 0 0; H 1.193 0 0; H -0.596 1.032 0; H -0.596 -1.032 0", 0, 1),
]

print("="*120)
print("v12: B3LYP/cc-pVTZ 几何优化后再分解 (6 关键化合物)")
print("="*120)

results = []
for label, atoms, A, B in KEY:
    try:
        print(f"\n>>> {label}")
        r = analyze_optimized(label, atoms, A, B)
        results.append(r)
        print(f"  ε = {r['eps']:.4f}, δ_σ = {r['delta_sigma_num']:.4f}, δ_π = {r['delta_pi_num']:.4f}")
    except Exception as e:
        print(f"!!! {label} FAILED: {str(e)[:200]}")

print("\n" + "="*120)
print("v12 优化几何 vs 手填几何")
print("="*120)
for r in results:
    eps = r['eps']; dpi = r['delta_pi_num']; dsi = r['delta_sigma_num']
    if eps > 0.05:
        sign = "✓" if (dsi>0 and dpi<0) else ("σ-only" if abs(dpi)<0.005 and dsi>0 else "✗")
    elif eps < 0.01:
        sign = "DEGEN" if abs(dpi+dsi)<0.05 else "?"
    else:
        sign = "border"
    print(f"{r['name']:25} ε={eps:+7.4f}  δ_σ={dsi:+7.4f}  δ_π={dpi:+7.4f}  {sign}")
print("="*120)
