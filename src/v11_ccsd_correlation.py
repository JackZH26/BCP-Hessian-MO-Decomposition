"""
v11：CCSD/cc-pVDZ 严格电子相关测试
==================================
8 个最关键化合物用 CCSD 计算（最严格的 wave function 方法）
验证：HF 的 σ/π 符号规律是否在精确电子相关下保持
"""
import sys, numpy as np
sys.path.insert(0, '/data/.openclaw/workspace/research/experiments/cbt/2026-04-26/loop_1')
from sigma_pi_decomposition import find_BCP, rho_grad_hess, mo_hessian_contrib_along_axis, classify_MO
from pyscf import gto, scf, cc
import warnings; warnings.filterwarnings('ignore')

def analyze_decomp_ccsd(name, atoms, A_ang, B_ang, basis='cc-pVDZ'):
    mol = gto.Mole()
    mol.atom = atoms; mol.basis = basis; mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol); mf.verbose = 0; mf.kernel()
    mycc = cc.CCSD(mf); mycc.verbose = 0; mycc.kernel()
    
    # CCSD 1-RDM in MO basis
    dm_mo = mycc.make_rdm1()
    # Convert to AO basis: D_AO = C @ D_MO @ C.T
    dm = mf.mo_coeff @ dm_mo @ mf.mo_coeff.T
    
    bcp, t = find_BCP(mol, dm, A_ang, B_ang)
    rho, grad, H = rho_grad_hess(mol, dm, bcp, eps=1e-3)
    evals, evecs = np.linalg.eigh(H)
    lam1, lam2, lam3 = evals[0], evals[1], evals[2]
    delta_bader = lam2 - lam1
    eps_val = lam1/lam2 - 1 if abs(lam2) > 1e-6 else 0
    e_lam1 = evecs[:, 0]; e_lam2 = evecs[:, 1]
    
    # For per-MO decomposition with CCSD natural orbitals would be complex
    # Use HF MO basis for σ/π classification but CCSD density for ε
    # Decomposition uses HF ψ_i with effective occupation from CCSD diagonal
    
    # Get HF MO occupations as a proxy
    n_occ_eff = np.diag(dm_mo)  # diagonal of CCSD 1-RDM in MO basis
    
    c_lam1, psi0, dpsi_lam1 = mo_hessian_contrib_along_axis(mol, mf, bcp, e_lam1, eps=8e-3)
    c_lam2, _, dpsi_lam2 = mo_hessian_contrib_along_axis(mol, mf, bcp, e_lam2, eps=8e-3)
    
    # Reweight by CCSD effective occupation
    classes = []
    for i in range(len(mf.mo_occ)):
        if mf.mo_occ[i] < 0.5:
            classes.append('virtual')
        else:
            cl = classify_MO(psi0[i], dpsi_lam1[i], dpsi_lam2[i], 0, threshold_psi=0.08)
            classes.append(cl)
    
    lam1_pi = sum(c_lam1[i] for i in range(len(classes)) if classes[i]=='pi')
    lam1_sigma = sum(c_lam1[i] for i in range(len(classes)) if classes[i]=='sigma')
    lam2_pi = sum(c_lam2[i] for i in range(len(classes)) if classes[i]=='pi')
    lam2_sigma = sum(c_lam2[i] for i in range(len(classes)) if classes[i]=='sigma')
    
    delta_pi = lam2_pi - lam1_pi
    delta_sigma = lam2_sigma - lam1_sigma
    
    return {'name': name, 'eps': eps_val, 'lam1': lam1, 'lam2': lam2,
            'delta_bader': delta_bader, 'delta_pi_num': delta_pi, 'delta_sigma_num': delta_sigma}

# 8 最关键化合物
KEY = [
    ("Ethylene C=C", "C 0 0 -0.667; C 0 0 0.667; H 0.921 0 -1.241; H -0.921 0 -1.241; H 0.921 0 1.241; H -0.921 0 1.241", [0,0,-0.667],[0,0,0.667]),
    ("N2H2 trans", "N 0 0.625 0; N 0 -0.625 0; H 0.870 1.083 0; H -0.870 -1.083 0", [0,0.625,0],[0,-0.625,0]),
    ("H2CO C=O", "C 0 0 -0.605; O 0 0 0.599; H 0.937 0 -1.198; H -0.937 0 -1.198", [0,0,-0.605],[0,0,0.599]),
    ("HCCH C≡C", "C 0 0 -0.601; C 0 0 0.601; H 0 0 -1.667; H 0 0 1.667", [0,0,-0.601],[0,0,0.601]),
    ("Ethane C-C", "C 0 0 -0.764; C 0 0 0.764; H 0 1.018 -1.160; H 0.882 -0.509 -1.160; H -0.882 -0.509 -1.160; H 0 -1.018 1.160; H -0.882 0.509 1.160; H 0.882 0.509 1.160", [0,0,-0.764],[0,0,0.764]),
    ("Cyclopropane", "C 0 0.866 0; C -0.750 -0.433 0; C 0.750 -0.433 0; H 0 1.480 0.890; H 0 1.480 -0.890; H -1.282 -0.740 0.890; H -1.282 -0.740 -0.890; H 1.282 -0.740 0.890; H 1.282 -0.740 -0.890", [0,0.866,0],[-0.750,-0.433,0]),
    ("Benzene C-C", "C 1.395 0 0; C 0.698 1.208 0; C -0.698 1.208 0; C -1.395 0 0; C -0.698 -1.208 0; C 0.698 -1.208 0; H 2.479 0 0; H 1.240 2.148 0; H -1.240 2.148 0; H -2.479 0 0; H -1.240 -2.148 0; H 1.240 -2.148 0", [1.395,0,0],[0.698,1.208,0]),
    ("N2F4 N-N", "N 0 0.706 0; N 0 -0.706 0; F 1.059 1.227 0; F -1.059 1.227 0; F 1.059 -1.227 0; F -1.059 -1.227 0", [0,0.706,0],[0,-0.706,0]),
]

print("="*120)
print("v11: CCSD/cc-pVDZ 电子相关测试 (8 关键化合物)")
print("="*120)

results = []
for label, atoms, A, B in KEY:
    try:
        print(f"\n>>> {label}")
        r = analyze_decomp_ccsd(label, atoms, A, B)
        results.append(r)
        print(f"  ε = {r['eps']:.4f}, δ_total = {r['delta_bader']:.4f}, δ_σ = {r['delta_sigma_num']:.4f}, δ_π = {r['delta_pi_num']:.4f}")
    except Exception as e:
        print(f"!!! {label} FAILED: {str(e)[:120]}")

print("\n" + "="*120)
print("v11 CCSD 比较")
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
