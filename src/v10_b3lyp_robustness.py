"""
v10：B3LYP/def2-TZVP 电子相关 robustness 测试
==============================================
验证 HF 结果是否会被电子相关 (DFT vs HF) 改变符号
- HF (v6/v7/v8): 已得 71/72 通过
- cc-pVTZ HF (v9): 16/16 通过
- B3LYP/def2-TZVP (v10, NEW): 验证 DFT 是否给同符号
"""
import sys, numpy as np
sys.path.insert(0, '/data/.openclaw/workspace/research/experiments/cbt/2026-04-26/loop_1')
from sigma_pi_decomposition import build, find_BCP, rho_grad_hess, mo_hessian_contrib_along_axis, classify_MO
from pyscf import gto, dft
import warnings; warnings.filterwarnings('ignore')

def analyze_decomp_dft(name, atoms, A_ang, B_ang, basis='def2-TZVP', xc='B3LYP'):
    mol = gto.Mole()
    mol.atom = atoms; mol.basis = basis; mol.verbose = 0
    mol.build()
    mf = dft.RKS(mol); mf.xc = xc; mf.verbose = 0; mf.kernel()
    dm = mf.make_rdm1()
    bcp, t = find_BCP(mol, dm, A_ang, B_ang)
    rho, grad, H = rho_grad_hess(mol, dm, bcp, eps=1e-3)
    evals, evecs = np.linalg.eigh(H)
    lam1, lam2, lam3 = evals[0], evals[1], evals[2]
    delta_bader = lam2 - lam1
    eps_val = lam1/lam2 - 1 if abs(lam2) > 1e-6 else 0
    e_lam1 = evecs[:, 0]; e_lam2 = evecs[:, 1]; e_lam3 = evecs[:, 2]
    
    c_lam1, psi0, dpsi_lam1 = mo_hessian_contrib_along_axis(mol, mf, bcp, e_lam1, eps=8e-3)
    c_lam2, _, dpsi_lam2 = mo_hessian_contrib_along_axis(mol, mf, bcp, e_lam2, eps=8e-3)
    _, _, dpsi_bond = mo_hessian_contrib_along_axis(mol, mf, bcp, e_lam3, eps=8e-3)
    
    classes = []
    for i in range(len(mf.mo_occ)):
        if mf.mo_occ[i] < 0.5:
            classes.append('virtual')
        else:
            cl = classify_MO(psi0[i], dpsi_lam1[i], dpsi_lam2[i], dpsi_bond[i], threshold_psi=0.08)
            classes.append(cl)
    
    lam1_pi = sum(c_lam1[i] for i in range(len(classes)) if classes[i]=='pi')
    lam1_sigma = sum(c_lam1[i] for i in range(len(classes)) if classes[i]=='sigma')
    lam2_pi = sum(c_lam2[i] for i in range(len(classes)) if classes[i]=='pi')
    lam2_sigma = sum(c_lam2[i] for i in range(len(classes)) if classes[i]=='sigma')
    
    delta_pi = lam2_pi - lam1_pi
    delta_sigma = lam2_sigma - lam1_sigma
    
    return {'name': name, 'eps': eps_val, 'lam1': lam1, 'lam2': lam2, 
            'delta_bader': delta_bader, 'delta_pi_num': delta_pi, 'delta_sigma_num': delta_sigma}

# 16 关键化合物（与 v9 一致）
KEY_TESTS = [
    ("Ethylene C=C", "C 0 0 -0.667; C 0 0 0.667; H 0.921 0 -1.241; H -0.921 0 -1.241; H 0.921 0 1.241; H -0.921 0 1.241", [0,0,-0.667],[0,0,0.667]),
    ("N2H2 trans N=N", "N 0 0.625 0; N 0 -0.625 0; H 0.870 1.083 0; H -0.870 -1.083 0", [0,0.625,0],[0,-0.625,0]),
    ("H2CO C=O", "C 0 0 -0.605; O 0 0 0.599; H 0.937 0 -1.198; H -0.937 0 -1.198", [0,0,-0.605],[0,0,0.599]),
    ("Acetone C=O", "C 0 0 0; O 0 0 1.221; C 1.538 0 -0.500; C -1.538 0 -0.500; H 1.916 1.018 -0.152; H 1.916 -1.018 -0.152; H 1.538 0 -1.590; H -1.916 1.018 -0.152; H -1.916 -1.018 -0.152; H -1.538 0 -1.590", [0,0,0],[0,0,1.221]),
    ("HCCH C≡C", "C 0 0 -0.601; C 0 0 0.601; H 0 0 -1.667; H 0 0 1.667", [0,0,-0.601],[0,0,0.601]),
    ("N2 N≡N", "N 0 0 -0.549; N 0 0 0.549", [0,0,-0.549],[0,0,0.549]),
    ("Ethane C-C", "C 0 0 -0.764; C 0 0 0.764; H 0 1.018 -1.160; H 0.882 -0.509 -1.160; H -0.882 -0.509 -1.160; H 0 -1.018 1.160; H -0.882 0.509 1.160; H 0.882 0.509 1.160", [0,0,-0.764],[0,0,0.764]),
    ("Cyclopropane C-C", "C 0 0.866 0; C -0.750 -0.433 0; C 0.750 -0.433 0; H 0 1.480 0.890; H 0 1.480 -0.890; H -1.282 -0.740 0.890; H -1.282 -0.740 -0.890; H 1.282 -0.740 0.890; H 1.282 -0.740 -0.890", [0,0.866,0],[-0.750,-0.433,0]),
    ("Allene C=C", "C 0 0 0; C 0 0 1.309; C 0 0 -1.309; H 0.926 0 1.839; H -0.926 0 1.839; H 0 0.926 -1.839; H 0 -0.926 -1.839", [0,0,0],[0,0,1.309]),
    ("BH3 B-H", "B 0 0 0; H 1.193 0 0; H -0.596 1.032 0; H -0.596 -1.032 0", [0,0,0],[1.193,0,0]),
    ("Benzene C-C", "C 1.395 0 0; C 0.698 1.208 0; C -0.698 1.208 0; C -1.395 0 0; C -0.698 -1.208 0; C 0.698 -1.208 0; H 2.479 0 0; H 1.240 2.148 0; H -1.240 2.148 0; H -2.479 0 0; H -1.240 -2.148 0; H 1.240 -2.148 0", [1.395,0,0],[0.698,1.208,0]),
    ("Pyridine C2-C3", "N 0 1.134 0; C 0.977 0.569 0; C -0.977 0.569 0; C 1.197 -0.820 0; C -1.197 -0.820 0; C 0 -1.599 0; H 1.737 1.222 0; H -1.737 1.222 0; H 2.196 -1.175 0; H -2.196 -1.175 0; H 0 -2.681 0", [0.977,0.569,0],[1.197,-0.820,0]),
    ("Furan C2-C3", "O 0 1.209 0; C 1.157 0.374 0; C -1.157 0.374 0; C 0.710 -0.994 0; C -0.710 -0.994 0; H 2.069 0.953 0; H -2.069 0.953 0; H 1.256 -1.989 0; H -1.256 -1.989 0", [1.157,0.374,0],[0.710,-0.994,0]),
    ("N2F4 N-N", "N 0 0.706 0; N 0 -0.706 0; F 1.059 1.227 0; F -1.059 1.227 0; F 1.059 -1.227 0; F -1.059 -1.227 0", [0,0.706,0],[0,-0.706,0]),
    ("NH3 N-H", "N 0 0 0.067; H 0.939 0 -0.312; H -0.470 0.813 -0.312; H -0.470 -0.813 -0.312", [0,0,0.067],[0.939,0,-0.312]),
    ("trans-CHF=CHF C=C", "C 0 -0.670 0; C 0 0.670 0; F 1.140 -1.290 0; F -1.140 1.290 0; H -0.940 -1.260 0; H 0.940 1.260 0", [0,-0.670,0],[0,0.670,0]),
]

print("="*120)
print("v10: B3LYP/def2-TZVP 电子相关 robustness 测试 (16 关键化合物)")
print("="*120)

results = []
for label, atoms, A, B in KEY_TESTS:
    try:
        print(f"\n>>> {label}")
        r = analyze_decomp_dft(label, atoms, A, B, basis='def2-TZVP', xc='B3LYP')
        results.append(r)
        print(f"  ε = {r['eps']:.4f}, δ_total = {r['delta_bader']:.4f}, δ_σ = {r['delta_sigma_num']:.4f}, δ_π = {r['delta_pi_num']:.4f}")
    except Exception as e:
        print(f"!!! {label} FAILED: {str(e)[:100]}")

print("\n" + "="*120)
print("v10 B3LYP vs HF 比较")
print("="*120)
print(f"{'分子':32} {'ε B3LYP':9} {'δ_σ':9} {'δ_π':9} {'符号'}")
print("-"*120)

h1_pass = 0; h1_total = 0
h2_pass = 0; h2_total = 0
for r in results:
    eps = r['eps']; dpi = r['delta_pi_num']; dsi = r['delta_sigma_num']
    if eps > 0.05:
        h1_total += 1
        if abs(dpi) < 0.005:
            sign = "σ-only"
            if dsi > 0: h1_pass += 1
        elif dsi > 0 and dpi < 0:
            sign = "✓"; h1_pass += 1
        else:
            sign = "✗"
    elif eps < 0.01:
        h2_total += 1
        if abs(dpi+dsi) < 0.01 + 0.01*max(abs(dpi),abs(dsi)):
            sign = "DEGEN✓"; h2_pass += 1
        else:
            sign = "DEGEN?"
    else:
        sign = "border"
    print(f"{r['name']:32} {eps:+8.4f}  {dsi:+8.4f}  {dpi:+8.4f}  {sign}")

print(f"\nv10 B3LYP: H1 {h1_pass}/{h1_total}, H2 {h2_pass}/{h2_total}")
print("="*120)
