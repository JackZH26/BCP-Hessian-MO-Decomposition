"""
v13：Localized MO 独立分类验证
==============================
用 Boys 和 Pipek-Mezey localized MOs 独立做 σ/π 分类
验证 δ_σ > 0, δ_π < 0 规律不依赖于 canonical MO 选择

理论基础：
- 对 closed-shell RHF：ρ = 2 Σ_i ψ_i² 在占据空间酉变换下不变
- 但 per-MO Hessian 贡献依赖具体 MO 集合
- 如果 LMO 分类后仍给同样符号 → 规律 robust
"""
import sys, numpy as np
sys.path.insert(0, '/data/.openclaw/workspace/research/experiments/cbt/2026-04-26/loop_1')
from sigma_pi_decomposition import find_BCP, rho_grad_hess, classify_MO
from pyscf import gto, scf, lo
import warnings; warnings.filterwarnings('ignore')

a0 = 0.529177

def mo_hessian_lmo(mol, lmo_coeff, mo_occ_lmo, bcp, axis_unit, eps=1e-3):
    """Compute per-LMO contribution to ∂²ρ/∂q² at bcp, where ρ = Σ n_a (ψ'_a)²."""
    r0 = bcp
    rp = r0 + eps*axis_unit
    rm = r0 - eps*axis_unit
    
    def lmo_at(r):
        ao = mol.eval_gto('GTOval', np.array([r]))[0]
        ao_ip = mol.eval_gto('GTOval_ip', np.array([r]))
        ao_ip = ao_ip[:, 0, :]
        psi = ao @ lmo_coeff
        d_psi = np.einsum('k,kn->n', axis_unit, ao_ip @ lmo_coeff)
        return psi, d_psi
    
    psi0, dpsi0 = lmo_at(r0)
    psip, dpsip = lmo_at(rp)
    psim, dpsim = lmo_at(rm)
    d2psi0 = (psip - 2*psi0 + psim) / eps**2
    
    contrib = np.zeros(len(mo_occ_lmo))
    for i, n in enumerate(mo_occ_lmo):
        if n < 0.5: continue
        contrib[i] = 2 * n * ((dpsi0[i])**2 + psi0[i] * d2psi0[i])
    return contrib, psi0, dpsi0


def analyze_lmo(name, atoms, A_ang, B_ang, basis='6-31G*', method='boys'):
    mol = gto.Mole()
    mol.atom = atoms; mol.basis = basis; mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol); mf.verbose = 0; mf.kernel()
    dm = mf.make_rdm1()
    
    # Find BCP using full density
    bcp, t = find_BCP(mol, dm, A_ang, B_ang)
    rho, grad, H = rho_grad_hess(mol, dm, bcp, eps=1e-3)
    evals, evecs = np.linalg.eigh(H)
    lam1, lam2, lam3 = evals[0], evals[1], evals[2]
    delta_bader = lam2 - lam1
    eps_val = lam1/lam2 - 1 if abs(lam2) > 1e-6 else 0
    e_lam1 = evecs[:, 0]; e_lam2 = evecs[:, 1]
    
    # Get occupied MO coefficients
    n_occ = (mf.mo_occ > 0.5).sum()
    occ_coeff = mf.mo_coeff[:, :n_occ]
    
    # Localize occupied MOs
    if method == 'boys':
        loc = lo.Boys(mol, occ_coeff)
        loc.verbose = 0
        lmo_coeff = loc.kernel()
    elif method == 'pm':
        loc = lo.PipekMezey(mol, occ_coeff)
        loc.verbose = 0
        lmo_coeff = loc.kernel()
    elif method == 'ibo':
        lmo_coeff = lo.ibo.ibo(mol, occ_coeff)
    else:
        lmo_coeff = occ_coeff  # canonical
    
    n_lmo = lmo_coeff.shape[1]
    occ_lmo = np.full(n_lmo, 2.0)  # closed shell
    
    # Compute LMO Hessian contributions
    c_lam1, psi0_l, dpsi_lam1_l = mo_hessian_lmo(mol, lmo_coeff, occ_lmo, bcp, e_lam1, eps=8e-3)
    c_lam2, _, dpsi_lam2_l = mo_hessian_lmo(mol, lmo_coeff, occ_lmo, bcp, e_lam2, eps=8e-3)
    
    # Classify each LMO using same heuristic
    classes = []
    for i in range(n_lmo):
        cl = classify_MO(psi0_l[i], dpsi_lam1_l[i], dpsi_lam2_l[i], 0, threshold_psi=0.08)
        classes.append(cl)
    
    lam1_pi = sum(c_lam1[i] for i in range(n_lmo) if classes[i]=='pi')
    lam1_sigma = sum(c_lam1[i] for i in range(n_lmo) if classes[i]=='sigma')
    lam2_pi = sum(c_lam2[i] for i in range(n_lmo) if classes[i]=='pi')
    lam2_sigma = sum(c_lam2[i] for i in range(n_lmo) if classes[i]=='sigma')
    
    delta_pi = lam2_pi - lam1_pi
    delta_sigma = lam2_sigma - lam1_sigma
    delta_total = delta_pi + delta_sigma
    
    return {'name': name, 'method': method, 'eps': eps_val,
            'delta_bader': delta_bader, 'delta_pi_num': delta_pi, 'delta_sigma_num': delta_sigma,
            'delta_total_recon': delta_total,
            'n_pi': sum(1 for c in classes if c=='pi'),
            'n_sigma': sum(1 for c in classes if c=='sigma')}


KEY = [
    ("Ethylene", "C 0 0 -0.667; C 0 0 0.667; H 0.921 0 -1.241; H -0.921 0 -1.241; H 0.921 0 1.241; H -0.921 0 1.241", [0,0,-0.667],[0,0,0.667]),
    ("N2H2", "N 0 0.625 0; N 0 -0.625 0; H 0.870 1.083 0; H -0.870 -1.083 0", [0,0.625,0],[0,-0.625,0]),
    ("H2CO", "C 0 0 -0.605; O 0 0 0.599; H 0.937 0 -1.198; H -0.937 0 -1.198", [0,0,-0.605],[0,0,0.599]),
    ("HCCH", "C 0 0 -0.601; C 0 0 0.601; H 0 0 -1.667; H 0 0 1.667", [0,0,-0.601],[0,0,0.601]),
    ("Ethane", "C 0 0 -0.764; C 0 0 0.764; H 0 1.018 -1.160; H 0.882 -0.509 -1.160; H -0.882 -0.509 -1.160; H 0 -1.018 1.160; H -0.882 0.509 1.160; H 0.882 0.509 1.160", [0,0,-0.764],[0,0,0.764]),
    ("Cyclopropane", "C 0 0.866 0; C -0.750 -0.433 0; C 0.750 -0.433 0; H 0 1.480 0.890; H 0 1.480 -0.890; H -1.282 -0.740 0.890; H -1.282 -0.740 -0.890; H 1.282 -0.740 0.890; H 1.282 -0.740 -0.890", [0,0.866,0],[-0.750,-0.433,0]),
    ("Allene", "C 0 0 0; C 0 0 1.309; C 0 0 -1.309; H 0.926 0 1.839; H -0.926 0 1.839; H 0 0.926 -1.839; H 0 -0.926 -1.839", [0,0,0],[0,0,1.309]),
    ("BH3", "B 0 0 0; H 1.193 0 0; H -0.596 1.032 0; H -0.596 -1.032 0", [0,0,0],[1.193,0,0]),
    ("Benzene", "C 1.395 0 0; C 0.698 1.208 0; C -0.698 1.208 0; C -1.395 0 0; C -0.698 -1.208 0; C 0.698 -1.208 0; H 2.479 0 0; H 1.240 2.148 0; H -1.240 2.148 0; H -2.479 0 0; H -1.240 -2.148 0; H 1.240 -2.148 0", [1.395,0,0],[0.698,1.208,0]),
    ("Furan", "O 0 1.209 0; C 1.157 0.374 0; C -1.157 0.374 0; C 0.710 -0.994 0; C -0.710 -0.994 0; H 2.069 0.953 0; H -2.069 0.953 0; H 1.256 -1.989 0; H -1.256 -1.989 0", [1.157,0.374,0],[0.710,-0.994,0]),
    ("N2F4", "N 0 0.706 0; N 0 -0.706 0; F 1.059 1.227 0; F -1.059 1.227 0; F 1.059 -1.227 0; F -1.059 -1.227 0", [0,0.706,0],[0,-0.706,0]),
    ("CHF=CHF trans", "C 0 -0.670 0; C 0 0.670 0; F 1.140 -1.290 0; F -1.140 1.290 0; H -0.940 -1.260 0; H 0.940 1.260 0", [0,-0.670,0],[0,0.670,0]),
]

print("="*120)
print("v13: Localized MO 独立分类 (Boys + Pipek-Mezey + IBO)")
print("="*120)

for label, atoms, A, B in KEY:
    print(f"\n>>> {label}")
    for method in ['boys', 'pm', 'ibo']:
        try:
            r = analyze_lmo(label, atoms, A, B, method=method)
            print(f"  {method:6}: ε={r['eps']:+7.4f}  δ_σ={r['delta_sigma_num']:+7.4f}  δ_π={r['delta_pi_num']:+7.4f}  recon={r['delta_total_recon']:+7.4f} (target {r['delta_bader']:+7.4f})  n_σ={r['n_sigma']} n_π={r['n_pi']}")
        except Exception as e:
            print(f"  {method:6}: FAILED {str(e)[:80]}")

print("\n" + "="*120)
print("v13 验证：LMO 是否复现 δ_σ > 0, δ_π < 0 规律")
print("="*120)
