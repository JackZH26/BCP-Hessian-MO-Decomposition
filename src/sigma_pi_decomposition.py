"""
Loop 1 Experiment: σ/π MO decomposition of BCP Hessian eigenvalues.

For each MO:
  - Compute ψ(BCP)
  - Compute (∂ψ/∂x_λ1)² and (∂ψ/∂x_λ2)² (along Hessian principal axes)
  - Compute (∂²ψ/∂x_λ1²) and (∂²ψ/∂x_λ2²)
  - Contribution to ∂²ρ/∂x_q² = 2 n_i [(∂ψ_i/∂x_q)² + ψ_i ∂²ψ_i/∂x_q²]
  - Classify MO as π / σ_CC / σ_CH / core based on ψ(BCP) and symmetry
  - Sum within each class to get λ_1^(class), λ_2^(class)

Then:
  δ_class = λ_2^(class) - λ_1^(class)
  δ_total = Σ δ_class
  Verify: δ_total ≈ δ_Bader (sanity check)

Output: a table of δ_π / δ_σCC / δ_σCH / δ_core for each system.
"""
import numpy as np
from pyscf import gto, scf
import warnings; warnings.filterwarnings('ignore')
a0 = 0.529177

def build(atoms, basis='6-31G*'):
    mol = gto.Mole()
    mol.atom = atoms; mol.basis = basis; mol.verbose = 0
    mol.build()
    return mol

def rho_grad_hess(mol, dm, r, eps=1e-3):
    r_arr = np.array([r])
    ao = mol.eval_gto('GTOval', r_arr)
    ao_ip = mol.eval_gto('GTOval_ip', r_arr)
    rho = float(np.einsum('xi,ij,xj->x', ao, dm, ao)[0])
    grad = np.zeros(3)
    for k in range(3):
        grad[k] = 2 * float(np.einsum('xi,ij,xj->x', ao_ip[k], dm, ao)[0])
    H = np.zeros((3,3))
    for k in range(3):
        rp = r.copy(); rp[k] += eps
        rm = r.copy(); rm[k] -= eps
        aop = mol.eval_gto('GTOval', np.array([rp]))
        aop_ip = mol.eval_gto('GTOval_ip', np.array([rp]))
        aom = mol.eval_gto('GTOval', np.array([rm]))
        aom_ip = mol.eval_gto('GTOval_ip', np.array([rm]))
        for j in range(3):
            gpj = 2*float(np.einsum('xi,ij,xj->x', aop_ip[j], dm, aop)[0])
            gmj = 2*float(np.einsum('xi,ij,xj->x', aom_ip[j], dm, aom)[0])
            H[j,k] = (gpj - gmj)/(2*eps)
    H = 0.5*(H + H.T)
    return rho, grad, H

def find_BCP(mol, dm, A_ang, B_ang):
    A = np.array(A_ang)/a0; B = np.array(B_ang)/a0
    best = None
    for t in np.linspace(0.25, 0.75, 50):
        r = A + t*(B-A)
        _, g, _ = rho_grad_hess(mol, dm, r, eps=5e-3)
        gn = np.linalg.norm(g)
        if best is None or gn < best[0]:
            best = (gn, r.copy(), t)
    return best[1], best[2]

def mo_hessian_contrib_along_axis(mol, mf, bcp, axis_unit, eps=1e-3):
    """
    For each MO i, compute the contribution to ∂²ρ/∂q² at bcp where q is along axis_unit.
    Returns array of n_MO contributions: c_i = 2 n_i [(∂ψ_i/∂q)² + ψ_i (∂²ψ_i/∂q²)]
    Also returns ψ_i(BCP) and (∂ψ_i/∂q) for classification.
    """
    r0 = bcp
    rp = r0 + eps*axis_unit
    rm = r0 - eps*axis_unit
    
    # ψ_i and ∂ψ_i/∂q at r0, rp, rm
    def mo_at(r):
        ao = mol.eval_gto('GTOval', np.array([r]))[0]   # (nbasis,)
        ao_ip = mol.eval_gto('GTOval_ip', np.array([r]))  # (3, 1, nbasis)
        ao_ip = ao_ip[:, 0, :]  # (3, nbasis)
        psi = ao @ mf.mo_coeff   # (n_MO,)
        # ∂ψ/∂q = sum_k axis_k * (∂ψ/∂x_k)
        d_psi = np.einsum('k,kn->n', axis_unit, ao_ip @ mf.mo_coeff)
        return psi, d_psi
    
    psi0, dpsi0 = mo_at(r0)
    psip, dpsip = mo_at(rp)
    psim, dpsim = mo_at(rm)
    # ∂²ψ_i/∂q² ≈ (ψ(rp) - 2ψ(r0) + ψ(rm))/eps²
    d2psi0 = (psip - 2*psi0 + psim) / eps**2
    
    n_occ = mf.mo_occ
    contrib = np.zeros(len(n_occ))
    for i, n in enumerate(n_occ):
        if n < 0.5: continue
        contrib[i] = 2 * n * ((dpsi0[i])**2 + psi0[i] * d2psi0[i])
    return contrib, psi0, dpsi0

def classify_MO(psi_at_bcp, dpsi_pi, dpsi_sigma_perp, dpsi_bond, threshold_psi=0.05):
    """
    Classify MO at the BCP:
    - 'pi'   : ψ(BCP) ≈ 0 AND |dψ_pi| > 2|dψ_sigma_perp|
    - 'sigma_CC' : large ψ(BCP), gradient mostly along bond axis  (|dψ_bond| dominates)
    - 'sigma_CH' or 'sigma_other': small/medium ψ(BCP), gradient distributed
    - 'core': very small (deep core)
    Note: simpler heuristic — 'pi' if ψ small AND dψ_pi dominant, else 'sigma'.
    """
    if abs(psi_at_bcp) < threshold_psi and abs(dpsi_pi) > abs(dpsi_sigma_perp):
        return 'pi'
    return 'sigma'

def analyze_decomp(name, atoms, A_ang, B_ang, basis='6-31G*'):
    mol = build(atoms, basis=basis)
    mf = scf.RHF(mol); mf.verbose = 0; mf.kernel()
    dm = mf.make_rdm1()
    bcp, t = find_BCP(mol, dm, A_ang, B_ang)
    rho, grad, H = rho_grad_hess(mol, dm, bcp, eps=1e-3)
    evals, evecs = np.linalg.eigh(H)
    lam1, lam2, lam3 = evals[0], evals[1], evals[2]
    delta_bader = lam2 - lam1
    eps_val = lam1/lam2 - 1 if abs(lam2) > 1e-6 else 0
    # axes
    e_lam1 = evecs[:, 0]   # most negative direction (presumed π for double bond)
    e_lam2 = evecs[:, 1]   # less negative perpendicular
    e_lam3 = evecs[:, 2]   # bond axis
    
    # MO contributions to λ_1 and λ_2 directions
    c_lam1, psi0, dpsi_lam1 = mo_hessian_contrib_along_axis(mol, mf, bcp, e_lam1, eps=8e-3)
    c_lam2, _, dpsi_lam2 = mo_hessian_contrib_along_axis(mol, mf, bcp, e_lam2, eps=8e-3)
    _, _, dpsi_bond = mo_hessian_contrib_along_axis(mol, mf, bcp, e_lam3, eps=8e-3)
    
    # Classify each MO
    classes = []
    for i in range(len(mf.mo_occ)):
        if mf.mo_occ[i] < 0.5:
            classes.append('virtual')
        else:
            cl = classify_MO(psi0[i], dpsi_lam1[i], dpsi_lam2[i], dpsi_bond[i],
                             threshold_psi=0.08)
            classes.append(cl)
    
    # Sum contributions per class
    lam1_pi = sum(c_lam1[i] for i in range(len(classes)) if classes[i]=='pi')
    lam1_sigma = sum(c_lam1[i] for i in range(len(classes)) if classes[i]=='sigma')
    lam2_pi = sum(c_lam2[i] for i in range(len(classes)) if classes[i]=='pi')
    lam2_sigma = sum(c_lam2[i] for i in range(len(classes)) if classes[i]=='sigma')
    
    delta_pi = lam2_pi - lam1_pi
    delta_sigma = lam2_sigma - lam1_sigma
    delta_total = delta_pi + delta_sigma
    
    # Sanity check: lam1_pi + lam1_sigma should ≈ lam1
    lam1_recon = lam1_pi + lam1_sigma
    lam2_recon = lam2_pi + lam2_sigma
    
    print(f"\n=== {name}  basis={basis} ===")
    print(f"  BCP: t={t:.3f}, ρ_BCP={rho:.4f}")
    print(f"  Hessian eigenvals: λ₁={lam1:+.4f} λ₂={lam2:+.4f} λ₃={lam3:+.4f}")
    print(f"  ε = λ₁/λ₂-1 = {eps_val:.4f},  δ_Bader = λ₂-λ₁ = {delta_bader:.4f}")
    print(f"  Reconstruction: λ₁_recon = {lam1_recon:+.4f} (target {lam1:+.4f})")
    print(f"                  λ₂_recon = {lam2_recon:+.4f} (target {lam2:+.4f})")
    print(f"  -- π MOs contribute: λ₁_π={lam1_pi:+.4f}  λ₂_π={lam2_pi:+.4f}  δ_π = {delta_pi:.4f}")
    print(f"  -- σ MOs contribute: λ₁_σ={lam1_sigma:+.4f}  λ₂_σ={lam2_sigma:+.4f}  δ_σ = {delta_sigma:.4f}")
    print(f"  Sum δ_π + δ_σ = {delta_total:.4f}  vs δ_Bader = {delta_bader:.4f}  (recon err {100*(delta_total-delta_bader)/delta_bader:+.1f}%)")
    
    # MO-level breakdown for the most important MOs (top 5 by |contribution|)
    print(f"  Top MO contributions to (λ₂-λ₁):")
    deltas = [(i, c_lam2[i]-c_lam1[i], psi0[i], classes[i]) for i in range(len(classes)) if classes[i] != 'virtual']
    deltas.sort(key=lambda x: -abs(x[1]))
    for (i, d, psi_i, cl) in deltas[:6]:
        print(f"    MO[{i}] ({cl})  ψ(BCP)={psi_i:+.3f}  contrib_to_δ = {d:+.4f}")
    
    return {
        'name': name, 'rho': rho, 'eps': eps_val,
        'lam1': lam1, 'lam2': lam2, 'delta_bader': delta_bader,
        'delta_pi_num': delta_pi, 'delta_sigma_num': delta_sigma,
        'delta_total_recon': delta_total,
        'classes': classes
    }

# ----- Test systems -----
results = []

# 1. Ethylene
results.append(analyze_decomp("Ethylene C=C",
    "C 0 0 -0.667; C 0 0 0.667; H 0.921 0 -1.241; H -0.921 0 -1.241; H 0.921 0 1.241; H -0.921 0 1.241",
    [0,0,-0.667], [0,0,0.667]))

# 2. Propene C=C terminal
results.append(analyze_decomp("Propene C=C",
    "C 0 0.000 -0.766; C 0 0.000 0.570; C 0 1.254 1.305; H 0 -0.930 -1.316; H 0.920 0.540 -1.316; H 0 -0.930 1.110; H 0 2.120 0.640; H 0.870 1.320 1.960; H -0.870 1.320 1.960",
    [0,0,-0.766], [0,0,0.570]))

# 3. Butadiene 1-2
buta_atoms = "C 1.853 0.682 0; C 0.611 0 0; C -0.611 0 0; C -1.853 0.682 0; H 1.853 1.770 0; H 2.810 0.220 0; H 0.611 -1.089 0; H -0.611 -1.089 0; H -1.853 1.770 0; H -2.810 0.220 0"
results.append(analyze_decomp("Butadiene 1-2", buta_atoms, [1.853, 0.682, 0], [0.611, 0, 0]))

# 4. Butadiene 2-3
results.append(analyze_decomp("Butadiene 2-3", buta_atoms, [0.611, 0, 0], [-0.611, 0, 0]))

# 5. Benzene
benz_atoms = "C 1.395 0 0; C 0.698 1.208 0; C -0.698 1.208 0; C -1.395 0 0; C -0.698 -1.208 0; C 0.698 -1.208 0; H 2.479 0 0; H 1.240 2.148 0; H -1.240 2.148 0; H -2.479 0 0; H -1.240 -2.148 0; H 1.240 -2.148 0"
results.append(analyze_decomp("Benzene C-C", benz_atoms, [1.395, 0, 0], [0.698, 1.208, 0]))

# Naphthalene (planar) — using clean geometry
napth_atoms = """
C  1.243 0.715 0
C  0.000 1.413 0
C -1.243 0.715 0
C -1.243 -0.715 0
C  0.000 -1.413 0
C  1.243 -0.715 0
C  2.484 1.420 0
C  2.484 2.840 0
C  1.243 3.553 0
C  0.000 2.840 0
H  3.444 0.876 0
H  3.444 3.380 0
H  1.243 4.638 0
H -1.000 3.380 0
H -2.205 1.255 0
H -2.205 -1.255 0
H -1.000 -2.483 0
H  1.000 -2.483 0
"""
# Naph 0-1 (peripheral C1-C2)
results.append(analyze_decomp("Naphthalene 0-1", napth_atoms, [1.243, 0.715, 0], [0.0, 1.413, 0]))
# Naph 0-5 (fusion C-C, indices 0 and 5)
results.append(analyze_decomp("Naphthalene 0-5", napth_atoms, [1.243, 0.715, 0], [1.243, -0.715, 0]))
# Naph 1-2 - between C0(1.243,0.715) and C6(2.484,1.420)
results.append(analyze_decomp("Naphthalene 0-6 (peri)", napth_atoms, [1.243, 0.715, 0], [2.484, 1.420, 0]))

# Summary
print("\n" + "="*100)
print("SIGMA / PI DECOMPOSITION SUMMARY")
print("="*100)
print(f"{'System':28} {'δ_Bader':>10} {'δ_π_num':>10} {'δ_σ_num':>10} {'recon%':>8} {'σ/Bader':>10}")
for r in results:
    rcn = 100*(r['delta_total_recon']-r['delta_bader'])/r['delta_bader']
    sigma_ratio = r['delta_sigma_num']/r['delta_bader']
    print(f"{r['name']:28} {r['delta_bader']:10.4f} {r['delta_pi_num']:10.4f} "
          f"{r['delta_sigma_num']:10.4f} {rcn:+8.1f} {sigma_ratio:+10.4f}")
