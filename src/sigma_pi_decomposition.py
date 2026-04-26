"""
sigma_pi_decomposition
======================

Per-Molecular-Orbital (MO) decomposition of the Quantum Theory of Atoms in
Molecules (QTAIM) Bond Critical Point (BCP) density Hessian.

Mathematical foundation
-----------------------
For a single-determinant electron density rho(r) = sum_i n_i psi_i^2(r), the
Hessian d^2 rho / dx_q^2 at any point can be decomposed as a sum over MOs:

    d^2 rho / dx_q^2 = sum_i 2 n_i [(d psi_i / dx_q)^2 + psi_i d^2 psi_i / dx_q^2]

This module evaluates that decomposition at the BCP, projects onto the Hessian
principal axes (eigenvectors of the eigenvalues lambda_1, lambda_2, lambda_3),
and aggregates the per-MO contributions into sigma vs pi groups.

Public API
----------
- find_BCP(mol, dm, A_ang, B_ang)
    Locate the bond critical point along an internuclear path.
- rho_grad_hess(mol, dm, r, eps=1e-3)
    Numerical density, gradient, Hessian at point r (Bohr).
- mo_hessian_contrib_along_axis(mol, mf, bcp, axis_unit, eps=1e-3)
    Per-MO contribution to d^2 rho / dq^2 along axis q.
- classify_MO(psi_at_bcp, dpsi_pi, dpsi_sigma_perp, dpsi_bond, threshold_psi)
    Sigma vs pi classification heuristic.
- analyze_decomp(name, atoms, A_ang, B_ang, basis="6-31G*")
    Full pipeline returning a result dict.

Author
------
Jian Zhou, JZ Institute of Science. <jack@jzis.org>

License
-------
CC BY 4.0
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


__all__ = [
    "find_BCP",
    "rho_grad_hess",
    "mo_hessian_contrib_along_axis",
    "classify_MO",
    "analyze_decomp",
]
