"""
Path B v14: Group-theoretic σ/π MO classification via σ_h reflection
=====================================================================
Replaces the geometric heuristic (criterion 2: |∂ψ/∂ξ_1|² > |∂ψ/∂ξ_2|²)
with a strictly symmetry-based classification:

  σ MO: σ_h ψ_i = +ψ_i  (symmetric under reflection through molecular plane)
  π MO: σ_h ψ_i = -ψ_i  (antisymmetric)

For planar molecules with reflection symmetry σ_h, this is a strict
group-theoretic definition INDEPENDENT of BCP-local quantities.

If sign rule (δ_σ > 0, δ_π < 0) still holds under this independent
classification, it is a TRUE empirical finding, not a tautology.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                'experiments', 'cbt', '2026-04-26', 'loop_1'))
from sigma_pi_decomposition import build, find_BCP, rho_grad_hess, mo_hessian_contrib_along_axis
from pyscf import gto, scf
import warnings; warnings.filterwarnings('ignore')

a0 = 0.529177


def reflect_through_plane(coord, plane='xy'):
    """Apply σ_h reflection: flip the coordinate normal to the mirror plane."""
    new = coord.copy()
    if plane == 'xy':   # mirror is xy plane → flip z
        new[..., 2] = -new[..., 2]
    elif plane == 'xz': # mirror is xz plane → flip y
        new[..., 1] = -new[..., 1]
    elif plane == 'yz': # mirror is yz plane → flip x
        new[..., 0] = -new[..., 0]
    return new


def compute_mo_reflection_eigenvalue(mol, mf, mirror_plane='xy', n_grid=30, grid_radius=4.0):
    """
    Compute ⟨ψ_i | σ_h ψ_i⟩ for each occupied MO.
    
    Returns:
        eigenvalues: array of shape (n_occ,) with values in [-1, +1]
                     +1 → σ (symmetric), -1 → π (antisymmetric)
    """
    n_occ = (mf.mo_occ > 0.5).sum()
    
    # Build a 3D grid in Bohr
    pts_per_axis = n_grid
    half = grid_radius / a0  # convert Å to Bohr
    grid_1d = np.linspace(-half, half, pts_per_axis)
    XX, YY, ZZ = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
    grid_pts = np.stack([XX.flatten(), YY.flatten(), ZZ.flatten()], axis=1)
    
    # Volume element
    dV = (2*half / (pts_per_axis - 1))**3
    
    # Evaluate AO on grid
    ao_grid = mol.eval_gto('GTOval', grid_pts)  # (n_pts, n_basis)
    psi_grid = ao_grid @ mf.mo_coeff  # (n_pts, n_mo)
    
    # Evaluate AO on reflected grid
    grid_reflected = reflect_through_plane(grid_pts, plane=mirror_plane)
    ao_grid_r = mol.eval_gto('GTOval', grid_reflected)
    psi_grid_r = ao_grid_r @ mf.mo_coeff
    
    # Compute ⟨ψ_i | σ_h ψ_i⟩ for each MO
    # σ_h ψ(r) = ψ(σ_h r), so we need ∫ ψ_i(r) ψ_i(σ_h r) dr
    eigenvalues = np.zeros(n_occ)
    norms = np.zeros(n_occ)
    for i in range(n_occ):
        # ⟨ψ_i | σ_h ψ_i⟩ ≈ Σ_pts ψ_i(r) · ψ_i(σ_h r) · dV
        integral = np.sum(psi_grid[:, i] * psi_grid_r[:, i]) * dV
        norm_squared = np.sum(psi_grid[:, i]**2) * dV  # should be ~1 for good grid
        eigenvalues[i] = integral / norm_squared if abs(norm_squared) > 1e-10 else 0
        norms[i] = norm_squared
    
    return eigenvalues, norms


def detect_mirror_plane(atoms_str):
    """Auto-detect the molecular plane from atom coordinates.
    Returns 'xy', 'xz', 'yz', or None if non-planar."""
    atom_lines = [a.strip() for a in atoms_str.replace('\n', ';').split(';') if a.strip()]
    coords = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) >= 4:
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    coords = np.array(coords)
    
    # Check which coordinate is closest to all-zero (within 0.05 Å)
    for axis_idx, plane_name in zip([2, 1, 0], ['xy', 'xz', 'yz']):
        if np.max(np.abs(coords[:, axis_idx])) < 0.05:
            return plane_name
    return None


def analyze_symmetry_decomp(name, atoms, A_ang, B_ang, basis='6-31G*',
                             reflection_threshold=0.7, mirror=None):
    """
    Full pipeline: SCF → BCP → Hessian → group-theoretic σ/π → δ_σ, δ_π
    
    Reflection threshold: classify as σ if eigenvalue > +threshold,
                          π if eigenvalue < -threshold,
                          'mixed' otherwise (excluded from σ/π statistics).
    """
    if mirror is None:
        mirror = detect_mirror_plane(atoms)
        if mirror is None:
            return {'name': name, 'error': 'No mirror plane detected (non-planar)'}
    
    mol = build(atoms, basis=basis)
    mf = scf.RHF(mol); mf.verbose = 0; mf.kernel()
    dm = mf.make_rdm1()
    
    # Find BCP
    bcp, t = find_BCP(mol, dm, A_ang, B_ang)
    rho, grad, H = rho_grad_hess(mol, dm, bcp, eps=1e-3)
    evals, evecs = np.linalg.eigh(H)
    lam1, lam2, lam3 = evals
    delta_bader = lam2 - lam1
    eps_val = lam1/lam2 - 1 if abs(lam2) > 1e-6 else 0
    e_lam1 = evecs[:, 0]; e_lam2 = evecs[:, 1]
    
    # Compute MO reflection eigenvalues (independent of BCP geometry!)
    refl_eig, norms = compute_mo_reflection_eigenvalue(mol, mf, mirror_plane=mirror)
    
    # Classify based on σ_h eigenvalue (group theory, NOT BCP geometry)
    n_occ = (mf.mo_occ > 0.5).sum()
    classes = []
    for i in range(n_occ):
        if refl_eig[i] > reflection_threshold:
            classes.append('sigma')
        elif refl_eig[i] < -reflection_threshold:
            classes.append('pi')
        else:
            classes.append('mixed')
    
    # Per-MO Hessian contributions
    c_lam1, psi0, dpsi_lam1 = mo_hessian_contrib_along_axis(mol, mf, bcp, e_lam1, eps=8e-3)
    c_lam2, _, dpsi_lam2 = mo_hessian_contrib_along_axis(mol, mf, bcp, e_lam2, eps=8e-3)
    
    delta_sigma = sum((c_lam2[i] - c_lam1[i]) for i in range(n_occ) if classes[i] == 'sigma')
    delta_pi = sum((c_lam2[i] - c_lam1[i]) for i in range(n_occ) if classes[i] == 'pi')
    delta_mixed = sum((c_lam2[i] - c_lam1[i]) for i in range(n_occ) if classes[i] == 'mixed')
    
    n_sigma = sum(1 for c in classes if c == 'sigma')
    n_pi = sum(1 for c in classes if c == 'pi')
    n_mixed = sum(1 for c in classes if c == 'mixed')
    
    return {
        'name': name, 'mirror': mirror, 'eps': eps_val,
        'lam1': lam1, 'lam2': lam2, 'delta_bader': delta_bader,
        'delta_sigma': delta_sigma, 'delta_pi': delta_pi, 'delta_mixed': delta_mixed,
        'n_sigma': n_sigma, 'n_pi': n_pi, 'n_mixed': n_mixed,
        'reflection_eigenvalues': refl_eig.tolist(),
        'mo_classes': classes,
    }


# =====================================================================
# Stage 1: Validate on ethylene (ground truth)
# =====================================================================
if __name__ == '__main__':
    print("="*100)
    print("v14 Path B Stage 1: σ_h reflection classification on ETHYLENE (ground truth)")
    print("="*100)
    print("Ethylene C2H4: D_2h symmetry, molecular plane = xz (y is normal)")
    print("Expected occupied MOs (HF/6-31G*): 1a_g, 1b_{1u}, 2a_g, 2b_{1u}, 1b_{2u}, 3a_g, 1b_{3g}, 1b_{3u}")
    print("Expected: 7 σ-type (in plane, ε_σh = +1), 1 π-type (1b_{3u}, ε_σh = -1)")
    print()
    
    # Ethylene: place in xz plane, y = 0
    atoms = "C 0 0 -0.667; C 0 0 0.667; H 0.921 0 -1.241; H -0.921 0 -1.241; H 0.921 0 1.241; H -0.921 0 1.241"
    
    # Mirror plane should be xz (y=0)
    print("Auto-detected mirror plane:", detect_mirror_plane(atoms))
    print()
    
    r = analyze_symmetry_decomp("Ethylene C=C", atoms, [0,0,-0.667], [0,0,0.667])
    
    print("="*100)
    print("RESULTS")
    print("="*100)
    print(f"  Mirror plane: {r['mirror']}")
    print(f"  ε = {r['eps']:.4f}")
    print(f"  δ_total = {r['delta_bader']:.4f}")
    print(f"  δ_σ = {r['delta_sigma']:.4f}  ({r['n_sigma']} σ MOs)")
    print(f"  δ_π = {r['delta_pi']:.4f}  ({r['n_pi']} π MOs)")
    print(f"  δ_mixed = {r['delta_mixed']:.4f}  ({r['n_mixed']} mixed)")
    print(f"  Reconstruction (σ+π+mixed): {r['delta_sigma']+r['delta_pi']+r['delta_mixed']:.4f}")
    print()
    print("Per-MO σ_h eigenvalues:")
    for i, (eig, cls) in enumerate(zip(r['reflection_eigenvalues'], r['mo_classes'])):
        marker = '✓' if abs(eig) > 0.7 else '?'
        print(f"  MO[{i}]  σ_h_eig = {eig:+.4f}  → {cls:6s} {marker}")
    
    print()
    print("Sign rule check:")
    if r['delta_sigma'] > 0 and r['delta_pi'] < 0:
        print("  ✓ δ_σ > 0 AND δ_π < 0 — sign rule HOLDS in symmetry-based classification")
    elif r['n_pi'] == 0:
        print("  σ-only system (no π MOs)")
    else:
        print(f"  ✗ Sign rule may not hold: δ_σ = {r['delta_sigma']:+.4f}, δ_π = {r['delta_pi']:+.4f}")
