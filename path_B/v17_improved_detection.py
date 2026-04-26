"""
v17: Improved mirror detection — handle ring planes and substituents
=====================================================================
The strict z=0 check fails for cyclopropane (CH atoms above/below ring plane).
We need to detect the molecular skeleton plane and project.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from v14_symmetry_projection import compute_mo_reflection_eigenvalue
from sigma_pi_decomposition import build, find_BCP, rho_grad_hess, mo_hessian_contrib_along_axis
from pyscf import scf
import warnings; warnings.filterwarnings('ignore')


def detect_mirror_plane_robust(atoms_str, tolerance=0.10):
    """Detect mirror plane by finding axis where the heavy-atom skeleton is approximately planar.
    For cyclopropane etc., heavy atoms (C/N/O/S) lie in a plane even if H atoms don't."""
    atom_lines = [a.strip() for a in atoms_str.replace('\n', ';').split(';') if a.strip()]
    
    heavy_coords = []
    all_coords = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) >= 4:
            el = parts[0]
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
            all_coords.append((el, xyz))
            if el != 'H':
                heavy_coords.append(xyz)
    
    if not heavy_coords:
        return None, None  # No heavy atoms (e.g. H2)
    
    heavy_coords = np.array(heavy_coords)
    all_coords_arr = np.array([c[1] for c in all_coords])
    
    # First, check strict planarity (all atoms): z=0, y=0, or x=0
    for axis_idx, plane_name in zip([2, 1, 0], ['xy', 'xz', 'yz']):
        if np.max(np.abs(all_coords_arr[:, axis_idx])) < 0.05:
            return plane_name, 'strict'
    
    # Then check heavy-atom planarity (allow H to be off plane)
    for axis_idx, plane_name in zip([2, 1, 0], ['xy', 'xz', 'yz']):
        if np.max(np.abs(heavy_coords[:, axis_idx])) < tolerance:
            # Check if H atoms are symmetric about this plane (so σ_h is still a true symmetry)
            h_atoms = [c[1] for c in all_coords if c[0] == 'H']
            if not h_atoms:
                return plane_name, 'heavy-only'
            
            h_coords = np.array(h_atoms)
            # For σ_h to be a true symmetry, H atoms must come in mirror pairs
            mirrored_h = h_coords.copy()
            mirrored_h[:, axis_idx] = -mirrored_h[:, axis_idx]
            
            # Check if for each H, there exists a matching mirror H
            n_matched = 0
            for h in h_coords:
                for mh in mirrored_h:
                    if np.allclose(h, mh, atol=0.05):
                        n_matched += 1
                        break
            
            if n_matched == len(h_coords):
                return plane_name, 'with-mirror-H'
    
    return None, None


def analyze_with_robust_mirror(name, atoms, A_ang, B_ang, basis='6-31G*', threshold=0.7):
    plane, mode = detect_mirror_plane_robust(atoms)
    if plane is None:
        return {'name': name, 'error': 'No mirror plane (3D non-planar)'}
    
    mol = build(atoms, basis=basis)
    mf = scf.RHF(mol); mf.verbose = 0; mf.kernel()
    dm = mf.make_rdm1()
    
    bcp, t = find_BCP(mol, dm, A_ang, B_ang)
    rho, grad, H = rho_grad_hess(mol, dm, bcp, eps=1e-3)
    evals, evecs = np.linalg.eigh(H)
    lam1, lam2 = evals[0], evals[1]
    delta_bader = lam2 - lam1
    eps_val = lam1/lam2 - 1 if abs(lam2) > 1e-6 else 0
    e_lam1 = evecs[:, 0]; e_lam2 = evecs[:, 1]
    
    refl_eig, _ = compute_mo_reflection_eigenvalue(mol, mf, mirror_plane=plane)
    
    n_occ = (mf.mo_occ > 0.5).sum()
    classes = []
    for i in range(n_occ):
        if refl_eig[i] > threshold:
            classes.append('sigma')
        elif refl_eig[i] < -threshold:
            classes.append('pi')
        else:
            classes.append('mixed')
    
    c_lam1, psi0, dpsi_lam1 = mo_hessian_contrib_along_axis(mol, mf, bcp, e_lam1, eps=8e-3)
    c_lam2, _, dpsi_lam2 = mo_hessian_contrib_along_axis(mol, mf, bcp, e_lam2, eps=8e-3)
    
    delta_sigma = sum((c_lam2[i] - c_lam1[i]) for i in range(n_occ) if classes[i] == 'sigma')
    delta_pi = sum((c_lam2[i] - c_lam1[i]) for i in range(n_occ) if classes[i] == 'pi')
    delta_mixed = sum((c_lam2[i] - c_lam1[i]) for i in range(n_occ) if classes[i] == 'mixed')
    
    return {'name': name, 'plane': plane, 'plane_mode': mode, 'eps': eps_val,
            'lam1': lam1, 'lam2': lam2, 'delta_bader': delta_bader,
            'delta_sigma': delta_sigma, 'delta_pi': delta_pi, 'delta_mixed': delta_mixed,
            'n_sigma': sum(1 for c in classes if c == 'sigma'),
            'n_pi': sum(1 for c in classes if c == 'pi'),
            'n_mixed': sum(1 for c in classes if c == 'mixed'),
            'reflection_eigenvalues': refl_eig.tolist()}


# ── Test on additional molecules where geometry needs correction ──
EXTRA_TESTS = [
    # B2 H2CNH (place in xz plane: H of N off-plane → fix it)
    ("B2 H2CNH planar", "C 0 0 -0.640; N 0 0 0.766; H 0.942 0 -1.254; H -0.942 0 -1.254; H 0.962 0 1.249", [0,0,-0.640],[0,0,0.766]),
    # H1 CH2=CHF: put F in plane
    ("H1 CH2=CHF (F in plane)", "C 0 0 -0.670; C 0 0 0.662; F 1.130 0 1.340; H 0.926 0 -1.237; H -0.926 0 -1.237; H -0.905 0 1.200", [0,0,-0.670],[0,0,0.662]),
    # K1 CH3CHO: planar acyl group, methyl Hs out of plane symmetric
    ("K1 CH3CHO C=O", "C 0 0 0; O 0 0 1.207; C -1.508 0 -0.482; H 0.941 0 -0.554; H -1.958 1.018 -0.200; H -1.958 -1.018 -0.200; H -1.508 0 -1.574", [0,0,0],[0,0,1.207]),
    # AC1 Oxirane: ring is planar (xz), Hs above/below symmetric
    ("AC1 Oxirane C-O ring", "O 0 0 0.520; C 0.730 0 -0.310; C -0.730 0 -0.310; H 1.260 0.910 -0.585; H 1.260 -0.910 -0.585; H -1.260 0.910 -0.585; H -1.260 -0.910 -0.585", [0.730,0,-0.310],[0,0,0.520]),
    # AG1 trans-azomethane (planar N=N + methyl)
    ("AG1 CH3N=NCH3 trans", "N 0 0.625 0; N 0 -0.625 0; C 1.300 1.255 0; C -1.300 -1.255 0; H 1.300 1.880 0.890; H 1.300 1.880 -0.890; H 2.180 0.620 0; H -1.300 -1.880 -0.890; H -1.300 -1.880 0.890; H -2.180 -0.620 0", [0,0.625,0],[0,-0.625,0]),
    # NEW: Cyclopropane ring is in xy plane (3 Cs at z=0, Hs above/below)
    ("Cyclopropane C-C ring", "C 0 0.866 0; C -0.750 -0.433 0; C 0.750 -0.433 0; H 0 1.480 0.890; H 0 1.480 -0.890; H -1.282 -0.740 0.890; H -1.282 -0.740 -0.890; H 1.282 -0.740 0.890; H 1.282 -0.740 -0.890", [0,0.866,0],[-0.750,-0.433,0]),
    # NEW: cyclopropene (planar ring)
    ("Cyclopropene C=C ring", "C 0 0.620 0; C 0 -0.620 0; C 0 0 1.260; H -0.930 1.260 0; H 0.930 -1.260 0; H 0 0 2.350; H 0.930 1.260 0; H -0.930 -1.260 0", [0,0.620,0],[0,-0.620,0]),
]

print("="*125)
print("v17 Improved detection: NEW planar/ring molecules + extras")
print("="*125)
print(f"{'Compound':36} {'plane':6} {'mode':18} {'ε':>8} {'δ_σ':>10} {'δ_π':>10} {'δ_mix':>8} {'pattern':>16}")
print("-"*125)

for label, atoms, A, B in EXTRA_TESTS:
    try:
        r = analyze_with_robust_mirror(label, atoms, A, B)
        if 'error' in r:
            print(f"{label[:36]:36} ERROR: {r['error']}")
            continue
        if r['n_pi'] == 0:
            pattern = "σ-only"
        elif r['delta_pi'] > 0 and r['delta_sigma'] < 0:
            pattern = "🔴 PI-DOM"
        elif r['delta_sigma'] > 0 and r['delta_pi'] < 0:
            pattern = "🔵 SIG-DOM"
        else:
            pattern = "OTHER"
        print(f"{label[:36]:36} {r['plane']:6} {r['plane_mode']:18} {r['eps']:+8.4f} "
              f"{r['delta_sigma']:+10.4f} {r['delta_pi']:+10.4f} {r['delta_mixed']:+8.4f} {pattern:>16}")
    except Exception as e:
        print(f"{label[:36]:36} ERROR: {str(e)[:50]}")
