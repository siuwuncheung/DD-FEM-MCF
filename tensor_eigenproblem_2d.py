import numpy as np
import scipy.linalg as la
from scipy.sparse import coo_matrix, csr_matrix
from form_system_2d import form_system_2d

def tensor_eigenproblem_2d(a_loc: np.ndarray, nx: int, ny: int, 
                           snapshot_matrix: np.ndarray, multiply_GFEM_local_basis: bool = False) -> np.ndarray:

    loc_A, loc_M = form_system_2d(a_loc, nx, ny, 2, 2)

    # --- Projection and Eigenproblem ---
    if snapshot_matrix is not None:
        #loc_A_red = snapshot_matrix.T @ loc_A @ snapshot_matrix
        #loc_M_red = snapshot_matrix.T @ loc_M @ snapshot_matrix
        U, S, _ = np.linalg.svd(snapshot_matrix, full_matrices=False)
        phi = U[:, S > 1e-10] 
        loc_A_red = phi.T @ loc_A @ phi
        loc_M_red = phi.T @ loc_M @ phi
    else:
        loc_A_red = loc_A.toarray()
        loc_M_red = loc_M.toarray()

    # Force symmetry
    loc_A_red = (loc_A_red + loc_A_red.T) / 2.0
    loc_M_red = (loc_M_red + loc_M_red.T) / 2.0
    #loc_M_red += 1.e-8 * np.eye(loc_M_red.shape[0])
    #print("Smallest eigenvalue of A:", np.min(np.linalg.eigvalsh(loc_A_red)))
    #print("Smallest eigenvalue of M:", np.min(np.linalg.eigvalsh(loc_M_red)))

    # Solve Generalized Eigenproblem
    # eigh is much safer for SPD matrices than eig
    eigvals, v = la.eigh(loc_A_red, loc_M_red)

    sort_indices = np.argsort(np.abs(eigvals))
    eigvals = eigvals[sort_indices]
    print(f'eigvals = {eigvals}')

    # Back-project the basis to the original space
    if snapshot_matrix is not None:
        loc_basis = phi @ v[:, sort_indices]

    if multiply_GFEM_local_basis:
        from GFEM_local_basis import GFEM_local_basis
        GFEM_basis = GFEM_local_basis(a_loc, nx, ny)
        loc_basis = GFEM_basis[:, None] * loc_basis
    
    return loc_basis.real
