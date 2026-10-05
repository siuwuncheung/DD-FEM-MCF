import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.linalg import eigh # Used for dense generalized eigenproblem
from scipy.sparse.linalg import eigs # Can be used for sparse eigenproblem
import time
from typing import Tuple

def scalar_eigenproblem_2d(a_loc: np.ndarray, nx: int, ny: int, Nx: int, Ny: int, 
                           num_basis: int, snapshot_matrix: np.ndarray) -> csr_matrix:
    # --- Grid Parameters ---
    hx = 1.0 / (Nx * nx)
    hy = 1.0 / (Ny * ny)
    ratiox = hy / hx  # Equivalent to (Ny*ny / Nx*nx)
    ratioy = hx / hy  # Equivalent to (Nx*nx / Ny*ny)

    N_rows_local = 2 * ny + 1
    N_cols_local = 2 * nx + 1
    N_dof_local = N_rows_local * N_cols_local
    
    N_fine_dof = (Nx * nx + 1) * (Ny * ny + 1)
    
    # Dimensions for final sparse assembly
    N_global_basis = (Nx - 1) * (Ny - 1) * num_basis

    # --- Pre-calculate Local Matrix Assembly Indices (I and J) ---
    # The local region is 2ny x 2nx fine elements.
    idx_local_1based = np.arange(1, N_dof_local + 1).reshape(N_rows_local, N_cols_local, order='F')
    idx_elem_1based = idx_local_1based[:-1, :-1]
    
    idx1 = idx_elem_1based
    idx2 = idx1 + N_rows_local
    idx3 = idx1 + 1
    idx4 = idx1 + 1 + N_rows_local
    
    I_list = [arr.flatten() for arr in [idx1, idx2, idx3, idx4]]
    I_loc_0based = np.concatenate(I_list * 4) - 1

    J_list = [arr.flatten() for arr in [idx1, idx2, idx3, idx4]]
    J_loc_0based = np.concatenate([
        J_list[0], J_list[0], J_list[0], J_list[0], 
        J_list[1], J_list[1], J_list[1], J_list[1], 
        J_list[2], J_list[2], J_list[2], J_list[2], 
        J_list[3], J_list[3], J_list[3], J_list[3], 
    ]) - 1

    # Local Stiffness components (K_x and K_y)
    Kx_local_flat = np.array([
        2, -2, 1, -1, -2, 2, -1, 1, 1, -1, 2, -2, -1, 1, -2, 2
    ]) * ratiox
    Ky_local_flat = np.array([
        2, 1, -2, -1, 1, 2, -1, -2, -2, -1, 2, 1, -1, -2, 1, 2
    ]) * ratioy
    K_local_flat = Kx_local_flat + Ky_local_flat

    # Local Mass components (M)
    M_local_flat = np.array([
        1/9, 1/18, 1/18, 1/36, 1/18, 1/9, 1/36, 1/18, 1/18, 1/36, 1/9, 1/18, 1/36, 1/18, 1/18, 1/9
    ])
    
    a_loc_flat = a_loc.flatten(order='F')
            
    V_DA = np.kron(K_local_flat, a_loc_flat) / 6
    loc_A = coo_matrix((V_DA, (I_loc_0based, J_loc_0based)), shape=(N_dof_local, N_dof_local)).tocsr()
            
    V_M_a = np.kron(M_local_flat, a_loc_flat) * hx * hy
    loc_M_a = coo_matrix((V_M_a, (I_loc_0based, J_loc_0based)), shape=(N_dof_local, N_dof_local)).tocsr()
            
    if snapshot_matrix is not None:
        loc_A = snapshot_matrix.T @ loc_A @ snapshot_matrix
        loc_M_a = snapshot_matrix.T @ loc_M_a @ snapshot_matrix

    loc_A = (loc_A + loc_A.T) / 2.0
    loc_M_a = (loc_M_a + loc_M_a.T) / 2.0

    loc_A = loc_A.real
    loc_M_a = loc_M_a.real

    eigvals, v = eigs(loc_A, k=num_basis, M=loc_M_a, which='SM') 
    sort_idx = np.argsort(np.abs(eigvals))
    eigvals = eigvals[sort_idx[0:num_basis]]
    print(f'eigvals = {eigvals}')
    loc_basis = v[:, sort_idx[0:num_basis]]
    if snapshot_matrix is not None:
        loc_basis = snapshot_matrix @ loc_basis
    
    return loc_basis
