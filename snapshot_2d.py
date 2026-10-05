import numpy as np
from scipy.sparse import coo_matrix, eye, spdiags
from scipy.sparse.linalg import spsolve
from typing import Tuple
from form_system_2d import form_system_2d

import numpy as np

def generate_sine_bcs(
    N_bdy_dof: int,
    N_local_problems: int,
    N_modes: int,
) -> np.ndarray:
    """
    Generate deterministic sine-based Dirichlet boundary conditions.

    For each frequency k, offsets that produce exactly the same
    periodic sine function are removed.

    Returns
    -------
    np.ndarray
        Array of shape (N_bdy_dof, N_BCs), where each column is
        one unique boundary condition.
    """

    theta = np.linspace(0.0, 2 * np.pi, N_bdy_dof, endpoint=False)
    offsets = np.linspace(0.0, np.pi, N_local_problems, endpoint=False)

    # Midpoints of the N_local_problems uniform subintervals

    bcs = []

    for k in range(1, N_modes + 1):
        for offset in offsets:
            bc = np.sin(k * theta + offset)
            bcs.append(bc)

    return np.column_stack(bcs)

def generate_random_fourier_bcs(
    N_bdy_dof: int, N_local_problems: int, N_modes: int 
) -> np.ndarray:
    """ 
    Generates 'N_local_problems' sets of smoothed Dirichlet boundary
    conditions using a truncated Fourier series with randomized spatial offsets.
    
    Returns:
        np.ndarray: Matrix of boundary values (N_bdy_dof x N_local_problems)
    """
    
    # Map boundary DOFs to a continuous coordinate theta (0 to 2*pi)
    theta = np.linspace(0, 2 * np.pi, N_bdy_dof, endpoint=False)
        
    f_bdy = np.zeros((N_bdy_dof, N_local_problems))
        
    # Generate random Fourier coefficients for N_local_problems
    N_coeff = 2 * N_modes + 1 
    coeffs = (np.random.rand(N_coeff, N_local_problems) - 0.5) * 2 # Scale to [-1, 1]
    
    # Shape: (1, N_local_problems) so it aligns perfectly during broadcasting
    offsets = np.random.rand(1, N_local_problems) * 2 * np.pi
        
    # a0 (mean value)
    a0 = coeffs[0, :]
    f_bdy += a0
            
    for m in range(1, N_modes + 1): 
        # Broadcasting yields a shape of (N_bdy_dof, N_local_problems)
        shifted_theta = m * (theta[:, None] + offsets)
        
        # am * cos(m * (theta + offset))
        am = coeffs[2 * m - 1, :]
        f_bdy += am * np.cos(shifted_theta)
            
        # bm * sin(m * (theta + offset))
        bm = coeffs[2 * m, :]
        f_bdy += bm * np.sin(shifted_theta)
            
    return f_bdy

def snapshot_2d_dbc(a_loc: np.ndarray, nx: int, ny: int, N_local_problems: int, N_modes: int, add_constant: bool = True) -> np.ndarray:
    # --- Grid Parameters ---
    N_rows_local = 2 * ny + 1
    N_cols_local = 2 * nx + 1
    N_dof_local = N_rows_local * N_cols_local
    N_bdy_dof = 4 * ny + 4 * nx

    hx = 1.0 / nx
    hy = 1.0 / ny

    ratiox = hy / hx 
    ratioy = hx / hy 

    # Handle the input shape: (ny, nx, 2, 2, n_samples)
    # The variable 'a' will now represent the tensor samples
    if a_loc.ndim == 5:
        a = a_loc
    else:
        # Fallback if a scalar field was passed (ny, nx, samples)
        # We transform it into a diagonal tensor [ [val, 0], [0, val] ]
        n_samples = a_loc.shape[2] if a_loc.ndim == 3 else 1
        temp_a = np.zeros((ny, nx, 2, 2, n_samples))
        for s in range(n_samples):
            val = a_loc[:, :, s] if a_loc.ndim == 3 else a_loc
            temp_a[:, :, 0, 0, s] = val
            temp_a[:, :, 1, 1, s] = val
        a = temp_a

    idx_local = np.arange(1, N_dof_local + 1).reshape(N_rows_local, N_cols_local, order='F')

    loc_boundary_1based = np.concatenate([
        idx_local[0, :-1].flatten(),
        idx_local[:-1, -1].flatten(),
        idx_local[-1, 1:][::-1].flatten(),
        idx_local[1:, 0][::-1].flatten()
    ])
    loc_boundary_0based = loc_boundary_1based - 1

    np.random.seed(42)

    f_bdy = generate_sine_bcs(N_bdy_dof, N_local_problems, N_modes)
    #f_bdy = generate_random_fourier_bcs(N_bdy_dof, N_local_problems, N_modes)
    #f_bdy = np.random.rand(N_bdy_dof, N_local_problems)
    N_bdy = f_bdy.shape[1]

    snapshot_matrix = np.zeros((N_dof_local, N_bdy, a.shape[4]))

    # Loop over the samples (last dimension of a)
    for k in range(0, a.shape[4]):
        f = np.zeros((N_dof_local, N_bdy))
        f[loc_boundary_0based, :] = f_bdy

        loc_A, _ = form_system_2d(a[:, :, :, :, k], nx, ny, 2, 2)

        # Impose Dirichlet BCs
        loc_A[loc_boundary_0based, :] = 0
        loc_A[loc_boundary_0based, loc_boundary_0based] = 1
        
        # Solve the system
        loc_U = spsolve(loc_A, f)
        snapshot_matrix[:, :, k] = loc_U
        
    snapshot_matrix = snapshot_matrix.reshape((N_dof_local, N_bdy * a.shape[4]))

    if add_constant:
        constant_vector = np.ones((N_dof_local, 1))
        snapshot_matrix = np.hstack((constant_vector, snapshot_matrix))

    return snapshot_matrix

