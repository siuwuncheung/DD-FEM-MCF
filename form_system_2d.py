import numpy as np
from scipy.sparse import coo_matrix

def form_system_2d(a: np.ndarray, nx: int, ny: int, Nx: int, Ny: int):
    # --- Grid Parameters ---
    hx = 1.0 / (Nx * nx)
    hy = 1.0 / (Ny * ny)
    ratiox = hy / hx
    ratioy = hx / hy

    N_rows = Ny * ny + 1
    N_cols = Nx * nx + 1
    N_dof = N_rows * N_cols

    # Order 'F' matches MATLAB's column-major indexing
    idx = np.arange(N_dof).reshape(N_rows, N_cols, order='F')

    # --- Element Node Indices ---
    # Nodes: 0:BL, 1:BR, 2:TL, 3:TR
    node_idx = [
        idx[:-1, :-1].flatten(), # BL
        idx[:-1, 1:].flatten(),  # BR
        idx[1:, :-1].flatten(),  # TL
        idx[1:, 1:].flatten()   # TR
    ]

    # --- Assembly I and J ---
    # We need all 16 combinations (4 rows x 4 cols)
    I = np.concatenate([node_idx[i] for i in range(4) for _ in range(4)])
    J = np.concatenate([node_idx[j] for _ in range(4) for j in range(4)])

    # --- Local Stiffness Matrix Components (4x4) ---
    # Corrected for ordering: [BL, BR, TL, TR]
    Kxx = np.array([
        [ 2, -2,  1, -1],
        [-2,  2, -1,  1],
        [ 1, -1,  2, -2],
        [-1,  1, -2,  2]
    ]) * ratiox / 6.0

    Kyy = np.array([
        [ 2,  1, -2, -1],
        [ 1,  2, -1, -2],
        [-2, -1,  2,  1],
        [-1, -2,  1,  2]
    ]) * ratioy / 6.0
    
    Kxy = np.array([
        [ 1,  1, -1, -1],
        [-1, -1,  1,  1],
        [ 1,  1, -1, -1],
        [-1, -1,  1,  1]
    ]) / 4.0

    Kxx_f = Kxx.flatten()
    Kyy_f = Kyy.flatten()
    Kxy_f = Kxy.flatten()
    Kyx_f = Kxy.T.flatten()

    a11 = a[:, :, 0, 0].flatten()
    a22 = a[:, :, 1, 1].flatten()
    a12 = a[:, :, 0, 1].flatten()

    # --- Stiffness Assembly ---
    V_DA = (np.kron(Kxx_f, a11) + 
            np.kron(Kyy_f, a22) + 
            np.kron(Kxy_f + Kyx_f, a12))

    Global_DA = coo_matrix((V_DA, (I, J)), shape=(N_dof, N_dof)).tocsr()

    # --- Mass Matrix Assembly ---
    M_local = np.array([
        [4, 2, 2, 1],
        [2, 4, 1, 2],
        [2, 1, 4, 2],
        [1, 2, 2, 4]
    ]) * (hx * hy / 36.0)
    
    V_M = np.kron(M_local.flatten(), np.ones(a11.size))
    Global_M = coo_matrix((V_M, (I, J)), shape=(N_dof, N_dof)).tocsr()

    return Global_DA, Global_M
