import numpy as np
from scipy.sparse import coo_matrix

def source_function(x, y, problem):
    if problem == "constant_field":
        return np.sin(np.pi * x)
    elif problem == "single_null" or problem == "double_null":
        xc, yc = 0.5, 0.5
        sigma = 1./8.
        return np.exp(-0.5 * (((x - xc) / sigma)**2 + ((y - yc) / sigma)**2))
    elif problem == "magnetic_island":
        return 0.0 * x
    else:
        return 1.0 + 0.0 * x

def form_source_2d(F: np.ndarray, nx: int, ny: int, Nx: int, Ny: int) -> np.ndarray:
    """
    Assembles the load vector 'f' from the source term 'F'
    using a mass lumping-like approximation where the total element source term
    (F * element_area) is distributed equally to the four element nodes.

    Args:
        F (np.ndarray): The source term values defined at the center of each fine element,
                        size (Ny*ny, Nx*nx).
        nx (int): Number of fine elements per coarse block in x.
        ny (int): Number of fine elements per coarse block in y.
        Nx (int): Number of coarse blocks in x.
        Ny (int): Number of coarse blocks in y.

    Returns:
        np.ndarray: The assembled fine-scale load vector 'f' (as a dense vector).
    """

    # --- Grid Parameters ---
    hx = 1.0 / (Nx * nx)
    hy = 1.0 / (Ny * ny)
    element_area = hx * hy

    # --- DOF Index Generation (1-based, MATLAB's column-major order) ---
    N_rows = Ny * ny + 1
    N_cols = Nx * nx + 1
    N_dof = N_rows * N_cols
    N_elements = Nx * nx * Ny * ny

    # Create 1-based DOF indices, ordered column-major (Fortran order in numpy)
    idx = np.arange(1, N_dof + 1).reshape(N_rows, N_cols, order='F')

    # --- Element Node Indices (1-based) ---
    # These represent the DOF indices for the four nodes of every element.
    idx1 = idx[:-1, :-1]  # Bottom-Left node
    idx2 = idx[:-1, 1:]   # Bottom-Right node
    idx3 = idx[1:, :-1]   # Top-Left node
    idx4 = idx[1:, 1:]    # Top-Right node

    # Flatten the element node indices. This forms the Row indices (I) for the COO matrix.
    # The 4 entries (one for each node) for every element must be mapped.
    # We use a list comprehension and concatenate for the row indices.
    I_list = [arr.flatten() for arr in [idx1, idx2, idx3, idx4]]
    # Combine the row indices and convert to 0-based for SciPy
    I = np.concatenate(I_list) - 1

    # --- Source Vector Assembly (f) ---

    # Local Source Contribution (Value V):
    # The term 'kron([1;1;1;1], F(:) )/4*hx*hy' means that the total source
    # contribution from an element (F_element * hx * hy) is split equally
    # into 4 parts (1/4) and distributed to each of the 4 nodes.
    source_per_element_node = F.flatten() * element_area / 4.0

    # Repeat the source contributions 4 times, corresponding to the 4 nodes
    V = np.concatenate([source_per_element_node] * 4)

    # --- Final Assembly ---
    # In MATLAB, 'ones(4*Nx*nx*Ny*ny,1)' as the column index (J)
    # is only used as a placeholder when assembling a sparse vector (a matrix with one column).
    # In SciPy, we can directly assemble a sparse matrix with shape (N_dof, 1),
    # or more commonly, we assemble the components and sum them into a dense vector.

    # Use coo_matrix to efficiently sum up contributions to the same DOF
    f_coo = coo_matrix((V, (I, np.zeros_like(I))), shape=(N_dof, 1))

    # Convert to a dense column vector
    f = f_coo.toarray().flatten()

    return f
