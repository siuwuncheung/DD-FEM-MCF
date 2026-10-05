import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def boundary_function(x, y, problem):
    if problem == "constant_field":
        return 0.0 * x
    elif problem == "single_null" or problem == "double_null":
        return 0.0 * x
    elif problem == "magnetic_island":
        conditions = [
            (x > 0.75) & (x <= 1.0),
            (x >= 0.25) & (x <= 0.75),
            (x >= 0.0) & (x < 0.25)
        ]
        choices = [
            lambda x: 2 * (1 - x),
            0.5,
            lambda x: 1 - 2 * x
        ]
        return np.piecewise(x, conditions, choices)
    else:
        return 0.0 * x


def apply_bc_2d(
    problem,
    Global_DA,
    Global_M,
    Global_f,
    nx,
    ny,
    Nx,
    Ny,
    check=True,
):
    # Grid parameters
    hx = 1.0 / (Nx * nx)
    hy = 1.0 / (Ny * ny)

    N_rows = Ny * ny + 1
    N_cols = Nx * nx + 1
    N_dof = N_rows * N_cols

    # MATLAB-compatible column-major indexing
    idx = np.arange(N_dof).reshape(
        N_rows, N_cols, order="F"
    )

    # Coordinates
    X_nodes = np.linspace(0.0, 1.0, N_cols)
    Y_nodes = np.linspace(0.0, 1.0, N_rows)

    X_nodes_mesh, Y_nodes_mesh = np.meshgrid(
        X_nodes, Y_nodes
    )

    # Construct Dirichlet boundary values
    bfc = np.zeros((N_rows, N_cols))

    if problem == "magnetic_island":
        # Dirichlet only on x = 0 and x = 1.
        bfc[:, 0] = boundary_function(
            X_nodes_mesh[:, 0],
            Y_nodes_mesh[:, 0],
            problem
        )
        bfc[:, -1] = boundary_function(
            X_nodes_mesh[:, -1],
            Y_nodes_mesh[:, -1],
            problem
        )
    else:
        # Dirichlet on all four sides.
        bfc = boundary_function(
            X_nodes_mesh,
            Y_nodes_mesh,
            problem
        )
        bfc[1:-1, 1:-1] = 0.0

    # Dirichlet mask
    dirichlet = np.zeros(
        (N_rows, N_cols),
        dtype=bool
    )

    if problem == "magnetic_island":
        dirichlet[:, 0] = True
        dirichlet[:, -1] = True
    else:
        dirichlet[0, :] = True
        dirichlet[-1, :] = True
        dirichlet[:, 0] = True
        dirichlet[:, -1] = True

    # Build reduced DOF map
    dof_map = -np.ones(
        (N_rows, N_cols),
        dtype=int
    )

    if problem == "magnetic_island":
        # Independent nodes:
        #   x = hx,...,1-hx
        #   y = 0,...,1-hy
        #
        # The row y = 1 is the periodic copy of y = 0.

        N_y_periodic = N_rows - 1
        N_x_free = N_cols - 2

        N_reduced = N_y_periodic * N_x_free

        for j in range(1, N_cols - 1):
            for i in range(N_rows - 1):
                dof_map[i, j] = (
                    (j - 1) * N_y_periodic + i
                )

        # Identify y = 1 with y = 0.
        for j in range(1, N_cols - 1):
            dof_map[-1, j] = dof_map[0, j]

    else:
        # Ordinary all-Dirichlet problem.
        is_DOF_grid = ~dirichlet
        N_reduced = np.count_nonzero(is_DOF_grid)

        count = 0
        for i in range(N_rows):
            for j in range(N_cols):
                if is_DOF_grid[i, j]:
                    dof_map[i, j] = count
                    count += 1

    # Flatten map using the same ordering as the global matrices.
    dof_map_flat = dof_map.flatten(order="F")

    # Construct transformation matrix
    rows = []
    cols = []
    data = []

    for full_dof in range(N_dof):
        reduced_dof = dof_map_flat[full_dof]

        if reduced_dof >= 0:
            rows.append(full_dof)
            cols.append(reduced_dof)
            data.append(1.0)

    T = csr_matrix(
        (data, (rows, cols)),
        shape=(N_dof, N_reduced)
    )

    bfc_flat = bfc.flatten(order="F")

    if problem == "magnetic_island":
        # Periodic reduced system
        FOM_DA = T.T @ Global_DA @ T
        FOM_M = T.T @ Global_M @ T
        FOM_RHS = T.T @ (
            Global_f - Global_DA @ bfc_flat
        )

        # Symmetrize
        FOM_DA = 0.5 * (FOM_DA + FOM_DA.T)
        FOM_M = 0.5 * (FOM_M + FOM_M.T)

        # One entry for each independent reduced DOF.
        DOF_idx_FOM = np.arange(N_reduced)
        is_DOF = dof_map_flat >= 0
    else:
        # Standard all-Dirichlet system
        is_DOF = ~dirichlet.flatten(order="F")
        DOF_idx_FOM = np.flatnonzero(is_DOF)

        FOM_DA = Global_DA[
            np.ix_(DOF_idx_FOM, DOF_idx_FOM)
        ]

        FOM_M = Global_M[
            np.ix_(DOF_idx_FOM, DOF_idx_FOM)
        ]

        FOM_DA = 0.5 * (FOM_DA + FOM_DA.T)
        FOM_M = 0.5 * (FOM_M + FOM_M.T)

        FOM_RHS = (
            Global_f[is_DOF]
            - Global_DA[is_DOF, :] @ bfc_flat
        )

    return (
        is_DOF,
        DOF_idx_FOM,
        FOM_DA,
        FOM_M,
        FOM_RHS,
        bfc,
        dof_map,
        T
    )
