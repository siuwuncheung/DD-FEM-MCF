import numpy as np
from scipy.sparse import coo_matrix
from GFEM_local_basis import GFEM_local_basis

def harmonic_partition(
    a,
    nx: int,
    ny: int,
    Nx: int,
    Ny: int,
    problem="constant_field",
):
    """
    Construct a global harmonic-extension basis.

    Parameters
    ----------
    a : ndarray
        Global coefficient field with shape

            (Ny * ny, Nx * nx, 2, 2)

        assuming the last two dimensions store the 2x2 tensor.

    nx, ny : int
        Number of fine elements per coarse element.

    Nx, Ny : int
        Number of coarse elements in x and y.

    problem : str
        "constant_field", "linear_model", etc.:
            Nonperiodic basis.

        "magnetic_island":
            Periodic basis in y.

    Returns
    -------
    POU_basis : scipy.sparse.coo_matrix
        Sparse global basis with shape

            (N_fine_dof, N_coarse_dof).
    """

    # --------------------------------------------------------------
    # Basic dimensions
    # --------------------------------------------------------------

    print(f'a.shape = {a.shape}')
    N_rows_fine = Ny * ny + 1
    N_cols_fine = Nx * nx + 1

    N_fine_dof = N_rows_fine * N_cols_fine

    if problem == "magnetic_island":
        N_coarse_dof = (Nx + 1) * Ny
    else:
        N_coarse_dof = (Nx + 1) * (Ny + 1)

    # --------------------------------------------------------------
    # Fine-grid numbering
    # --------------------------------------------------------------

    idx_fine = np.arange(
        1,
        N_fine_dof + 1
    ).reshape(
        N_rows_fine,
        N_cols_fine,
        order="F"
    )

    I = []
    J = []
    V = []

    # --------------------------------------------------------------
    # Loop over coarse nodes
    #
    # Each basis function is defined on a 2 x 2 coarse-element patch.
    # --------------------------------------------------------------

    if problem == "magnetic_island":

        # Periodic fine-grid size in y
        Ny_fine = Ny * ny

        for i in range(Nx + 1):

            for j in range(Ny):

                # --------------------------------------------------
                # Periodic coarse-node numbering
                # --------------------------------------------------

                coarse_idx = i * Ny + j

                # --------------------------------------------------
                # Patch starts at coarse node (i-1, j-1)
                # --------------------------------------------------

                x_start = (i - 1) * nx
                x_end = (i + 1) * nx

                y_start = (j - 1) * ny

                # --------------------------------------------------
                # Fine-grid indices
                # --------------------------------------------------

                x_idx = np.arange(
                    x_start,
                    x_end + 1
                )

                y_idx = (
                    y_start
                    + np.arange(2 * ny + 1)
                ) % Ny_fine

                # --------------------------------------------------
                # Extract coefficient patch
                # --------------------------------------------------

                a_patch = a[
                    y_idx[:-1],
                    x_idx[:-1],
                    :, :
                ]
                print(f'a_patch.shape = {a_patch.shape}')

                # --------------------------------------------------
                # Solve harmonic extension
                # --------------------------------------------------

                basis_local = GFEM_local_basis(
                    a_patch,
                    nx,
                    ny
                )

                # --------------------------------------------------
                # Fine-node numbering
                # --------------------------------------------------

                loc_idx = idx_fine[
                    np.ix_(
                        y_idx,
                        x_idx
                    )
                ]

                I_local = loc_idx.flatten(
                    order="F"
                ) - 1

                J_local = np.full(
                    I_local.shape,
                    coarse_idx
                )

                # --------------------------------------------------
                # Store sparse entries
                # --------------------------------------------------

                I.append(I_local)
                J.append(J_local)
                V.append(basis_local)

    else:

        for i in range(Nx + 1):

            for j in range(Ny + 1):

                # --------------------------------------------------
                # Coarse-node numbering
                # --------------------------------------------------

                coarse_idx = i * (Ny + 1) + j

                # --------------------------------------------------
                # Patch boundaries
                # --------------------------------------------------

                x_start = (i - 1) * nx
                x_end = (i + 1) * nx

                y_start = (j - 1) * ny
                y_end = (j + 1) * ny

                # --------------------------------------------------
                # Skip boundary coarse nodes for now
                # --------------------------------------------------

                if (
                    i == 0
                    or i == Nx
                    or j == 0
                    or j == Ny
                ):
                    continue

                # --------------------------------------------------
                # Fine-grid indices
                # --------------------------------------------------

                x_idx = np.arange(
                    x_start, x_end + 1
                )

                y_idx = np.arange(
                    y_start, y_end + 1
                )

                # --------------------------------------------------
                # Extract coefficient patch
                # --------------------------------------------------

                a_patch = a[
                    y_idx,
                    x_idx,
                    :, :
                ]
                print(f'a_patch.shape = {a_patch.shape}')

                # --------------------------------------------------
                # Solve harmonic extension
                # --------------------------------------------------

                basis_local = GFEM_local_basis(
                    a_patch,
                    nx,
                    ny
                )

                # --------------------------------------------------
                # Fine-node numbering
                # --------------------------------------------------

                loc_idx = idx_fine[
                    np.ix_(
                        y_idx,
                        x_idx
                    )
                ]

                I_local = loc_idx.flatten(
                    order="F"
                ) - 1

                J_local = np.full(
                    I_local.shape,
                    coarse_idx
                )

                # --------------------------------------------------
                # Store sparse entries
                # --------------------------------------------------

                I.append(I_local)
                J.append(J_local)
                V.append(basis_local)

    # --------------------------------------------------------------
    # Assemble sparse matrix
    # --------------------------------------------------------------

    I_global = np.concatenate(I)
    J_global = np.concatenate(J)
    V_global = np.concatenate(V)

    POU_basis = coo_matrix(
        (
            V_global,
            (I_global, J_global)
        ),
        shape=(
            N_fine_dof,
            N_coarse_dof
        )
    )

    return POU_basis
