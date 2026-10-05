import numpy as np
from scipy.sparse import coo_matrix


def bilinear_partition(nx: int, ny: int, Nx: int, Ny: int,
                       problem="constant_field"):
    """
    Construct the bilinear partition-of-unity basis.

    Standard problems:
        Dirichlet boundary conditions on all boundaries.
        There are (Nx+1)*(Ny+1) coarse nodes.

    magnetic_island:
        Periodic boundary conditions in y.
        The coarse nodes at y=0 and y=1 are identified.
        There are (Nx+1)*Ny distinct coarse nodes.
        The fine nodes at y=0 and y=1 are assigned consistently
        to the same periodic coarse basis functions.
    """

    N_rows_fine = Ny * ny + 1
    N_cols_fine = Nx * nx + 1
    N_fine_dof = N_rows_fine * N_cols_fine

    if problem == "magnetic_island":
        N_coarse_dof = (Nx + 1) * Ny
    else:
        N_coarse_dof = (Nx + 1) * (Ny + 1)

    idx_fine = np.arange(
        1, N_fine_dof + 1
    ).reshape(
        N_rows_fine,
        N_cols_fine,
        order="F"
    )

    x_local = np.linspace(0, 1, nx + 1)
    y_local = np.linspace(0, 1, ny + 1)
    loc_X, loc_Y = np.meshgrid(x_local, y_local)

    X_flat_F = loc_X.flatten(order="F")
    Y_flat_F = loc_Y.flatten(order="F")

    local_basis_values_list = [
        (1 - X_flat_F) * (1 - Y_flat_F),  # SW
        (1 - X_flat_F) * Y_flat_F,        # NW
        X_flat_F * (1 - Y_flat_F),        # SE
        X_flat_F * Y_flat_F               # NE
    ]

    I = []
    J = []
    V = []

    if problem == "magnetic_island":
        Ny_fine = Ny * ny

        for i in range(1, Nx + 1):
            for j in range(1, Ny + 1):

                x_start = (i - 1) * nx
                x_end = i * nx

                x_idx = np.arange(
                    x_start,
                    x_end + 1
                )

                # The local coarse element runs from
                # j-1 to j in the periodic y direction.
                y_start = (j - 1) * ny

                y_relative = np.arange(
                    0,
                    ny + 1
                )

                y_idx = (
                    y_start + y_relative
                ) % Ny_fine

                loc_idx = idx_fine[
                    np.ix_(
                        y_idx,
                        x_idx
                    )
                ]

                I_local_flat = loc_idx.flatten(
                    order="F"
                )

                # Periodic coarse-node numbering:
                #
                #   j=1: south=0, north=1
                #   ...
                #   j=Ny-1: south=Ny-2, north=Ny-1
                #   j=Ny: south=Ny-1, north=0
                #
                # Thus y=0 and y=1 share the same coarse node.
                south_j = j - 1
                north_j = j % Ny

                coarse_node_idx_1based = np.array([
                    (i - 1) * Ny + south_j + 1,  # SW
                    (i - 1) * Ny + north_j + 1,  # NW
                    i * Ny + south_j + 1,        # SE
                    i * Ny + north_j + 1         # NE
                ])

                for k in range(4):
                    I.append(I_local_flat)
                    J.append(
                        np.full_like(
                            I_local_flat,
                            coarse_node_idx_1based[k]
                        )
                    )
                    V.append(local_basis_values_list[k])

    else:
        for i in range(1, Nx + 1):
            for j in range(1, Ny + 1):

                row_start = (j - 1) * ny
                row_end = j * ny + 1

                col_start = (i - 1) * nx
                col_end = i * nx + 1

                loc_idx = idx_fine[
                    row_start:row_end,
                    col_start:col_end
                ]

                I_local_flat = loc_idx.flatten(
                    order="F"
                )

                coarse_node_idx_1based = np.array([
                    (i - 1) * (Ny + 1) + j,
                    (i - 1) * (Ny + 1) + j + 1,
                    i * (Ny + 1) + j,
                    i * (Ny + 1) + j + 1
                ])

                for k in range(4):
                    I.append(I_local_flat)
                    J.append(
                        np.full_like(
                            I_local_flat,
                            coarse_node_idx_1based[k]
                        )
                    )
                    V.append(local_basis_values_list[k])

    I_global = np.concatenate(I) - 1
    J_global = np.concatenate(J) - 1
    V_global = np.concatenate(V)

    pairs = np.column_stack((I_global, J_global))
    _, unique_indices = np.unique(
        pairs,
        axis=0,
        return_index=True
    )

    I_global = I_global[unique_indices]
    J_global = J_global[unique_indices]
    V_global = V_global[unique_indices]

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
