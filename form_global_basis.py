import os
import re
import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import least_squares


def linear_model(params, X, Y):
    base_theta_pi, curv_x_pi, curv_y_pi, mag_grad_x, mag_grad_y = params
    Bp = 1 / (1 + np.exp(mag_grad_x * X + mag_grad_y * Y))
    theta = (base_theta_pi + curv_x_pi * X + curv_y_pi * Y) * np.pi
    bx = Bp * np.cos(theta)
    by = Bp * np.sin(theta)
    return np.concatenate([bx.flatten(), by.flatten()])


def residuals(params, X, Y, b_true):
    b_current = linear_model(params, X, Y)
    return b_current - b_true


def invert_params(X, Y, b_true):
    curr_guess = [0.0, 0.0, 0.0, 0.0, 0.0]
    res = least_squares(
        residuals,
        curr_guess,
        args=(X, Y, b_true),
        ftol=1e-9,
        xtol=1e-9
    )
    return res


def find_best_directory(samp_dir, X, Y, b_true, verbosity):
    res = invert_params(X, Y, b_true)
    best_r = float("inf")
    dir_name = None

    pattern = re.compile(
        r"mbt(?P<mbt>\-?[\d\.]+)_"
        r"mcx(?P<mcx>\-?[\d\.]+)_"
        r"mcy(?P<mcy>\-?[\d\.]+).*"
        r"mmgx(?P<mmgx>\-?[\d\.]+)_"
        r"mmgy(?P<mmgy>\-?[\d\.]+)"
    )

    subdirs = [
        d for d in os.listdir(samp_dir)
        if os.path.isdir(os.path.join(samp_dir, d))
    ]

    for subdir in subdirs:
        match = pattern.search(subdir)

        if match:
            try:
                params = (
                    float(match.group("mbt")),
                    float(match.group("mcx")),
                    float(match.group("mcy")),
                    float(match.group("mmgx")),
                    float(match.group("mmgy"))
                )

                b_current = linear_model(params, X, Y)
                res_vec = b_current - b_true
                r = np.mean(res_vec**2)

                if r < best_r:
                    dir_name = subdir
                    best_r = r

            except ValueError:
                continue

    if verbosity >= 2:
        base_theta_pi, curv_x_pi, curv_y_pi, mag_grad_x, mag_grad_y = res.x
        print(f"base_theta = {base_theta_pi:8.4f} pi")
        print(f"curv_x = {curv_x_pi:8.4f} pi")
        print(f"curv_y = {curv_y_pi:8.4f} pi")
        print(f"mag_grad_x = {mag_grad_x}")
        print(f"mag_grad_y = {mag_grad_y}")
        print(f"dir_name = {dir_name}")
        print(f"distance = {best_r}")

    return dir_name

def form_global_basis(
    verbosity,
    samp_dir,
    nx,
    ny,
    Nx,
    Ny,
    bx,
    by,
    num_loc_basis,
    POU_basis=None,
    problem="constant_field",
):
    """
    Construct global localized ROM basis functions in the full
    spatial-grid representation.

    Standard problems:
        Dirichlet BC on all boundaries.
        Basis functions are associated with interior coarse nodes.

    magnetic_island:
        Dirichlet BC on x = 0 and x = 1.
        Periodic BC in y.
        Basis functions are associated with
            i = 1, ..., Nx-1
            j = 0, ..., Ny-1.

        The returned basis is ALWAYS represented on the full
        spatial grid of size
            (Ny*ny + 1) * (Nx*nx + 1).

        The periodic DOF reduction is NOT performed here.
        It is done afterward using T.T.
    """

    N_rows_fine = Ny * ny + 1
    N_cols_fine = Nx * nx + 1
    N_fine_dof = N_rows_fine * N_cols_fine

    N_dof_local = (2 * ny + 1) * (2 * nx + 1)

    if problem == "magnetic_island":
        N_global_basis = (Nx - 1) * Ny * num_loc_basis
    else:
        N_global_basis = (Nx - 1) * (Ny - 1) * num_loc_basis

    idx_global_1based = np.arange(
        1,
        N_fine_dof + 1
    ).reshape(
        N_rows_fine,
        N_cols_fine,
        order="F"
    )

    # Local coordinates used by find_best_directory.
    x = np.linspace(-1, 1, 2 * nx)
    y = np.linspace(-1, 1, 2 * ny)
    X, Y = np.meshgrid(x, y)

    I_global = []
    J_global = []
    V_global = []

    Ny_fine = Ny * ny
    count = 0

    for i in range(1, Nx):

        if problem == "magnetic_island":
            j_range = range(0, Ny)
        else:
            j_range = range(1, Ny)

        for j in j_range:

            if problem == "magnetic_island":
                coarse_node_idx = i * Ny + j
            else:
                coarse_node_idx = i * (Ny + 1) + j

            subdomain_verbosity = (
                verbosity
                + int(
                    (i == 1 or i == Nx - 1)
                    and (
                        j == 0
                        if problem == "magnetic_island"
                        else (j == 1 or j == Ny - 1)
                    )
                )
            )

            if subdomain_verbosity >= 2:
                print(f"Loading spectral basis ({i},{j})")

            # x-direction local patch.
            x_start_elem = (i - 1) * nx
            x_end_elem = (i + 1) * nx

            x_idx = np.arange(
                x_start_elem,
                x_end_elem + 1
            )

            if problem == "magnetic_island":

                if j == 0:
                    # Local patch crosses y = 0/y = 1.
                    #
                    # Local ordering corresponds to
                    #
                    #   y = -ny,...,-1,0,...,ny
                    #
                    # which is represented spatially by
                    #
                    #   y = Ny_fine-ny,...,Ny_fine-1,
                    #       0,...,ny.

                    y_idx_top = np.arange(
                        Ny_fine - ny,
                        Ny_fine
                    )

                    y_idx_bottom = np.arange(
                        0,
                        ny + 1
                    )

                    y_idx = np.concatenate([
                        y_idx_top,
                        y_idx_bottom
                    ])

                    loc_idx = idx_global_1based[
                        np.ix_(y_idx, x_idx)
                    ]

                    # Element-based magnetic field.
                    y_field_idx = np.concatenate([
                        np.arange(
                            Ny_fine - ny,
                            Ny_fine
                        ),
                        np.arange(
                            0,
                            ny
                        )
                    ])

                    x_field_idx = np.arange(
                        x_start_elem,
                        x_end_elem
                    )

                    bx_loc = bx[
                        np.ix_(
                            y_field_idx,
                            x_field_idx
                        )
                    ]

                    by_loc = by[
                        np.ix_(
                            y_field_idx,
                            x_field_idx
                        )
                    ]

                else:
                    # Ordinary patch.
                    y_start_elem = (j - 1) * ny
                    y_end_elem = (j + 1) * ny

                    y_idx = np.arange(
                        y_start_elem,
                        y_end_elem + 1
                    )

                    loc_idx = idx_global_1based[
                        np.ix_(y_idx, x_idx)
                    ]

                    bx_loc = bx[
                        y_start_elem:y_end_elem,
                        x_start_elem:x_end_elem
                    ]

                    by_loc = by[
                        y_start_elem:y_end_elem,
                        x_start_elem:x_end_elem
                    ]

            else:
                # Standard Dirichlet case.
                y_start_elem = (j - 1) * ny
                y_end_elem = (j + 1) * ny

                y_idx = np.arange(
                    y_start_elem,
                    y_end_elem + 1
                )

                loc_idx = idx_global_1based[
                    y_start_elem:y_end_elem + 1,
                    x_start_elem:x_end_elem + 1
                ]

                bx_loc = bx[
                    y_start_elem:y_end_elem,
                    x_start_elem:x_end_elem
                ]

                by_loc = by[
                    y_start_elem:y_end_elem,
                    x_start_elem:x_end_elem
                ]

            loc_idx_1based = loc_idx.flatten(order="F")

            if subdomain_verbosity >= 2:
                print(f"X.shape = {X.shape}")
                print(f"Y.shape = {Y.shape}")
                print(f"bx_loc.shape = {bx_loc.shape}")
                print(f"by_loc.shape = {by_loc.shape}")

            b_true = np.concatenate([
                bx_loc.flatten(),
                by_loc.flatten()
            ])

            samp_subdir = find_best_directory(
                samp_dir,
                X,
                Y,
                b_true,
                subdomain_verbosity
            )

            loc_basis = np.load(
                f"{samp_dir}/{samp_subdir}/loc_basis.npy",
                allow_pickle=True
            ).real

            loc_basis = loc_basis[:, :num_loc_basis]

            if loc_basis.shape[0] != N_dof_local:
                raise ValueError(
                    f"Local basis has {loc_basis.shape[0]} rows, "
                    f"expected {N_dof_local}."
                )

            # POU is already represented on the full spatial grid.
            if POU_basis is not None:
                chi_col = POU_basis[
                    :,
                    coarse_node_idx
                ].toarray().ravel()

                loc_chi = chi_col[
                    loc_idx_1based - 1
                ]

                loc_basis *= loc_chi[:, np.newaxis]

            # Assemble directly into the full spatial grid.
            I_element = np.tile(
                loc_idx_1based,
                num_loc_basis
            )

            J_element = np.repeat(
                count + np.arange(
                    1,
                    num_loc_basis + 1
                ),
                N_dof_local
            )

            V_element = loc_basis.flatten(
                order="F"
            )

            I_global.append(I_element)
            J_global.append(J_element)
            V_global.append(V_element)

            count += num_loc_basis

    I_final = np.concatenate(I_global) - 1
    J_final = np.concatenate(J_global) - 1
    V_final = np.concatenate(V_global)

    global_basis = coo_matrix(
        (
            V_final,
            (I_final, J_final)
        ),
        shape=(
            N_fine_dof,
            N_global_basis
        )
    ).tocsr()

    return global_basis
