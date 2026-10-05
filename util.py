import argparse
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description="Problem Setting and ROM Parameters")

    parser.add_argument("--verbosity", type=int, default=1, help="Verbosity (Silent/Default/Debug)")
    parser.add_argument("--root_dir", type=str, default="local_basis", help="Root directory of sampling")

    # Problem setting
    parser.add_argument("--medium_type", type=str, default="magnetic_island", 
                        choices=["constant_field", "linear_model", "single_null", "double_null", "magnetic_island"])
    parser.add_argument("--test_anisotropy_order", type=int, default=6)

    # Fine mesh
    parser.add_argument("--n_FOM_x", type=int, default=256)
    parser.add_argument("--n_FOM_y", type=int, help="Defaults to n_FOM_x")

    # Coarse mesh
    parser.add_argument("--Nx", type=int, default=16)
    parser.add_argument("--Ny", type=int, help="Defaults to Nx")

    # FOM
    parser.add_argument("--solve_FOM", type=int, default=1, help="-1: skip, 0: load, 1: solve")
    parser.add_argument("--plot_solution_FOM", type=int, default=1)

    # ROM
    parser.add_argument("--solve_ROM", type=int, default=1, help="0: load, 1: solve")
    parser.add_argument("--plot_solution_rom", type=int, default=1)
    parser.add_argument("--num_loc_basis", type=int, default=5)

    # Sampling
    parser.add_argument("--num_realizations", type=int, default=1)
    parser.add_argument("--num_loc_bc_mode", type=int, default=10)
    parser.add_argument("--num_loc_snap", type=int, default=2)
    parser.add_argument("--train_anisotropy_order", type=int, default=6)

    parser.add_argument("--range_base_theta_pi", type=float, default=0.0, help="Multiple of pi")
    parser.add_argument("--range_curv_x_pi", type=float, default=0.0, help="Multiple of pi")
    parser.add_argument("--range_curv_y_pi", type=float, default=0.0, help="Multiple of pi")
    parser.add_argument("--range_mag_grad_x", type=float, default=0.0)
    parser.add_argument("--range_mag_grad_y", type=float, default=0.0)

    # Online
    parser.add_argument("--multiply_POU", type=int, default=1, help="0: no POU multiplication, 1: bilinear POU, 2: multiscale POU")
    parser.add_argument("--corr_patch_size", type=int, default=0, help="Source correction patch size")

    args = parser.parse_args()

    if args.n_FOM_y is None:
        args.n_FOM_y = args.n_FOM_x
    if args.Ny is None:
        args.Ny = args.Nx

    return args

def reconstruct_field(
    values,
    bfc,
    medium_type,
    dof_map=None,
    ny=None,
    Ny=None,
    nx=None,
    Nx=None,
):
    """
    Reconstruct a full spatial-grid field from BC-reduced DOFs.

    For magnetic_island:
        dof_map maps each full spatial-grid node to its reduced DOF.
        Entries with dof_map < 0 are prescribed/boundary nodes.

    For standard Dirichlet problems:
        values contains only the interior nodes.
    """
    field = bfc.copy()

    if medium_type == "magnetic_island":
        if dof_map is None:
            raise ValueError("dof_map is required for magnetic_island.")

        mask = dof_map >= 0
        field[mask] = values[dof_map[mask]]

    else:
        if ny is None or Ny is None or nx is None or Nx is None:
            raise ValueError(
                "nx, ny, Nx, Ny are required for standard Dirichlet problems."
            )

        field[1:-1, 1:-1] = values.reshape(
            ny * Ny - 1,
            nx * Nx - 1,
            order="F",
        )

    return field


def plot_scalar_field(
    field,
    filename,
    title=None,
    extent=(0, 1, 0, 1),
    cmap="jet",
    figsize=None,
    transparent=True,
):
    """Plot a 2D scalar field."""
    plt.figure(figsize=figsize)

    plt.imshow(
        field,
        origin="lower",
        extent=extent,
        cmap=cmap,
    )

    plt.colorbar()

    if title is not None:
        plt.title(title)

    plt.savefig(
        filename,
        bbox_inches="tight",
        pad_inches=0,
        transparent=transparent,
    )
    plt.close()


def plot_scalar_field_with_contour(
    field,
    filename,
    X,
    Y,
    contour_data=None,
    contour_level=None,
    contour_levels=20,
    title=None,
    extent=(0, 1, 0, 1),
    cmap="jet",
    transparent=True,
):
    """Plot a scalar field with optional contour overlay."""
    plt.figure()

    if contour_data is not None:
        if contour_level is not None:
            plt.contour(
                X,
                Y,
                contour_data,
                levels=[contour_level],
                colors="w",
            )
        else:
            plt.contour(
                X,
                Y,
                contour_data,
                levels=contour_levels,
                colors="w",
            )

    plt.imshow(
        field,
        origin="lower",
        extent=extent,
        cmap=cmap,
    )

    plt.colorbar()

    if title is not None:
        plt.title(title)

    plt.savefig(
        filename,
        bbox_inches="tight",
        pad_inches=0,
        transparent=transparent,
    )
    plt.close()


def plot_surface(
    field,
    X,
    Y,
    filename,
    title=None,
    cmap="viridis",
    figsize=(12, 8),
    elev=30,
    azim=225,
):
    """Plot a scalar field as a 3D surface."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X,
        Y,
        field,
        cmap=cmap,
        edgecolor="none",
        antialiased=True,
    )

    if title is not None:
        ax.set_title(title, fontsize=15)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    fig.colorbar(
        surf,
        shrink=0.5,
        aspect=10,
    )

    ax.view_init(
        elev=elev,
        azim=azim,
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_tensor_components(
    a,
    filename,
    extent=(0, 1, 0, 1),
    cmap="jet",
    figsize=(10, 8),
):
    """Plot the four components of a 2x2 tensor field."""
    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize,
    )

    for i in range(2):
        for j in range(2):
            ax = axes[i, j]

            d_min = np.min(a[:, :, i, j])
            d_max = np.max(a[:, :, i, j])

            center = 0.5 * (d_max + d_min)
            min_range = 1e-6

            if d_max - d_min < min_range:
                v_min = center - 0.5 * min_range
                v_max = center + 0.5 * min_range
            else:
                v_min = d_min
                v_max = d_max

            im = ax.imshow(
                a[:, :, i, j],
                origin="lower",
                extent=extent,
                vmin=v_min,
                vmax=v_max,
                cmap=cmap,
            )

            fig.colorbar(
                im,
                ax=ax,
            )

    plt.tight_layout()
    plt.savefig(
        filename,
        bbox_inches="tight",
    )
    plt.close()


def plot_global_basis(
    glob_basis,
    basis_idx,
    subdomain_idx,
    num_loc_basis,
    filename,
    extent=(0, 1, 0, 1),
    cmap="jet",
):
    """
    Plot one global basis function.

    glob_basis is assumed to live on the full spatial grid:
        ((ny*Ny+1)*(nx*Nx+1), N_basis)
    """
    column = basis_idx + subdomain_idx * num_loc_basis

    basis = glob_basis[:, column].toarray().ravel()

    N_dof_spatial = basis.size
    N_rows = int(np.sqrt(N_dof_spatial))

    if N_rows * N_rows != N_dof_spatial:
        raise ValueError(
            "Cannot infer spatial-grid dimensions from basis size. "
            "Pass a basis with a known rectangular grid."
        )

    plot_basis = basis.reshape(
        N_rows,
        N_rows,
        order="F",
    )

    plot_scalar_field(
        plot_basis,
        filename,
        title=None,
        extent=extent,
        cmap=cmap,
    )


def plot_global_basis_grid(
    glob_basis,
    basis_idx,
    subdomain_idx,
    num_loc_basis,
    n_rows,
    n_cols,
    filename,
    extent=(0, 1, 0, 1),
    cmap="jet",
):
    """
    Plot one global basis when the spatial-grid dimensions are known.
    """
    column = basis_idx + subdomain_idx * num_loc_basis

    basis = glob_basis[:, column].toarray().ravel()

    plot_basis = basis.reshape(
        n_rows,
        n_cols,
        order="F",
    )

    plot_scalar_field(
        plot_basis,
        filename,
        title=None,
        extent=extent,
        cmap=cmap,
    )


def plot_patch_solution(
    field,
    filename,
    i_center,
    j_center,
    patch_size,
    Nx,
    Ny,
    title=None,
    cmap="jet",
):
    """Plot a solution together with a rectangular coarse-grid patch."""
    plt.figure()

    plt.imshow(
        field,
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
    )

    ax = plt.gca()

    i_min = max(0, i_center - patch_size)
    i_max = min(Nx - 1, i_center + patch_size - 1)
    j_min = max(0, j_center - patch_size)
    j_max = min(Ny - 1, j_center + patch_size - 1)
 
    xL = i_min / Nx
    xR = (i_max + 1) / Nx
    yB = j_min / Ny
    yT = (j_max + 1) / Ny

    ax.plot(
        [xL, xR, xR, xL, xL],
        [yB, yB, yT, yT, yB],
        color="white",
        linewidth=2,
    )

    if title is not None:
        plt.title(title)

    plt.colorbar()

    plt.savefig(
        filename,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close()

def smallest_eigenvalue(A):
    A = 0.5 * (A + A.T)
    return eigsh(
        A,
        k=1,
        which="SA",
        return_eigenvectors=False
    )[0]
