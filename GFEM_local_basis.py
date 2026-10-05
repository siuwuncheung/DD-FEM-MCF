import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse import coo_matrix, csr_matrix
from form_system_2d import form_system_2d

def GFEM_local_basis(
    a_loc: np.ndarray,
    nx: int,
    ny: int,
    Lx: float = 1.0,
    Ly: float = 1.0,
) -> np.ndarray:
    # ------------------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------------------

    expected_shape = (2 * ny, 2 * nx, 2, 2)

    if a_loc.shape != expected_shape:
        raise ValueError(
            f"a_loc must have shape {expected_shape}, "
            f"but received {a_loc.shape}."
        )

    hx = Lx / nx
    hy = Ly / ny

    # ------------------------------------------------------------------
    # Helper: solve one quadrant
    #
    # quadrant:
    #
    #   "TR" = top-right
    #   "TL" = top-left
    #   "BL" = bottom-left
    #   "BR" = bottom-right
    #
    # The returned array has shape (ny+1, nx+1).
    # ------------------------------------------------------------------

    def solve_quadrant(a_quad, quadrant):

        A, _ = form_system_2d(a_quad, nx, ny, 1, 1)

        n_rows = ny + 1
        n_cols = nx + 1
        n_dof = n_rows * n_cols

        # --------------------------------------------------------------
        # Coordinates of local quadrant nodes.
        #
        # We use local coordinates s,t in [0,1] x [0,1], measured
        # outward from the central corner.
        # --------------------------------------------------------------

        s = np.linspace(0.0, 1.0, nx + 1)
        t = np.linspace(0.0, 1.0, ny + 1)

        if quadrant == "TR": 
            boundary_values = np.outer(1.0 - t, 1.0 - s)
        elif quadrant == "TL": 
            boundary_values = np.outer(1.0 - t, s)
        elif quadrant == "BL": 
            boundary_values = np.outer(t, s)
        elif quadrant == "BR": 
            boundary_values = np.outer(t, 1.0 - s)
        else:
            raise ValueError(f"'{quadrant}' is not a valid input for quadrant.")

        # --------------------------------------------------------------
        # Identify boundary and interior DOFs.
        #
        # The array is indexed as [row, col], but the FE vector uses
        # Fortran ordering.
        # --------------------------------------------------------------

        boundary_mask = np.zeros((n_rows, n_cols), dtype=bool)

        boundary_mask[0, :] = True
        boundary_mask[-1, :] = True
        boundary_mask[:, 0] = True
        boundary_mask[:, -1] = True

        interior_mask = ~boundary_mask

        boundary_dofs = np.flatnonzero(
            boundary_mask.flatten(order="F")
        )

        interior_dofs = np.flatnonzero(
            interior_mask.flatten(order="F")
        )

        g = boundary_values.flatten(order="F")

        # --------------------------------------------------------------
        # Dirichlet elimination:
        #
        # A_II u_I + A_IB u_B = 0
        #
        # so
        #
        # A_II u_I = -A_IB g_B.
        # --------------------------------------------------------------

        A_II = A[interior_dofs][:, interior_dofs]
        A_IB = A[interior_dofs][:, boundary_dofs]

        rhs = -A_IB @ g[boundary_dofs]

        u = np.zeros(n_dof)

        u[boundary_dofs] = g[boundary_dofs]

        if len(interior_dofs) > 0:
            u[interior_dofs] = spla.spsolve(A_II, rhs)

        # Convert back to nodal array.
        u = u.reshape((n_rows, n_cols), order="F")

        return u

    # ------------------------------------------------------------------
    # Extract the four coefficient fields.
    #
    # Original element layout:
    #
    #          y
    #          ^
    #
    #       TL | TR
    #       ---+---
    #       BL | BR
    #
    # Each quadrant contains ny x nx elements.
    # ------------------------------------------------------------------

    a_BL = a_loc[:ny, :nx, :, :]
    a_BR = a_loc[:ny, nx:, :, :]
    a_TL = a_loc[ny:, :nx, :, :]
    a_TR = a_loc[ny:, nx:, :, :]

    # ------------------------------------------------------------------
    # Solve the four independent Dirichlet problems.
    # ------------------------------------------------------------------

    u_TR = solve_quadrant(a_TR, "TR")
    u_TL = solve_quadrant(a_TL, "TL")
    u_BL = solve_quadrant(a_BL, "BL")
    u_BR = solve_quadrant(a_BR, "BR")

    # ------------------------------------------------------------------
    # Glue the four solutions.
    #
    # Global array:
    #
    #       TL | TR
    #       ---+---
    #       BL | BR
    #
    # Each quadrant has nx+1 by ny+1 nodes, so the central row/column
    # are shared.
    # ------------------------------------------------------------------

    basis = np.zeros(
        (2 * ny + 1, 2 * nx + 1),
        dtype=float
    )

    # Bottom-left
    basis[:ny + 1, :nx + 1] = u_BL

    # Bottom-right
    basis[:ny + 1, nx:] = u_BR

    # Top-left
    basis[ny:, :nx + 1] = u_TL

    # Top-right
    basis[ny:, nx:] = u_TR

    return basis.flatten(order="F") 
