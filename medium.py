import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def compute_anisotropic_tensor(
    Ny: int, 
    Nx: int, 
    anisotropy_ratio: float, 
    problem: str,
    save_dir: str = "."
) -> np.ndarray:
    """
    Computes the 2D anisotropic conductivity tensor K for specific prototype 
    problems from the Vogl et al. (2023) paper.
    
    Returns:
        np.ndarray: Shape (Ny, Nx, 2, 2) where K[i, j] is the 2x2 tensor.
    """
    kappa_par = 1.0
    kappa_perp = kappa_par / anisotropy_ratio
    diff = kappa_par - kappa_perp
    x0, y0 = 0.5, 0.5
    
    x_coords = np.linspace(1/Nx/2, 1 - 1/Nx/2, Nx)
    y_coords = np.linspace(1/Ny/2, 1 - 1/Ny/2, Ny)
    if problem == "constant_field":
        x_coords = x_coords * np.pi
        
    X, Y = np.meshgrid(x_coords, y_coords)
    
    if problem == "constant_field":
        Az = Y
    elif problem == "single_null":
        d1_sq = (X - 0.5)**2 + (Y - 0.75)**2
        d2_sq = (X - 0.5)**2 + (Y + 0.25)**2
        Az = 0.5 * np.log(d1_sq * d2_sq + 1e-9)
    elif problem == "double_null":
        Az = 0.5*(X - x0)**2 + 0.5*(0.25 * np.sin(2*np.pi*(Y - y0)))**2
    elif problem == "magnetic_island":
        L = 1.0
        envelope = (1.0 - ((X - x0)/(L/2))**2)
        Az = 0.5*(X - x0)**2 + 0.5*(envelope * 0.25 * np.sin(2*np.pi*(Y - y0)))**2
    else:
        raise ValueError(f"Unknown problem type: {problem}")

    dAz_dy, dAz_dx = np.gradient(Az, y_coords, x_coords)
    X_idx, Y_idx = np.unravel_index(np.argmin(np.abs(dAz_dx) + np.abs(dAz_dy)), Az.shape)
    psi_sep = Az[Y_idx, X_idx]

    Bpx, Bpy = dAz_dy, -dAz_dx
    
    B_mag = np.sqrt(Bpx**2 + Bpy**2 + 1.0)
    bx, by = Bpx/B_mag, Bpy/B_mag
    
    K = np.zeros((Ny, Nx, 2, 2))
    
    K[:, :, 0, 0] = kappa_perp + diff * bx**2
    K[:, :, 1, 1] = kappa_perp + diff * by**2
    K[:, :, 0, 1] = diff * bx * by
    K[:, :, 1, 0] = K[:, :, 0, 1] 
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Benchmark problem: {problem.replace('_', ' ').title()}")

    def clip_vectors(vx, vy, percentile=95):
        mag = np.sqrt(vx**2 + vy**2)
        thresh = np.percentile(mag, percentile)
        # Avoid division by zero
        scale = np.ones_like(mag)
        mask = mag > thresh
        scale[mask] = thresh / mag[mask]
        return vx * scale, vy * scale

    def get_vrange(data):
        dmin, dmax = np.min(data), np.max(data)
        if np.isclose(dmin, dmax, atol=1e-12):
            return dmin - 0.1, dmax + 0.1
        return dmin, dmax

    # --- Top Row ---
    # Az
    im0 = axes[0, 0].contourf(X, Y, Az, levels=20, cmap='viridis')
    axes[0, 0].set_title("$A_z$")
    fig.colorbar(im0, ax=axes[0, 0])
        
    #skip = 7
    skip = Nx // 8 - 1
    Bpx_c, Bpy_c = clip_vectors(Bpx, Bpy)
    axes[0, 1].quiver(X[::skip, ::skip], Y[::skip, ::skip], Bpx_c[::skip, ::skip], Bpy_c[::skip, ::skip], color='red')
    axes[0, 1].set_title("$B_p$")
        
    bx_c, by_c = clip_vectors(bx, by)
    axes[0, 2].quiver(X[::skip, ::skip], Y[::skip, ::skip], bx_c[::skip, ::skip], by_c[::skip, ::skip], color='blue')
    axes[0, 2].set_title("$\hat{b}$")

    # --- Bottom Row ---
    vmin, vmax = get_vrange(K[:, :, 0, 0])
    im3 = axes[1, 0].pcolormesh(X, Y, K[:, :, 0, 0], shading='auto', cmap='magma', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("$K_{xx}$")
    fig.colorbar(im3, ax=axes[1, 0])
        
    # Kyy
    vmin, vmax = get_vrange(K[:, :, 1, 1])
    im4 = axes[1, 1].pcolormesh(X, Y, K[:, :, 1, 1], shading='auto', cmap='magma', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("$K_{yy}$")
    fig.colorbar(im4, ax=axes[1, 1])
        
    # Kxy
    v_xy = max(abs(np.min(K[:, :, 0, 1])), abs(np.max(K[:, :, 0, 1])))
    if v_xy < 1e-12: v_xy = 0.1
    im5 = axes[1, 2].pcolormesh(X, Y, K[:, :, 0, 1], shading='auto', cmap='RdBu_r', vmin=-v_xy, vmax=v_xy)
    axes[1, 2].set_title("$K_{xy}$")
    fig.colorbar(im5, ax=axes[1, 2])

    for ax in axes.flat:
        ax.set_aspect('auto')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{save_dir}/{problem}.png")
    plt.close(fig)

    return Az, psi_sep, bx, by, K

def compute_local_anisotropic_tensor(
    Ny: int, 
    Nx: int, 
    anisotropy_ratio: float,
    base_theta_pi: float,
    curv_x_pi: float,
    curv_y_pi: float,
    mag_grad_x: float,
    mag_grad_y: float,
    field_type: str = "",
    save_dir: str = "."
) -> np.ndarray:
    """
    Computes a 2D anisotropic conductivity tensor K using a local log-linear 
    parameterization of magnetic field angle and magnitude.

    The magnetic field is modeled as B = (Bp*cos(theta), Bp*sin(theta), 1.0),
    where the poloidal magnitude Bp follows an exponential profile to ensure 
    positivity and mimic physical field gradients (e.g., near null points).

    Args:
        Ny, Nx (int): Grid resolution.
        anisotropy_ratio (float): Ratio of parallel (kappa_par) to 
            perpendicular (kappa_perp) conductivity.
        base_theta_pi (float): The poloidal angle theta (pi radians) at the 
            origin (center of the patch).
        curv_x_pi, curv_y_pi (float): Spatial derivatives of the angle (d_theta/dx, 
            d_theta/dy), representing field curvature or shear.
        mag_grad_x, mag_grad_y (float): Spatial derivatives of the log-magnitude 
            (d(ln Bp)/dx, d(ln Bp)/dy), representing field strength gradients.

    Returns:
        np.ndarray: Shape (Ny, Nx, 2, 2) where K[i, j] is the 2x2 local 
            anisotropic conductivity tensor.
    """
    kappa_par = 1.0
    kappa_perp = kappa_par / anisotropy_ratio
    diff = kappa_par - kappa_perp
    
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y)
    
    Bp = 1 / (1 + np.exp(mag_grad_x * X + mag_grad_y * Y))
    theta = (base_theta_pi + (curv_x_pi * X) + (curv_y_pi * Y)) * np.pi
    bx = Bp * np.cos(theta)
    by = Bp * np.sin(theta)
    
    K = np.zeros((Ny, Nx, 2, 2))
    
    K[:, :, 0, 0] = kappa_perp + diff * bx**2
    K[:, :, 1, 1] = kappa_perp + diff * by**2
    K[:, :, 0, 1] = diff * bx * by
    K[:, :, 1, 0] = K[:, :, 0, 1]
    
    if not field_type == "":
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(fr"{field_type} Parameter: $\varphi_0$={base_theta_pi:.2f} pi, $\alpha$={curv_x_pi:.2f} pi, $\beta$={curv_y_pi:.2f} pi, $\gamma$={mag_grad_x:.2f}, $\delta$={mag_grad_y:.2f}")

        def clip_vectors(vx, vy, percentile=95):
            mag = np.sqrt(vx**2 + vy**2)
            thresh = np.percentile(mag, percentile)
            # Avoid division by zero
            scale = np.ones_like(mag)
            mask = mag > thresh
            scale[mask] = thresh / mag[mask]
            return vx * scale, vy * scale

        def get_vrange(data):
            dmin, dmax = np.min(data), np.max(data)
            if np.isclose(dmin, dmax, atol=1e-12):
                return dmin - 0.1, dmax + 0.1
            return dmin, dmax

        # --- Top Row ---
        #if field_type == "Global":
        #    skip = 7
        #else:
        #    skip = 1
        skip = Nx // 8 - 1
        bx_c, by_c = clip_vectors(bx, by)
        axes[0, 0].quiver(X[::skip, ::skip], Y[::skip, ::skip], bx_c[::skip, ::skip], by_c[::skip, ::skip], color='blue')
        axes[0, 0].set_title("$\hat{b}$")

        # --- Bottom Row ---
        vmin, vmax = get_vrange(K[:, :, 0, 0])
        im3 = axes[0, 1].pcolormesh(X, Y, K[:, :, 0, 0], shading='auto', cmap='magma', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title("$K_{xx}$")
        fig.colorbar(im3, ax=axes[0, 1])
        
        # Kyy
        vmin, vmax = get_vrange(K[:, :, 1, 1])
        im4 = axes[1, 0].pcolormesh(X, Y, K[:, :, 1, 1], shading='auto', cmap='magma', vmin=vmin, vmax=vmax)
        axes[1, 0].set_title("$K_{yy}$")
        fig.colorbar(im4, ax=axes[1, 0])
        
        # Kxy (Diverging Colormap)
        v_xy = max(abs(np.min(K[:, :, 0, 1])), abs(np.max(K[:, :, 0, 1])))
        if v_xy < 1e-12: v_xy = 0.1
        im5 = axes[1, 1].pcolormesh(X, Y, K[:, :, 0, 1], shading='auto', cmap='RdBu_r', vmin=-v_xy, vmax=v_xy)
        axes[1, 1].set_title("$K_{xy}$")
        fig.colorbar(im5, ax=axes[1, 1])

        for ax in axes.flat:
            ax.set_aspect('auto')
            ax.set_xlabel('x')
            ax.set_ylabel('y')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{save_dir}/{field_type}_parametric_{base_theta_pi:.2f}_{curv_x_pi:.2f}_{curv_y_pi:.2f}_{mag_grad_x:.2f}_{mag_grad_y:.2f}.png")
        plt.close(fig)

    return bx, by, K
