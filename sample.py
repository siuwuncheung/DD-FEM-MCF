import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_sampling_plots(n_samples=10, anisotropy_ratio=100.0):
    kappa_par = 1.0
    kappa_perp = kappa_par / anisotropy_ratio
    diff = kappa_par - kappa_perp
    
    # Create output directory
    output_dir = "tensor_samples"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Local grid coordinates (a small patch)
    res = 50
    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(x, y)

    print(f"Generating {n_samples} local sampling plots...")

    for s in range(n_samples):
        # 1. Randomly parameterize a local magnetic field orientation
        # We simulate a field that might be rotating or uniform across the patch
        base_theta = np.random.uniform(0, 2*np.pi)
        curvature_x = np.random.uniform(-np.pi/2, np.pi/2) # How much the field twists locally
        curvature_y = np.random.uniform(-np.pi/2, np.pi/2) # How much the field twists locally
        
        theta_local = base_theta + curvature_x * X + curvature_y * Y
        
        # 2. Derive b_hat and Tensor Components
        bx = np.cos(theta_local)
        by = np.sin(theta_local)
        
        Kxx = kappa_perp + diff * bx**2
        Kyy = kappa_perp + diff * by**2
        Kxy = diff * bx * by

        # 3. Plotting
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Sample {s+1}: Base $\\theta$ = {np.degrees(base_theta):.1f}°, Curvature = ({curvature_x:.2f}, {curvature_y:.2f})", fontsize=16)

        # Subplot: b_hat direction
        skip = 5
        axes[0, 0].quiver(X[::skip, ::skip], Y[::skip, ::skip], bx[::skip, ::skip], by[::skip, ::skip], color='blue')
        axes[0, 0].set_title("Local Field Direction ($\hat{b}$)")
        
        # Subplot: Kxx
        im1 = axes[0, 1].pcolormesh(X, Y, Kxx, shading='auto', cmap='magma')
        axes[0, 1].set_title("$K_{xx}$")
        fig.colorbar(im1, ax=axes[0, 1])
        
        # Subplot: Kyy
        im2 = axes[1, 0].pcolormesh(X, Y, Kyy, shading='auto', cmap='magma')
        axes[1, 0].set_title("$K_{yy}$")
        fig.colorbar(im2, ax=axes[1, 0])
        
        # Subplot: Kxy
        # We use a symmetric colorbar for Kxy as it can be positive or negative
        v_limit = max(abs(Kxy.min()), abs(Kxy.max()), 1e-5)
        im3 = axes[1, 1].pcolormesh(X, Y, Kxy, shading='auto', cmap='RdBu_r', vmin=-v_limit, vmax=v_limit)
        axes[1, 1].set_title("$K_{xy}$")
        fig.colorbar(im3, ax=axes[1, 1])

        for ax in axes.flat:
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, f"sample_{s+1:02d}.png")
        plt.savefig(save_path)
        plt.close(fig)
        
    print(f"All samples saved to '{output_dir}/'")

if __name__ == "__main__":
    generate_sampling_plots()
