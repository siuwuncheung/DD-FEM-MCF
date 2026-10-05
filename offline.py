import os
import sys
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from medium import compute_anisotropic_tensor, compute_local_anisotropic_tensor
    from snapshot_2d import snapshot_2d_dbc
    from scalar_eigenproblem_2d import scalar_eigenproblem_2d
    from tensor_eigenproblem_2d import tensor_eigenproblem_2d
except ImportError as e:
    print(f"FATAL ERROR: Could not import a necessary module: {e}")
    print("Please ensure all translated Python files (.py) are in the same directory.")
    sys.exit(1)

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Mesh and Sampling Parameters Parser")

    parser.add_argument("--root_dir", type=str, default="local_basis", help="Root directory of sampling")

    # Fine mesh
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)

    # Sampling
    parser.add_argument("--num_realizations", type=int, default=1)
    parser.add_argument("--num_loc_bc_mode", type=int, default=10)
    parser.add_argument("--num_loc_snap", type=int, default=2)
    parser.add_argument("--train_anisotropy_order", type=int, default=6)

    parser.add_argument("--mean_base_theta_pi", type=float, default=0.0, help="Multiple of pi")
    parser.add_argument("--mean_curv_x_pi", type=float, default=0.0, help="Multiple of pi")
    parser.add_argument("--mean_curv_y_pi", type=float, default=0.0, help="Multiple of pi")
    parser.add_argument("--mean_mag_grad_x", type=float, default=0.0, help="Multiple of pi")
    parser.add_argument("--mean_mag_grad_y", type=float, default=0.0, help="Multiple of pi")

    parser.add_argument("--range_base_theta_pi", type=float, default=0.0, help="Multiple of pi")
    parser.add_argument("--range_curv_x_pi", type=float, default=0.0, help="Multiple of pi")
    parser.add_argument("--range_curv_y_pi", type=float, default=0.0, help="Multiple of pi")
    parser.add_argument("--range_mag_grad_x", type=float, default=0.0, help="Multiple of pi")
    parser.add_argument("--range_mag_grad_y", type=float, default=0.0, help="Multiple of pi")

    parser.add_argument("--multiply_GFEM_local_basis", action="store_true")

    args = parser.parse_args()

    return args

def plot_loc_field(loc_field_vec, ny, nx, idx, samp_dir, tag, colorbar=True, axis=False):
    loc_field_2d = loc_field_vec.reshape(2*ny + 1, 2*nx + 1, order='F')
    plt.figure()
    plt.imshow(loc_field_2d, origin='lower', cmap='jet')
    if colorbar:
        plt.colorbar()
    if not axis:
        plt.axis('off')
    plt.savefig(f'{samp_dir}/{tag}_{idx}_{nx}.png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

params = get_args()

# Fine mesh
nx = params.nx
ny = params.ny

# Sampling
num_realizations = params.num_realizations
num_loc_bc_mode = params.num_loc_bc_mode
num_loc_snap = params.num_loc_snap
train_anisotropy_order = params.train_anisotropy_order

mean_base_theta_pi = params.mean_base_theta_pi
mean_curv_x_pi = params.mean_curv_x_pi
mean_curv_y_pi = params.mean_curv_y_pi
mean_mag_grad_x = params.mean_mag_grad_x
mean_mag_grad_y = params.mean_mag_grad_y

range_base_theta_pi = params.range_base_theta_pi
range_curv_x_pi = params.range_curv_x_pi
range_curv_y_pi = params.range_curv_y_pi
range_mag_grad_x = params.range_mag_grad_x
range_mag_grad_y = params.range_mag_grad_y

multiply_GFEM_local_basis = params.multiply_GFEM_local_basis

root_dir = params.root_dir

np.random.seed(42)

samp_dir = (
    f"{root_dir}/anis{train_anisotropy_order}/"
    f"nx{nx}_ny{ny}_real{num_realizations}_bc{num_loc_bc_mode}_snap{num_loc_snap}/"
    f"rbt{range_base_theta_pi:.2f}_rcx{range_curv_x_pi:.2f}_rcy{range_curv_y_pi:.2f}_"
    f"rmgx{range_mag_grad_x:.2f}_rmgy{range_mag_grad_y:.2f}/"
    f"mbt{mean_base_theta_pi:.2f}_mcx{mean_curv_x_pi:.2f}_mcy{mean_curv_y_pi:.2f}_"
    f"mmgx{mean_mag_grad_x:.2f}_mmgy{mean_mag_grad_y:.2f}"
)
os.makedirs(samp_dir, exist_ok=True)

a_loc_list = []
for k in range(num_realizations):
    base_theta_pi = mean_base_theta_pi + np.random.uniform(-range_base_theta_pi, range_base_theta_pi)
    curv_x_pi = mean_curv_x_pi + np.random.uniform(-range_curv_x_pi, range_curv_x_pi)
    curv_y_pi = mean_curv_y_pi + np.random.uniform(-range_curv_y_pi, range_curv_y_pi)
    mag_grad_x = mean_mag_grad_x + np.random.uniform(-range_mag_grad_x, range_mag_grad_x)
    mag_grad_y = mean_mag_grad_y + np.random.uniform(-range_mag_grad_y, range_mag_grad_y)
    bx_loc_realizaiton, by_loc_realization, a_loc_realization = compute_local_anisotropic_tensor(
        Ny=2*ny, 
        Nx=2*nx, 
        anisotropy_ratio=10.0**train_anisotropy_order, 
        base_theta_pi=base_theta_pi, 
        curv_x_pi=curv_x_pi, 
        curv_y_pi=curv_y_pi,
        mag_grad_x=mag_grad_x, 
        mag_grad_y=mag_grad_y,
        field_type="Snapshot",
        save_dir=samp_dir
    )
    a_loc_list.append(a_loc_realization)
a_loc = np.stack(a_loc_list, axis=4)

print(f'a_loc.shape = {a_loc.shape}')

print('Computing Local Snapshot')
loc_snapshots = snapshot_2d_dbc(a_loc, nx, ny,  num_loc_snap, num_loc_bc_mode)
print(f'Local snapshots shape = {loc_snapshots.shape}')
np.savez(f'{samp_dir}/loc_snapshots.npz', loc_snapshots=loc_snapshots)
print(f'Saved snapshots to {samp_dir}')

for k in range(loc_snapshots.shape[1]):
    plot_loc_field(loc_snapshots[:,k], ny, nx, k, samp_dir, 'snap')

print('Forming Local Basis')
bx_loc_sp, by_loc_sp, a_loc_sp = compute_local_anisotropic_tensor(
    Ny=2*ny, 
    Nx=2*nx, 
    anisotropy_ratio=10.0**train_anisotropy_order, 
    base_theta_pi=mean_base_theta_pi, 
    curv_x_pi=mean_curv_x_pi, 
    curv_y_pi=mean_curv_y_pi,
    mag_grad_x=mean_mag_grad_x, 
    mag_grad_y=mean_mag_grad_y,
    field_type="Spectral",
    save_dir=samp_dir
)
loc_basis = tensor_eigenproblem_2d(a_loc_sp, nx, ny, loc_snapshots, multiply_GFEM_local_basis)
#loc_basis = scalar_eigenproblem_2d(np.ones((2*ny, 2*nx)), nx, ny, loc_snapshots)
np.save(f'{samp_dir}/loc_basis.npy', loc_basis)

for k in range(loc_basis.shape[1]):
    plot_loc_field(loc_basis[:,k], ny, nx, k, samp_dir, 'basis')

print(f'Saved spectral basis to {samp_dir}')
