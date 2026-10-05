#!/bin/bash
#SBATCH -J offline_single
#SBATCH -p pdebug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00

root_dir=fourier_mode_harmonic_extension
resolution=8
num_mode=20
order=6

t=-0.5
cx=0
cy=0
gx=0
gy=0

echo "Running task $SLURM_ARRAY_TASK_ID: t=$t, cx=$cx, cy=$cy, gx=$gx, gy=$gy"

python3 offline.py \
    --root_dir ${root_dir} \
    --nx ${resolution} \
    --ny ${resolution} \
    --num_loc_bc_mode ${num_mode} \
    --train_anisotropy_order ${order} \
    --mean_base_theta_pi "$t" \
    --mean_curv_x_pi "$cx" \
    --mean_curv_y_pi "$cy" \
    --mean_mag_grad_x "$gx" \
    --mean_mag_grad_y "$gy" \
    #--multiply_GFEM_local_basis \
