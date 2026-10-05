#!/bin/bash
#SBATCH -J offline_parallel
#SBATCH -A romdft
#SBATCH --qos=normal
#SBATCH -p pbatch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --array=0-3124%50     # 5x5x5x5x5 = 3125 total jobs. Run 50 concurrently.

root_dir=fourier_mode_harmonic_extension
resolution=64
num_mode=80
order=4

# Define the arrays
theta_vals=(-1 -0.5 0 0.5 1)
curv_x_vals=(-0.5 -0.25 0 0.25 0.5)
curv_y_vals=(-0.5 -0.25 0 0.25 0.5)
grad_x_vals=(-1 -0.5 0 0.5 1)
grad_y_vals=(-1 -0.5 0 0.5 1)

# Calculate total combinations
num_t=${#theta_vals[@]}
num_cx=${#curv_x_vals[@]}
num_cy=${#curv_y_vals[@]}
num_gx=${#grad_x_vals[@]}
num_gy=${#grad_y_vals[@]}

# Decode SLURM_ARRAY_TASK_ID into multi-dimensional indices
i_t=$(( SLURM_ARRAY_TASK_ID / (num_cx * num_cy * num_gx * num_gy) % num_t ))
i_cx=$(( SLURM_ARRAY_TASK_ID / (num_cy * num_gx * num_gy) % num_cx ))
i_cy=$(( SLURM_ARRAY_TASK_ID / (num_gx * num_gy) % num_cy ))
i_gx=$(( SLURM_ARRAY_TASK_ID / num_gy % num_gx ))
i_gy=$(( SLURM_ARRAY_TASK_ID % num_gy ))

t=${theta_vals[$i_t]}
cx=${curv_x_vals[$i_cx]}
cy=${curv_y_vals[$i_cy]}
gx=${grad_x_vals[$i_gx]}
gy=${grad_y_vals[$i_gy]}

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
